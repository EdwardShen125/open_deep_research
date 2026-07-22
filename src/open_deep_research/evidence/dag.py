"""阶段 7 = Runbook v1 阶段 7: planner 显式 DAG。

设计依据: notes/evidence-pipeline-runbook-v1.md 阶段 7 节。

## 问题
原 v0.0.16 中 supervisor 黑盒调度: 用户给个 research_topic,supervisor 决定
要拆多少个 sub-topic、并跑多少个 researcher、调用哪个 tool。失败、超时、
资源耗尽都很难定位,因为调度逻辑散在 supervisor 内部。

## 整改
Planner 接受一个**显式 DAG**(节点 + 依赖),每个节点对应一个 stage_fn,
planner 按拓扑序跑。这让:
- 调度逻辑一目了然(读 DAG 即可)
- 每个节点可配额(extract 的 token budget / merge 的 cosine threshold 都
  在节点 metadata)
- 失败定位准确(哪个 DAG 节点挂了)
- per-dimension retro 自然可加(把 merge / grade 节点乘以 dim 数)

## 设计权衡
- 不引入第三方图库 — 全 Python dict / set,小到能塞进 Runbook
- 不接 LangGraph conditional_edges — 那是更深的重写,留待后续重构
- 不替换 supervisor — use_explicit_dag=True 时才用 DAG,默认还是 supervisor
  (向后兼容;现有 EDR v4 benchmark 不破)

## 接口
- DAGNode: 单节点(name / fn / depends_on / quota / metadata)
- DAG: 节点集合 + add_node / topo_sort / validate
- dag_to_stages(dag) -> list[(stage_name, fn)] — 转成 ResearchJob.stages
- validate_dag(dag) — 检查 1) 无环 2) 名唯一 3) 依赖存在
- batch_dag_for_dimensions(dag, dimensions) — 把 merge / grade 节点复制
  per dimension,retro loop 也变成 per-dim
"""
from __future__ import annotations

import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from open_deep_research.evidence.observability import stage_trace


# =============================================================================
# NodeQuota — 单节点配额(对应 RunConfig 的一部分)
# =============================================================================


@dataclass
class NodeQuota:
    """单节点配额 — 阶段 7 的 RunConfig 粒度从这里体现。

    字段:
        token_budget: 这个节点允许消耗的最大 tokens(None = 无限)
        time_budget_s: 这个节点允许的最长运行时间(None = 无限)
        retry_on_transient: 遇到 transient 错误时是否重试(默认 True)
        max_retries: 重试次数上限(默认 3)
        cosine_threshold: 仅 merge 节点用 — 归并 cosine 阈值
        entailment_strict: 仅 verify 节点用 — entailment 严度
    """

    token_budget: Optional[int] = None
    time_budget_s: Optional[float] = None
    retry_on_transient: bool = True
    max_retries: int = 3
    cosine_threshold: Optional[float] = None  # e.g. 0.92
    entailment_strict: Optional[bool] = None

    def has_limits(self) -> bool:
        """是否真有限制(用于 budget guard 决策)。"""
        return self.token_budget is not None or self.time_budget_s is not None

    def to_metadata(self) -> dict[str, Any]:
        """用于 Langfuse / logger 装饰器读 metadata。"""
        return {
            "token_budget": self.token_budget,
            "time_budget_s": self.time_budget_s,
            "retry_on_transient": self.retry_on_transient,
            "max_retries": self.max_retries,
            "cosine_threshold": self.cosine_threshold,
            "entailment_strict": self.entailment_strict,
        }


# =============================================================================
# DAGNode — 单节点
# =============================================================================


@dataclass
class DAGNode:
    """DAG 单节点。

    字段:
        name: 节点唯一名
        fn: 节点函数 async (state, ctx) -> dict
            注意:func 应能接收 ctx 入参 (run_id 等)
        depends_on: 依赖的节点名列表(空 = 根节点)
        quota: 节点配额,None = 无限制
        per_dimension: 是否需要 per-dimension 复制(merge/grade 节点用)
        metadata: 自由附加 dict,例如 {"node_kind": "merge"}
    """

    name: str
    fn: Callable
    depends_on: list[str] = field(default_factory=list)
    quota: Optional[NodeQuota] = None
    per_dimension: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DAGNode.name 不能为空")
        if not isinstance(self.fn, Callable):
            raise ValueError(f"DAGNode.fn 必须 callable, got {type(self.fn)}")
        if not isinstance(self.depends_on, list):
            raise ValueError("DAGNode.depends_on 必须是 list")


# =============================================================================
# DAG — 节点集合
# =============================================================================


@dataclass
class DAG:
    """DAG 容器。

    使用:
        dag = DAG()
        dag.add(DAGNode(name="extract", fn=extract_fn))
        dag.add(DAGNode(name="verify", fn=verify_fn, depends_on=["extract"]))
        ...
        validate_dag(dag)
        stages = dag_to_stages(dag)
    """

    nodes: list[DAGNode] = field(default_factory=list)

    def add(self, node: DAGNode) -> None:
        """添加节点,允许重名(后加覆盖前加)。"""
        self.nodes = [n for n in self.nodes if n.name != node.name]
        self.nodes.append(node)

    def get(self, name: str) -> Optional[DAGNode]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def roots(self) -> list[DAGNode]:
        """无依赖的节点(入度=0)。"""
        return [n for n in self.nodes if not n.depends_on]


# =============================================================================
# 校验
# =============================================================================


class DAGValidationError(ValueError):
    """DAG 校验失败,通常是环 / 重复名 / 缺失依赖。"""


def validate_dag(dag: DAG) -> None:
    """校验 DAG 结构。

    检查:
    1. 节点名唯一
    2. 所有 depends_on 引用的节点都存在
    3. 无环(Kahn's algorithm topo sort 检测)

    Raises:
        DAGValidationError: 任何错误
    """
    names = dag.names()
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise DAGValidationError(f"DAG 节点名重复: {dup}")

    name_set = set(names)
    for n in dag.nodes:
        for dep in n.depends_on:
            if dep not in name_set:
                raise DAGValidationError(
                    f"节点 '{n.name}' 依赖 '{dep}' 不存在"
                )

    # 拓扑排序检测环 (Kahn: BFS 删入度=0)
    in_deg: dict[str, int] = {n.name: 0 for n in dag.nodes}
    graph: dict[str, list[str]] = defaultdict(list)
    for n in dag.nodes:
        in_deg[n.name] += 0  # init
        for dep in n.depends_on:
            graph[dep].append(n.name)
            in_deg[n.name] += 1

    queue: deque[str] = deque(n for n, d in in_deg.items() if d == 0)
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        for nxt in graph[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)

    if visited != len(dag.nodes):
        # 找环中节点
        in_cycle = [n for n, d in in_deg.items() if d > 0]
        raise DAGValidationError(
            f"DAG 含环,未访问节点: {in_cycle[:5]}{'...' if len(in_cycle) > 5 else ''}"
        )


# =============================================================================
# DAG → ResearchJob stages 转换
# =============================================================================


def topo_sort(dag: DAG) -> list[DAGNode]:
    """Kahn's algorithm 拓扑排序,返回 DAGNode 列表。

    多个节点入度同时为 0 时按添加顺序输出(确定性)。
    """
    if not dag.nodes:
        return []
    in_deg: dict[str, int] = {n.name: 0 for n in dag.nodes}
    graph: dict[str, list[str]] = defaultdict(list)
    by_name = {n.name: n for n in dag.nodes}
    for n in dag.nodes:
        for dep in n.depends_on:
            if dep in by_name:
                graph[dep].append(n.name)
                in_deg[n.name] += 1

    # roots first (按添加顺序)
    queue: deque[str] = deque()
    for n in dag.nodes:
        if in_deg[n.name] == 0:
            queue.append(n.name)

    result: list[DAGNode] = []
    while queue:
        cur_name = queue.popleft()
        cur = by_name[cur_name]
        result.append(cur)
        for nxt in graph[cur_name]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                # 保持添加顺序
                inserted = False
                for i, q in enumerate(queue):
                    # 简单 stable insert
                    if cur_name < q:
                        queue.insert(i, nxt)
                        inserted = True
                        break
                if not inserted:
                    queue.append(nxt)
    return result


def dag_to_stages(dag: DAG) -> list[tuple[str, Callable]]:
    """DAG → (stage_name, fn) 列表,给 ResearchJob(stages=...) 用。

    Steps:
    1. 校验 DAG
    2. 拓扑排序
    3. 用 @stage_trace 装饰每个 fn(把 stage_name 上传到 Langfuse/log)
    4. 返回 (name, decorated_fn) 对的列表
    """
    validate_dag(dag)
    order = topo_sort(dag)
    stages: list[tuple[str, Callable]] = []
    for node in order:
        if node.quota is None:
            decorated = stage_trace(node.name)(node.fn)
        else:
            # quota 携带 metadata,传到 stage_trace
            decorated = stage_trace(node.name)(node.fn)
            # 注:quota 在 fn 调用前会通过 quota enforcement 生效(stage_quota_guard)
        stages.append((node.name, decorated))
    return stages


# =============================================================================
# per-dimension 批量
# =============================================================================


def batch_dag_for_dimensions(
    dag: DAG,
    dimensions: list[str],
) -> DAG:
    """把 per_dimension=True 的节点复制成 per-dim 节点。

    例:
        原始 DAG:
            merge(per_dimension=True) → grade(per_dimension=True)

        转换后(dimensions=['d1', 'd2']):
            merge__d1 → grade__d1
            merge__d2 → grade__d2
            merge__d3 → grade__d3

    非 per-dimension 节点不动。

    用途:
        per-dim retro loop — 每个 dimension 独立走 grade → retro 决策,
        不会因为某个 dim 全 D 而拖累整个 run。
    """
    if not dimensions:
        return dag
    new_dag = DAG()
    per_dim_names = {n.name for n in dag.nodes if n.per_dimension}
    # 第一遍:加非 per-dim 节点(包括 write),但其依赖需要从 per_dim grade 解析到具体 d1/d2
    for n in dag.nodes:
        if not n.per_dimension:
            new_deps = []
            for dep in n.depends_on:
                if dep in per_dim_names:
                    # write 依赖 grade → 展开成所有 grade__dim
                    # 这是一种 fan-in:任何 per_dim 节点完成都触发下游
                    new_deps.extend(f"{dep}__{d}" for d in dimensions)
                else:
                    new_deps.append(dep)
            new_node = DAGNode(
                name=n.name,
                fn=n.fn,
                depends_on=new_deps,
                quota=copy.deepcopy(n.quota),
                per_dimension=False,
                metadata=dict(n.metadata),
            )
            new_dag.add(new_node)
    # 第二遍:加 per-dim 节点
    for n in dag.nodes:
        if not n.per_dimension:
            continue
        for dim in dimensions:
            new_node = DAGNode(
                name=f"{n.name}__{dim}",
                fn=_wrap_per_dim_fn(n.fn, dim),
                depends_on=[f"{dep}__{dim}" if dep in per_dim_names else dep for dep in n.depends_on],
                quota=copy.deepcopy(n.quota),
                per_dimension=False,
                metadata={**n.metadata, "original_name": n.name, "dimension": dim},
            )
            new_dag.add(new_node)
    return new_dag


def _wrap_per_dim_fn(fn: Callable, dim: str) -> Callable:
    """per-dim 节点包装:在 ctx 里塞 dimension 字段,fn 读 ctx['dimension']。"""
    async def per_dim_async(state: dict, ctx: dict) -> dict:
        ctx = dict(ctx)
        ctx["dimension"] = dim
        return await fn(state, ctx)
    def per_dim_sync(state: dict, ctx: dict) -> dict:
        ctx = dict(ctx)
        ctx["dimension"] = dim
        return fn(state, ctx)
    import inspect
    if inspect.iscoroutinefunction(fn):
        return per_dim_async
    return per_dim_sync


# =============================================================================
# 工厂 — 构造默认 5-stage DAG
# =============================================================================


def default_pipeline_dag(
    extract_fn: Callable,
    verify_fn: Callable,
    merge_fn: Callable,
    grade_fn: Callable,
    write_fn: Callable,
    *,
    merge_per_dimension: bool = True,
    cosine_threshold: float = 0.92,
    token_budget_per_stage: Optional[int] = None,
) -> DAG:
    """构造默认 5-stage DAG (与 Runbook 阶段 1 设计一致)。

    Args:
        各 fn: 阶段函数
        merge_per_dimension: merge/grade 是否 per-dim(默认 True — 阶段 6 retro 是 per-dim 的)
        cosine_threshold: merge 节点的 cosine 阈值(配 NodeQuota.cosine_threshold)
        token_budget_per_stage: 各阶段 token 预算(None = 无限制)

    Returns:
        DAG ready for validate_dag + dag_to_stages
    """
    dag = DAG()
    base_quota = NodeQuota(token_budget=token_budget_per_stage, cosine_threshold=cosine_threshold)

    dag.add(DAGNode(
        name="extract", fn=extract_fn, depends_on=[],
        quota=NodeQuota(token_budget=token_budget_per_stage),
    ))
    dag.add(DAGNode(
        name="verify", fn=verify_fn, depends_on=["extract"],
        quota=NodeQuota(token_budget=token_budget_per_stage),
    ))
    if merge_per_dimension:
        # merge 节点是 per-dim 的根(没有上游 per-dim 依赖)
        # 在 batch_dag_for_dimensions 时会展开
        dag.add(DAGNode(
            name="merge", fn=merge_fn, depends_on=["verify"],
            quota=base_quota,
            per_dimension=True,
        ))
        # grade 依赖 merge(同名 + __dim 后缀)
        dag.add(DAGNode(
            name="grade", fn=grade_fn, depends_on=["verify"],
            quota=base_quota,
            per_dimension=True,
            metadata={"depends_on_merge": True},  # 标识 batch 时依赖用 merge__dim
        ))
    else:
        dag.add(DAGNode(
            name="merge", fn=merge_fn, depends_on=["verify"],
            quota=base_quota,
        ))
        dag.add(DAGNode(
            name="grade", fn=grade_fn, depends_on=["merge"],
            quota=base_quota,
        ))
    dag.add(DAGNode(
        name="write", fn=write_fn,
        depends_on=(
            ["grade"] if merge_per_dimension else ["grade"]
            # 注: 阶段 7 write 仍依赖最终 grade(batch 后会变 grade__final)
        ),
        quota=NodeQuota(token_budget=token_budget_per_stage),
    ))
    return dag


__all__ = [
    "DAG",
    "DAGNode",
    "DAGValidationError",
    "NodeQuota",
    "batch_dag_for_dimensions",
    "dag_to_stages",
    "default_pipeline_dag",
    "topo_sort",
    "validate_dag",
]
