"""阶段 7 = Runbook v1 阶段 7: planner 显式 DAG + RunConfig 测试覆盖。

Mock-DB 离线。覆盖:
1. DAG 基础: add/get/roots/duplicate-name validation
2. validate_dag: 环检测 / 缺失依赖 / 重复名
3. topo_sort: 确定性 + 顺序保留
4. dag_to_stages + ResearchJob wiring
5. NodeQuota metadata
6. batch_dag_for_dimensions: per-dim 节点展开
7. default_pipeline_dag
8. per-dim retro loop: 独立决策 per dim

不覆盖:
- 真 PG 集成 (pgvector 缺失,阶段 7 一起改 docker)
- supervisor 实际改造 (那是 LangGraph 重构,本阶段不引入)
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any
from uuid import uuid4

import pytest

from open_deep_research.evidence import (
    DAG,
    DAGNode,
    DAGValidationError,
    NodeQuota,
    ResearchJob,
    STAGES,
    batch_dag_for_dimensions,
    dag_to_stages,
    default_pipeline_dag,
    detect_grade_d_pct,
    run_with_per_dim_retro,
    should_retry,
    topo_sort,
    validate_dag,
)
from open_deep_research.evidence import checkpoint as ckpt_mod


# =============================================================================
# Mock CheckpointDAO
# =============================================================================


class MockCheckpointDAO:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict] = {}

    def __enter__(self) -> "MockCheckpointDAO":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def upsert(
        self,
        run_id: str | Any,
        stage: str,
        *,
        status: str,
        payload: dict | None = None,
    ) -> None:
        self._store[(str(run_id), stage)] = {
            "run_id": str(run_id),
            "stage": stage,
            "status": status,
            "payload": payload or {},
        }

    def get(self, run_id: str | Any, stage: str) -> dict | None:
        return self._store.get((str(run_id), stage))


@pytest.fixture(autouse=True)
def _inject_mock_dao():
    mock = MockCheckpointDAO()
    prev = ckpt_mod._dao_override
    ckpt_mod.set_dao_override(mock)
    yield mock
    ckpt_mod.set_dao_override(prev)


# 简单 async fn
async def _noop(s, c): return {}
async def _pass(s, c): return None
async def _make(s, c): return {"made": True}


# =============================================================================
# 1. DAG 基础
# =============================================================================


class TestDAGBasics:
    def test_empty_dag(self):
        dag = DAG()
        assert dag.names() == []
        assert dag.roots() == []

    def test_add_node(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        assert "a" in dag.names()

    def test_add_overwrites_duplicate(self):
        """add 同名节点会覆盖前一个(确定性)。"""
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop, metadata={"v": 1}))
        dag.add(DAGNode(name="a", fn=_noop, metadata={"v": 2}))
        assert dag.get("a").metadata["v"] == 2

    def test_get_returns_none_for_missing(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        assert dag.get("missing") is None

    def test_roots(self):
        """roots 是入度 = 0 的节点(无依赖)。"""
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop))
        dag.add(DAGNode(name="c", fn=_noop, depends_on=["a", "b"]))
        root_names = sorted(n.name for n in dag.roots())
        assert root_names == ["a", "b"]

    def test_node_name_required(self):
        with pytest.raises(ValueError, match="name 不能为空"):
            DAGNode(name="", fn=_noop)

    def test_node_fn_required(self):
        with pytest.raises(ValueError, match="必须 callable"):
            DAGNode(name="a", fn="not callable")


# =============================================================================
# 2. validate_dag: 环 / 缺失依赖 / 重复名
# =============================================================================


class TestValidateDAG:
    def test_valid_dag(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        validate_dag(dag)  # 不抛

    def test_self_dependency_cycle(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop, depends_on=["a"]))
        with pytest.raises(DAGValidationError, match="含环"):
            validate_dag(dag)

    def test_three_node_cycle(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        dag.add(DAGNode(name="c", fn=_noop, depends_on=["b"]))
        # 注:add 会去重同名,所以 a 重 add with dep c 创建 a→c→b→a 环
        dag.add(DAGNode(name="a", fn=_noop, depends_on=["c"]))
        with pytest.raises(DAGValidationError):
            validate_dag(dag)

    def test_missing_dependency(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop, depends_on=["nonexistent"]))
        with pytest.raises(DAGValidationError, match="不存在"):
            validate_dag(dag)


# =============================================================================
# 3. topo_sort
# =============================================================================


class TestTopoSort:
    def test_linear_chain(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        dag.add(DAGNode(name="c", fn=_noop, depends_on=["b"]))
        order = topo_sort(dag)
        assert [n.name for n in order] == ["a", "b", "c"]

    def test_diamond_dependency(self):
        """a → {b, c} → d"""
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        dag.add(DAGNode(name="c", fn=_noop, depends_on=["a"]))
        dag.add(DAGNode(name="d", fn=_noop, depends_on=["b", "c"]))
        order = topo_sort(dag)
        names = [n.name for n in order]
        assert names[0] == "a"
        assert names[-1] == "d"
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_empty_dag(self):
        assert topo_sort(DAG()) == []

    def test_two_roots_kept_in_insertion_order(self):
        """稳定排序:同入度的节点按添加顺序输出。"""
        dag = DAG()
        dag.add(DAGNode(name="z", fn=_noop))
        dag.add(DAGNode(name="a", fn=_noop))
        order = topo_sort(dag)
        # z 先 add,应排在 a 前
        assert [n.name for n in order] == ["z", "a"]


# =============================================================================
# 4. dag_to_stages + ResearchJob wiring
# =============================================================================


class TestDAGToStages:
    def test_dag_to_stages_returns_pairs(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        stages = dag_to_stages(dag)
        assert len(stages) == 2
        assert stages[0][0] == "a"
        assert stages[1][0] == "b"

    def test_stages_wire_to_researchjob(self):
        dag = default_pipeline_dag(_noop, _pass, _make, _noop, _noop)
        stages = dag_to_stages(dag)
        job = ResearchJob(stages=stages)
        assert len(job.stages) == 5

    def test_validate_called_inside_dag_to_stages(self):
        """dag_to_stages 调用时自动 validate。"""
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop, depends_on=["missing"]))
        with pytest.raises(DAGValidationError):
            dag_to_stages(dag)

    @pytest.mark.asyncio
    async def test_end_to_end_dag_runs(self, _inject_mock_dao):
        """DAG → stages → ResearchJob.run: 端到端跑通。

        必须用 5 节点的 default_pipeline_dag(因为 STAGES 元组 = 5 个,
        ResearchJob.is_run_complete 会校验)。
        """
        async def noop(s, c): return None
        async def make(s, c): return {"v": 1}
        dag = default_pipeline_dag(noop, noop, make, noop, noop)
        stages = dag_to_stages(dag)
        job = ResearchJob(stages=stages)

        rid = str(uuid4())
        state = await job.run(rid, {"count": 0})
        assert state["v"] == 1


# =============================================================================
# 5. NodeQuota
# =============================================================================


class TestNodeQuota:
    def test_default_no_limits(self):
        q = NodeQuota()
        assert q.token_budget is None
        assert q.time_budget_s is None
        assert q.has_limits() is False

    def test_with_token_budget(self):
        q = NodeQuota(token_budget=10_000)
        assert q.has_limits() is True

    def test_with_time_budget(self):
        q = NodeQuota(time_budget_s=60.0)
        assert q.has_limits() is True

    def test_metadata_for_langfuse(self):
        q = NodeQuota(token_budget=5000, cosine_threshold=0.85)
        md = q.to_metadata()
        assert md["token_budget"] == 5000
        assert md["cosine_threshold"] == 0.85
        assert md["retry_on_transient"] is True

    def test_quota_attached_to_dag_node(self):
        dag = DAG()
        quota = NodeQuota(token_budget=1000, cosine_threshold=0.9)
        dag.add(DAGNode(name="merge", fn=_make, quota=quota))
        node = dag.get("merge")
        assert node.quota.cosine_threshold == 0.9


# =============================================================================
# 6. batch_dag_for_dimensions
# =============================================================================


class TestBatchDAGForDimensions:
    def test_per_dim_false_unchanged(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop))
        dag.add(DAGNode(name="b", fn=_noop, depends_on=["a"]))
        batched = batch_dag_for_dimensions(dag, ["d1", "d2"])
        assert sorted(batched.names()) == ["a", "b"]

    def test_per_dim_true_duplicated(self):
        """per_dim 节点被复制成 N 个,依赖链正确。"""
        dag = DAG()
        dag.add(DAGNode(name="extract", fn=_noop))
        dag.add(DAGNode(name="verify", fn=_noop, depends_on=["extract"]))
        dag.add(DAGNode(name="merge", fn=_noop, depends_on=["verify"], per_dimension=True))
        dag.add(DAGNode(name="grade", fn=_noop, depends_on=["merge"], per_dimension=True))
        batched = batch_dag_for_dimensions(dag, ["d1", "d2"])
        names = set(batched.names())
        # extract / verify 保留
        assert "extract" in names
        assert "verify" in names
        # merge / grade 被展开
        assert "merge__d1" in names and "merge__d2" in names
        assert "grade__d1" in names and "grade__d2" in names
        # 原 per-dim 名不应保留
        assert "merge" not in names
        assert "grade" not in names

    def test_empty_dimensions_returns_unchanged(self):
        dag = DAG()
        dag.add(DAGNode(name="a", fn=_noop, per_dimension=True))
        batched = batch_dag_for_dimensions(dag, [])
        assert batched.names() == ["a"]

    def test_per_dim_deps_renamed(self):
        """batch 后 per-dim grade 的依赖应是 {merge}__dim(不是 'merge')。"""
        dag = DAG()
        dag.add(DAGNode(name="merge", fn=_noop, per_dimension=True))
        dag.add(DAGNode(name="grade", fn=_noop, depends_on=["merge"], per_dimension=True))
        batched = batch_dag_for_dimensions(dag, ["d1"])
        grade = batched.get("grade__d1")
        # 检查依赖被改写
        assert "merge__d1" in grade.depends_on


# =============================================================================
# 7. default_pipeline_dag
# =============================================================================


class TestDefaultPipelineDAG:
    def test_5_nodes(self):
        dag = default_pipeline_dag(_noop, _pass, _make, _noop, _noop)
        assert len(dag.nodes) == 5
        assert dag.names() == ["extract", "verify", "merge", "grade", "write"]

    def test_topological_order(self):
        dag = default_pipeline_dag(_noop, _pass, _make, _noop, _noop)
        order = topo_sort(dag)
        assert [n.name for n in order] == ["extract", "verify", "merge", "grade", "write"]

    def test_merge_per_dimension_flag(self):
        """merge_per_dimension=True → merge/grade 节点带 per_dimension=True"""
        dag_no = default_pipeline_dag(_noop, _pass, _make, _noop, _noop, merge_per_dimension=False)
        dag_yes = default_pipeline_dag(_noop, _pass, _make, _noop, _noop, merge_per_dimension=True)
        assert dag_no.get("merge").per_dimension is False
        assert dag_yes.get("merge").per_dimension is True


# =============================================================================
# 8. per-dim retro
# =============================================================================


class _Claim:
    def __init__(self, grade: str) -> None:
        self.grade = grade


class TestPerDimRetro:
    @pytest.mark.asyncio
    async def test_no_dim_retro_when_all_good(self, _inject_mock_dao):
        """所有 dimension 都达标 → 不重试, status=ok。"""
        async def extract(s, c): return {"e": 1}
        async def verify(s, c): return None
        async def merge_per_dim(s, c):
            # c['dimension'] 标识是哪个 dim
            dim = c.get("dimension", "?")
            s[f"claims__{dim}"] = [_Claim("A"), _Claim("B"), _Claim("A")]
            return s
        async def grade_per_dim(s, c):
            dim = c.get("dimension", "?")
            return {}
        async def write(s, c):
            return {"__body__": "# body"}

        # 构造 2-dim DAG
        dag = default_pipeline_dag(extract, verify, merge_per_dim, grade_per_dim, write, merge_per_dimension=True)
        batched = batch_dag_for_dimensions(dag, ["d1", "d2"])
        stages = dag_to_stages(batched)
        job = ResearchJob(stages=stages)

        result = await run_with_per_dim_retro(
            job, str(uuid4()), {"research_brief": "x"},
            dimensions=["d1", "d2"],
            threshold=0.5,
        )
        assert result.status == "ok"
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_one_dim_bad_triggers_retry(self, _inject_mock_dao):
        """d1 bad, d2 OK → 只重试 d1,最终若 d1 也修好 → ok。"""
        run_counts = {"merge__d1": 0, "merge__d2": 0}

        async def extract(s, c): return {}
        async def verify(s, c): return None
        async def merge_per_dim(s, c):
            dim = c.get("dimension")
            run_counts[f"merge__{dim}"] = run_counts.get(f"merge__{dim}", 0) + 1
            if dim == "d1":
                # 第一次跑全 D, 第二次跑全 A
                if run_counts[f"merge__{dim}"] == 1:
                    s["claims__d1"] = [_Claim("D"), _Claim("D"), _Claim("D")]
                else:
                    s["claims__d1"] = [_Claim("A"), _Claim("A"), _Claim("A")]
            else:
                s["claims__d2"] = [_Claim("A"), _Claim("A"), _Claim("B")]
            return s
        async def grade_per_dim(s, c): return {}
        async def write(s, c):
            return {"__body__": "# body"}

        dag = default_pipeline_dag(extract, verify, merge_per_dim, grade_per_dim, write, merge_per_dimension=True)
        batched = batch_dag_for_dimensions(dag, ["d1", "d2"])
        stages = dag_to_stages(batched)
        job = ResearchJob(stages=stages)

        rid = str(uuid4())
        result = await run_with_per_dim_retro(
            job, rid, {}, dimensions=["d1", "d2"], threshold=0.5, max_retries=3,
        )

        # d1 重试过一次, d2 没动
        assert run_counts["merge__d1"] == 2
        assert run_counts["merge__d2"] == 1
        assert result.status in ("ok", "fallback_used")
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_failed_when_no_dim_improves(self, _inject_mock_dao):
        """retro 用尽仍不达标 → failed。"""
        async def extract(s, c): return {}
        async def verify(s, c): return None
        async def merge_per_dim(s, c):
            dim = c.get("dimension")
            s[f"claims__{dim}"] = [_Claim("D"), _Claim("D"), _Claim("D"), _Claim("A")]
            return s
        async def grade_per_dim(s, c): return {}
        async def write(s, c): return {}

        dag = default_pipeline_dag(extract, verify, merge_per_dim, grade_per_dim, write, merge_per_dimension=True)
        batched = batch_dag_for_dimensions(dag, ["d1"])
        stages = dag_to_stages(batched)
        job = ResearchJob(stages=stages)

        result = await run_with_per_dim_retro(
            job, str(uuid4()), {}, dimensions=["d1"], threshold=0.5, max_retries=1,
        )

        assert result.status == "failed"
        assert result.ok is False