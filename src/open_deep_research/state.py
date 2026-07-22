"""Graph state definitions and data structures for the Deep Research agent.

Phase 3 (= Runbook v1 阶段 1.3) 变更:
- `raw_notes` 字段从 AgentState / SupervisorState / ResearcherState 中删除。
  理由:Runbook 1.3 "state 只保留引用,不保留内容"。原 raw_notes 持有
  23K+ 字符的聚合文本,直接导致 supervisor state 序列化超 50KB(验收 3)。
- 新增 `eu_counts: dict[str, int]` / `claim_counts: dict[str, int]`,
  让 supervisor 不再持有 EU 内容,只持有 dimension_id → 计数 的引用。
- `dimension_ids: list[str]` 让 supervisor 知道本 run 跑了哪些 dimension
  (阶段 3 归并 / 阶段 7 planner DAG 都要用)。
- `compressed_research: str` 保留(decision D2-B:阶段 1 staged 削除)。
  阶段 4 job 化时再彻底删除。
- 字段 `evidence_units` 保留但标注 DEPRECATED,阶段 4 才删。原因:
  supervisor 现有的 O(n) dedup 在 EU 进 PG 前需要一份 in-memory 副本
  (避免边读 PG 边 dedup 的开销)。阶段 4 改为去 PG 读 + checkpoint。
"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""

    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data.

    Plan v2 / Phase 3 fields:
      - evidence_units : LEGACY in-memory EU pool (DEPRECATED, 阶段 4 删除)
      - eu_counts      : {dimension_id: count} (阶段 1 新增,Phase 3 唯一引用)
      - claim_counts   : {grade: count} (阶段 3 归并后回填)
      - dimension_ids  : [str] 本 run 的 dimension 列表
      - cited_report   : writer's structured claim↔EU output
      - verification   : verifier engine output (rules 1/2/3/C)
      - url_compliance : Rule 4 audit (page-level URL enforcement)
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    # Phase 3: raw_notes 已删 (decision D2-B)
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    # Phase 7 (= Runbook v1 阶段 5.2): 结构化结果
    # - ok: 硬信号 (True / False), 调用方只看这个字段
    # - status: 详细分类 (ok / partial / fallback_used / failed)
    # - failures: 失败列表供诊断
    report_result: Optional[dict] = None
    evidence_units: Annotated[list, override_reducer] = []
    cited_report: Optional[dict] = None
    verification: Optional[dict] = None
    url_compliance: Annotated[list, override_reducer] = []  # noqa: E501
    # Phase 3 新增:state 瘦身后的引用层
    eu_counts: Annotated[dict[str, int], override_reducer] = {}
    claim_counts: Annotated[dict[str, int], override_reducer] = {}
    dimension_ids: Annotated[list[str], override_reducer] = []


class SupervisorState(TypedDict):

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    # Phase 2 / supervisor-convergence: explicit operator.add reducer so the
    # counter accumulates across supervisor→supervisor_tools loops. Without
    # this Annotated reducer, LangGraph's default last-write-wins behavior on
    # plain int fields caused research_iterations to silently reset to 0 on
    # every Command.update round-trip — making the supervisor_tools
    # exceeded_allowed_iterations guard never fire (and the supervisor ran
    # 45+ times in the EDR v4 run before timeout). See deep_researcher.py
    # supervisor_tools for the corresponding enforcement.
    research_iterations: Annotated[int, operator.add] = 0
    # Phase 3: raw_notes 已删 (decision D2-B)
    # Plan v2: aggregated EU pool across all researchers. The supervisor
    # aggregates per-researcher EUs here before final_report_generation
    # consumes them. Without this field the supervisor's update would be
    # dropped silently and the EU pool would never reach the writer.
    # DEPRECATED in 阶段 1;阶段 4 改为 EuDAO.upsert_many。
    evidence_units: Annotated[list, override_reducer] = []
    # Phase 3 新增(state 瘦身)
    eu_counts: Annotated[dict[str, int], override_reducer] = {}
    dimension_ids: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    # Phase 3: raw_notes 已删 (decision D2-B)
    # Plan v2: per-researcher EU accumulator (forwarded via ResearcherOutputState).
    # Without this, `researcher_tools` writes EU into the update dict but the
    # researcher subgraph schema drops it on return, so the supervisor never
    # sees the structured citations and final_report_generation runs with an
    # empty EU pool — triggering the legacy fallback even when EUs were
    # successfully extracted from Tavily observations.
    # DEPRECATED in 阶段 1;阶段 4 改为 EuDAO.upsert_many。
    evidence_units: Annotated[list, override_reducer] = []
    # Phase 3 新增:研究者维度 ID + EU 计数引用
    dimension_id: Optional[str]
    eu_count: int = 0

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    # Phase 3: raw_notes 已删 (decision D2-B)
    # Plan v2: surface the EU pool so the supervisor can aggregate it from
    # every parallel researcher invocation.
    # DEPRECATED in 阶段 1;阶段 4 由 EuDAO 提供。
    evidence_units: Annotated[list, override_reducer] = []
    # Phase 3 新增
    dimension_id: Optional[str]
    eu_count: int = 0