"""Graph state definitions and data structures for the Deep Research agent."""

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

    Plan v2 adds 4 fields beyond the v1 baseline:
      - evidence_units : EU pool across all researchers
      - cited_report   : writer's structured claim↔EU output
      - verification   : verifier engine output (rules 1/2/3/C)
      - url_compliance : Rule 4 audit (page-level URL enforcement)
    These default to empty / None so v1 callers still see the same surface.
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    evidence_units: Annotated[list, override_reducer] = []
    cited_report: Optional[dict] = None
    verification: Optional[dict] = None
    url_compliance: Annotated[list, override_reducer] = []


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
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: aggregated EU pool across all researchers. The supervisor
    # aggregates per-researcher EUs here before final_report_generation
    # consumes them. Without this field the supervisor's update would be
    # dropped silently and the EU pool would never reach the writer.
    evidence_units: Annotated[list, override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: per-researcher EU accumulator (forwarded via ResearcherOutputState).
    # Without this, `researcher_tools` writes EU into the update dict but the
    # researcher subgraph schema drops it on return, so the supervisor never
    # sees the structured citations and final_report_generation runs with an
    # empty EU pool — triggering the legacy fallback even when EUs were
    # successfully extracted from Tavily observations.
    evidence_units: Annotated[list, override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: surface the EU pool so the supervisor can aggregate it from
    # every parallel researcher invocation.
    evidence_units: Annotated[list, override_reducer] = []