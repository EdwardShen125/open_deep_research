# State Schema v1

**Source**: `src/open_deep_research/state.py` (commit `pre-prompts-migration`)
**Date**: 2026-07-18
**Status**: Living document — schema is captured exactly as it exists in code

This file is the **canonical reference** for Phase 1/2/3 migrations. If code and
this document disagree, the document wins (per plan v2 §"已知风险与注意事项").

---

## Pydantic Structured Outputs (5)

### `ConductResearch`
Used by: `supervisor` node (delegation)
```python
class ConductResearch(BaseModel):
    research_topic: str  # single topic, >=1 paragraph detail
```

### `ResearchComplete`
Used by: `supervisor` node (termination signal)
```python
class ResearchComplete(BaseModel):
    pass  # no fields, presence is the signal
```

### `Summary`
Used by: `summarize_webpage` (utils.py) — webpage compression for downstream researcher
```python
class Summary(BaseModel):
    summary: str
    key_excerpts: str
```

### `ClarifyWithUser`
Used by: `clarify_with_user` node (main graph entry)
```python
class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str        # only used if need_clarification=True
    verification: str    # acknowledgement if need_clarification=False
```

### `ResearchQuestion`
Used by: `write_research_brief` node (rewrites user messages)
```python
class ResearchQuestion(BaseModel):
    research_brief: str
```

---

## Graph States (4)

### `AgentInputState`
Role: LangGraph graph input contract
```python
class AgentInputState(MessagesState):
    """Only messages. No additional fields."""
```
Inherits `messages: list[MessageLikeRepresentation]` from `MessagesState`.

### `AgentState`
Role: Main graph state (4-node pipeline: clarify → brief → research → report)
```python
class AgentState(MessagesState):
    supervisor_messages: Annotated[list, override_reducer]   # accumulated from subgraphs
    research_brief: Optional[str]                            # set by write_research_brief
    raw_notes: Annotated[list[str], override_reducer] = []   # unprocessed researcher outputs
    notes: Annotated[list[str], override_reducer] = []       # compressed by researcher
    final_report: str                                        # set by final_report_generation
```

### `SupervisorState`
Role: Supervisor subgraph (Send API dispatches to researcher subgraphs)
```python
class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list, override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
```

### `ResearcherState`
Role: Individual researcher subgraph (one per ConductResearch topic)
```python
class ResearcherState(TypedDict):
    researcher_messages: Annotated[list, operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
```

### `ResearcherOutputState`
Role: Researcher subgraph return contract (what Send API forwards to parent)
```python
class ResearcherOutputState(BaseModel):
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
```

---

## Reducers

### `override_reducer(current_value, new_value)`
```python
def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
```
Behavior: if new_value is a dict with `type="override"`, replace. Otherwise append (list sem).

This is the trick used by writer / supervisor nodes to set fields like `final_report`
without going through messages reducers. **Subtle gotcha**: most callers `Annotated[list, override_reducer]` but the override protocol requires explicit dict wrappers, which is rarely used.

---

## Implications for Plan v2

Plan v2 §"Phase 2 — Evidence Extraction 重写" requires migrating from `compressed_research: str`
to `list[EvidenceUnit]`. This means:

- `ResearcherState.compressed_research: str` → `evidence_units: list[EvidenceUnit]`
- `ResearcherOutputState.compressed_research: str` → `evidence_units: list[EvidenceUnit]`
- `SupervisorState.notes: list[str]` → `evidence_units: list[EvidenceUnit]` (aggregated)
- `AgentState.raw_notes: list[str]` → `evidence_units: list[EvidenceUnit]`
- `AgentState.notes: list[str]` → removed (compression step is gone)
- `AgentState.final_report: str` → `final_report_blocks: list[ReportBlock]` (per Phase 3b)

Plan v2 §"Phase 3b" replaces `final_report: str` with structured blocks.

---

## v1 Defenses (None for evidence)

The current state schema has **no native field for**:
- per-claim evidence provenance
- source URL binding
- verification status (single/multi source)
- claim clustering
- number binding (for rule 一 in Phase 3a)
- block-level claim anchors (for Phase 3b)

These will all be added in Phase 2 EvidenceUnit schema and Phase 3b ReportBlock schema.

---

## What's NOT in v1 but will be in v2

| v1 | v2 (plan) |
|---|---|
| `compressed_research: str` | `evidence_units: list[EvidenceUnit]` (Phase 2) |
| `notes: list[str]` (compressed text) | `evidence_units: list[EvidenceUnit]` (Phase 2) |
| `final_report: str` | `final_report: Report` with `blocks: list[ReportBlock]` (Phase 3b) |
| (no verification field) | `verification_distribution: dict[str, int]` (Phase 3 trust score) |
| (no claim cluster) | `cluster_id` per EU (Phase 2) |
| (no entity normalization) | `entity_ids: list[str]` per EU (Phase 2) |

See `docs/state_v2.md` (TODO: write in Phase 2a).
