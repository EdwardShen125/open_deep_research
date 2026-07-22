"""Phase 3 evidence subpackage."""
from open_deep_research.evidence.eu_dao import (
    ClaimDAO,
    EuDAO,
    RunCheckpointDAO,
    host_of,
)
from open_deep_research.evidence.llm_entailment import (
    DEFAULT_BATCH_SIZE,
    verify_entailment_batch,
    verify_entailment_batch_sync,
)
from open_deep_research.evidence.llm_extractor import (
    extract_from_content_with_llm,
    extract_from_search_results_with_llm,
)
from open_deep_research.evidence.schema import (
    ClaimV2,
    ClaimType,
    EvidenceUnitV2,
    Grade,
    SourceTier,
    Verdict,
)
from open_deep_research.evidence.verify import (
    ENTAILMENT_PROMPT,
    EntailmentBatchResult,
    EntailmentResult,
    GateStats,
    _normalize,
    _numbers,
    _render_entailment_items,
    has_numeric_drift,
    parse_entailment_response,
    run_gate1_span,
    run_gate2_numeric_drift,
    verify_span,
)

__all__ = [
    "ClaimDAO",
    "ClaimV2",
    "ClaimType",
    "DEFAULT_BATCH_SIZE",
    "ENTAILMENT_PROMPT",
    "EntailmentBatchResult",
    "EntailmentResult",
    "EuDAO",
    "EvidenceUnitV2",
    "GateStats",
    "Grade",
    "RunCheckpointDAO",
    "SourceTier",
    "Verdict",
    "_normalize",
    "_numbers",
    "_render_entailment_items",
    "extract_from_content_with_llm",
    "extract_from_search_results_with_llm",
    "has_numeric_drift",
    "host_of",
    "parse_entailment_response",
    "run_gate1_span",
    "run_gate2_numeric_drift",
    "verify_entailment_batch",
    "verify_entailment_batch_sync",
    "verify_span",
]