"""Unit tests for writer resilience helpers added in Phase 4.

Covers:
- `_is_transient_writer_error` — classifies HTTP timeouts / 429 / 5xx as retryable
- `_render_eu_digest` — produces a valid markdown digest from an EU pool,
  even when the writer LLM is unreachable

These tests run by default (no env var) because they're cheap and guard
against regressions in the fallback path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.deep_researcher import (
    _is_transient_writer_error,
    _render_eu_digest,
    _eu_attr,
)


def test_transient_detection_httpx_timeouts():
    """httpx.Timeout variants should be classified as transient."""
    import httpx

    assert _is_transient_writer_error(httpx.ReadTimeout("read timed out"))
    assert _is_transient_writer_error(httpx.ConnectTimeout("connect timed out"))
    assert _is_transient_writer_error(httpx.WriteTimeout("write timed out"))
    assert _is_transient_writer_error(httpx.PoolTimeout("pool timed out"))


def test_transient_detection_message_based():
    """Message-based detection covers wrapped / non-standard exceptions."""
    class FakeRateLimitError(Exception):
        pass

    e = FakeRateLimitError("429 Too Many Requests: rate limit exceeded")
    assert _is_transient_writer_error(e)

    e2 = FakeRateLimitError("Server returned 503 Service Unavailable")
    assert _is_transient_writer_error(e2)

    e3 = FakeRateLimitError("connection refused while reading response")
    assert _is_transient_writer_error(e3)


def test_transient_detection_non_transient():
    """Non-transient errors should NOT trigger retry."""
    class ValueErr(Exception):
        pass

    e = ValueErr("invalid prompt: missing required field")
    assert not _is_transient_writer_error(e)

    e2 = ValueErr("KeyError: 'state'")
    assert not _is_transient_writer_error(e2)


def test_render_digest_empty_pool():
    """Empty EU pool should produce empty string (caller falls back to error msg)."""
    assert _render_eu_digest([], "test-model") == ""
    assert _render_eu_digest(None, "test-model") == ""


def test_render_digest_basic_structure():
    """Digest must include stats + per-domain sections + EU rows."""
    eu_pool = [
        {
            "id": "eu-aaaa1111",
            "claim": "CrowdStrike ARR grew 29% YoY in FY2025.",
            "source_url": "https://ir.crowdstrike.com/news/2025",
            "source_title": "CrowdStrike Q4 FY2025 Earnings",
            "confidence": 0.92,
            "numbers": [
                {"text": "29%", "unit": "% YoY", "value_min": 29, "value_max": 29},
            ],
            "entities": [
                {"name": "CrowdStrike", "type": "company"},
            ],
        },
        {
            "id": "eu-bbbb2222",
            "claim": "SentinelOne ARR up 27% YoY per Q4 announcement.",
            "source_url": "https://investors.sentinelone.com/news/2025",
            "source_title": "SentinelOne Q4 FY2025 Earnings",
            "confidence": 0.88,
            "numbers": [
                {"text": "27%", "unit": "% YoY", "value_min": 27, "value_max": 27},
            ],
            "entities": [
                {"name": "SentinelOne", "type": "company"},
            ],
        },
        {
            "id": "eu-cccc3333",
            "claim": "Microsoft Defender for Endpoint is bundled with E5 license.",
            "source_url": "https://learn.microsoft.com/defender-endpoint",
            "source_title": "Microsoft Defender for Endpoint overview",
            "confidence": 0.75,
            "numbers": [],
            "entities": [
                {"name": "Microsoft", "type": "company"},
            ],
        },
    ]

    md = _render_eu_digest(eu_pool, "minimax:MiniMax-M3")
    assert "# Raw Evidence Digest" in md
    assert "Evidence units:** 3" in md
    assert "Unique domains:** 3" in md
    assert "Numeric anchors:** 2" in md
    assert "ir.crowdstrike.com" in md
    assert "investors.sentinelone.com" in md
    assert "learn.microsoft.com" in md
    assert "eu-aaaa1111" in md
    assert "CrowdStrike ARR grew 29%" in md
    assert "confidence: `0.92`" in md
    assert "29%" in md


def test_render_digest_handles_dataclass_pool():
    """Digest must work with both dict EUs and EvidenceUnit dataclass instances."""
    from open_deep_research.evidence_units import EvidenceUnit, NumberBinding, EntityRef

    eu = EvidenceUnit(
        claim="Bitdefender GravityZone holds 12% market share in EMEA.",
        source_url="https://bitdefender.com/gz",
        source_title="GravityZone overview",
        confidence=0.80,
        numbers=[NumberBinding(text="12%", value_min=12.0, value_max=12.0, unit="%")],
        entities=[EntityRef(name="Bitdefender", entity_type="company")],
    )

    md = _render_eu_digest([eu], "minimax:MiniMax-M3")
    assert "Bitdefender GravityZone holds 12%" in md
    assert "bitdefender.com" in md
    assert "Evidence units:** 1" in md


def test_eu_attr_polymorphism():
    """_eu_attr should support both dict and dataclass EU shapes."""
    d = {"id": "eu-x", "claim": "test"}
    obj = type("EU", (), {"id": "eu-y", "claim": "obj claim"})()

    assert _eu_attr(d, "id") == "eu-x"
    assert _eu_attr(d, "missing", "fallback") == "fallback"
    assert _eu_attr(obj, "id") == "eu-y"
    assert _eu_attr(obj, "missing", "fallback") == "fallback"


if __name__ == "__main__":
    test_transient_detection_httpx_timeouts()
    test_transient_detection_message_based()
    test_transient_detection_non_transient()
    test_render_digest_empty_pool()
    test_render_digest_basic_structure()
    test_render_digest_handles_dataclass_pool()
    test_eu_attr_polymorphism()
    print("✓ all writer-resilience tests passed")