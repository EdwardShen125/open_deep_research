"""Phase 2.1 — EvidenceUnit schema tests."""
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.evidence_units import (  # noqa: E402
    EvidenceUnit, NumberBinding, EntityRef,
    extract_numbers, dedup_eus, eus_as_dicts, make_eu_id,
)


# ---------------------------------------------------------------------------
# NumberBinding parsing
# ---------------------------------------------------------------------------

def test_number_binding_chinese_range():
    nb = NumberBinding.from_text("30-60 亿美元")
    assert nb.text == "30-60 亿美元"
    assert nb.value_min is not None and nb.value_max is not None
    assert nb.value_min < nb.value_max
    assert nb.unit == "USD"
    print(f"  ✓ NumberBinding: '30-60 亿美元' → {nb.value_min:.0f}-{nb.value_max:.0f} {nb.unit}")


def test_number_binding_single():
    nb = NumberBinding.from_text("约 8.5 亿")
    assert nb.value_min is not None
    assert nb.is_estimated is True
    print(f"  ✓ NumberBinding: '约 8.5 亿' → {nb.value_min:.1f} estimated")


def test_number_binding_english_percent():
    nb = NumberBinding.from_text("1-2% of TAM")
    assert nb.value_min == 1.0 and nb.value_max == 2.0
    assert nb.unit is None  # no scaling for plain %
    print(f"  ✓ NumberBinding: '1-2%' → {nb.value_min}-{nb.value_max}")


def test_number_binding_no_match():
    nb = NumberBinding.from_text("no numbers here, just prose")
    assert nb.text == "no numbers here, just prose"
    assert nb.value_min is None and nb.value_max is None
    print(f"  ✓ NumberBinding: prose without numbers parsed cleanly")


def test_extract_numbers_multi():
    nums = extract_numbers("Klue 在 2022 Series C 估值约 8-10 亿美元,竞品 Crayon 估值约 5 亿美元")
    assert len(nums) >= 2
    texts = [n.text for n in nums]
    assert any("8-10" in t and "亿" in t for t in texts)
    print(f"  ✓ extract_numbers found {len(nums)} anchors: {texts[:3]}")


def test_extract_numbers_estimate_marker():
    """Approximation marker should set is_estimated even when range absent."""
    nb = NumberBinding.from_text("约 30 亿美元")
    assert nb.is_estimated is True
    print(f"  ✓ estimate marker detected: '{nb.text}'")


# ---------------------------------------------------------------------------
# EntityRef
# ---------------------------------------------------------------------------

def test_entity_ref_roundtrip():
    er = EntityRef(name="Kompyte", entity_type="company", extra={"acquired_by": "Crayon"})
    d = er.to_dict()
    assert d["name"] == "Kompyte"
    assert d["extra"] == {"acquired_by": "Crayon"}
    er2 = EntityRef(**d)
    assert er2 == er
    print(f"  ✓ EntityRef to_dict/from_dict roundtrip")


# ---------------------------------------------------------------------------
# EvidenceUnit invariants
# ---------------------------------------------------------------------------

def _make_eu(**overrides):
    defaults = dict(
        claim="Klue 收购 Algorithmia 旗下 win/loss 分析业务",
        quote="Klue has acquired Algorithmia's win/loss analysis business.",
        source_url="https://example.com/klue-acquired-algorithmia",
        source_title="Klue Acquisition Note",
        source_id=42,
        confidence=0.7,
    )
    defaults.update(overrides)
    return EvidenceUnit(**defaults)


def test_eu_constructs_with_minimum():
    eu = EvidenceUnit(
        claim="Klue 是竞争情报工具",
        source_url="https://klue.com/product",
    )
    assert eu.id is not None
    assert eu.content_hash != ""
    print(f"  ✓ EU constructs with minimum fields (id={eu.id})")


def test_eu_rejects_empty_claim():
    try:
        EvidenceUnit(claim="", source_url="https://klue.com")
    except ValueError as e:
        print(f"  ✓ EU rejects empty claim: {e}")
        return
    raise AssertionError("expected ValueError on empty claim")


def test_eu_rejects_invalid_confidence():
    try:
        _make_eu(confidence=1.5)
    except ValueError as e:
        print(f"  ✓ EU rejects confidence > 1.0: {e}")
        return
    raise AssertionError("expected ValueError on out-of-range confidence")


def test_eu_truncates_oversized_claim():
    long_claim = "x" * 1000
    eu = _make_eu(claim=long_claim)
    assert len(eu.claim) <= 500
    assert eu.claim.endswith("...")
    print(f"  ✓ EU truncates claim to 500 chars + ellipsis")


def test_eu_content_hash_dedup():
    a = _make_eu()
    b = _make_eu()        # identical content, different extracted_at
    assert a.content_hash == b.content_hash, "Same content → same hash"
    assert a.id == b.id
    print(f"  ✓ EU content_hash deterministic across runs (id={a.id})")


def test_eu_content_hash_differs_on_quote():
    a = _make_eu(quote="alpha")
    b = _make_eu(quote="beta")
    assert a.content_hash != b.content_hash
    print(f"  ✓ EU content_hash differs on quote (a={a.content_hash[:8]}, b={b.content_hash[:8]})")


def test_eu_from_search_summary_mines_numbers():
    eu = EvidenceUnit.from_search_summary(
        claim="Klue 2022 Series C 估值约 8-10 亿美元",
        quote="Series C funding round, ~$800M-$1B valuation",
        source_url="https://klue.com/series-c",
        text_context="Klue 拿到 Series C 约 8-10 亿美元投资,Crayon 估值约 5 亿美元",
    )
    assert len(eu.numbers) >= 1, eu.numbers
    assert any("亿" in n.text for n in eu.numbers)
    print(f"  ✓ from_search_summary mined {len(eu.numbers)} numeric bindings")


# ---------------------------------------------------------------------------
# dedup + collection
# ---------------------------------------------------------------------------

def test_dedup_eus_collapses_duplicates():
    eus = [_make_eu() for _ in range(3)]
    eus.append(_make_eu(quote="different"))
    out = dedup_eus(eus)
    assert len(out) == 2
    assert len(dedup_eus([])) == 0
    print(f"  ✓ dedup_eus collapsed 4 → {len(out)}")


def test_eus_as_dicts_serializes_datetime():
    eus = [_make_eu()]
    out = eus_as_dicts(eus)
    assert isinstance(out[0]["extracted_at"], str)
    assert "T" in out[0]["extracted_at"]
    print(f"  ✓ eus_as_dicts ISO-serializes datetime")


# ---------------------------------------------------------------------------
# id stability
# ---------------------------------------------------------------------------

def test_make_eu_id_format():
    cid = "0" * 64
    eid = make_eu_id(cid)
    assert eid.startswith("eu-")
    assert len(eid) == len("eu-") + 12
    print(f"  ✓ make_eu_id format: {eid}")


def test_eu_roundtrip():
    eu = _make_eu()
    d = eu.to_dict()
    eu2 = EvidenceUnit.from_dict(d)
    assert eu2.claim == eu.claim
    assert eu2.quote == eu.quote
    assert eu2.source_url == eu.source_url
    assert len(eu2.numbers) == len(eu.numbers)
    print(f"  ✓ EU to_dict/from_dict roundtrip")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("number_binding_chinese_range", test_number_binding_chinese_range),
        ("number_binding_single", test_number_binding_single),
        ("number_binding_english_percent", test_number_binding_english_percent),
        ("number_binding_no_match", test_number_binding_no_match),
        ("extract_numbers_multi", test_extract_numbers_multi),
        ("extract_numbers_estimate_marker", test_extract_numbers_estimate_marker),
        ("entity_ref_roundtrip", test_entity_ref_roundtrip),
        ("eu_constructs_with_minimum", test_eu_constructs_with_minimum),
        ("eu_rejects_empty_claim", test_eu_rejects_empty_claim),
        ("eu_rejects_invalid_confidence", test_eu_rejects_invalid_confidence),
        ("eu_truncates_oversized_claim", test_eu_truncates_oversized_claim),
        ("eu_content_hash_dedup", test_eu_content_hash_dedup),
        ("eu_content_hash_differs_on_quote", test_eu_content_hash_differs_on_quote),
        ("eu_from_search_summary_mines_numbers", test_eu_from_search_summary_mines_numbers),
        ("dedup_eus_collapses_duplicates", test_dedup_eus_collapses_duplicates),
        ("eus_as_dicts_serializes_datetime", test_eus_as_dicts_serializes_datetime),
        ("make_eu_id_format", test_make_eu_id_format),
        ("eu_roundtrip", test_eu_roundtrip),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} EvidenceUnit tests")
    print("=" * 70)
    failed = []
    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
