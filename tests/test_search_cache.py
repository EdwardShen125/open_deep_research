"""Unit tests for SearchCache (Phase 1.2)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.search_cache import (  # noqa: E402
    SearchCache, query_key, is_fresh, compute_expires_at,
)


# Fake monotonic clock for deterministic tests.
class _Clock:
    def __init__(self):
        self.t = 1000.0
    def __call__(self):
        return self.t
    def advance(self, dt: float):
        self.t += dt


def _payload(name: str):
    return {"results": [{"url": f"https://{name}.com/a", "title": name, "score": 0.9}]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_query_key_stable_and_distinguishes_topic():
    a = query_key("Klue vs Crayon", "ci")
    b = query_key("KLUE   vs   CRAYON", "ci")
    c = query_key("Klue vs Crayon", "news")
    assert a == b, "whitespace + case must collapse"
    assert a != c, "topic must distinguish"
    assert len(a) == 32
    print("  ✓ query_key is stable across whitespace/case")


def test_l1_put_get_roundtrip():
    clock = _Clock()
    c = SearchCache(ttl_seconds=60, l1_max_entries=4, clock=clock)
    assert c.get("Klue Crayon") is None
    c.put("Klue Crayon", _payload("klue"))
    got = c.get("Klue Crayon")
    assert got is not None
    assert got["results"][0]["url"] == "https://klue.com/a"
    s = c.stats()
    assert s["l1_hits"] == 1
    assert s["l1_misses"] == 1
    assert s["puts"] == 1
    print("  ✓ L1 put/get roundtrip + stats update")


def test_l1_ttl_expiry():
    clock = _Clock()
    c = SearchCache(ttl_seconds=60, clock=clock)
    c.put("q1", _payload("a"))
    assert c.get("q1") is not None
    clock.advance(30)
    assert c.get("q1") is not None
    clock.advance(31)  # now past expiry
    assert c.get("q1") is None
    s = c.stats()
    assert s["l1_invalidations"] == 1
    print("  ✓ L1 entry expires after TTL")


def test_l1_lru_eviction():
    clock = _Clock()
    c = SearchCache(ttl_seconds=600, l1_max_entries=3, clock=clock)
    for k in ["q1", "q2", "q3"]:
        c.put(k, _payload(k))
    c.get("q1"); c.get("q1")  # touches q1
    c.put("q4", _payload("q4"))  # evicts q2 (LRU)
    assert c.get("q1") is not None
    assert c.get("q2") is None
    assert c.get("q3") is not None
    assert c.get("q4") is not None
    print("  ✓ LRU eviction: q2 dropped, q1/q3/q4 retained")


def test_l2_invoked_when_dao_provided():
    """Smoke test: L2 path doesn't crash when dao is None or passed."""
    clock = _Clock()
    # No DAO: still works, just L1 only.
    c1 = SearchCache(ttl_seconds=60, clock=clock)
    c1.put("z", _payload("z"))
    assert c1.get("z") is not None
    # Pass a mock DAO to ensure no crash on put (urls=[]).
    class _FakeDAO:
        def upsert(self, _rec):
            return 1
    c2 = SearchCache(sources_dao=_FakeDAO(), ttl_seconds=60, clock=clock)
    c2.put("z2", _payload("z2"), urls=[{"url": "https://z2.com/x", "title": "x"}])
    assert c2.get("z2") is not None
    print("  ✓ cache accepts (and tolerates) SourcesDAO injection")


def test_is_fresh_and_expires_helpers():
    import datetime as dt
    now = dt.datetime(2026, 7, 19, 12, 0, 0, tzinfo=dt.timezone.utc)
    future = compute_expires_at(60, now=now)
    assert is_fresh(future, now=now)
    assert not is_fresh(None, now=now)
    # Past expiry → not fresh
    past = compute_expires_at(-10, now=now)
    assert not is_fresh(past, now=now)
    print("  ✓ is_fresh + compute_expires_at roundtrip")


def test_invalidate_and_clear():
    clock = _Clock()
    c = SearchCache(ttl_seconds=60, clock=clock)
    c.put("q1", _payload("a"))
    c.put("q2", _payload("b"))
    assert c.invalidate("q1") is True
    assert c.invalidate("q1") is False  # already gone
    assert c.get("q1") is None
    assert c.get("q2") is not None
    n = c.clear_l1()
    assert n >= 1
    assert c.get("q2") is None
    print("  ✓ invalidate + clear_l1 behave as expected")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("query_key_stable", test_query_key_stable_and_distinguishes_topic),
        ("l1_roundtrip", test_l1_put_get_roundtrip),
        ("l1_ttl_expiry", test_l1_ttl_expiry),
        ("l1_lru_eviction", test_l1_lru_eviction),
        ("l2_invoked_when_dao_provided", test_l2_invoked_when_dao_provided),
        ("is_fresh_helpers", test_is_fresh_and_expires_helpers),
        ("invalidate_and_clear", test_invalidate_and_clear),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} SearchCache tests")
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
            print(f"  ✗ ERROR: {e!r}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
