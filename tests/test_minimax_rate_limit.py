"""Tests for the MiniMax RPM rate limiter."""

import asyncio
import time

import pytest

from open_deep_research.minimax_rate_limit import RPMRateLimiter


def _run(coro):
    """Helper: drive a coroutine on a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def setup_function(_):
    """Reset the singleton before each test."""
    RPMRateLimiter.reset()


def teardown_function(_):
    """Reset the singleton after each test."""
    RPMRateLimiter.reset()


def test_get_singleton_returns_same_instance():
    a = RPMRateLimiter.get(rpm=10)
    b = RPMRateLimiter.get(rpm=999)  # second call should NOT override
    assert a is b
    assert a.rpm == 10


def test_default_rpm_from_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "7")
    lim = RPMRateLimiter.get()
    assert lim.rpm == 7
    # max_concurrent defaults to min(rpm, 4) at the production default of
    # 15. With explicit override MINIMAX_RPM_LIMIT=7, the helper code
    # still uses min(env_rpm, 4) for max_concurrent → 4.
    assert lim.max_concurrent == 4


def test_default_rpm_15_when_no_env():
    """Production default is 15 RPM / 4 concurrent."""
    monkey = pytest.MonkeyPatch()
    monkey.delenv("MINIMAX_RPM_LIMIT", raising=False)
    monkey.delenv("MINIMAX_MAX_CONCURRENT", raising=False)
    try:
        lim = RPMRateLimiter.get()
        assert lim.rpm == 15
        assert lim.max_concurrent == 4
    finally:
        monkey.undo()


def test_acquire_below_rpm_does_not_wait():
    """5 acquires at rpm=10 should all return instantly."""
    lim = RPMRateLimiter.get(rpm=10, max_concurrent=10)

    async def go():
        for _ in range(5):
            await lim.acquire()
            lim.release()
            # Reset window — release() doesn't drain it; under a tight
            # loop without window eviction we'd trip the rpm cap anyway.
            lim._window.clear()

    t0 = time.monotonic()
    _run(go())
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"5 acquires at rpm=10 took {elapsed:.3f}s"


def test_acquire_at_rpm_with_release_does_not_wait():
    """Single acquire+release at rpm=3 should return instantly."""
    lim = RPMRateLimiter.get(rpm=3, max_concurrent=10)

    async def go():
        t0 = time.monotonic()
        for _ in range(5):
            await lim.acquire()
            lim.release()
            # Reset the window to avoid timestamp accumulation — this
            # test exercises the semaphore path, not the sliding window.
            lim._window.clear()
        return time.monotonic() - t0

    elapsed = _run(go())
    assert elapsed < 0.5, f"acquire-release loop at rpm=3 took {elapsed:.3f}s"


def test_acquire_over_rpm_waits():
    """Without releasing, the 4th acquire at rpm=3 must wait (capped at 2s)."""
    lim = RPMRateLimiter.get(rpm=3, max_concurrent=10)

    async def go():
        ts = []
        for _ in range(4):
            t = time.monotonic()
            await lim.acquire()
            ts.append(time.monotonic() - t)
        return ts

    # 4th acquire would wait ~60s without releasing — we cap at 2s and
    # expect asyncio.TimeoutError to fire (proving the limit kicked in).
    raised = False
    try:
        _run(asyncio.wait_for(go(), timeout=2.0))
    except asyncio.TimeoutError:
        raised = True
    # First 3 were admitted (we can check the window depth).
    # If timeout fired, only 3 made it in.
    assert raised, "4th acquire should have blocked past 2s timeout"
    assert lim.total_acquired == 3, (
        f"expected 3 acquires admitted, got {lim.total_acquired}"
    )


def test_concurrent_semaphore_caps_inflight():
    """max_concurrent=2 means at most 2 acquires proceed past the sem gate."""
    lim = RPMRateLimiter.get(rpm=100, max_concurrent=2)

    in_flight = 0
    peak_in_flight = 0

    async def worker():
        nonlocal in_flight, peak_in_flight
        await lim.acquire()
        in_flight += 1
        # pylint: disable=global-variable-not-assigned
        # (peak is nonlocal-captured above)
        # Use a list-as-cell trick for nested mutability in closures.
        in_flight_box[0] = max(in_flight_box[0], in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        lim.release()

    in_flight_box = [0]

    async def go():
        await asyncio.gather(*[worker() for _ in range(6)])

    _run(go())
    # Semaphore limits in-flight to max_concurrent=2
    assert in_flight_box[0] <= 2, (
        f"peak in-flight {in_flight_box[0]} > max_concurrent 2"
    )
    assert in_flight_box[0] == 2, "should actually saturate the cap"


def test_stats_reports_actual_usage():
    lim = RPMRateLimiter.get(rpm=5, max_concurrent=5)

    async def go():
        for _ in range(3):
            await lim.acquire()
            lim.release()

    _run(go())
    s = lim.stats()
    assert s["rpm"] == 5
    assert s["max_concurrent"] == 5
    assert s["total_acquired"] == 3
    # Window holds all 3 timestamps (release() only frees the concurrency
    # semaphore, not the window — eviction is time-based, 60s).
    assert s["peak_window_depth"] == 3
    assert s["window_depth_now"] == 3
    # Sanity: total_waited_seconds should be 0 — well below the 5-rpm cap.
    assert s["total_waited_seconds"] == 0.0


def test_window_evicts_old_entries():
    """After >60s, old timestamps fall out of the window."""
    lim = RPMRateLimiter.get(rpm=2, max_concurrent=2)

    async def go():
        # Inject a fake "old" timestamp directly into the window
        lim._window.append(time.monotonic() - 65.0)
        # Now acquire — window eviction should clear the old entry
        # and admit the new one without blocking.
        t0 = time.monotonic()
        await lim.acquire()
        elapsed = time.monotonic() - t0
        lim.release()
        return elapsed

    elapsed = _run(go())
    assert elapsed < 0.1, f"old entry not evicted: {elapsed:.3f}s wait"


def test_context_manager_releases_on_exit():
    """async with limiter: release on exit, permits reusable."""
    lim = RPMRateLimiter.get(rpm=10, max_concurrent=1)

    async def go():
        # Take the only permit via context manager
        async with lim:
            assert lim._sem._value == 0
        # After exit, the permit is back
        assert lim._sem._value == 1
        # Re-enter to confirm we can reuse
        async with lim:
            assert lim._sem._value == 0
        assert lim._sem._value == 1

    _run(go())
