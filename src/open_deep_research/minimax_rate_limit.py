"""RPM (Requests-Per-Minute) rate limiter for MiniMax-M3 API calls.

Why this exists:
- MiniMax's Token Plan has a HARD RPM cap. Hitting it returns HTTP 429 with
  the MiniMax-specific error code ``2062``, which terminates the entire
  LangGraph run.
- LangChain's default ``with_retry`` does not backoff for 429/2062 specifically,
  so the only safe approach is to *proactively throttle* before the cap is hit.

Design:
- Process-wide singleton (one limiter per process is enough — all ChatMiniMax
  instances share the same budget).
- Sliding window of timestamps; ``acquire()`` waits until the oldest
  timestamp falls outside the 60-second window before granting a permit.
- Async-friendly: callers do ``await limiter.acquire()`` (no time.sleep).
- Concurrency limit via an ``asyncio.Semaphore`` so we don't burst N>>rpm
  in-flight requests that would *all* resolve against an empty window.

Config (env vars):
- ``MINIMAX_RPM_LIMIT`` (default 15) — 30 was too aggressive against
  MiniMax's actual Token Plan cap and still triggered 2062 under load.
- ``MINIMAX_MAX_CONCURRENT`` (default min(rpm, 4))

Usage::

    limiter = RPMRateLimiter.get()
    async with limiter:
        resp = await client.post(...)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Optional


class RPMRateLimiter:
    """Sliding-window RPM limiter with concurrency cap.

    Sliding window: each ``acquire()`` records a timestamp at grant time.
    Before granting, we evict timestamps older than 60s. If the window is
    still at capacity ``rpm``, we compute how long until the oldest entry
    falls out and sleep that long.

    Concurrency cap: an asyncio.Semaphore limits in-flight calls. ``acquire``
    takes a permit; the context manager ``__aexit__`` releases it. This
    prevents thundering-herd where 30 calls land in the same millisecond
    and 25 of them burst past the sliding window check.
    """

    _instance: Optional["RPMRateLimiter"] = None

    def __init__(self, rpm: int, max_concurrent: int):
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self.rpm = rpm
        self.max_concurrent = max(1, max_concurrent)
        self._window: deque[float] = deque()
        self._window_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(self.max_concurrent)
        # Stats
        self.total_acquired = 0
        self.total_waited_seconds = 0.0
        self.peak_window_depth = 0

    @classmethod
    def get(
        cls,
        rpm: Optional[int] = None,
        max_concurrent: Optional[int] = None,
    ) -> "RPMRateLimiter":
        """Return the process-wide singleton, creating it on first call.

        Env vars (only consulted on first init):
        - ``MINIMAX_RPM_LIMIT`` (default 30)
        - ``MINIMAX_MAX_CONCURRENT`` (default min(rpm, 8))
        """
        if cls._instance is None:
            env_rpm = int(os.getenv("MINIMAX_RPM_LIMIT", "15"))
            env_conc = int(
                os.getenv("MINIMAX_MAX_CONCURRENT", str(min(env_rpm, 4)))
            )
            cls._instance = cls(
                rpm=rpm if rpm is not None else env_rpm,
                max_concurrent=(
                    max_concurrent if max_concurrent is not None else env_conc
                ),
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton. Test-only."""
        cls._instance = None

    async def acquire(self) -> None:
        """Block until a permit is available, then record a timestamp."""
        # Gate 1: concurrency semaphore — caps in-flight calls.
        await self._sem.acquire()
        try:
            # Gate 2: sliding window — caps calls per 60s.
            async with self._window_lock:
                now = time.monotonic()
                # Evict entries older than 60s.
                while self._window and (now - self._window[0]) > 60.0:
                    self._window.popleft()
                if len(self._window) >= self.rpm:
                    sleep_for = 60.0 - (now - self._window[0])
                else:
                    sleep_for = 0.0

            if sleep_for > 0:
                self.total_waited_seconds += sleep_for
                # Sleep OUTSIDE the window lock so other coroutines that
                # arrive during the wait aren't blocked on the lock —
                # they will queue on the semaphore instead.
                await asyncio.sleep(sleep_for)
                # Re-acquire the lock, re-evict, re-check.
                async with self._window_lock:
                    now = time.monotonic()
                    while self._window and (now - self._window[0]) > 60.0:
                        self._window.popleft()
                    # If other coroutines slipped in front during the sleep,
                    # sleep again. In practice with max_concurrent << rpm this
                    # is rare.
                    while len(self._window) >= self.rpm:
                        next_sleep = 60.0 - (now - self._window[0])
                        if next_sleep <= 0:
                            break
                        self.total_waited_seconds += next_sleep
                        await asyncio.sleep(next_sleep)
                        now = time.monotonic()
                        while self._window and (now - self._window[0]) > 60.0:
                            self._window.popleft()
                    self._window.append(now)
            else:
                async with self._window_lock:
                    self._window.append(now)

            self.total_acquired += 1
            depth = len(self._window)
            if depth > self.peak_window_depth:
                self.peak_window_depth = depth
        except BaseException:
            # If we failed before recording the acquire, release the
            # semaphore permit so we don't leak it.
            self._sem.release()
            raise

    def release(self) -> None:
        """Release the concurrency permit.

        Note: window timestamps are recorded at *grant* time (in acquire),
        not at release time. This means a long-running call doesn't "hold"
        its window slot indefinitely — once granted, it counts as one of
        the rpm calls regardless of how long the HTTP roundtrip takes.
        The semaphore (in-flight cap) handles the burst control instead.
        """
        self._sem.release()

    async def __aenter__(self) -> "RPMRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def stats(self) -> dict:
        """Snapshot of limiter state — useful for /info and tests."""
        return {
            "rpm": self.rpm,
            "max_concurrent": self.max_concurrent,
            "window_depth_now": len(self._window),
            "total_acquired": self.total_acquired,
            "total_waited_seconds": round(self.total_waited_seconds, 3),
            "peak_window_depth": self.peak_window_depth,
        }
