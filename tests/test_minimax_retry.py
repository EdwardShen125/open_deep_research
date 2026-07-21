"""Tests for MiniMax-M3 ChatModel — 429 retry, rate-limit integration, sanity."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_deep_research.minimax_chat import ChatMiniMax
from open_deep_research.minimax_rate_limit import RPMRateLimiter


# ---------------------------------------------------------------------------
# _is_retryable_response
# ---------------------------------------------------------------------------

def test_is_retryable_429():
    assert ChatMiniMax._is_retryable_response(429, "") is True


def test_is_retryable_408():
    assert ChatMiniMax._is_retryable_response(408, "") is True


def test_is_retryable_500():
    assert ChatMiniMax._is_retryable_response(500, "") is True
    assert ChatMiniMax._is_retryable_response(503, "") is True


def test_is_retryable_200_with_2062_in_body():
    """MiniMax sometimes returns 200 with error code 2062 inside the body."""
    body = '{"error":{"code":2062,"message":"rate limit exceeded"}}'
    assert ChatMiniMax._is_retryable_response(200, body) is True


def test_is_retryable_200_with_normal_body():
    assert ChatMiniMax._is_retryable_response(200, '{"content":[]}') is False


def test_is_retryable_400_not_retried():
    assert ChatMiniMax._is_retryable_response(400, "bad request") is False


def test_is_retryable_401_not_retried():
    assert ChatMiniMax._is_retryable_response(401, "auth") is False


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------

def test_parse_retry_after_seconds():
    resp = MagicMock()
    resp.headers = {"retry-after": "5"}
    assert ChatMiniMax._parse_retry_after(resp) == 5.0


def test_parse_retry_after_missing():
    resp = MagicMock()
    resp.headers = {}
    assert ChatMiniMax._parse_retry_after(resp) is None


def test_parse_retry_after_http_date_fallback_to_60():
    resp = MagicMock()
    resp.headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert ChatMiniMax._parse_retry_after(resp) == 60.0


# ---------------------------------------------------------------------------
# _completion_with_retry — mocked HTTP
# ---------------------------------------------------------------------------

def _make_chat():
    """Construct a ChatMiniMax with a fake API key (no real call)."""
    return ChatMiniMax(api_key="test-key-not-real")


def _mock_response(status_code, body_dict=None, headers=None):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if body_dict is not None:
        import json
        resp.text = json.dumps(body_dict)
        resp.json.return_value = body_dict
    else:
        resp.text = ""
    return resp


def test_completion_retries_on_429_then_succeeds(monkeypatch):
    """First call returns 429, second call returns 200 — should retry once."""
    RPMRateLimiter.reset()
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "60")  # don't block on limiter
    chat = _make_chat()

    responses = [
        _mock_response(429, {"error": {"code": 2062}}, {"retry-after": "0.05"}),
        _mock_response(200, {"content": [{"type": "text", "text": "hi"}]}),
    ]
    call_count = [0]

    def fake_post(url, headers, json):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r

    with patch.object(chat, "_get_client") as gc:
        gc.return_value.post.side_effect = fake_post
        data = chat._completion_with_retry({"model": "MiniMax-M3"})
    assert call_count[0] == 2
    assert data == {"content": [{"type": "text", "text": "hi"}]}


def test_completion_gives_up_after_5_retries(monkeypatch):
    """All 5 attempts return 429 — should raise RuntimeError."""
    RPMRateLimiter.reset()
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "60")
    chat = _make_chat()

    call_count = [0]

    def fake_post(url, headers, json):
        call_count[0] += 1
        return _mock_response(429, {}, {"retry-after": "0.01"})

    with patch.object(chat, "_get_client") as gc:
        gc.return_value.post.side_effect = fake_post
        with pytest.raises(RuntimeError, match="rate-limited 5x"):
            chat._completion_with_retry({"model": "MiniMax-M3"})
    assert call_count[0] == 5


def test_completion_no_retry_on_400(monkeypatch):
    """400 is NOT retryable — single attempt then raise."""
    RPMRateLimiter.reset()
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "60")
    chat = _make_chat()

    call_count = [0]

    def fake_post(url, headers, json):
        call_count[0] += 1
        return _mock_response(400, {"error": "bad request"})

    with patch.object(chat, "_get_client") as gc:
        gc.return_value.post.side_effect = fake_post
        with pytest.raises(RuntimeError, match="400"):
            chat._completion_with_retry({"model": "MiniMax-M3"})
    assert call_count[0] == 1  # only one attempt


def test_completion_uses_retry_after_header(monkeypatch):
    """When Retry-After is present, backoff follows it (not exponential)."""
    RPMRateLimiter.reset()
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "60")
    chat = _make_chat()

    responses = [
        _mock_response(429, {}, {"retry-after": "0.5"}),
        _mock_response(200, {"content": []}),
    ]
    call_count = [0]

    def fake_post(url, headers, json):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r

    with patch.object(chat, "_get_client") as gc:
        gc.return_value.post.side_effect = fake_post
        t0 = time.monotonic()
        chat._completion_with_retry({"model": "MiniMax-M3"})
        elapsed = time.monotonic() - t0
    # First retry-after was 0.5s; we should have slept ~0.5s, not 1s (exp).
    assert 0.4 < elapsed < 1.5, f"Retry-After sleep not honored: {elapsed:.3f}s"
    assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Integration: rate-limit + retry should work together
# ---------------------------------------------------------------------------

def test_completion_does_not_acquire_rate_limit(monkeypatch):
    """_completion_with_retry is a thin HTTP helper — rate limiting
    lives at the _generate / _agenerate / _stream / _astream layer.
    Direct callers of _completion bypass the limiter by design."""
    RPMRateLimiter.reset()
    monkeypatch.setenv("MINIMAX_RPM_LIMIT", "60")
    monkeypatch.setenv("MINIMAX_MAX_CONCURRENT", "60")
    chat = _make_chat()

    def fake_post(url, headers, json):
        return _mock_response(200, {"content": []})

    with patch.object(chat, "_get_client") as gc:
        gc.return_value.post.side_effect = fake_post
        chat._completion_with_retry({"model": "MiniMax-M3"})

    # Limiter was NOT consulted by _completion alone.
    lim = RPMRateLimiter.get()
    assert lim.total_acquired == 0, (
        "_completion_with_retry should not touch the rate limiter; "
        f"got total_acquired={lim.total_acquired}"
    )
