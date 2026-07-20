"""LLM access layer - single entry point for all model calls.

Phase 0a: this module is added to centralize LLM access and add Langfuse tracing.
It does NOT yet replace existing call sites in deep_researcher.py - that's the
goal of a later phase. It provides:
  - get_llm(role, config): unified LLM factory (wraps __init__._resolve_chat_model)
  - get_prompt(role): load prompt by role from prompts.registry
  - get_prompt_version(role): return version string for metadata
  - trace_llm(role, ...): decorator that wraps an LLM call with a Langfuse span
    whose metadata contains prompt_version and node name

Backward compatibility: existing code keeps importing from
``open_deep_research`` (``create_configurable_model`` etc.). New code should use
``from open_deep_research.llm import get_llm, get_prompt``.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from open_deep_research import _resolve_chat_model
from open_deep_research.prompts import REGISTRY, load_prompt

logger = logging.getLogger(__name__)

# --- Langfuse (optional, gracefully degrades when not configured) ---
_LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)

_langfuse_client = None


def _get_langfuse():
    """Lazily import and init the Langfuse client.

    Returns None if Langfuse is not configured (LANGFUSE_* env vars absent)
    or if the SDK fails to import / init. All callers must handle None.
    """
    global _langfuse_client
    if not _LANGFUSE_ENABLED:
        return None
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
        logger.info("Langfuse client initialized (host=%s)", os.environ.get("LANGFUSE_HOST"))
        return _langfuse_client
    except Exception as e:
        logger.warning("Langfuse init failed, tracing disabled: %s", e)
        return None


# --- Public API ---

def get_llm(config: Optional[RunnableConfig] = None, tags: Optional[list[str]] = None) -> BaseChatModel:
    """Return the chat model for the given LangGraph config.

    This is a thin wrapper around the existing ``_resolve_chat_model`` so we
    don't break behavior. In Phase 2/3 this will be replaced by a role-aware
    router that picks a model per phase (planner/extractor/validator/writer).

    ``tags`` (optional) are forwarded to the underlying model as a kwarg.
    Historically we passed ``["langsmith:nostream"]`` to suppress streaming
    in summarization. With Langfuse we capture tags via span metadata instead,
    so this argument is kept for backward compatibility.
    """
    cfg = dict(config or {})
    if tags:
        cfg["tags"] = tags
    return _resolve_chat_model(cfg)


def get_prompt(role: str) -> str:
    """Load the prompt text for a given role.

    Roles are registered in open_deep_research.prompts.REGISTRY.
    """
    text, _version = load_prompt(role)
    return text


def get_prompt_version(role: str) -> str:
    """Return the version string (e.g. 'researcher_v1') for metadata tagging."""
    _, version = load_prompt(role)
    return version


def trace_llm(role: str, node_name: Optional[str] = None) -> Callable:
    """Decorator: trace an LLM call with Langfuse, tagging prompt_version.

    Usage:
        @trace_llm(role="researcher", node_name="researcher_node")
        async def call_researcher(...):
            ...

    If Langfuse is not configured, this is a no-op pass-through.

    Note: Langfuse 4.x exposes tracing through OpenTelemetry. We use
    ``client._otel_tracer.start_as_current_span`` as a context manager and
    then ``update_current_span`` to attach metadata.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            lf = _get_langfuse()
            if lf is None:
                return await fn(*args, **kwargs)
            prompt_version = get_prompt_version(role)
            span_name = node_name or fn.__name__
            tracer = lf._otel_tracer
            with tracer.start_as_current_span(span_name):
                lf.update_current_span(metadata={
                    "prompt_role": role,
                    "prompt_version": prompt_version,
                    "node": span_name,
                })
                return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            lf = _get_langfuse()
            if lf is None:
                return fn(*args, **kwargs)
            prompt_version = get_prompt_version(role)
            span_name = node_name or fn.__name__
            tracer = lf._otel_tracer
            with tracer.start_as_current_span(span_name):
                lf.update_current_span(metadata={
                    "prompt_role": role,
                    "prompt_version": prompt_version,
                    "node": span_name,
                })
                return fn(*args, **kwargs)

        # Pick async vs sync based on the function
        import inspect
        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def flush_langfuse() -> None:
    """Flush pending Langfuse events. Call at end of run / before exit."""
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)


def langfuse_status() -> dict[str, Any]:
    """Diagnostic: is Langfuse configured and reachable?"""
    lf = _get_langfuse()
    if lf is None:
        return {
            "enabled": False,
            "reason": "env vars missing" if not _LANGFUSE_ENABLED else "init failed",
        }
    return {
        "enabled": True,
        "host": os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        "client_id": id(lf),
    }
