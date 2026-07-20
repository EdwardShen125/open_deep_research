"""Prompt registry - single source of truth for all prompt versions.

Maps role names to (module_path, prompt_attribute, version) tuples.
Adding a new role? Create a new prompts/{role}_v{N}.py and register it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PromptSpec:
    role: str               # logical name, e.g. "clarify"
    module: str             # dotted path, e.g. "open_deep_research.prompts.clarify_v1"
    attribute: str          # exported name in module, e.g. "clarify_with_user_instructions"
    version: str            # matches PROMPT_VERSION in module, used as metadata tag


REGISTRY: dict[str, PromptSpec] = {
    # Phase 0a: keep v1 prompts identical to legacy prompts.py for behavior parity
    "clarify": PromptSpec(
        role="clarify",
        module="open_deep_research.prompts.clarify_v1",
        attribute="clarify_with_user_instructions",
        version="clarify_v1",
    ),
    "research_brief": PromptSpec(
        role="research_brief",
        module="open_deep_research.prompts.research_brief_v1",
        attribute="transform_messages_into_research_topic_prompt",
        version="research_brief_v1",
    ),
    "supervisor": PromptSpec(
        role="supervisor",
        module="open_deep_research.prompts.supervisor_v1",
        attribute="lead_researcher_prompt",
        version="supervisor_v1",
    ),
    "researcher": PromptSpec(
        role="researcher",
        module="open_deep_research.prompts.researcher_v1",
        attribute="research_system_prompt",
        version="researcher_v1",
    ),
    "compressor": PromptSpec(
        role="compressor",
        module="open_deep_research.prompts.compressor_v1",
        attribute="compress_research_system_prompt",
        version="compressor_v1",
    ),
    "writer": PromptSpec(
        role="writer",
        module="open_deep_research.prompts.writer_v1",
        attribute="final_report_generation_prompt",
        version="writer_v1",
    ),
    "webpage_summarizer": PromptSpec(
        role="webpage_summarizer",
        module="open_deep_research.prompts.webpage_summarizer_v1",
        attribute="summarize_webpage_prompt",
        version="webpage_summarizer_v1",
    ),
}


def load_prompt(role: str) -> tuple[str, str]:
    """Resolve a role to (prompt_text, version).

    Raises KeyError if role is not registered.
    """
    spec = REGISTRY[role]
    mod = import_module(spec.module)
    text = getattr(mod, spec.attribute)
    return text, spec.version


# Backward-compat re-exports for legacy callers that import from prompts directly.
# These keep the old import path working during the transition.
# Phase 0a: prefer get_prompt(role) from llm.py.
_LEGACY_NAMES = {
    "clarify_with_user_instructions": "clarify",
    "transform_messages_into_research_topic_prompt": "research_brief",
    "lead_researcher_prompt": "supervisor",
    "research_system_prompt": "researcher",
    "compress_research_system_prompt": "compressor",
    "compress_research_simple_human_message": "compressor",
    "final_report_generation_prompt": "writer",
    "summarize_webpage_prompt": "webpage_summarizer",
}


def __getattr__(name):
    """Module-level __getattr__ for backward-compat with `from .prompts import X`."""
    if name in _LEGACY_NAMES:
        role = _LEGACY_NAMES[name]
        spec = REGISTRY[role]
        mod = import_module(spec.module)
        if name in ("compress_research_system_prompt", "compress_research_simple_human_message"):
            # Both live in compressor_v1 module
            return getattr(mod, name)
        return getattr(mod, spec.attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
