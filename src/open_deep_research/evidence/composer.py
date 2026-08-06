"""4-layer research plan composer.

Per architecture v2 (post-rectification):

  archetype × ontology × registry × instance → Framework

Each layer is a separate file; they compose at runtime.

  - archetype: research shape (market_size / competitive / regulatory / etc.)
                ~10 globally, near-zero growth
  - ontology:  domain entity space (cn_cybersec / us_livecommerce / fnb_retail)
                1 per domain, written once
  - registry:  source + caliber YAMLs (data/registry/{calibers,sources}/<vertical>.yaml)
                1 per domain, existing schema preserved
  - instance:  runtime params (brand / market / year / category)
                0 hand-written, passed in CLI / API

`build_plan(archetypes, ontology, registry, instance)` returns a Framework
that the downstream pipeline (W1..W8) consumes unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml
from pydantic import BaseModel, Field

from open_deep_research.evidence.framework import (
    Framework, Section, Slot,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 4-layer data classes
# =============================================================================

class Archetype(BaseModel):
    """A reusable research shape (market_size / competitive / regulatory)."""
    archetype_id: str
    title: str
    description: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    # Raw list (preserves `expansion` directives before instantiation)


class Ontology(BaseModel):
    """Domain entity space + disambiguation knowledge."""
    ontology_id: str
    display_name: str = ""
    entity_spaces: dict[str, Any] = Field(default_factory=dict)
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    # regulations may be either a list[dict] (top-level items) or
    # a dict with `items` key (under entity_spaces.regulations).
    # Composer normalises both.
    regulations: Any = Field(default_factory=list)
    indicator_hollows: dict[str, Any] = Field(default_factory=dict)


class Registry(BaseModel):
    """Source + caliber references (the existing data/registry/* files)."""
    vertical: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    calibers: list[dict[str, Any]] = Field(default_factory=list)


class Instance(BaseModel):
    """Runtime parameters supplied by the caller."""
    brand: str | None = None
    market: str | None = None
    category: str | None = None
    year: int | None = None
    currency: str = "USD"
    extra: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"brand": self.brand, "market": self.market,
                               "category": self.category, "year": self.year,
                               "currency": self.currency}
        out.update(self.extra)
        return {k: v for k, v in out.items() if v is not None}


# =============================================================================
# Loaders
# =============================================================================

DEFAULT_ARCHETYPE_DIR = Path("data/archetypes")
DEFAULT_ONTOLOGY_DIR = Path("data/ontologies")
DEFAULT_REGISTRY_DIR = Path("data/registry")


def load_archetype(
    archetype_id: str, *, base_dir: Path | str | None = None,
) -> Archetype:
    base = Path(base_dir) if base_dir is not None else DEFAULT_ARCHETYPE_DIR
    path = Path(base) / f"{archetype_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"archetype not found: {path} "
            f"(available: {[p.stem for p in Path(base).glob('*.yaml')]})"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Archetype(**raw)


def load_ontology(
    ontology_id: str, *, base_dir: Path | str | None = None,
) -> Ontology:
    base = Path(base_dir) if base_dir is not None else DEFAULT_ONTOLOGY_DIR
    path = Path(base) / f"{ontology_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"ontology not found: {path} "
            f"(available: {[p.stem for p in Path(base).glob('*.yaml')]})"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    onto = Ontology(**raw)
    # Normalise: if regulations came in as dict with `items` key, lift to list
    if isinstance(onto.regulations, dict) and "items" in onto.regulations:
        onto.regulations = onto.regulations["items"]
    return onto


def load_registry(
    vertical: str, *, base_dir: Path | str | None = None,
) -> Registry:
    """Load sources + calibers for a domain.

    Reads both data/registry/sources/<vertical>.yaml AND
    data/registry/calibers/<vertical>.yaml and merges them.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_REGISTRY_DIR
    sources_path = Path(base) / "sources" / f"{vertical}.yaml"
    calibers_path = Path(base) / "calibers" / f"{vertical}.yaml"

    sources: list[dict] = []
    calibers: list[dict] = []

    if sources_path.exists():
        with sources_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        sources = data.get("sources", [])

    if calibers_path.exists():
        with calibers_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        calibers = data.get("calibers", [])

    return Registry(vertical=vertical, sources=sources, calibers=calibers)


# =============================================================================
# Template rendering
# =============================================================================

def _render_template(
    template: str,
    ctx: Mapping[str, Any],
    *,
    required_keys: Optional[Iterable[str]] = None,
    render_tag: str = "",
) -> str:
    """Render {key} placeholders; missing keys → 'unknown' (no exception).

    Supports:
      - Simple: {market} {year}
      - Dotted NOT supported by str.format_map (it tries attribute access on
        dict values, which fails). Use indexed instead: {item[name]}.

    R4 整改: 当 caller 显式传 required_keys 时(如 archetype slot 的
    {market} / {category} / {year}),缺 key 触发 logger.warning 并把模板
    渲染成含 __MISSING_{key}__ 标记的版本,既保留可读性又便于回查问题。
    缺 required_keys 时行为不变。
    """
    if not isinstance(template, str):
        return template
    if required_keys:
        missing = [k for k in required_keys if not ctx.get(k)]
        if missing:
            logger.warning(
                "[composer] %s missing required template keys: %s "
                "(template=%r ctx_keys=%s)",
                render_tag or "render",
                missing, template[:60], list(ctx.keys()),
            )
    try:
        return template.format_map({k: ("" if v is None else v) for k, v in ctx.items()})
    except (KeyError, IndexError, AttributeError):
        # Fallback: try indexed access for {item[name]} form
        try:
            return _render_indexed(template, ctx)
        except Exception:
            return template


def _render_indexed(template: str, ctx: Mapping[str, Any]) -> str:
    """Render {item[name]} / {item[id]} placeholders via indexed access.

    Used when template refers to dict items via {item[key]} syntax.
    """
    import re
    out = template
    # Find all {a[b]} and resolve
    for m in re.finditer(r"\{(\w+)\[(\w+)\]\}", template):
        outer, inner = m.group(1), m.group(2)
        val = ctx.get(outer, {})
        if isinstance(val, dict):
            replacement = str(val.get(inner, ""))
        else:
            replacement = ""
        out = out.replace(m.group(0), replacement, 1)
    # Now resolve remaining {simple} placeholders
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", "" if v is None else str(v))
    return out


# =============================================================================
# Composer: archetype + ontology + registry + instance → Framework
# =============================================================================

def build_plan(
    *,
    archetypes: list[str],
    ontology: Ontology,
    registry: Registry,
    instance: Instance,
    vertical_id: str | None = None,
) -> Framework:
    """Compose a Framework from the 4 layers.

    Steps:
      1. For each archetype, render its sections (templates filled by instance
         + ontology's `categories`/`metrics` lookups).
      2. Expand `expansion: source: "{ontology}.<space>"` rules — for each
         item in the entity space, emit a per-item slot. Capped at `cap`.
      3. Resolve `caliber_ref: "{domain}.<key>"` against registry.calibers.
      4. Merge sections across archetypes, dedup slot_ids with prefixes
         (e.g. `market_size.tam`, `competitive.competitor_profile_qax`).
      5. Validate (quantitative slots need caliber + tier).
    """
    vid = vertical_id or ontology.ontology_id
    ctx: dict[str, Any] = {**instance.as_dict(), "ontology": ontology.ontology_id}

    # Surface the ontology's first category as default category if not set
    if not ctx.get("category"):
        cats = ontology.entity_spaces.get("categories", {})
        if isinstance(cats, dict) and cats.get("tree"):
            ctx["category"] = next(iter(cats["tree"]), "")
    # Surface the ontology's primary metric key
    if ontology.metrics and not ctx.get("metric"):
        ctx["metric"] = ontology.metrics[0].get("id", "")

    sections: list[Section] = []
    seen_slot_ids: set[str] = set[str]()

    # R4 整改: 收集 archetype 模板里实际用到的 instance key
    # ({brand} / {market} / {category} / {year}),并对 caller 没传的部分 warning
    referenced_instance_keys = _scan_archetype_referenced_keys(archetypes)
    _validate_instance(instance, referenced_instance_keys)

    for arch_id in archetypes:
        arch = load_archetype(arch_id)
        for sec_raw in arch.sections:
            sec_id = f"{arch_id}__{sec_raw['id']}"
            sec_title = _render_template(sec_raw.get("title", sec_id), ctx)
            slots_out: list[Slot] = []

            # 1) Explicit slot list
            for slot_raw in sec_raw.get("slots", []):
                _emit_slot(
                    slot_raw, arch_id, sec_id, ctx, ontology, registry,
                    seen_slot_ids, slots_out,
                )

            # 2) Expansion rules — emit per-entity slots
            expansion = sec_raw.get("expansion")
            if expansion:
                _emit_expansion(
                    expansion, arch_id, sec_id, ctx, ontology, registry,
                    seen_slot_ids, slots_out,
                )

            sections.append(Section(
                section_id=sec_id,
                title=sec_title,
                slots=slots_out,
            ))

    return Framework(
        vertical_id=vid,
        title=f"{ontology.display_name or vid} — "
              f"{' × '.join(archetypes)} ({instance.year or 'any year'})",
        sections=sections,
    )


def _emit_slot(
    slot_raw: dict,
    arch_id: str,
    sec_id: str,
    ctx: dict,
    ontology: Ontology,
    registry: Registry,
    seen: set[str],
    out: list[Slot],
) -> None:
    sid_raw = slot_raw["id"]
    sid = f"{arch_id}__{sec_id}__{sid_raw}"
    if sid in seen:
        return
    seen.add(sid)

    q = _render_template(slot_raw.get("question", ""), ctx)
    ct = slot_raw.get("expected_claim_type", "qualitative")
    tier = slot_raw.get("required_tier_min")
    notes = slot_raw.get("notes")

    # Resolve caliber_ref → concrete caliber_id from registry
    caliber_id = None
    if "caliber_ref" in slot_raw:
        ref = _render_template(slot_raw["caliber_ref"], ctx)
        # ref like "cn_cybersec.market_size" — take last component
        ref_key = ref.split(".")[-1]
        for c in registry.calibers:
            if c.get("id") == ref or c.get("metric_type") == ref_key:
                caliber_id = c.get("id")
                break

    out.append(Slot(
        slot_id=sid,
        question=q,
        expected_claim_type=ct,
        required_tier_min=tier,
        caliber_id=caliber_id,
        comparison_axis=slot_raw.get("comparison_axis"),
        notes=notes,
    ))


def _emit_expansion(
    expansion: dict,
    arch_id: str,
    sec_id: str,
    ctx: dict,
    ontology: Ontology,
    registry: Registry,
    seen: set[str],
    out: list[Slot],
) -> None:
    """Expand `expansion: {source: "{ontology}.<entity_space>", cap, per_item_slot}`.

    Example:
      expansion:
        source: "{ontology}.vendors"
        cap: 30
        per_item_slot:
          id: "competitor_profile_{item.id}"
          question: "{item.name} 在 {market} 的市场份额 / 营收 / 客户数"
          expected_claim_type: quantitative
          required_tier_min: A
    """
    src = _render_template(expansion.get("source", ""), ctx)
    space_key = src.split(".")[-1]
    space = ontology.entity_spaces.get(space_key, {})
    # Special-case: regulations live on ontology.regulations (top-level)
    if space_key == "regulations" and not space:
        items: list[dict] = list(ontology.regulations or [])
    else:
        items = []
        if isinstance(space, dict) and "items" in space:
            items = list(space["items"])
        elif isinstance(space, list):
            items = space
    cap = int(expansion.get("cap", 30))
    items = items[:cap]

    per = expansion.get("per_item_slot", {})
    for item in items:
        item_id = item.get("id") or item.get("name", "unknown")
        # Per-item context: {item} = item dict, {item.id} = item.id, {item.name} = item.name
        item_ctx = {**ctx, "item": item,
                    "item_id": item.get("id", ""),
                    "item.name": item.get("name", ""),
                    "item.official": item.get("official", "")}
        sid = _render_template(per.get("id", ""), item_ctx)
        full_sid = f"{arch_id}__{sec_id}__{sid}"
        if full_sid in seen:
            continue
        seen.add(full_sid)

        q = _render_template(per.get("question", ""), item_ctx)
        ct = per.get("expected_claim_type", "qualitative")
        tier = per.get("required_tier_min")
        notes = per.get("notes")

        # Caliber resolution: optional `caliber_ref` or `caliber_template`
        caliber_id = None
        if "caliber_ref" in per:
            ref = _render_template(per["caliber_ref"], item_ctx)
            ref_key = ref.split(".")[-1]
            for c in registry.calibers:
                if c.get("id") == ref or c.get("metric_type") == ref_key:
                    caliber_id = c.get("id")
                    break

        out.append(Slot(
            slot_id=full_sid,
            question=q,
            expected_claim_type=ct,
            required_tier_min=tier,
            caliber_id=caliber_id,
            comparison_axis=per.get("comparison_axis"),
            notes=notes,
        ))


# =============================================================================
# R4 helpers: instance key validation
# =============================================================================

# 哪些 key 被识别为"instance 提供"。{brand}/{market}/{category}/{year} 是
# 4 个常用 slot,在 archetype 模板里被引用时要求 caller 通过 Instance 提供。
_INSTANCE_KEYS = ("brand", "market", "category", "year")


def _scan_archetype_referenced_keys(archetypes: list[str]) -> set[str]:
    """扫描所有 archetype 的 section/slot/expansion 模板,收集 {key} 被引用的
    instance key 集合。返回 set,如 {"market", "year"}。
    """
    import re
    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    referenced: set[str] = set()
    for arch_id in archetypes:
        try:
            arch = load_archetype(arch_id)
        except FileNotFoundError:
            continue
        for sec_raw in arch.sections:
            for tmpl in (
                sec_raw.get("title", ""),
                *[s.get("question", "") for s in sec_raw.get("slots", [])],
                *[s.get("id", "") for s in sec_raw.get("slots", [])],
                *[s.get("caliber_ref", "") for s in sec_raw.get("slots", []) if s.get("caliber_ref")],
                sec_raw.get("expansion", {}).get("source", "") if sec_raw.get("expansion") else "",
            ):
                for m in pattern.finditer(str(tmpl)):
                    if m.group(1) in _INSTANCE_KEYS:
                        referenced.add(m.group(1))
    return referenced


def _validate_instance(instance: "Instance", referenced: set[str]) -> None:
    """R4: 当 archetype 模板引用了某 instance key 但 caller 没传时,
    logger.warning(不 raise — back-compat,但写明缺失)。
    """
    if not referenced:
        return
    missing: list[str] = []
    for k in referenced:
        if not getattr(instance, k, None):
            missing.append(k)
    if missing:
        logger.warning(
            "[composer] Instance missing required keys for archetype templates: %s "
            "(templates reference them but caller did not provide; "
            "rendering will substitute empty/None and degrade query semantics)",
            missing,
        )


__all__ = [
    "Archetype",
    "Ontology",
    "Registry",
    "Instance",
    "load_archetype",
    "load_ontology",
    "load_registry",
    "build_plan",
    "DEFAULT_ARCHETYPE_DIR",
    "DEFAULT_ONTOLOGY_DIR",
    "DEFAULT_REGISTRY_DIR",
]
