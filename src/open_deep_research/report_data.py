"""Phase 3b: ReportDataObject — single-source-of-truth between text + tables.

## Why

v1 baseline defect (D1-D3 in v1_defects.md): the report contains prose
that says one thing and a table that says another (e.g. "Klue ranks #1
in X" in prose, table shows Klue at #3). The two views drift because the
LLM regenerates each independently.

Plan v2 fix: every numeric / categorical claim in the report is stored
as a typed `DataRow`; both the prose renderer and the table renderer read
from this same list. The prose wording is generated *from* the DataRow
("Klue ranks #2 in CI platforms" derives from DataRow(product="Klue",
rank=2)). The table is just `DataRow` → grid. Any drift is now
impossible by construction.

Rule 4 (URL page-level) — final reports MUST NOT contain domain-only
URLs as references. Domain-only URLs are flagged at write time and
either re-resolved (Crawl4AI) or replaced with `[UNVERIFIED_DOMAIN_ONLY]`
so they can be revised by a downstream editor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

from open_deep_research.sources_dao import classify_page_level, PageLevel


# =============================================================================
# DataRow
# =============================================================================

@dataclass
class DataRow:
    """A single data point in the report. Every numeric / categorical claim
    referenced in prose also appears as one of these (or as a derived form).

    `prose_template` is a `{key}` templated string used to render the prose
    paragraph for this row — same template fills the cell text in tables.
    """

    key: str                          # stable id (e.g. "klue_rank_2026")
    label: str                        # human label (e.g. "Klue")
    category: str                     # 'ranking' / 'price' / 'feature' / 'market_share'
    values: dict[str, Any]            # numeric or categorical fields
    unit: Optional[str] = None        # 'USD/year' etc.
    provenance: str = ""              # human-readable source attribution
    source_url: str = ""              # URL the row is grounded in
    source_id: Optional[int] = None
    eu_ids: list[str] = field(default_factory=list)
    confidence: float = 0.7
    prose_template: str = ""          # e.g. "{label} ranks #{values[rank]} in {category}"
    table_columns: list[str] = field(default_factory=list)
    is_estimated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def render_prose(self) -> str:
        """Render the prose sentence for this row from the template."""
        if not self.prose_template:
            return ""
        try:
            return self.prose_template.format(label=self.label, **self.values)
        except Exception:
            # Fallback: include values verbatim.
            return f"{self.label}: {self.values}"


# =============================================================================
# ReportDataObject
# =============================================================================

@dataclass
class ReportSection:
    heading: str
    rows: list[DataRow] = field(default_factory=list)
    prose_lead: str = ""              # short intro paragraph
    prose_footer: str = ""            # optional footnote

    def add_row(self, row: DataRow) -> None:
        self.rows.append(row)

    def to_markdown_table(self) -> str:
        """Render rows as a Markdown table. Header is derived from the
        union of `table_columns` across rows; missing cells → '—'."""
        cols: list[str] = []
        for r in self.rows:
            for c in r.table_columns:
                if c not in cols:
                    cols.append(c)
        if not cols:
            return ""
        # Always start with the label
        ordered = ["label"] + [c for c in cols if c != "label"]
        header = "| " + " | ".join(ordered) + " |"
        sep = "| " + " | ".join("---" for _ in ordered) + " |"
        body = []
        for r in self.rows:
            cells = ["**" + r.label + "**"]
            for c in ordered[1:]:
                v = r.values.get(c, "—")
                cells.append(str(v) if v is not None else "—")
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep, *body])

    def to_prose(self) -> str:
        """Render prose paragraph(s) — one sentence per row, sharing prose_template."""
        parts: list[str] = []
        if self.prose_lead:
            parts.append(self.prose_lead)
        for r in self.rows:
            txt = r.render_prose()
            if txt:
                parts.append(txt)
        if self.prose_footer:
            parts.append(self.prose_footer)
        return "\n\n".join(parts)


@dataclass
class ReportDataObject:
    """Single source of truth for the final report.

    Holds:
      - title
      - ordered sections (each section has DataRow[] + prose lead/footer)
      - a flat index of *all* rows (for cross-section lookups by key)

    The renderer (`to_markdown`) is the *only* path that produces the final
    report; it derives both prose and table views from the same DataRow set.
    """

    title: str
    sections: list[ReportSection] = field(default_factory=list)
    _row_index: dict[str, DataRow] = field(default_factory=dict)

    def add_section(self, heading: str, prose_lead: str = "") -> ReportSection:
        sec = ReportSection(heading=heading, prose_lead=prose_lead)
        self.sections.append(sec)
        return sec

    def add_row(self, section: ReportSection, row: DataRow) -> None:
        if not row.key:
            row.key = f"r_{len(self._row_index)}"
        section.add_row(row)
        self._row_index[row.key] = row

    def get_row(self, key: str) -> Optional[DataRow]:
        return self._row_index.get(key)

    def to_markdown(self) -> str:
        out: list[str] = [f"# {self.title}", ""]
        for sec in self.sections:
            out.append(f"## {sec.heading}")
            out.append("")
            if sec.prose_lead:
                out.append(sec.prose_lead)
                out.append("")
            for r in sec.rows:
                if r.render_prose():
                    out.append("- " + r.render_prose())
            out.append("")
            tbl = sec.to_markdown_table()
            if tbl:
                out.append(tbl)
                out.append("")
            if sec.prose_footer:
                out.append(sec.prose_footer)
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "sections": [
                {
                    "heading": s.heading,
                    "prose_lead": s.prose_lead,
                    "prose_footer": s.prose_footer,
                    "rows": [r.to_dict() for r in s.rows],
                }
                for s in self.sections
            ],
        }


# =============================================================================
# Rule 4 — page-level URL enforcement
# =============================================================================

@dataclass
class UrlComplianceIssue:
    raw_url: str
    new_url_or_label: str   # '' if domain-only, no replacement available
    severity: str           # 'high' / 'medium'
    where: str              # 'prose' / 'table' / 'source_url'
    detail: str
    # Number of distinct sites in the RDO where this raw_url appeared.
    # Dedup key is (section_heading, raw_url); if the writer reuses the
    # same EU across N rows, we emit ONE issue with occurrences=N
    # instead of N identical issues.
    occurrences: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def enforce_page_level(
    rdo: ReportDataObject,
    *,
    resolver: Any = None,
    placeholder: str = "[UNVERIFIED_DOMAIN_ONLY]",
) -> list[UrlComplianceIssue]:
    """Walk every URL in `rdo` and flag domain-only ones.

    `resolver(url) -> str` is an optional function that re-resolves a
    domain-only URL to a page-level URL (via Crawl4AI, for example).
    If the resolver returns a page-level URL, the substitution happens.
    Otherwise the URL is replaced by the placeholder.

    Dedup: identical (section, raw_url) hits collapse into one issue
    with an `occurrences` counter. The in-place replacement still
    visits every row so each row's source_url is rewritten exactly
    once (regardless of how many issues we emit).
    """
    # Phase 1 — discover all audit candidates keyed by (section, raw_url)
    # so the same URL appearing in N rows of one section yields one issue.
    pending: dict[tuple[str, str], UrlComplianceIssue] = {}
    for sec in rdo.sections:
        # ----- DataRow.source_url -----
        for r in sec.rows:
            cls = classify_page_level(r.source_url)
            if cls is PageLevel.DOMAIN_ONLY and r.source_url:
                key = (sec.heading, r.source_url)
                if key not in pending:
                    pending[key] = _audit(
                        r.source_url, sec.heading, "source_url", "high",
                        resolver, placeholder,
                    )
                else:
                    pending[key].occurrences += 1
        # ----- prose_lead / prose_footer (text scan) -----
        for field_name in ("prose_lead", "prose_footer"):
            blob = getattr(sec, field_name) or ""
            for url in re.findall(r"https?://\S+", blob):
                if classify_page_level(url) is PageLevel.DOMAIN_ONLY:
                    # prose positions are distinct audit sites even if
                    # they happen to share the same URL — keep them
                    # separate keys so the writer sees both.
                    key = (sec.heading, f"prose:{url}")
                    pending[key] = _audit(
                        url, sec.heading, "prose", "medium",
                        resolver, placeholder,
                    )

    issues = list(pending.values())

    # Phase 2 — mutate the rdo in-place. We walk the full RDO (not the
    # deduped issues) so every row gets the placeholder substitution,
    # but we use each issue's new_url_or_label so behaviour matches
    # the issue we emitted.
    by_url = {i.raw_url: i.new_url_or_label for i in issues}
    for sec in rdo.sections:
        for r in sec.rows:
            if r.source_url in by_url:
                r.source_url = by_url[r.source_url]
        for field_name in ("prose_lead", "prose_footer"):
            blob = getattr(sec, field_name) or ""
            for raw_url, replacement in by_url.items():
                if raw_url in blob:
                    setattr(
                        sec, field_name,
                        blob.replace(raw_url, replacement),
                    )
                    blob = getattr(sec, field_name) or ""
    return issues


def _audit(url, section, where, severity, resolver, placeholder):
    """Apply a single audit step: resolve if possible, else replace with placeholder."""
    new = ""
    if resolver is not None:
        try:
            new = resolver(url)
        except Exception:
            new = ""
    if not new or classify_page_level(new) is not PageLevel.PAGE:
        new = placeholder
    return UrlComplianceIssue(
        raw_url=url,
        new_url_or_label=new,
        severity=severity,
        where=f'{where} (section "{section}")',
        detail=(
            f"Domain-only URL detected in {where}; replaced with: {new}"
            if new == placeholder else
            f"Domain-only URL resolved to page-level: {new}"
        ),
    )
