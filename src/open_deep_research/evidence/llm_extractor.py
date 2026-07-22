"""Phase 4 (= Runbook v1 阶段 2.1) LLM 抽取器。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 2.1

按方案 A:废弃 deterministic 抽取的唯一性,改为每页调 LLM 抽 EU。
为兼容旧调用方,保留旧 eu_extractor.extract_from_search_results(),
并在 pipeline 中通过 use_llm_extractor 切换。

输出形态:EvidenceUnitV2(已含 source_span + claim_type + entities)。
下游 eu_dao.upsert_many + EuDAO.update_verification 直接消费。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from open_deep_research.evidence.schema import ClaimType, EvidenceUnitV2, SourceTier
from open_deep_research.llm import get_prompt
from open_deep_research.prompts import EXTRACT_PROMPT

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[str]:
    """抽取首个 JSON 对象(支持 ```json fence 或裸 brace 匹配)。"""
    if not text:
        return None
    # 先尝试 ```json 围栏
    if "```" in text:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("```") and s.endswith("```"):
                inner = s[3:-3].strip()
                if inner.startswith("json"):
                    inner = inner[4:].strip()
                if inner.startswith("{"):
                    return inner
    # 退化:首 { 到末 }
    if "{" in text:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _parse_one(
    raw_eu: dict[str, Any],
    *,
    source_url: str,
    source_title: Optional[str],
    run_id: UUID,
    extractor_model: str,
    extracted_at: datetime,
) -> Optional[EvidenceUnitV2]:
    """把 LLM 输出的一条 dict 转换为 EvidenceUnitV2。

    失败的转换(关键字段缺失 / 类型错)返回 None,调用方决定是 skip 还是 raise。
    """
    claim = (raw_eu.get("claim") or "").strip()
    span = (raw_eu.get("source_span") or "").strip()
    if not claim or not span:
        return None
    if len(span) < 10:
        # Runbook 1.1 schema 要求 source_span ≥ 10 字符
        return None

    claim_type_raw = (raw_eu.get("claim_type") or "attribute").strip()
    valid_types = ("numeric", "event", "attribute", "relation", "opinion")
    claim_type: ClaimType = claim_type_raw if claim_type_raw in valid_types else "attribute"

    entities = raw_eu.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if e]

    norm_value = raw_eu.get("norm_value")
    unit = raw_eu.get("unit")
    value_as_of = raw_eu.get("value_as_of")  # str 'YYYY-MM-DD' or None

    domain = (urlsplit(source_url).hostname or "").lower()

    # claim_type=numeric 但 norm_value 缺失 → Pydantic 会拒绝,这里先标记
    if claim_type == "numeric" and norm_value is None:
        # 让 schema 校验拦下,跳过
        return None

    # source_tier 默认 tertiary — 阶段 3 白名单驱动升级
    source_tier: SourceTier = "tertiary"

    # value_as_of:接受 'YYYY-MM-DD' 字符串,留 Pydantic 校验
    from datetime import date
    value_as_of_date = None
    if isinstance(value_as_of, str):
        try:
            value_as_of_date = date.fromisoformat(value_as_of)
        except ValueError:
            value_as_of_date = None

    from decimal import Decimal
    norm_value_dec = None
    if isinstance(norm_value, (int, float, str)):
        try:
            norm_value_dec = Decimal(str(norm_value))
        except Exception:
            norm_value_dec = None

    return EvidenceUnitV2(
        eu_id=uuid4(),
        run_id=run_id,
        dimension_id=None,  # 阶段 3 才接 planner
        claim=claim,
        claim_type=claim_type,
        entities=entities,
        norm_value=norm_value_dec,
        unit=str(unit) if unit else None,
        value_as_of=value_as_of_date,
        source_url=source_url,
        source_domain=domain,
        source_title=source_title,
        published_at=None,
        source_tier=source_tier,
        source_span=span,
        span_start=None,
        span_end=None,
        extractor_model=extractor_model,
        extracted_at=extracted_at,
        span_verified=False,
        numeric_drift=False,
        entailment_verdict=None,
        entailment_score=None,
        claim_id=None,
        content_hash=None,
    )


def _parse_response(raw: str, *, n_expected: int = 0) -> list[dict[str, Any]]:
    """解析 LLM 输出 JSON,返回 raw_eu 列表。失败时返回空 + warning。"""
    if not raw:
        return []
    payload = _extract_json(raw)
    if payload is None:
        logger.warning("llm_extractor: no JSON found in response (len=%d)", len(raw))
        return []
    try:
        obj = json.loads(payload)
    except Exception as e:
        logger.warning("llm_extractor: JSON parse error: %s", e)
        return []
    items = obj.get("evidence_units") or []
    if not isinstance(items, list):
        return []
    return items


async def extract_from_content_with_llm(
    *,
    content: str,
    source_url: str,
    source_title: Optional[str],
    run_id: UUID | str,
    sub_query: str = "",
    llm: Any,
    extractor_model: str = "extractor_v1",
) -> list[EvidenceUnitV2]:
    """对单页正文调 LLM 抽 EU。

    失败(LLM 异常 / JSON 解析失败)返回空列表 — 调用方决定是否降级。
    """
    rid = UUID(run_id) if isinstance(run_id, str) else run_id
    try:
        prompt = EXTRACT_PROMPT.format(sub_query=sub_query, content=content)
        messages = [
            {"role": "system", "content": "You are a strict evidence extractor."},
            {"role": "user", "content": prompt},
        ]
        response = await llm.ainvoke(messages)
        raw = getattr(response, "content", str(response))
        if not isinstance(raw, str):
            raw = str(raw)
    except Exception as e:
        logger.warning("llm_extractor: LLM call failed (url=%s): %s", source_url, e)
        return []

    raws = _parse_response(raw)
    extracted_at = datetime.now(timezone.utc)
    eus: list[EvidenceUnitV2] = []
    for r in raws:
        try:
            eu = _parse_one(
                r,
                source_url=source_url,
                source_title=source_title,
                run_id=rid,
                extractor_model=extractor_model,
                extracted_at=extracted_at,
            )
        except Exception as e:
            logger.debug("llm_extractor: parse_one failed: %s", e)
            eu = None
        if eu is not None:
            eus.append(eu)
    return eus


async def extract_from_search_results_with_llm(
    results: Iterable[dict[str, Any]],
    *,
    run_id: UUID | str,
    llm: Any,
    sub_query: str = "",
    extractor_model: str = "extractor_v1",
    content_key: str = "raw_content",
    fallback_content_keys: tuple[str, ...] = ("summary", "content"),
) -> list[EvidenceUnitV2]:
    """对一批 search result 抽 EU。

    content 优先取 `raw_content`(Crawl4AI 抓全文后的字段),降级到
    `summary` / `content`(Tavily 已有)。
    """
    rid = UUID(run_id) if isinstance(run_id, str) else run_id
    out: list[EvidenceUnitV2] = []
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        title = r.get("title")
        content = r.get(content_key) or ""
        if not content:
            for k in fallback_content_keys:
                v = r.get(k)
                if v:
                    content = str(v)
                    break
        if not content:
            # 没正文:跳过(覆盖率为 0,但不强行"创造"EU)
            continue
        eus = await extract_from_content_with_llm(
            content=content,
            source_url=url,
            source_title=title,
            run_id=rid,
            sub_query=sub_query,
            llm=llm,
            extractor_model=extractor_model,
        )
        out.extend(eus)
    return out


__all__ = [
    "extract_from_content_with_llm",
    "extract_from_search_results_with_llm",
    "_parse_one",
    "_parse_response",
]