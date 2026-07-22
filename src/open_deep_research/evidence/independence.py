"""Phase 5 (= Runbook v1 阶段 3.2 + 3.3 + 3.4) 源独立性与置信度分级。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 3.2-3.4

3.2 源独立性:把互相依附的源折叠成簇,返回独立簇数。
  - 注册域归一(同一 registrable_domain 视为同簇)
  - 通稿转载:正文相似度高 + 发布时间接近 → 同一簇
  - 引用依附:A 正文提及 B 机构名/域名 且 B 发布更早 → A 依附 B

3.3 置信度分级(纯规则):
  A: ≥2 个独立源一致
  B: 单一一手权威源
  C: 多源数值冲突 或 单一二手源
  D: 无任何 EU 通过 entailment

3.4 source_tier 白名单:domain → tier 的硬编码表;未命中默认 tertiary。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from open_deep_research.evidence.merge import ClaimDraft
from open_deep_research.evidence.schema import ClaimV2, EvidenceUnitV2, Grade


# =============================================================================
# 3.2 源独立性
# =============================================================================

def registrable_domain(domain: str) -> str:
    """简化版:eTLD+1 抽取。

    真实 PSL(Public Suffix List)在 production 需要 publicsuffix2 包,
    这里是简化版 — 取最后两段(覆盖 com/org/net/cn/co 等绝大多数场景)。
    如需精确(github.io, s3.amazonaws.com),接入 publicsuffix2。
    """
    if not domain:
        return ""
    d = domain.lower().strip().rstrip(".")
    parts = d.split(".")
    if len(parts) <= 2:
        return d
    return ".".join(parts[-2:])


def _content_emb_url(
    url: str,
    page_emb: dict[str, Any],
) -> Optional[Any]:
    """查 URL 在 page_emb 里的 embedding;找不到返回 None。"""
    if url in page_emb:
        return page_emb[url]
    # 兼容:按规范化 URL 查
    norm = urlsplit(url)
    base = f"{norm.scheme}://{norm.netloc}{norm.path}"
    if base in page_emb:
        return page_emb[base]
    return None


def _cosine(a: Any, b: Any) -> float:
    """两个向量余弦相似度,容忍 numpy / list / tuple 输入。"""
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
        va, vb = np.asarray(a), np.asarray(b)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(va @ vb / (na * nb))
    except Exception:
        # 退化 list 实现
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(min(la, lb)))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def _cites(a: EvidenceUnitV2, b: EvidenceUnitV2) -> bool:
    """A 的正文是否提及 B 的机构名或域名(简化版:substring 检查)。

    严格版本需要 NER + URL mention detection;这里用 source_domain substring
    作为弱信号。生产环境应接真 NER(LAC / stanza)。
    """
    if not a.source_span or not b.source_domain:
        return False
    return b.source_domain in a.source_span or (b.source_title or "") in a.source_span


def _earlier(a: EvidenceUnitV2, b: EvidenceUnitV2) -> bool:
    """A 是否早于 B 发布时间。"""
    if not a.published_at or not b.published_at:
        return False
    return a.published_at < b.published_at


def independent_source_count(
    eus: list[EvidenceUnitV2],
    *,
    page_emb: Optional[dict[str, Any]] = None,
    wire_threshold: float = 0.85,
    wire_time_window_hours: float = 72.0,
) -> int:
    """返回 eus 涉及到的独立源簇数。

    三层折叠:
      1) 同一 registrable_domain 视为同簇
      2) 正文相似 > 0.85 且发布时间差 ≤ 72h → 通稿转载,同簇
      3) A 引用 B 且 B 更早 → A 依附 B,同簇

    返回独立簇数(support_count 字段语义)。
    """
    if not eus:
        return 0

    # 1) 注册域归一
    clusters: dict[str, list[EvidenceUnitV2]] = defaultdict(list)
    for eu in eus:
        rd = registrable_domain(eu.source_domain)
        clusters[rd].append(eu)

    # 每个 cluster 取一个代表 EU(发布时间最早的)
    reps: list[EvidenceUnitV2] = [
        min(v, key=lambda e: e.published_at or datetime.max.replace(tzinfo=timezone.utc))
        for v in clusters.values()
    ]
    n = len(reps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 2) 通稿转载
    if page_emb is not None:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = reps[i], reps[j]
                ea = _content_emb_url(a.source_url, page_emb)
                eb = _content_emb_url(b.source_url, page_emb)
                if ea is None or eb is None:
                    continue
                sim = _cosine(ea, eb)
                if sim <= wire_threshold:
                    continue
                gap = float("inf")
                if a.published_at and b.published_at:
                    gap = abs((a.published_at - b.published_at).total_seconds())
                if gap <= wire_time_window_hours * 3600:
                    union(i, j)

    # 3) 引用依附
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = reps[i], reps[j]
            if _cites(a, b) and _earlier(b, a):
                # A(i) 引用 B(j) 且 B 更早 → A 依附 B(并簇到 B)
                union(i, j)

    return len({find(i) for i in range(n)})


def primary_source_count(eus: list[EvidenceUnitV2]) -> int:
    """返回涉及到的 primary tier 源数量(简化:不去重 cluster)。"""
    return sum(1 for eu in eus if eu.source_tier == "primary")


# =============================================================================
# 3.3 置信度分级
# =============================================================================

def grade_claim(
    claim: ClaimDraft,
    *,
    independent_count: int,
    primary_count: int,
    has_any_entailed: bool,
) -> tuple[Grade, str]:
    """纯规则分级,Runbook 3.3 原文。

    A: ≥2 个独立源一致
    B: 单一一手权威源
    C: 多源数值冲突 或 单一二手源
    D: 无任何 EU 通过 entailment
    """
    if not has_any_entailed:
        return "D", "无任何 EU 通过 entailment 校验"

    if claim.has_conflict:
        spread = f"{claim.value_spread:.1%}" if claim.value_spread is not None else "?"
        return "C", f"多源数值冲突,最大相对偏差 {spread}"

    if independent_count >= 2:
        return "A", f"{independent_count} 个独立源一致"

    if primary_count >= 1:
        return "B", "单一一手权威源"

    return "C", "单一二手源,未获独立证实"


# =============================================================================
# 3.4 source_tier 白名单
# =============================================================================

# 阶段 1 默认 tertiary;阶段 3 引入白名单驱动升级。
# 简化的手工列表(200 条目标)。production 应接数据库 / 配置文件。

PRIMARY_DOMAINS: frozenset[str] = frozenset({
    # 公司官网 / 公告 / 年报 / 招股书
    "kompyte.com", "crayon.com", "klue.com", "alphasense.com", "zoominfo.com",
    "6sense.com", "datarobot.com", "salesforce.com", "hubspot.com",
    "microsoft.com", "oracle.com", "sap.com",
    # 监管 / 统计部门
    "sec.gov", "edgar.sec.gov", "fda.gov", "cdc.gov", "europa.eu",
    "stats.gov.cn", "stats.gov", "gov.cn", "miit.gov.cn", "csrc.gov.cn",
    "pbc.gov.cn", "cbirc.gov.cn", "nmpa.gov.cn",
    # 法院文书 / 司法
    "wenshu.court.gov.cn", "court.gov.cn",
    # 中文垂直源(接 Runbook 提到的)
    "tianyancha.com", "qcc.com", "qichacha.com", "7md.cn", "qimai.cn",
    "iresearch.com.cn", "iresearch.cn",
})

SECONDARY_DOMAINS: frozenset[str] = frozenset({
    # 主流媒体
    "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com", "bloomberg.com",
    "reuters.com", "apnews.com", "bbc.com", "cnn.com", "theguardian.com",
    "techcrunch.com", "wired.com", "theverge.com", "arstechnica.com",
    "forbes.com", "businessinsider.com", "cnbc.com",
    # 中文主流
    "people.com.cn", "xinhuanet.com", "qq.com", "163.com", "sina.com.cn",
    "sohu.com", "ifeng.com", "thepaper.cn", "jiemian.com", "yicai.com",
    "caixin.com", "stcn.com", "21jingji.com", "cls.cn", "wallstreetcn.com",
    "36kr.com", "huxiu.com",
    # 行业媒体 / 券商研报
    "pitchbook.com", "cbinsights.com", "tracxn.com", "betakit.com",
    "prnewswire.com", "businesswire.com", "globenewswire.com",
})

UGC_DOMAINS: frozenset[str] = frozenset({
    # 论坛 / 问答 / 自媒体
    "reddit.com", "quora.com", "stackoverflow.com",
    "zhihu.com", "csdn.net", "jianshu.com", "bilibili.com",
    "weibo.com", "douban.com", "tieba.baidu.com",
    "medium.com", "substack.com",
})

# (anything not above) → tertiary (default)


def classify_source_tier(domain: str) -> str:
    """Runbook 3.4:domain → tier。命中白名单 → 对应 tier;未命中 → tertiary。"""
    d = (domain or "").lower().strip()
    if not d:
        return "tertiary"
    rd = registrable_domain(d)
    if rd in PRIMARY_DOMAINS or d in PRIMARY_DOMAINS:
        return "primary"
    if rd in SECONDARY_DOMAINS or d in SECONDARY_DOMAINS:
        return "secondary"
    if rd in UGC_DOMAINS or d in UGC_DOMAINS:
        return "ugc"
    return "tertiary"


def upgrade_source_tier(eu: EvidenceUnitV2) -> EvidenceUnitV2:
    """把 EU 的 source_tier 按白名单升级(默认 tertiary → 实际 tier)。"""
    new_tier = classify_source_tier(eu.source_domain)
    if new_tier == eu.source_tier:
        return eu
    return eu.model_copy(update={"source_tier": new_tier})  # type: ignore[arg-type]


__all__ = [
    "registrable_domain",
    "independent_source_count",
    "primary_source_count",
    "grade_claim",
    "classify_source_tier",
    "upgrade_source_tier",
    "PRIMARY_DOMAINS",
    "SECONDARY_DOMAINS",
    "UGC_DOMAINS",
]