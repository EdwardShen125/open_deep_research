"""A3a · 确定性验证门(span + numeric drift)

按 plan A3a:三门里前两门无需 LLM、确定性、便宜。
应先落地并**独立于 entailment 门**——即使 LLM 门挂了(MiniMax 401),
这两门仍在把关,不会全盘静默通过。

接口:
    span_gate(claim, source_text) -> bool
        字面命中:claim 中的数值 / 实体是否在 source_text 中找到。
        找不到 → False

    drift_gate(claim, source_text, *, threshold=0.05) -> float
        数值漂移:claim 中的数值 vs source_text 中近邻数值的最大相对偏差。
        超阈值(如 5%)→ 调用方标 verification_status='to_verify'(不丢)

失败处理:
    span=False → GateResults.span=False → verification_status='to_verify'
    drift > threshold → GateResults.drift=实际值 → verification_status='to_verify'
    任何路径都不静默吞掉。
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DRIFT_THRESHOLD = 0.05  # 5% 相对偏差


# =============================================================================
# Numeric extraction
# =============================================================================

# 匹配整数、小数、带千分位、带百分号、带单位简写
_NUMBER_RE = re.compile(
    r"""
    (?<!\d)                        # 前面不是数字
    -?\d{1,3}(?:,\d{3})+           # 千分位整数
    (?: \.\d+)?                    # 可选小数
    (?:%)?                         # 可选百分号(不带空格)
    |
    -?\d+\.\d+                     # 普通小数
    |
    -?\d+                          # 整数
    """,
    re.VERBOSE,
)


def extract_numbers(text: str) -> list[Decimal]:
    """从文本里抽出所有数值(返回 Decimal 列表,去千分位)。"""
    if not text:
        return []
    out: list[Decimal] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "").replace("%", "").strip()
        try:
            out.append(Decimal(raw))
        except (InvalidOperation, ValueError):
            continue
    return out


def normalize_number_token(s: str) -> Optional[str]:
    """把数字字符串标准化(去千分位、去百分号)用于字面命中比较。"""
    if not s:
        return None
    cleaned = s.replace(",", "").replace("%", "").strip()
    try:
        # 转 float 再回 str,做格式规范化("146.40" → "146.4")
        return str(float(cleaned))
    except (ValueError, TypeError):
        return None


# =============================================================================
# span_gate
# =============================================================================

def span_gate(
    claim: str,
    source_text: str,
    *,
    require_number: bool = True,
) -> bool:
    """字面命中:claim 中的"主张元素"是否在 source_text 里出现。

    检测 3 类元素(任一在 source_text 中找到即可):
        1. claim 中的所有数字(若 require_number=True)
        2. claim 中的 ≥4 字符连续英文/中文 token(实体近似)
        3. claim 中的整句(全字符串包含)——fallback

    Args:
        claim: 抽取出的断言文本
        source_text: 源文本(已爬到的原文或 source_span 之外的全文)
        require_number: numeric claim 必须找到数字才过

    Returns:
        True / False。False 时调用方标 verification_status='to_verify'。
    """
    if not claim or not source_text:
        return False

    claim_numbers = extract_numbers(claim)
    source_numbers = extract_numbers(source_text)
    source_numbers_normalized = {normalize_number_token(str(n)) for n in source_numbers}

    # 1. 数字字面命中
    if claim_numbers:
        for n in claim_numbers:
            n_norm = normalize_number_token(str(n))
            if n_norm and n_norm in source_numbers_normalized:
                return True
        # numeric claim 但数字未命中 → False
        if require_number:
            return False

    # 2. 实体近似:≥4 字符的英文 token
    entity_hits = 0
    for token in re.findall(r"\b[A-Za-z]{4,}\b", claim):
        if token.lower() in source_text.lower():
            entity_hits += 1
    if entity_hits >= 2:
        return True

    # 3. 中文实体近似:≥3 字符的中文串
    zh_hits = 0
    for token in re.findall(r"[\u4e00-\u9fff]{3,}", claim):
        if token in source_text:
            zh_hits += 1
    if zh_hits >= 2:
        return True

    # 4. 全字符串包含(fallback,最宽松)
    if claim.strip() in source_text:
        return True

    return False


# =============================================================================
# drift_gate
# =============================================================================

def drift_gate(
    claim: str,
    source_text: str,
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> float:
    """数值漂移:claim 中数值 vs source_text 中近邻数值的最大相对偏差。

    找出 claim 中的每个数值,在 source_text 数值集合中找最近的(绝对距离最小),
    计算相对偏差 = |claim_n - nearest_source_n| / max(|nearest_source_n|, 1)。

    返回:
        所有 claim 数值的"最近邻相对偏差"的**最大值**。
        0.0 表示 claim 中每个数值都能在 source 中找到精确匹配;
        0.10 表示某个 claim 数值漂移 10%(超阈值)。

    注:返回值不强制 threshold 检查;调用方对比 `drift > threshold` 自行判定。
    """
    claim_numbers = extract_numbers(claim)
    source_numbers = [float(n) for n in extract_numbers(source_text)]

    if not claim_numbers:
        return 0.0  # 无数字 → 无漂移概念
    if not source_numbers:
        # claim 有数字但 source 没有 → 视为无穷大漂移
        return float("inf")

    max_drift = 0.0
    for cn in claim_numbers:
        cn_f = float(cn)
        # 找最近邻
        nearest = min(source_numbers, key=lambda x: abs(x - cn_f))
        denom = max(abs(nearest), 1.0)  # 防除零
        drift = abs(cn_f - nearest) / denom
        if drift > max_drift:
            max_drift = drift

    return max_drift


def drift_passes(
    claim: str,
    source_text: str,
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> bool:
    """drift_gate 的便捷 boolean 包装。"""
    return drift_gate(claim, source_text, threshold=threshold) <= threshold


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DEFAULT_DRIFT_THRESHOLD",
    "extract_numbers",
    "normalize_number_token",
    "span_gate",
    "drift_gate",
    "drift_passes",
]
