"""Phase 4 (= Runbook v1 阶段 2) 三道闸。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 2.2-2.4

闸 1:verify_span — 字面/归一/模糊匹配,零 LLM
闸 2:has_numeric_drift — 数值漂移检测,零 LLM
闸 3:verify_entailment_batch — 批量 LLM 蕴含校验

闸 1 和闸 2 共享 _NUM / _SCALE / _numbers 帮助函数。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from open_deep_research.evidence.schema import Verdict


# =============================================================================
# 闸 1:span 字面校验 (Runbook 2.2)
# =============================================================================

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """去掉空白 + 统一全角标点,用于跨语言/标点形态的命中。"""
    text = _WS.sub("", text)
    return text.replace("，", ",").replace("。", ".").replace("％", "%")


def verify_span(
    span: str,
    content: str,
    fuzzy_threshold: float = 0.92,
    fuzzy_max_len: int = 400,
) -> tuple[bool, Optional[int]]:
    """字面命中优先,退化到归一化匹配,再退化到滑窗模糊匹配。

    返回 (is_hit, position):
      is_hit=True  position=命中位置(content 内 offset) or None(模糊命中,位置无意义)
      is_hit=False position=None

    阈值与最大长度通过参数暴露,便于测试。
    """
    if not span or not content:
        return False, None
    idx = content.find(span)
    if idx >= 0:
        return True, idx

    ns, nc = _normalize(span), _normalize(content)
    idx = nc.find(ns)
    if idx >= 0:
        # 模糊命中:在归一化串上找到,不能直接反推回原文 offset
        return True, None

    if len(ns) > fuzzy_max_len:
        # 过长片段不做模糊匹配(避免 O(n*window) 慢),直接判失败
        return False, None

    window = len(ns)
    best = 0.0
    # 步长 window // 4(25% 滑窗)—— 兼顾精度和性能
    step = max(1, window // 4)
    for start in range(0, max(1, len(nc) - window + 1), step):
        ratio = SequenceMatcher(None, ns, nc[start : start + window]).ratio()
        if ratio > best:
            best = ratio
        if best >= fuzzy_threshold:
            return True, None
    return False, None


# =============================================================================
# 闸 2:数值漂移检测 (Runbook 2.3)
# =============================================================================

_NUM = re.compile(r"(\d+(?:[,，]\d{3})*(?:\.\d+)?)\s*(万亿|亿|万|千|百分点|%|％)?")

_SCALE: dict[str | None, float] = {
    "万": 1e4,
    "亿": 1e8,
    "万亿": 1e12,
    "千": 1e3,
    "百分点": 1.0,
    "%": 1.0,
    "％": 1.0,
    None: 1.0,
    "": 1.0,
}


def _numbers(text: str) -> list[float]:
    """从文本中抽出所有数值(单位归一到 base)。

    兼容 CJK 单位(万/亿/万亿/千/百分点/%)和纯数字。
    注意:年份 (1900-2099) 不参与单位换算 — 因为 "2024" 旁边的 magnitude
    keyword (如 million USD) 不会被误归一。
    """
    out: list[float] = []
    for raw, suffix in _NUM.findall(text or ""):
        try:
            v = float(raw.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        out.append(v * _SCALE.get(suffix, 1.0))
    return out


def has_numeric_drift(
    claim: str,
    span: str,
    rel_tol: float = 0.005,
    exclude_year_range: tuple[int, int] = (1900, 2100),
) -> bool:
    """判断 claim 中的数字是否都能在 span 里找到(归一到 base 后相对偏差 < rel_tol)。

    rel_tol=0.005 = 0.5%(单位换算 12000万 vs 1.2亿 应被正确归一到同一 base)。

    Year-range exclusion:span 中出现的年份(1900-2100)不算"数值"——它们是
    时间锚,不是 claim 中的财务/统计数字。
    """
    span_nums = [
        n for n in _numbers(span)
        if not (exclude_year_range[0] <= n <= exclude_year_range[1])
    ]
    for n in _numbers(claim):
        # 跳过 claim 中也显然是年份的数字
        if exclude_year_range[0] <= n <= exclude_year_range[1]:
            continue
        if not any(
            abs(n - s) <= rel_tol * max(abs(n), abs(s), 1.0) for s in span_nums
        ):
            return True
    return False


# =============================================================================
# 闸 3:entailment 批量 (Runbook 2.4)
# =============================================================================

@dataclass
class EntailmentResult:
    """单条 EU 的 entailment 校验结果。"""
    index: int
    verdict: Verdict
    score: float = 0.0
    reason: str = ""


@dataclass
class EntailmentBatchResult:
    """一批 entailment 调用结果(可含解析失败的兜底)。"""
    results: list[EntailmentResult]
    raw_response: str = ""
    parse_warnings: list[str] = field(default_factory=list)


ENTAILMENT_PROMPT: str = """判断每个 claim 是否被其配对的 span 蕴含。

verdict 取值:
- entailed:     span 直接、完整地支持 claim
- partial:      span 支持 claim 的一部分,或 claim 强于 span
- contradicted: span 与 claim 矛盾
- unverifiable: span 不涉及 claim 所述内容

以下情形一律**不得**判为 entailed:
- claim 中的数字/日期/比例在 span 中不存在或不同
- claim 把 span 中的"预计/计划/据称/有望"表述为确定事实
- claim 把 span 中某主体的行为归给了另一主体
- claim 做了 span 未做的因果、排名或对比推断

待判条目:
{items}

输出严格 JSON(无 markdown 代码块标记),形如:

{{
  "results": [
    {{
      "index": 0,
      "verdict": "entailed|partial|contradicted|unverifiable",
      "score": 0.0,
      "reason": "≤200 字符简短理由"
    }}
  ]
}}
"""


def _render_entailment_items(
    items: list[dict[str, Any]],
) -> str:
    """格式化待校验条目(claim + span)成 prompt 片段。

    items 中每个元素至少含 claim / span 两个 key;index 由调用方注入。
    """
    lines: list[str] = []
    for i, it in enumerate(items):
        lines.append(f"--- [{i}] ---")
        lines.append(f"claim: {it['claim']}")
        lines.append(f"span:  {it['span']}")
    return "\n".join(lines)


def parse_entailment_response(
    raw: str,
    n_items: int,
) -> EntailmentBatchResult:
    """解析 LLM 输出的 entailment JSON 响应。

    失败模式:
      - raw 空 / 非 JSON → 整批 unverifiable,score=0,reason='parse_failed'
      - 部分 index 缺失 → 缺的那条 unverifiable,score=0,reason='missing_in_response'
      - verdict / score 类型错 → 该条 unverifiable,warning 记录
    """
    warnings: list[str] = []
    results: list[EntailmentResult] = []
    if not raw:
        return EntailmentBatchResult(
            results=[
                EntailmentResult(index=i, verdict="unverifiable", score=0.0,
                                 reason="empty response")
                for i in range(n_items)
            ],
            raw_response="",
            parse_warnings=["empty response"],
        )

    # 抽取首个 JSON 对象
    text = raw.strip()
    payload = None
    if "{" in text:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    payload = text[start : i + 1]
                    break
    if payload is None:
        return EntailmentBatchResult(
            results=[
                EntailmentResult(index=i, verdict="unverifiable", score=0.0,
                                 reason="no json found")
                for i in range(n_items)
            ],
            raw_response=raw,
            parse_warnings=["no JSON object found"],
        )

    try:
        obj = json.loads(payload)
    except Exception as e:
        warnings.append(f"json parse: {e}")
        obj = {}

    raw_results = obj.get("results") or []
    by_index: dict[int, dict[str, Any]] = {}
    for r in raw_results:
        try:
            idx = int(r.get("index", -1))
        except Exception:
            warnings.append(f"bad index: {r.get('index')!r}")
            continue
        by_index[idx] = r

    for i in range(n_items):
        r = by_index.get(i)
        if r is None:
            results.append(EntailmentResult(
                index=i, verdict="unverifiable", score=0.0,
                reason="missing in response",
            ))
            warnings.append(f"missing index {i}")
            continue
        verdict_raw = r.get("verdict", "unverifiable")
        if verdict_raw not in ("entailed", "partial", "contradicted", "unverifiable"):
            warnings.append(f"bad verdict {verdict_raw!r} at index {i}")
            verdict = "unverifiable"
        else:
            verdict = verdict_raw  # type: ignore[assignment]
        try:
            score = float(r.get("score", 0.0))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.0
        reason = str(r.get("reason", ""))[:200]
        results.append(EntailmentResult(
            index=i, verdict=verdict, score=score, reason=reason,
        ))

    return EntailmentBatchResult(
        results=results,
        raw_response=raw,
        parse_warnings=warnings,
    )


# =============================================================================
# 综合:对 EU 列表跑三道闸
# =============================================================================

@dataclass
class GateStats:
    """闸后统计(rejected_stats 验收标准:各类目计数非零)。"""
    total: int = 0
    span_rejected: int = 0
    numeric_drift_rejected: int = 0
    entailment_contradicted: int = 0
    entailment_unverifiable: int = 0
    entailment_partial_kept: int = 0
    entailment_entailed_kept: int = 0
    span_fuzzy_hit: int = 0
    span_normalized_hit: int = 0

    def rejected_count(self) -> int:
        return self.span_rejected + self.numeric_drift_rejected + \
            self.entailment_contradicted + self.entailment_unverifiable

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "span_rejected": self.span_rejected,
            "numeric_drift_rejected": self.numeric_drift_rejected,
            "entailment_contradicted": self.entailment_contradicted,
            "entailment_unverifiable": self.entailment_unverifiable,
            "entailment_partial_kept": self.entailment_partial_kept,
            "entailment_entailed_kept": self.entailment_entailed_kept,
            "span_fuzzy_hit": self.span_fuzzy_hit,
            "span_normalized_hit": self.span_normalized_hit,
            "rejected_count": self.rejected_count(),
        }


def run_gate1_span(
    eus: list[dict[str, Any]],
    contents_by_url: dict[str, str],
    *,
    fuzzy_threshold: float = 0.92,
) -> tuple[list[bool], GateStats]:
    """闸 1:对每个 EU 校验 source_span 是否在对应 source_url 的正文中命中。

    返回 (passed: list[bool], stats):
      passed[i] = True 表示该 EU 通过闸 1
      stats 累计命中形态分布

    eus 元素: dict 形态,需要含 source_url / source_span
    """
    passed: list[bool] = []
    stats = GateStats()
    stats.total = len(eus)
    for eu in eus:
        url = eu.get("source_url") or ""
        span = eu.get("source_span") or ""
        content = contents_by_url.get(url, "")
        ok, pos = verify_span(span, content, fuzzy_threshold=fuzzy_threshold)
        if ok:
            passed.append(True)
            if pos is None:
                # 字面或归一化命中位置丢失(归一化路径)或模糊命中
                # 进一步区分需要看 content 是否真的有原文 span;
                # 这里粗分类:归一化命中也算 span_normalized_hit
                stats.span_normalized_hit += 1
        else:
            passed.append(False)
            stats.span_rejected += 1
    return passed, stats


def run_gate2_numeric_drift(
    eus: list[dict[str, Any]],
    gate1_passed: list[bool],
    *,
    rel_tol: float = 0.005,
) -> tuple[list[bool], GateStats]:
    """闸 2:对通过闸 1 的 EU 校验 claim 中的数值是否都能在 span 里找到。

    闸 2 只对闸 1 通过的 EU 跑(闸 1 失败的 EU 不再有"对照 baseline")。
    返回 (passed: list[bool], stats):stats.entailment_* 字段这里不使用。
    """
    passed: list[bool] = []
    stats = GateStats()
    stats.total = len(eus)
    for i, eu in enumerate(eus):
        if not gate1_passed[i]:
            passed.append(False)
            continue
        claim = eu.get("claim") or ""
        span = eu.get("source_span") or ""
        drift = has_numeric_drift(claim, span, rel_tol=rel_tol)
        passed.append(not drift)
        if drift:
            stats.numeric_drift_rejected += 1
    return passed, stats


__all__ = [
    "verify_span",
    "has_numeric_drift",
    "_normalize",
    "_numbers",
    "EntailmentResult",
    "EntailmentBatchResult",
    "ENTAILMENT_PROMPT",
    "_render_entailment_items",
    "parse_entailment_response",
    "run_gate1_span",
    "run_gate2_numeric_drift",
    "GateStats",
]