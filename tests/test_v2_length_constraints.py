"""V2 acceptance — length / count constraints for QueryIntent.queries.

V2 spec: rotation needs ≥5 variants per slot. Site-scoped queries can
legitimately exceed 200 chars. Hard raise on length is the wrong
behavior — it fails the whole intent when one variant is too long.
"""
import pytest
from open_deep_research.query_constructor import QueryIntent


def test_250_char_site_scoped_query_accepted():
    """The v41 regression query must not raise."""
    big_q = (
        "site:qihoo.com OR site:360.cn OR site:sangfor.com OR site:nsfocus.com "
        "OR site:dbappsecurity.com.cn OR site:venustech.com.cn OR site:qianxin.com.cn "
        "OR site:antiy.cn OR site:cnnics.cn OR site:topsec.com.cn "
        "中国 EDR 终端安全 厂商 市场份额 2026 收入 年同比 增长率 IDC 报告 行业总规模"
    )
    assert len(big_q) >= 250, f"sanity: query should be ≥250 chars, got {len(big_q)}"
    qi = QueryIntent(queries=[big_q], engines=["bing", "360search"], language="zh-CN")
    assert len(qi.queries) == 1
    # site-scoped → cap 300, should pass through whole or be truncated to 300
    assert len(qi.queries[0]) <= 300


def test_non_site_long_query_truncated():
    """Non-site query >200 chars should be truncated, not raise."""
    long_q = "a " * 150  # 300 chars
    qi = QueryIntent(queries=[long_q], engines=["bing"])
    # non-site cap is 200
    assert len(qi.queries[0]) <= 200


def test_eight_queries_accepted():
    """V2: max 8 queries per intent (was 3, blocks rotation)."""
    eight = [f"query variant {i} site:foo.com" for i in range(8)]
    qi = QueryIntent(queries=eight)
    assert len(qi.queries) == 8


def test_nine_queries_capped_at_eight():
    nine = [f"variant {i}" for i in range(9)]
    qi = QueryIntent(queries=nine)
    assert len(qi.queries) == 8


def test_eight_intents_per_plan_allowed():
    """V2: intents list now caps at 8 (was 4)."""
    from open_deep_research.query_constructor import ExecutionPlan
    plan = ExecutionPlan(
        intents=[
            QueryIntent(queries=[f"q{i}"], engines=["bing"]) for i in range(8)
        ]
    )
    assert len(plan.intents) == 8


def test_v41_regression_query_passes():
    """The exact v41 LLM output that raised — must now succeed."""
    # Reconstructed from v41_verify.log
    v41_evil_q = (
        "site:qihoo.com OR site:360.cn OR site:sangfor.com OR "
        "site:nsfocus.com OR site:dbappsecurity.com.cn OR site:venustech.com.cn "
        "OR site:qianxin.com.cn OR site:antiy.cn 营收 市占率"
    )
    qi = QueryIntent(
        queries=[v41_evil_q, "中国 EDR 市场 2026 总规模"],
        engines=["bing", "360search", "chinaso news"],
        language="zh-CN",
    )
    assert len(qi.queries) == 2