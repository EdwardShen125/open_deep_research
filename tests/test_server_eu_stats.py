"""Server endpoint regression tests — eu_stats 端点聚合。

eu_stats 让 /runs/{id} 立刻可观测 5 维度 EU 分布 + top source_domain,
不需要 claim_stats(phase 3 闸 2 才填)。
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def server_module():
    """需要 POSTGRES_* env 才能 import server(它会立刻 try EuDAO._connect)。"""
    if not os.environ.get("POSTGRES_HOST"):
        pytest.skip("POSTGRES_HOST 未设置 — server endpoint 测试需要真 PG")
    from open_deep_research.api import server
    return server


def test_build_eu_stats_returns_typed_dict(server_module):
    """_build_eu_stats 必须返固定 schema:total/by_dimension/top_source_domains。"""
    # 用一个不会有的 run_id —— EuDAO 应该返 total=0,by_dimension={},top=[]
    fake_run = "00000000-0000-0000-0000-000000000000"
    stats = server_module._build_eu_stats(fake_run)
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {
        "total", "by_dimension", "top_source_domains", "source_domain_count", "by_source_tier",
    }
    assert stats["total"] == 0
    assert stats["by_dimension"] == {}
    assert stats["top_source_domains"] == []
    assert stats["source_domain_count"] == 0
    print("  ✓ _build_eu_stats empty-run schema correct")


def test_get_run_status_404_returns_404(server_module):
    """/runs/{id} 对不存在的 run_id 不应崩 — meta=None 走 'not_found' 分支。"""
    from fastapi.testclient import TestClient
    client = TestClient(server_module.app)
    fake_run = "11111111-2222-3333-4444-555555555555"
    r = client.get(f"/runs/{fake_run}")
    assert r.status_code == 200  # endpoint 设计上返 200 + status='not_found'
    body = r.json()
    assert body["status"] == "not_found"
    assert body["claim_stats"] is None
    # eu_stats 可能返 0 EU(empty)或 None(若 EuDAO 抛错)
    assert body["eu_stats"] is None or body["eu_stats"]["total"] == 0
    print(f"  ✓ /runs/{{id}} not_found: status=200, status_field='not_found'")


def test_get_run_status_400_on_bad_uuid(server_module):
    """/runs/{not-a-uuid} 必须返 400。"""
    from fastapi.testclient import TestClient
    client = TestClient(server_module.app)
    r = client.get("/runs/not-a-uuid")
    assert r.status_code == 400
    print("  ✓ /runs/{bad-uuid} → 400")


def test_get_run_report_404_when_no_data(server_module):
    """/runs/{id}/report 对空 run_id 必须返 404(没有 EU 也没有 claim)。"""
    from fastapi.testclient import TestClient
    client = TestClient(server_module.app)
    fake_run = "22222222-3333-4444-5555-666666666666"
    r = client.get(f"/runs/{fake_run}/report")
    assert r.status_code == 404
    print("  ✓ /runs/{id}/report empty → 404")