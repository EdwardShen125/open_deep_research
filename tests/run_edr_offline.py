"""Run the EDR market-research query through plan_v2_pipeline in
placeholder-writer mode (no LLM writer call → no MiniMax rate-limit risk).

Output:  edr_market_v2_zh_v5_pipeline.md  (markdown from RDO.to_markdown())
         edr_market_v2_zh_v5_pipeline.meta.json  (full PlanV2RunResult summary)

This is the offline counterpart to `tests/run_bench_item.py` (which
goes through the full deep_researcher graph and hits the writer LLM).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from open_deep_research.plan_v2_pipeline import run_pipeline  # noqa: E402
from open_deep_research.search_providers import (  # noqa: E402
    TavilyProvider, SearchQuery,
)
from open_deep_research.search_cache import SearchCache  # noqa: E402
from open_deep_research.sources_dao import SourceRecord  # noqa: E402

# In-memory SQLite for sources DAO (so we don't need docker postgres here)
sys.path.insert(0, str(ROOT / "tests"))
from test_sources_dao_sqlite import _SQLiteConnection, _DAOTest  # type: ignore

EDR_QUERY = (
    "请撰写一份截至 2026 年中期的全球终端检测与响应(Endpoint Detection and "
    "Response, EDR)软件市场综合调研报告。报告需覆盖以下 10 个维度:\n"
    "1. 市场规模与增长(Gartner/IDC/Forrester 三家 2025 TAM 数据 + 2030 CAGR 预测)\n"
    "2. 主要厂商画像(CrowdStrike Falcon、Microsoft Defender for Endpoint、"
    "SentinelOne Singularity、Palo Alto Cortex XDR、Trend Micro Vision One、"
    "Sophos Intercept X、Trellix、Bitdefender GravityZone、ESET PROTECT、"
    "VMware/Carbon Black、Elastic EDR 等)\n"
    "3. 能力差异化(EDR vs XDR vs EPP, AI/ML 检测引擎, 行为分析, 勒索防护, "
    "威胁情报订阅, MDR 捆绑)\n"
    "4. 部署模式(云原生 vs 本地 vs 混合, 多租户 SaaS, Agent 架构)\n"
    "5. 定价与许可(订阅分层, 平台捆绑, MSSP, 免费层)\n"
    "6. 竞争格局(市场份额, M&A, 战略合作, 裁员, 厂商整合)\n"
    "7. 终端用户分层(大型企业 vs 中端 vs SMB vs 政府/国防, 行业垂直差异)\n"
    "8. 区域分布(北美/EMEA/亚太/拉美)\n"
    "9. 监管环境(NIS2, DORA, SEC 披露, CISA, MLPS 2.0)\n"
    "10. 2026-2028 战略展望(平台化, AI SOC 替代)"
)

OUT_DIR = ROOT / "tests/expt_results"


async def _run() -> Any:
    primary = TavilyProvider()  # uses TAVILY_API_KEY from env
    dao = _DAOTest(_SQLiteConnection())
    cache = SearchCache(ttl_seconds=60, sources_dao=dao)
    out = await run_pipeline(
        EDR_QUERY,
        run_id="edr-v5-offline",
        primary=primary,
        sources_dao=dao,
        cache=cache,
        writer_response=None,    # placeholder mode — no LLM
        title="全球 EDR 市场综合调研报告 (2026 中期)",
    )
    return out


def main() -> int:
    print("== EDR market research (placeholder writer mode) ==")
    print(f"prompt: {len(EDR_QUERY)} chars")
    t0 = time.time()
    try:
        out = asyncio.run(_run())
    except Exception as e:
        print(f"✗ run failed: {type(e).__name__}: {str(e)[:500]}")
        return 1
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    # summary
    eu_count = len(out.evidence_units)
    cr = out.cited_report
    v = out.verification
    rdo = out.report_data
    print(f"  evidence_units: {eu_count}")
    print(f"  cited_report.sections: {len(cr.sections) if cr else 0}")
    print(f"  verification.issues: {len(v.issues) if v else 0}")
    print(f"  url_compliance: {len(out.url_compliance)}")
    print(f"  report_data.sections: {len(rdo.sections) if rdo else 0}")

    # Write markdown
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = OUT_DIR / f"edr_market_v2_zh_v5_pipeline_{ts}.md"
    if rdo is not None:
        md_path.write_text(rdo.to_markdown(), encoding="utf-8")
    else:
        md_path.write_text("# (no RDO produced)\n", encoding="utf-8")
    print(f"\nwrote {md_path}")

    # Write meta
    meta = {
        "bench_id": "edr-v5-offline",
        "mode": "placeholder writer (no LLM)",
        "prompt_len": len(EDR_QUERY),
        "prompt": EDR_QUERY,
        "elapsed_seconds": elapsed,
        "summary": {
            "evidence_units": eu_count,
            "cited_report_sections": len(cr.sections) if cr else 0,
            "verification_issues": len(v.issues) if v else 0,
            "url_compliance_issues": len(out.url_compliance),
            "report_data_sections": len(rdo.sections) if rdo else 0,
        },
        "verification_by_severity": v.by_severity if v else {},
        "first_5_eus": [
            {
                "id": eu.id,
                "claim": eu.claim[:200],
                "source_url": eu.source_url,
                "source_title": eu.source_title,
                "numbers": [
                    {"text": n.text, "value_min": n.value_min, "unit": n.unit}
                    for n in (eu.numbers or [])
                ],
            }
            for eu in out.evidence_units[:5]
        ],
        "unique_source_urls": sorted({
            eu.source_url for eu in out.evidence_units if eu.source_url
        }),
        "state_keys": sorted(out.to_dict().keys()),
    }
    meta_path = OUT_DIR / f"edr_market_v2_zh_v5_pipeline_{ts}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())