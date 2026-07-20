"""Run one Deep Research Bench prompt through the v2 graph directly.

Bypasses langgraph dev (which has SSE streaming instability under sustained
load) and invokes the deep_researcher graph directly in-process. This is
also closer to what `tests/run_evaluate.py` does, which makes the bench
output reproducible by either path.

Usage:
    python tests/run_bench_item.py --id 93 [--question "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import UsageMetadataCallbackHandler
from open_deep_research.deep_researcher import deep_researcher_builder

BENCH_JSONL = ROOT / "tests/expt_results/deep_research_bench_gpt-5.jsonl"
OUT_DIR = ROOT / "tests/expt_results"


def _load_bench_prompt(bench_id: int) -> tuple[str, str]:
    with open(BENCH_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("id") == bench_id:
                return d["prompt"], d.get("article", "")
    raise SystemExit(f"id={bench_id} not found in {BENCH_JSONL}")


def _summarize(state: dict) -> dict:
    eu = state.get("evidence_units") or []
    cr = state.get("cited_report") or {}
    v = state.get("verification") or {}
    uc = state.get("url_compliance") or []
    fr = state.get("final_report") or ""
    domains = set()
    urls = set()
    nums = 0
    for e in eu:
        # EU may be Pydantic (has .source_url / .numbers attrs) or dict
        u = None
        ns = None
        if isinstance(e, dict):
            u = e.get("source_url")
            ns = e.get("numbers") or []
        else:
            u = getattr(e, "source_url", None)
            ns = getattr(e, "numbers", None) or []
        if u:
            urls.add(u)
            try:
                domains.add("/".join(u.split("/")[:3]))
            except Exception:
                pass
        if isinstance(ns, list):
            nums += len(ns)
    sections = cr.get("sections") or []
    claims_total = sum(len(s.get("claims") or []) for s in sections)
    return {
        "evidence_units": len(eu),
        "unique_urls": len(urls),
        "unique_domains": len(domains),
        "numeric_anchors": nums,
        "cited_sections": len(sections),
        "cited_claims": claims_total,
        "verification_issues": len(v.get("issues") or []),
        "verification_by_severity": v.get("by_severity") or {},
        "url_compliance_issues": len(uc),
        "final_report_chars": len(fr),
        "final_report_starts_with_error_json": fr.lstrip().startswith('{"error"'),
    }


def _collect_token_usage(callback: UsageMetadataCallbackHandler) -> dict:
    """Aggregate per-call token counts from UsageMetadataCallbackHandler.

    Returns a flat dict suitable for meta.json. `cost_usd` is intentionally
    omitted — we don't ship model pricing in the repo (drift risk); the
    consumer can compute cost from `tokens_in` / `tokens_out` against
    their own price sheet.
    """
    total_in = 0
    total_out = 0
    per_model: dict[str, dict[str, int]] = {}
    for model_name, meta in callback.usage_metadata.items():
        # meta is a langchain_core.messages.ai.UsageMetadata (TypedDict)
        try:
            in_t = int(meta.get("input_tokens", 0))
            out_t = int(meta.get("output_tokens", 0))
        except Exception:
            in_t, out_t = 0, 0
        total_in += in_t
        total_out += out_t
        per_model[model_name or "unknown"] = {
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_tokens": in_t + out_t,
            "calls": 1,
        }
    # Collapse identical model entries (callback stores per-call — multiple
    # calls to the same model show up as separate keys). We aggregate by
    # model_name prefix to handle trace-id-suffixed keys.
    collapsed: dict[str, dict[str, int]] = {}
    for k, v in per_model.items():
        # Strip any trailing "(name=...)" or "[id=N]" suffix the callback
        # appends to disambiguate concurrent calls.
        base = k.split(" ")[0]
        if base not in collapsed:
            collapsed[base] = {"input_tokens": 0, "output_tokens": 0,
                              "total_tokens": 0, "calls": 0}
        collapsed[base]["input_tokens"] += v["input_tokens"]
        collapsed[base]["output_tokens"] += v["output_tokens"]
        collapsed[base]["total_tokens"] += v["total_tokens"]
        collapsed[base]["calls"] += v["calls"]
    return {
        "tokens_in": total_in,
        "tokens_out": total_out,
        "tokens_total": total_in + total_out,
        "llm_calls": len(callback.usage_metadata),
        "by_model": collapsed,
    }


async def _run(question: str, usage_cb: UsageMetadataCallbackHandler) -> dict:
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            "allow_clarification": False,
            "max_concurrent_research_units": 1,
            "max_researcher_iterations": 2,
            "max_react_tool_calls": 3,
            # Bump writer model max tokens — default 10000 truncates long
            # market-analysis JSON. 32000 fits a full PhD-style brief with
            # ~6 sections, 50+ claims, EU citations and numeric anchors.
            "final_report_model_max_tokens": 32000,
            "research_model_max_tokens": 16000,
        },
        "callbacks": [usage_cb],
    }
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--question", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default=None)
    args = ap.parse_args()

    if args.question:
        prompt = args.question
        baseline_article = ""
    else:
        prompt, baseline_article = _load_bench_prompt(args.id)

    print(f"== bench item id={args.id} ==")
    print(f"prompt ({len(prompt)} chars):")
    print(f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    if baseline_article:
        print(f"\nbaseline (gpt-5) article: {len(baseline_article)} chars")

    usage_cb = UsageMetadataCallbackHandler()
    print("\nrunning graph.ainvoke() ...")
    t0 = time.time()
    try:
        state = asyncio.run(_run(prompt, usage_cb))
    except Exception as e:
        print(f"✗ run failed: {type(e).__name__}: {str(e)[:500]}")
        return 1
    elapsed = time.time() - t0
    summary = _summarize(state)
    usage = _collect_token_usage(usage_cb)
    print(f"\nelapsed: {elapsed:.1f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nusage: {usage['tokens_in']} in + {usage['tokens_out']} out "
          f"= {usage['tokens_total']} total across {usage['llm_calls']} LLM calls")
    if usage["by_model"]:
        print(f"  by_model:")
        for m, mv in usage["by_model"].items():
            print(f"    {m}: {mv['input_tokens']} in / "
                  f"{mv['output_tokens']} out / {mv['calls']} calls")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"bench_v2_{args.id}_{ts}"
    md_path = OUT_DIR / f"{prefix}.md"
    meta_path = OUT_DIR / f"{prefix}.meta.json"

    fr = state.get("final_report") or ""
    md_path.write_text(fr, encoding="utf-8")
    print(f"\nwrote {md_path}")

    meta = {
        "bench_id": args.id,
        "prompt_len": len(prompt),
        "prompt": prompt,
        "baseline_article_len": len(baseline_article),
        "elapsed_seconds": elapsed,
        "model_provider": "minimax:MiniMax-M3 (default)",
        "summary": summary,
        "usage": usage,
        "state_keys": sorted(state.keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())