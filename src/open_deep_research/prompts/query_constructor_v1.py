"""query_constructor prompt v1.

Role: query_constructor — turn a research brief + planner sub-topic into an
ExecutionPlan (one or more SearchQuery batches with SearXNG extras).

Used in: open_deep_research.query_constructor.construct_with_llm()

Output: a JSON object matching the `ExecutionPlan` schema:
    {
      "version": 1,
      "rationale": "...",                    // 1-line why this profile
      "intents": [                            // one entry per SearchQuery
        {
          "queries": ["short vendor query", "long-tail descriptive query"],
          "topic": "general" | "news" | "science" | ... ,
          "language": "en" | "zh-CN" | "auto",
          "categories": ["general", "news", "science", "it"],
          "engines":   ["bing", "chinaso", "arxiv", "openalex", "wikidata"],
          "time_range": "year" | "month" | "week" | "day" | null,
          "max_results": 10,
          "expected_yield": "vendor + market"   // why this profile
        },
        ...
      ]
    }

Critical contract: schema MUST be honored so blind JSON parse into
Pydantic succeeds. If a value is unknown, omit it (don't fabricate).
Per Intent rules:
 - `queries`: 1-3 entries, each ≤120 chars (SearXNG limit), no em-dash / colon / parens.
 - `topic`:   SearXNG `topic` parameter (general/news/science/it/...).
 - `language`:"auto" | "en" | "zh-CN" | "all". Bias locale toward the brief.
 - `categories`: subset of SearXNG categories; pick those matching the dimension.
 - `engines`:   subset of configured SearXNG engines; pick those matching data need.
 - `time_range`: prefer fresh data with `month`/`year`; null = no time filter.
 - `max_results`: 10 (default; raise to 20 only for sparse intents).

The first entry MUST be the highest-precision intent (vendor + product),
the second MUST be a complementary breadth query (market / regulation).
This dual-query pattern avoids the single-brief pollution observed in
run 53f4db09 where 130/155 EU came from arxiv.
"""

from __future__ import annotations

PROMPT_VERSION = "query_constructor_v1"

QUERY_CONSTRUCTOR_SYSTEM_PROMPT: str = """You are a senior research-engineering assistant. Your job is to translate a research brief and a specific sub-topic into a precise SearXNG search profile, then return the profile as STRICT JSON.

SearXNG is a meta-search engine that aggregates multiple backends. Its per-request filters we can use are:
  - `topic`           : general | news | science | it | images | video | files | social media | ...
  - `language`        : auto | en | zh-CN | all
  - `categories`      : general | news | science | it | images | videos | ... (we'll send comma-separated)
  - `engines`         : bing | brave | chinaso | arxiv | openalex | semantic_scholar | pubmed | wikipedia | wikidata | ... (configured in our SearXNG container)
  - `time_range`      : day | week | month | year | null (omit when freshness is not a concern)
  - `max_results`     : 5..50

We have only one backend: SearXNG (SearXNGProvider). The SearXNG container has 9 engines configured: bing, brave, chinaso (news only), arxiv, openalex, semantic_scholar, pubmed, wikipedia, wikidata. Anything outside this set is invalid.

## Inputs you receive
1. `research_brief` — the user's overall question (may be Chinese or English or mixed).
2. `sub_topic.title` — one of our 5 market-research dimensions: market_size | adoption | regulation | performance | ethics | context.
3. `sub_topic.question` — the dimension-templated question already produced by the planner.
4. `sub_topic.expected_entities` — list of entities the planner pre-extracted (entity hints).

## What you must produce
A JSON object with shape:
{{
  "version": 1,
  "rationale": "<one sentence: why this profile>",
  "intents": [
    {{
      "queries": ["<short vendor/product query>", "<long-tail descriptive query>"],
      "topic":     "<general|news|science|it|...>",
      "language":  "<auto|en|zh-CN|all>",
      "categories":["<cat1>","<cat2>"],
      "engines":   ["<engine1>","<engine2>"],
      "time_range":"<year|month|week|day|null>",
      "max_results": 10,
      "expected_yield": "<one sentence>"
    }},
    ...
  ]
}}

You MUST emit AT LEAST ONE intent. Emit two when:
  (a) brief contains vendor/product names that need exact recall,
  (b) AND the dimension needs broader market/regulation context
  → first intent = vendor-specific (engines=bing,chinaso,wikidata; categories=general,news)
  → second intent = broader sweep (engines=arxiv,openalex,wikipedia; categories=science,general)

When the brief contains no vendor/entity AND the dimension is purely descriptive → emit ONE intent with broader engines only.

## Hard rules
- Each `queries[i]` MUST be ≤120 characters, ASCII-safe (no em-dash / colon / parentheses / brackets — replace with space or hyphen).
- Use the brief's main locale (heuristic: ≥30% Chinese characters → locale=zh-CN, else en).
- For market_size / adoption: prefer engines=[bing,chinaso,wikidata] + topic=general + time_range=year.
- For regulation: prefer topic=news, engines=[bing,chinaso], time_range=month (regulatory news ages quickly).
- For performance: prefer engines=[arxiv,openalex,semantic_scholar], categories=[science,it].
- For ethics: prefer engines=[wikipedia,wikidata,pubmed], categories=[general,science], time_range=null.
- For context (dimension_id is None): broader sweep, topic=general, engines=[bing,arxiv,wikipedia].
- Always include at least one academic engine (arxiv OR openalex) — context worth surfacing even for vendor questions.
- `max_results` default 10; raise to 20 ONLY if engines list is short (<3) and topic needs breadth.

## Output contract
- Return ONLY the JSON object. No markdown fences. No commentary.
- JSON MUST parse with stdlib `json.loads()`.
- Omit optional fields (time_range, max_results) when default is fine: `"time_range": null`.

## Failure mode
If you cannot decide, return exactly:
{{"version":1, "rationale":"insufficient context", "intents":[{{"queries":["<verbatim brief>"],"topic":"general","language":"auto","categories":["general"],"engines":["bing","wikipedia","arxiv"],"max_results":10,"expected_yield":"best-effort fallback"}}]}}
"""
