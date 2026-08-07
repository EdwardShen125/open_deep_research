"""extractor prompt v1.

Role: extractor - 从正文抽取 EvidenceUnit (Phase 4 Phase 2.1)

依据: notes/evidence-pipeline-runbook-v1.md 阶段 2.1

约束前置:硬性规则全部塞进 prompt,把幻觉面从源头堵住。
三道闸(span/numeric drift/entailment)的入口在这里。
"""
from __future__ import annotations

PROMPT_VERSION = "extractor_v1"

EXTRACT_PROMPT: str = """从下面的网页正文中抽取证据单元(EU)。

当前查询目标 claim_type:{expected_claim_type}
查询意图:{sub_query}

硬性规则:
1. 每条 EU 必须含 source_span —— 从正文中**逐字复制**的连续片段(≥10 字符)。
   不得改写、不得省略中间文字、不得拼接不相邻的句子。
2. claim 必须自足:不含"该公司/其/上述/这一"等指代,主语写全称。
3. 只抽取正文**明确陈述**的内容。不推断、不补全、不换算单位、不折算币种。
4. claim_type 决定必有字段:
   - numeric:必须填 norm_value(数字,无单位数字) / unit(单位字符串,如"亿元"/"%"/"USD billion") / value_as_of(YYYY-MM-DD,原文无点则填 null,**不要用发布时间代替**)
   - event:必有 entities(主语),value_as_of
   - attribute / relation / opinion:严禁捏造 norm_value(norm_value=null)
5. metric_type(必填)用于对账聚簇,必须从下列标准化词汇选一个:
   - "market_size"(市场规模/营收/销售额/产业规模)
   - "penetration"(渗透率/装机率/安装率)
   - "share"(市占率/份额/占比)
   - "gmv_estimate"(GMV/订单量/用户量估算)
   - "sku_rank"(SKU排名/榜单/排行)
   - "regulation"(法规/标准/合规要求/政策)
   - "financial"(财务数据,如营收/利润/估值)
   - "m_and_a"(融资/并购/上市/轮次)
   - "trends"(行业趋势/CAGR/增长率)
   - "other"(以上都不适用)
   若正文是导语/SEO标题/定义句/"啥是 X"开头,不抽取,跳过。
6. 原文的"预计/计划/据称/有望"必须在 claim 中保留,不得写成既成事实。
7. 一段话含多个独立事实 → 拆成多条;同一事实的不同表述 → 只留一条。
8. 正文无符合条件内容时返回空列表。宁缺毋滥。

子查询上下文:
{sub_query}

正文:
{content}

输出严格 JSON(无 markdown 代码块标记),形如:

{{
  "evidence_units": [
    {{
      "claim": "自足陈述句",
      "claim_type": "numeric|event|attribute|relation|opinion",
      "entities": ["主体1", "主体2"],
      "metric_type": "market_size|penetration|share|gmv_estimate|sku_rank|regulation|financial|m_and_a|trends|other",
      "norm_value": null,
      "unit": null,
      "value_as_of": null,
      "source_span": "从正文逐字复制的片段,≥10 字符"
    }}
  ]
}}
"""