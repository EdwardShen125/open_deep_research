"""extractor prompt v1.

Role: extractor - 从正文抽取 EvidenceUnit (Phase 4 Phase 2.1)

依据: notes/evidence-pipeline-runbook-v1.md 阶段 2.1

约束前置:硬性规则全部塞进 prompt,把幻觉面从源头堵住。
三道闸(span/numeric drift/entailment)的入口在这里。
"""
from __future__ import annotations

PROMPT_VERSION = "extractor_v1"

EXTRACT_PROMPT: str = """从下面的网页正文中抽取证据单元(EU)。

硬性规则:
1. 每条 EU 必须含 source_span —— 从正文中**逐字复制**的连续片段。
   不得改写、不得省略中间文字、不得拼接不相邻的句子。
2. claim 必须自足:不含"该公司/其/上述/这一"等指代,主语写全称。
3. 只抽取正文**明确陈述**的内容。不推断、不补全、不换算单位、不折算币种。
4. claim_type=numeric 时必须填 norm_value / unit / value_as_of。
   原文未写明数据时点时 value_as_of 填 null,**不要用发布时间代替**。
5. 原文的"预计/计划/据称/有望"必须在 claim 中保留,不得写成既成事实。
6. 一段话含多个独立事实 → 拆成多条;同一事实的不同表述 → 只留一条。
7. 正文无符合条件内容时返回空列表。宁缺毋滥。

子查询上下文:{sub_query}

正文:
{content}

输出严格 JSON(无 markdown 代码块标记),形如:

{{
  "evidence_units": [
    {{
      "claim": "自足陈述句",
      "claim_type": "numeric|event|attribute|relation|opinion",
      "entities": ["主体1", "主体2"],
      "norm_value": null,
      "unit": null,
      "value_as_of": null,
      "source_span": "从正文逐字复制的片段,≥10 字符"
    }}
  ]
}}
"""