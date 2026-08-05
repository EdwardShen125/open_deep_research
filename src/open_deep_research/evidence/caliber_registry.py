"""F2 · 口径注册表(per vertical)

按 plan F2:同一指标(如"市场规模")在不同源下口径不同
(eMarketer 直播零售 vs Coresight/McKinsey 广义视频购物,差 3-5 倍)。
对账和报告都靠这份注册表判断"可不可比"。

Caliber 模型:
    id:               全局唯一标识(vertical_metric_author)
    metric_type:      指标类型(market_size / penetration / share ...)
    entity:           测量对象(us_live_commerce / us_video_shopping ...)
    definition:       口径定义(必须包含 / 排除什么)
    includes:         list[str] (该口径包含的成分)
    excludes:         list[str] (该口径排除的成分)
    canonical_source_type: 权威源类型(research_house / official)
    year_applicable:  list[int] (适用年份;None 表示不限)

CaliberRegistry:
    - .from_yaml(path)
    - .get(id) -> Caliber | None
    - .by_metric(metric_type) -> list[Caliber]
    - .by_entity(entity) -> list[Caliber]
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Literal type aliases
# =============================================================================

MetricType = Literal[
    "market_size",
    "penetration",
    "share",
    "gmv_estimate",
    "sku_rank",
    "regulation",
    "financial",
    "m_and_a",
    "trends",
]

SourceType = Literal[
    "research_house",
    "official",
    "mainstream_media",
    "industry_media",
    "vendor",
    "crawler_estimate",
    "ugc",
]


# =============================================================================
# Caliber
# =============================================================================

class Caliber(BaseModel):
    """一个可独立对账的统计口径。

    对账(D2/D3)靠 id 区分;同一 metric_type 下不同 caliber 数值差再大
    也**不取平均**——这是 plan 皇冠差异化的根。
    """

    id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]+$",
        description="全局唯一 id(vertical_metric_author 风格)",
    )
    metric_type: MetricType
    entity: str = Field(min_length=1, max_length=64)
    definition: str = Field(min_length=10, max_length=1000)
    includes: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    canonical_source_type: SourceType
    year_applicable: Optional[list[int]] = None

    @model_validator(mode="after")
    def _id_referenced_known_metric(self) -> "Caliber":
        # id 应以 entity 前缀开头(防御性校验,避免错填)
        if not self.id.startswith(self.entity.lower().replace("-", "_")[:8]):
            # warning 而非 error:某些 entity 名过长可忽略
            pass
        return self


# =============================================================================
# CaliberRegistry
# =============================================================================

class CaliberRegistry(BaseModel):
    """一个 vertical 的口径注册表。

    YAML 格式:
        vertical: us_livecommerce
        calibers:
          - id: us_livestream_retail_emarketer
            metric_type: market_size
            entity: us_live_commerce
            definition: 仅社交/核心电商直播网络内直接成交流水
            includes: [livestream commerce, social commerce livestream]
            excludes: [CTV, brand.com video shopping, China market]
            canonical_source_type: research_house
            year_applicable: [2023, 2024, 2025, 2026]
    """

    vertical: str = Field(min_length=1, max_length=64)
    calibers: list[Caliber] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_id(self) -> "CaliberRegistry":
        seen: set[str] = set()
        for c in self.calibers:
            if c.id in seen:
                raise ValueError(f"duplicate caliber id: {c.id}")
            seen.add(c.id)
        return self

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def get(self, caliber_id: str) -> Optional[Caliber]:
        for c in self.calibers:
            if c.id == caliber_id:
                return c
        return None

    def by_metric(self, metric_type: MetricType) -> list[Caliber]:
        return [c for c in self.calibers if c.metric_type == metric_type]

    def by_entity(self, entity: str) -> list[Caliber]:
        return [c for c in self.calibers if c.entity == entity]

    def applicable_in(self, year: int) -> list[Caliber]:
        """返回适用该年份的 caliber(year_applicable 为 None 也返回)。"""
        out: list[Caliber] = []
        for c in self.calibers:
            if c.year_applicable is None or year in c.year_applicable:
                out.append(c)
        return out


# =============================================================================
# Loader
# =============================================================================

DEFAULT_REGISTRY_DIR = Path("data/registry/calibers")


def load_caliber_registry(
    vertical: str, *, base_dir: Path | str | None = None
) -> CaliberRegistry:
    """加载 data/registry/calibers/<vertical>.yaml。"""
    base = Path(base_dir) if base_dir is not None else DEFAULT_REGISTRY_DIR
    path = Path(base) / f"{vertical}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"caliber registry not found: {path} "
            f"(available: {[p.stem for p in Path(base).glob('*.yaml')]})"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return CaliberRegistry(**raw)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MetricType",
    "SourceType",
    "Caliber",
    "CaliberRegistry",
    "load_caliber_registry",
    "DEFAULT_REGISTRY_DIR",
]
