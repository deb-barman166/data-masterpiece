"""
data_masterpiece.config  --  Unified configuration system.

Provides a typed Config dataclass with JSON I/O, sensible defaults,
and convenience builders for both auto and manual pipeline modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """Unified configuration for the full Data Masterpiece pipeline."""

    # -- pipeline-level settings --
    mode: str = "auto"
    active_agents: list = field(default_factory=lambda: [
        "cleaning", "type_conversion", "missing",
        "encoding", "feature", "validation",
    ])
    run_intelligence: bool = True  # run intelligence engine after preprocessing

    # -- preprocessing thresholds --
    drop_duplicates: bool = True
    null_drop_threshold: float = 0.60
    variance_threshold: float = 1e-10
    low_card_threshold: int = 10
    med_card_threshold: int = 50
    normalize: bool = False
    scale_method: str = "minmax"
    log_transform_skewed: bool = False
    datetime_include_time: bool = False
    normalize_text: bool = True

    # -- per-column overrides --
    cleaning: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)
    encoding: dict = field(default_factory=dict)
    type_conversion: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)

    # -- intelligence engine settings --
    outlier_method: str = "auto"
    outlier_strategy: str = "clip"
    iqr_factor: float = 1.5
    zscore_thresh: float = 3.0
    intelligence_variance_threshold: float = 0.01
    intelligence_corr_threshold: float = 0.90
    intelligence_top_k: int = 0
    test_size: float = 0.20
    val_size: float = 0.0
    stratify: bool = True
    max_viz_cols: int = 20
    skip_outlier: bool = False
    skip_selection: bool = False
    skip_report: bool = False

    # -- output --
    output_path: str = "output/processed.csv"
    plot_dir: str = "output/plots"
    report_path: str = "output/report.html"

    # -- class methods --

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to nested dict for pipeline consumption."""
        return {
            "mode": self.mode,
            "active_agents": self.active_agents,
            "global": {
                "drop_duplicates": self.drop_duplicates,
                "null_drop_threshold": self.null_drop_threshold,
                "variance_threshold": self.variance_threshold,
                "low_card_threshold": self.low_card_threshold,
                "med_card_threshold": self.med_card_threshold,
                "normalize": self.normalize,
                "scale_method": self.scale_method,
                "log_transform_skewed": self.log_transform_skewed,
                "datetime_include_time": self.datetime_include_time,
                "normalize_text": self.normalize_text,
            },
            "cleaning": self.cleaning,
            "missing": self.missing,
            "encoding": self.encoding,
            "type_conversion": self.type_conversion,
            "features": self.features,
            "output_path": self.output_path,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_auto_config() -> dict:
    """Return default AUTO mode config dict."""
    return Config().to_dict()


def build_manual_config(
    drop_columns: list = None,
    missing_strategies: dict = None,
    encoding_strategies: dict = None,
    derived_features: list = None,
    **kwargs,
) -> dict:
    """Convenience builder for manual mode configs."""
    cfg = Config(mode="manual", **kwargs)
    if drop_columns:
        cfg.cleaning["drop_columns"] = drop_columns
    if missing_strategies:
        cfg.missing = missing_strategies
    if encoding_strategies:
        cfg.encoding = encoding_strategies
    if derived_features:
        cfg.features["derived"] = derived_features
    return cfg.to_dict()
