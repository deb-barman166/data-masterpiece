"""
data_masterpiece_v3.config
───────────────────────────
Unified Configuration System — Version 3.

Supports:
  • AUTO mode  : everything decided automatically from data
  • MANUAL mode: user controls every parameter via JSON file or Python dict

The JSON file is your control panel — modify it however you like!
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Main Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Master configuration for Data Masterpiece v3.

    Quick start
    -----------
    Auto mode (let the pipeline decide everything):
        cfg = Config()

    Manual mode (you control everything):
        cfg = Config.from_json("my_config.json")

    Save current config to JSON (so you can edit it):
        cfg.save_json("my_config.json")
    """

    # ── Pipeline mode ─────────────────────────────────────────────────────────
    mode: str = "auto"           # "auto" or "manual"
    active_agents: List[str] = field(default_factory=lambda: [
        "cleaning", "type_conversion", "missing",
        "encoding", "feature", "validation",
    ])

    # ── Global preprocessing ──────────────────────────────────────────────────
    drop_duplicates: bool = True
    null_drop_threshold: float = 0.60      # drop col if >60% null
    variance_threshold: float = 1e-10      # drop near-zero-variance cols
    low_card_threshold: int = 10           # ≤ this → categorical
    med_card_threshold: int = 50           # ≤ this → one-hot candidate
    normalize: bool = False
    scale_method: str = "minmax"           # "minmax" | "standard" | "robust"
    log_transform_skewed: bool = False
    datetime_include_time: bool = False
    normalize_text: bool = True

    # ── Per-column overrides (manual mode) ────────────────────────────────────
    cleaning: Dict[str, Any] = field(default_factory=dict)
    missing: Dict[str, Any] = field(default_factory=dict)
    encoding: Dict[str, Any] = field(default_factory=dict)
    type_conversion: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)

    # ── Intelligence engine ───────────────────────────────────────────────────
    run_intelligence: bool = True
    outlier_method: str = "auto"           # "iqr" | "zscore" | "auto" | "none"
    outlier_strategy: str = "clip"         # "clip" | "remove"
    iqr_factor: float = 1.5
    zscore_thresh: float = 3.0
    intelligence_variance_threshold: float = 0.01
    intelligence_corr_threshold: float = 0.90
    intelligence_top_k: int = 0            # 0 = keep all
    test_size: float = 0.20
    val_size: float = 0.10
    stratify: bool = True
    max_viz_cols: int = 25
    skip_outlier: bool = False
    skip_selection: bool = False
    skip_report: bool = False

    # ── Relationship analysis ─────────────────────────────────────────────────
    relationship_columns: List[List[str]] = field(default_factory=list)
    # Example: [["age", "income"], ["price", "quantity", "revenue"]]

    # ── AutoML settings ───────────────────────────────────────────────────────
    run_automl: bool = False               # requires user permission
    automl_task: str = "auto"             # "auto" | "classification" | "regression"
    automl_backends: List[str] = field(default_factory=lambda: ["sklearn"])
    # Options: ["sklearn", "pytorch"]
    automl_max_models: int = 8
    automl_cv_folds: int = 5
    automl_time_limit: float = 300.0      # seconds
    automl_output_dir: str = "output/models"
    pytorch_epochs: int = 50
    pytorch_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    pytorch_lr: float = 1e-3
    pytorch_batch_size: int = 32

    # ── Output paths ──────────────────────────────────────────────────────────
    output_path: str = "output/processed.csv"
    plot_dir: str = "output/plots"
    report_path: str = "output/report.html"
    ml_ready_dir: str = "output/ml_ready"

    # ── Visualization ─────────────────────────────────────────────────────────
    chart_style: str = "dark"             # "dark" | "light"
    chart_dpi: int = 150
    interactive_charts: bool = True       # generate Plotly HTML charts in report

    # ─────────────────────────────────────────────────────────────────────────
    #  Class methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from a JSON file (your personal control panel!)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build config from a plain Python dictionary."""
        # Flatten nested "global" key into top-level
        if "global" in data:
            for k, v in data["global"].items():
                data.setdefault(k, v)
            del data["global"]

        # Only pass known fields
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Export config as a nested dict (pipeline-friendly format)."""
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
            "relationship_columns": self.relationship_columns,
            "output_path": self.output_path,
        }

    def save_json(self, path: str) -> None:
        """Save full config to JSON so you can edit and reload it."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        full = {
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
            "relationship_columns": self.relationship_columns,
            "run_intelligence": self.run_intelligence,
            "outlier_method": self.outlier_method,
            "outlier_strategy": self.outlier_strategy,
            "iqr_factor": self.iqr_factor,
            "zscore_thresh": self.zscore_thresh,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "stratify": self.stratify,
            "run_automl": self.run_automl,
            "automl_task": self.automl_task,
            "automl_backends": self.automl_backends,
            "automl_max_models": self.automl_max_models,
            "automl_cv_folds": self.automl_cv_folds,
            "pytorch_epochs": self.pytorch_epochs,
            "pytorch_hidden_sizes": self.pytorch_hidden_sizes,
            "pytorch_lr": self.pytorch_lr,
            "pytorch_batch_size": self.pytorch_batch_size,
            "output_path": self.output_path,
            "plot_dir": self.plot_dir,
            "report_path": self.report_path,
            "ml_ready_dir": self.ml_ready_dir,
            "chart_style": self.chart_style,
            "interactive_charts": self.interactive_charts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"Config(mode={self.mode!r}, "
            f"agents={self.active_agents}, "
            f"automl={self.run_automl}, "
            f"intelligence={self.run_intelligence})"
        )
