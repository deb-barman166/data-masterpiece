"""
data_masterpiece.preprocessing.controller  --  Central Pipeline Controller

Orchestrates all preprocessing agents in sequence:
  Cleaning -> TypeConversion -> MissingValues -> Encoding -> FeatureTransform -> Validation
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from data_masterpiece.preprocessing.agents.cleaning import CleaningAgent
from data_masterpiece.preprocessing.agents.missing import MissingValueAgent
from data_masterpiece.preprocessing.agents.type_conversion import TypeConversionAgent
from data_masterpiece.preprocessing.agents.encoding import EncodingAgent
from data_masterpiece.preprocessing.agents.feature import FeatureTransformationAgent
from data_masterpiece.preprocessing.agents.validation import ValidationAgent
from data_masterpiece.utils.logger import get_logger


class PipelineController:
    """
    Runs agents in the following default order:

      1. CleaningAgent
      2. TypeConversionAgent
      3. MissingValueAgent
      4. EncodingAgent
      5. FeatureTransformationAgent
      6. ValidationAgent
    """

    AGENT_ORDER = [
        "cleaning",
        "type_conversion",
        "missing",
        "encoding",
        "feature",
        "validation",
    ]

    _AGENT_MAP = {
        "cleaning": CleaningAgent,
        "type_conversion": TypeConversionAgent,
        "missing": MissingValueAgent,
        "encoding": EncodingAgent,
        "feature": FeatureTransformationAgent,
        "validation": ValidationAgent,
    }

    def __init__(self, config: dict):
        self.config = config
        self.log = get_logger("PreprocessController")
        self._active_agents: list = config.get(
            "active_agents", self.AGENT_ORDER,
        )
        self._mode: str = config.get("mode", "sequential")
        self.summary: dict = {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        original_shape = df.shape
        self.log.info("=" * 60)
        self.log.info("  PREPROCESSING PIPELINE  --  START")
        self.log.info(f"  Input shape : {original_shape}")
        self.log.info(f"  Mode        : {self._mode}")
        self.log.info(f"  Agents      : {self._active_agents}")
        self.log.info("=" * 60)

        t_start = time.perf_counter()
        agent_reports: dict = {}

        for name in self.AGENT_ORDER:
            if name not in self._active_agents:
                self.log.info(f"[{name}] -- skipped (not in active_agents)")
                continue

            agent_cls = self._AGENT_MAP[name]
            agent = agent_cls()

            if self._mode == "safe":
                try:
                    df = agent.execute(df, self.config)
                except Exception as exc:
                    self.log.error(
                        f"[{name}] crashed in safe mode: {exc}. Continuing..."
                    )
            else:
                df = agent.execute(df, self.config)

            agent_reports[name] = agent.report

        total_s = round(time.perf_counter() - t_start, 3)
        self.summary = self._build_summary(
            original_shape, df.shape, agent_reports, total_s,
        )
        self._print_summary()
        return df

    def _build_summary(self, orig, final, reports, elapsed) -> dict:
        clean_rep = reports.get("cleaning", {})
        miss_rep = reports.get("missing", {})
        enc_rep = reports.get("encoding", {})
        feat_rep = reports.get("feature", {})
        val_rep = reports.get("validation", {})

        return {
            "original_shape": orig,
            "final_shape": final,
            "rows_removed": orig[0] - final[0],
            "dropped_cols": clean_rep.get("dropped_cols", []),
            "duplicate_rows_removed": clean_rep.get("dropped_rows", 0),
            "columns_imputed": list(miss_rep.get("filled_columns", {}).keys()),
            "encoding_log": enc_rep.get("encoding_log", {}),
            "feature_transforms": feat_rep.get("feature_transforms", []),
            "validation_errors": val_rep.get("validation_errors", []),
            "validation_warnings": val_rep.get("validation_warnings", []),
            "memory_mb": val_rep.get("memory_mb", 0),
            "total_elapsed_s": elapsed,
        }

    def _print_summary(self):
        s = self.summary
        self.log.info("=" * 60)
        self.log.info("  PREPROCESSING PIPELINE  --  COMPLETE")
        self.log.info("=" * 60)
        self.log.info(f"  Original shape   : {s['original_shape']}")
        self.log.info(f"  Final shape      : {s['final_shape']}")
        self.log.info(f"  Duplicate rows   : {s['duplicate_rows_removed']}")
        self.log.info(f"  Dropped columns  : {s['dropped_cols']}")
        self.log.info(f"  Imputed columns  : {s['columns_imputed']}")
        self.log.info(f"  Encodings applied: {len(s['encoding_log'])}")
        self.log.info(f"  Derived features : {s['feature_transforms']}")
        self.log.info(f"  Validation errors: {s['validation_errors']}")
        self.log.info(f"  Memory           : {s['memory_mb']} MB")
        self.log.info(f"  Total time       : {s['total_elapsed_s']}s")
        self.log.info("=" * 60)
