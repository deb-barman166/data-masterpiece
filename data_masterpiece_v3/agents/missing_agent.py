"""
data_masterpiece_v3.agents.missing_agent
─────────────────────────────────────────
Missing Value Agent — fills in the blanks intelligently.
AUTO mode: picks the best strategy per column automatically.
MANUAL mode: you tell it exactly what to do per column.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from ..utils.logger import get_logger

log = get_logger("MissingAgent")


class MissingAgent:
    """
    Imputes missing values.

    AUTO strategies chosen:
    • Numeric with low skew  → mean
    • Numeric with high skew → median
    • Categorical            → mode
    • Time series            → forward-fill

    MANUAL: specify per column in JSON:
        "missing": { "age": "median", "city": "unknown", "price": "mean" }

    Supported strategies: mean, median, mode, ffill, bfill,
                          zero, unknown, drop, constant:<value>
    """

    def __init__(self, cfg: dict):
        self.per_col: dict  = cfg.get("missing", {})
        self.mode: str      = cfg.get("mode", "auto")
        self.skew_thresh    = 1.0
        self.summary: dict  = {}

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("🔧 MissingAgent started")
        df = df.copy()
        imputed = {}

        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()

        if not cols_with_nulls:
            log.info("  No missing values found — nothing to impute!")
            self.summary = {"columns_imputed": []}
            return df

        for col in cols_with_nulls:
            strategy = self._choose_strategy(df, col)
            df, applied = self._apply_strategy(df, col, strategy)
            if applied:
                imputed[col] = strategy

        self.summary = {
            "columns_imputed": list(imputed.keys()),
            "strategies": imputed,
        }
        log.info(f"  Imputed {len(imputed)} columns")
        return df

    # ─────────────────────────────────────────────────────────────────────────

    def _choose_strategy(self, df: pd.DataFrame, col: str) -> str:
        """Auto-detect the best imputation strategy for a column."""
        # User override
        if col in self.per_col:
            return self.per_col[col]

        if self.mode == "manual":
            # In manual mode, unspecified cols get auto treatment anyway
            pass

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            skewness = abs(series.skew())
            return "median" if skewness > self.skew_thresh else "mean"
        else:
            return "mode"

    def _apply_strategy(self, df: pd.DataFrame, col: str, strategy: str):
        """Apply the strategy and return updated df + success flag."""
        series = df[col]

        try:
            if strategy == "mean":
                val = series.mean()
                df[col] = series.fillna(val)
                log.info(f"  [{col}] mean={val:.4f}")

            elif strategy == "median":
                val = series.median()
                df[col] = series.fillna(val)
                log.info(f"  [{col}] median={val:.4f}")

            elif strategy == "mode":
                mode_vals = series.mode()
                if len(mode_vals) > 0:
                    df[col] = series.fillna(mode_vals[0])
                    log.info(f"  [{col}] mode={mode_vals[0]!r}")

            elif strategy == "ffill":
                df[col] = series.ffill()
                log.info(f"  [{col}] forward-fill")

            elif strategy == "bfill":
                df[col] = series.bfill()
                log.info(f"  [{col}] back-fill")

            elif strategy == "zero":
                df[col] = series.fillna(0)
                log.info(f"  [{col}] filled with 0")

            elif strategy == "unknown":
                df[col] = series.fillna("unknown")
                log.info(f"  [{col}] filled with 'unknown'")

            elif strategy == "drop":
                df = df.dropna(subset=[col])
                log.info(f"  [{col}] rows with null dropped")

            elif strategy.startswith("constant:"):
                val = strategy.split(":", 1)[1]
                df[col] = series.fillna(val)
                log.info(f"  [{col}] constant={val!r}")

            else:
                log.warning(f"  [{col}] Unknown strategy '{strategy}' — skipping")
                return df, False

            return df, True

        except Exception as e:
            log.warning(f"  [{col}] Strategy failed: {e}")
            return df, False
