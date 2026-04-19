"""
data_masterpiece_v3.agents.cleaning_agent
──────────────────────────────────────────
Cleaning Agent — removes noise, duplicates, garbage columns.
Think of this as the "sweeper" that tidies up your data!
"""

from __future__ import annotations

import re
import pandas as pd
from ..utils.logger import get_logger

log = get_logger("CleaningAgent")


class CleaningAgent:
    """
    Cleans raw DataFrame:
    • Removes exact duplicate rows
    • Drops columns with too many nulls
    • Drops near-zero-variance columns
    • Strips whitespace from column names
    • Normalises string text (optional)
    • Drops user-specified columns
    """

    def __init__(self, cfg: dict):
        g = cfg.get("global", {})
        c = cfg.get("cleaning", {})

        self.drop_duplicates      = g.get("drop_duplicates", True)
        self.null_drop_threshold  = g.get("null_drop_threshold", 0.60)
        self.variance_threshold   = g.get("variance_threshold", 1e-10)
        self.normalize_text       = g.get("normalize_text", True)
        self.drop_columns         = c.get("drop_columns", [])
        self.summary: dict        = {}

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("🧹 CleaningAgent started")
        df = df.copy()
        initial_shape = df.shape

        df = self._clean_col_names(df)
        df = self._drop_user_columns(df)
        df = self._remove_duplicates(df)
        df = self._drop_high_null_cols(df)
        df = self._drop_zero_variance(df)
        if self.normalize_text:
            df = self._normalize_text(df)

        removed_rows = initial_shape[0] - df.shape[0]
        removed_cols = initial_shape[1] - df.shape[1]

        self.summary = {
            "rows_before": initial_shape[0],
            "cols_before": initial_shape[1],
            "rows_after": df.shape[0],
            "cols_after": df.shape[1],
            "rows_removed": removed_rows,
            "cols_removed": removed_cols,
        }

        log.info(f"  Rows removed : {removed_rows}")
        log.info(f"  Cols removed : {removed_cols}")
        log.info(f"  Shape after  : {df.shape}")
        return df

    # ─────────────────────────────────────────────────────────────────────────

    def _clean_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            re.sub(r"\s+", "_", str(c).strip().lower())
            for c in df.columns
        ]
        return df

    def _drop_user_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self.drop_columns if c in df.columns]
        if to_drop:
            log.info(f"  Dropping user-specified cols: {to_drop}")
            df = df.drop(columns=to_drop)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            n = before - len(df)
            if n:
                log.info(f"  Duplicate rows removed: {n}")
        return df

    def _drop_high_null_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        null_ratio = df.isnull().mean()
        to_drop = null_ratio[null_ratio > self.null_drop_threshold].index.tolist()
        if to_drop:
            log.info(f"  High-null cols dropped (>{self.null_drop_threshold*100:.0f}%): {to_drop}")
            df = df.drop(columns=to_drop)
        return df

    def _drop_zero_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns
        variances = df[num_cols].var(numeric_only=True)
        to_drop = variances[variances <= self.variance_threshold].index.tolist()
        if to_drop:
            log.info(f"  Near-zero-variance cols dropped: {to_drop}")
            df = df.drop(columns=to_drop)
        return df

    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        obj_cols = df.select_dtypes(include="object").columns
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df
