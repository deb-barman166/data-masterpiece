"""
data_masterpiece_v3.intelligence.stats
───────────────────────────────────────
Deep statistical analysis engine.
Computes everything you'd ever want to know about your dataset.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from ..utils.logger import get_logger

log = get_logger("StatsEngine")


class StatsEngine:
    """Computes comprehensive statistics for every column in a DataFrame."""

    def __init__(self):
        self.results: dict = {}

    def run(self, df: pd.DataFrame, target: str = None) -> dict:
        log.info("📊 StatsEngine running deep analysis ...")
        num_cols = df.select_dtypes(include="number").columns.tolist()

        overview = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "numeric_cols": len(num_cols),
            "total_nulls": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        }

        column_stats = {}
        for col in df.columns:
            column_stats[col] = self._col_stats(df[col])

        correlation = {}
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            correlation["pearson"] = corr_matrix.round(4).to_dict()
            if target and target in num_cols:
                target_corr = corr_matrix[target].drop(target).sort_values(
                    key=abs, ascending=False
                )
                correlation["target_correlation"] = target_corr.round(4).to_dict()

        # Distribution tests on numeric cols
        normality = {}
        for col in num_cols[:15]:  # limit to 15 for speed
            series = df[col].dropna()
            if len(series) >= 8:
                try:
                    stat, pval = sp_stats.shapiro(series.sample(min(len(series), 5000), random_state=42))
                    normality[col] = {
                        "shapiro_stat": round(float(stat), 4),
                        "shapiro_pval": round(float(pval), 6),
                        "is_normal": bool(pval > 0.05),
                        "skewness": round(float(series.skew()), 4),
                        "kurtosis": round(float(series.kurtosis()), 4),
                    }
                except Exception:
                    pass

        self.results = {
            "overview": overview,
            "column_stats": column_stats,
            "correlation": correlation,
            "normality": normality,
        }

        log.info(f"  Rows: {overview['n_rows']} | Cols: {overview['n_cols']}")
        log.info(f"  Total nulls: {overview['total_nulls']}")
        return self.results

    def _col_stats(self, series: pd.Series) -> dict:
        stats = {
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(float(series.isnull().mean() * 100), 2),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            stats.update({
                "mean":   round(float(desc.get("mean", 0)), 4),
                "std":    round(float(desc.get("std", 0)), 4),
                "min":    round(float(desc.get("min", 0)), 4),
                "q25":    round(float(desc.get("25%", 0)), 4),
                "median": round(float(desc.get("50%", 0)), 4),
                "q75":    round(float(desc.get("75%", 0)), 4),
                "max":    round(float(desc.get("max", 0)), 4),
                "skewness":  round(float(series.skew()), 4),
                "kurtosis":  round(float(series.kurtosis()), 4),
                "iqr":    round(float(desc.get("75%", 0) - desc.get("25%", 0)), 4),
            })
        else:
            top_vals = series.value_counts().head(5)
            stats.update({
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
                "mode": str(series.mode()[0]) if series.mode().any() else "N/A",
            })

        return stats
