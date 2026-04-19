"""
data_masterpiece_v3.intelligence.outlier
──────────────────────────────────────────
Outlier Detection and Treatment — finds unusual data points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..utils.logger import get_logger

log = get_logger("OutlierEngine")


class OutlierEngine:
    """
    Detects and handles outliers.
    Methods: IQR (box-plot rule), Z-Score, or AUTO (picks best per column).
    Strategy: clip (replace extreme values) or remove (drop rows).
    """

    def __init__(self, method="auto", strategy="clip", iqr_factor=1.5, zscore_thresh=3.0):
        self.method        = method
        self.strategy      = strategy
        self.iqr_factor    = iqr_factor
        self.zscore_thresh = zscore_thresh
        self.summary: dict = {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info(f"🔭 OutlierEngine: method={self.method}, strategy={self.strategy}")
        df = df.copy()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        outlier_counts = {}
        rows_removed = 0

        mask_remove = pd.Series([False] * len(df), index=df.index)

        for col in num_cols:
            method = self._pick_method(df[col])
            outlier_mask = self._detect(df[col], method)
            n_outliers = outlier_mask.sum()
            if n_outliers == 0:
                continue
            outlier_counts[col] = int(n_outliers)

            if self.strategy == "clip":
                if method == "iqr":
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower = q1 - self.iqr_factor * iqr
                    upper = q3 + self.iqr_factor * iqr
                    df[col] = df[col].clip(lower, upper)
                else:
                    mean, std = df[col].mean(), df[col].std()
                    df[col] = df[col].clip(
                        mean - self.zscore_thresh * std,
                        mean + self.zscore_thresh * std
                    )
                log.info(f"  [{col}] {n_outliers} outliers clipped")
            elif self.strategy == "remove":
                mask_remove |= outlier_mask

        if self.strategy == "remove" and mask_remove.any():
            rows_removed = mask_remove.sum()
            df = df[~mask_remove]
            log.info(f"  Removed {rows_removed} outlier rows")

        self.summary = {
            "outlier_counts": outlier_counts,
            "total_outlier_cols": len(outlier_counts),
            "rows_removed": int(rows_removed),
            "method": self.method,
            "strategy": self.strategy,
        }
        return df

    def _pick_method(self, series: pd.Series) -> str:
        if self.method != "auto":
            return self.method
        skew = abs(series.skew())
        return "iqr" if skew > 1.0 else "zscore"

    def _detect(self, series: pd.Series, method: str) -> pd.Series:
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (series < q1 - self.iqr_factor * iqr) | (series > q3 + self.iqr_factor * iqr)
        else:
            mean, std = series.mean(), series.std()
            if std == 0:
                return pd.Series([False] * len(series), index=series.index)
            z = (series - mean) / std
            return z.abs() > self.zscore_thresh
