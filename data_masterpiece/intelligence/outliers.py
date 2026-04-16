"""
data_masterpiece.intelligence.outliers  --  OutlierDetectionEngine

Rule-based outlier detection and treatment.  No ML.

Detection methods (per column):
  - IQR    (Interquartile Range)  -- default for skewed distributions
  - ZSCORE (Z-score)              -- default for near-normal distributions
  - BOTH   -- flag if either method triggers

Treatment strategies:
  - "clip"   -- winsorise to [lower_fence, upper_fence]
  - "drop"   -- remove the entire row
  - "flag"   -- add a boolean indicator column
  - "impute" -- replace outlier with column median

Auto-mode selects IQR for |skew| > 1.0, Z-score otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger

Strategy = Literal["clip", "drop", "flag", "impute"]
Method   = Literal["iqr", "zscore", "both", "auto"]


@dataclass
class OutlierReport:
    """Summary of all outlier operations."""
    column_stats: dict = field(default_factory=dict)
    total_outlier_cells: int = 0
    rows_dropped: int = 0
    columns_flagged: list = field(default_factory=list)
    shape_before: tuple = (0, 0)
    shape_after: tuple = (0, 0)

    def print_report(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print("  OUTLIER DETECTION REPORT")
        print(sep)
        print(f"  Shape before : {self.shape_before}  ->  after : {self.shape_after}")
        print(f"  Rows dropped : {self.rows_dropped}")
        print(f"  Outlier cells found : {self.total_outlier_cells}")
        if self.columns_flagged:
            print(f"  Flag columns added  : {self.columns_flagged}")
        print(f"\n  Per-column breakdown:")
        for col, stats in self.column_stats.items():
            n = stats.get("n_outliers", 0)
            if n == 0:
                continue
            print(
                f"    - {col:30s}  outliers={n:4d}  "
                f"method={stats['method']:6s}  "
                f"strategy={stats['strategy']:6s}  "
                f"bounds=[{stats['lower']:.2f}, {stats['upper']:.2f}]"
            )
        print(sep + "\n")


class OutlierDetectionEngine:
    """
    Detect and treat outliers in every numeric column of a DataFrame.

    Parameters
    ----------
    method        : "iqr" | "zscore" | "both" | "auto"
    strategy      : "clip" | "drop" | "flag" | "impute"
    iqr_factor    : IQR fence multiplier (default 1.5).
    zscore_thresh : Z-score threshold (default 3.0).
    col_overrides : {col_name: {"method": ..., "strategy": ...}}
    skip_cols     : Columns to skip entirely.
    """

    def __init__(
        self,
        method: Method = "auto",
        strategy: Strategy = "clip",
        iqr_factor: float = 1.5,
        zscore_thresh: float = 3.0,
        col_overrides: dict = None,
        skip_cols: list = None,
    ):
        self.method = method
        self.strategy = strategy
        self.iqr_factor = iqr_factor
        self.zscore_thresh = zscore_thresh
        self.col_overrides = col_overrides or {}
        self.skip_cols = set(skip_cols or [])
        self.log = get_logger("OutlierDetectionEngine")

    def run(
        self, df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, OutlierReport]:
        """Detect and treat outliers.  Returns (cleaned_df, OutlierReport)."""
        df = df.copy()
        report = OutlierReport(shape_before=df.shape)
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.skip_cols
        ]

        rows_to_drop: set = set()
        flag_cols: list = []
        total_outliers = 0

        for col in num_cols:
            override = self.col_overrides.get(col, {})
            method = override.get("method", self.method)
            strategy = override.get("strategy", self.strategy)

            series = df[col].dropna()
            if len(series) < 10:
                self.log.debug(f"[{col}] skipped -- fewer than 10 non-null values.")
                continue

            if method == "auto":
                try:
                    skew = float(series.skew())
                except Exception:
                    skew = 0.0
                method = "iqr" if abs(skew) > 1.0 else "zscore"

            lower, upper = self._compute_bounds(series, method)
            mask = (df[col] < lower) | (df[col] > upper)
            n_out = int(mask.sum())
            total_outliers += n_out

            report.column_stats[col] = {
                "method": method,
                "strategy": strategy,
                "lower": lower,
                "upper": upper,
                "n_outliers": n_out,
            }

            if n_out == 0:
                continue

            self.log.info(
                f"[{col}] {n_out} outliers | method={method} | "
                f"strategy={strategy} | bounds=[{lower:.2f}, {upper:.2f}]"
            )

            if strategy == "clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif strategy == "drop":
                rows_to_drop.update(df.index[mask].tolist())
            elif strategy == "flag":
                flag_col = f"{col}_is_outlier"
                df[flag_col] = mask.astype(np.uint8)
                flag_cols.append(flag_col)
            elif strategy == "impute":
                median_val = series.median()
                df.loc[mask, col] = median_val

        if rows_to_drop:
            df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

        report.total_outlier_cells = total_outliers
        report.rows_dropped = len(rows_to_drop)
        report.columns_flagged = flag_cols
        report.shape_after = df.shape
        report.print_report()
        return df, report

    def _compute_bounds(
        self, series: pd.Series, method: str,
    ) -> tuple[float, float]:
        """Return (lower_fence, upper_fence) for the chosen method."""
        lower_iqr = upper_iqr = lower_z = upper_z = 0.0

        if method in ("iqr", "both"):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_iqr = q1 - self.iqr_factor * iqr
            upper_iqr = q3 + self.iqr_factor * iqr
            if method == "iqr":
                return lower_iqr, upper_iqr

        if method in ("zscore", "both"):
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                return float(series.min()), float(series.max())
            lower_z = mu - self.zscore_thresh * sigma
            upper_z = mu + self.zscore_thresh * sigma
            if method == "zscore":
                return lower_z, upper_z

        # "both" -- union (wider = more conservative)
        return min(lower_iqr, lower_z), max(upper_iqr, upper_z)
