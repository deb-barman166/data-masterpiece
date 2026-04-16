"""
data_masterpiece.intelligence.profiler  --  StatisticalProfiler

Deep, column-level statistical profiling.  No ML -- pure numpy/pandas statistics.

For every numeric column computes:
  Central tendency : mean, median, mode
  Spread           : std, variance, range, IQR
  Shape            : skewness, kurtosis
  Extremes         : min, max, p1, p5, p25, p75, p95, p99
  Nulls & zeros    : null_count, zero_count, unique_count
  Distribution     : normal / right-skewed / left-skewed / uniform / bimodal-heuristic
  Outlier estimate  : IQR-based fence count
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger

_NORMAL       = "normal"
_RIGHT_SKEWED = "right-skewed"
_LEFT_SKEWED  = "left-skewed"
_UNIFORM      = "uniform"
_BIMODAL_HINT = "possibly-bimodal"
_UNKNOWN      = "unknown"


@dataclass
class ColumnProfile:
    """Full statistical profile of one column."""
    name: str
    dtype: str
    n_total: int
    n_null: int
    n_zero: int
    n_unique: int
    null_pct: float
    mean: float
    median: float
    mode: float
    std: float
    variance: float
    minimum: float
    maximum: float
    range_val: float
    p1: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float
    n_iqr_outliers: int
    distribution: str

    def summary_line(self) -> str:
        return (
            f"  {self.name:30s}  "
            f"dist={self.distribution:16s}  "
            f"mean={self.mean:10.3f}  "
            f"std={self.std:9.3f}  "
            f"skew={self.skewness:6.2f}  "
            f"outliers={self.n_iqr_outliers:4d}  "
            f"null={self.null_pct:.1%}"
        )


class StatisticalProfiler:
    """
    Compute a rich statistical profile for every numeric column.

    Parameters
    ----------
    percentiles : Quantiles to compute (default covers standard percentiles).
    """

    _PERCENTILES = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]

    def __init__(self, percentiles: list = None):
        self.percentiles = percentiles or self._PERCENTILES
        self.log = get_logger("StatisticalProfiler")

    def profile(
        self, df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[ColumnProfile]]:
        """
        Profile all numeric columns.

        Returns (summary_df, [ColumnProfile, ...]).
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            self.log.warning("No numeric columns to profile.")
            return pd.DataFrame(), []

        self.log.info(f"Profiling {len(num_cols)} numeric columns ...")
        profiles: list[ColumnProfile] = []

        for col in num_cols:
            series = df[col]
            profile = self._profile_column(col, series, str(df[col].dtype))
            profiles.append(profile)

        summary_df = self._to_dataframe(profiles)
        self._print_report(profiles)
        return summary_df, profiles

    def to_dict(self, profiles: list[ColumnProfile]) -> dict:
        """Return a JSON-serialisable nested dict of all profiles."""
        return {
            p.name: {
                "dtype": p.dtype, "n_total": p.n_total, "n_null": p.n_null,
                "n_zero": p.n_zero, "n_unique": p.n_unique,
                "null_pct": round(p.null_pct, 4),
                "mean": round(p.mean, 6), "median": round(p.median, 6),
                "mode": round(p.mode, 6), "std": round(p.std, 6),
                "variance": round(p.variance, 6), "min": round(p.minimum, 6),
                "max": round(p.maximum, 6), "range": round(p.range_val, 6),
                "p1": round(p.p1, 6), "p5": round(p.p5, 6),
                "p25": round(p.p25, 6), "p75": round(p.p75, 6),
                "p95": round(p.p95, 6), "p99": round(p.p99, 6),
                "iqr": round(p.iqr, 6), "skewness": round(p.skewness, 4),
                "kurtosis": round(p.kurtosis, 4), "cv": round(p.cv, 4),
                "n_iqr_outliers": p.n_iqr_outliers,
                "distribution": p.distribution,
            }
            for p in profiles
        }

    # -- private helpers -------------------------------------------------------

    def _profile_column(
        self, name: str, series: pd.Series, dtype: str,
    ) -> ColumnProfile:
        clean = series.dropna()
        n_total = len(series)
        n_null = int(series.isna().sum())

        if len(clean) == 0:
            return ColumnProfile(
                name=name, dtype=dtype, n_total=n_total, n_null=n_null,
                n_zero=0, n_unique=0, null_pct=1.0,
                mean=np.nan, median=np.nan, mode=np.nan, std=np.nan,
                variance=np.nan, minimum=np.nan, maximum=np.nan,
                range_val=np.nan,
                p1=np.nan, p5=np.nan, p25=np.nan, p75=np.nan,
                p95=np.nan, p99=np.nan,
                iqr=np.nan, skewness=np.nan, kurtosis=np.nan,
                cv=np.nan, n_iqr_outliers=0, distribution=_UNKNOWN,
            )

        vals = clean.astype(float).values
        n_zero = int((clean == 0).sum())
        n_uniq = int(clean.nunique())

        pcts = np.nanpercentile(vals, [p * 100 for p in self.percentiles])
        p1, p5, p25, p75, p95, p99 = pcts

        iqr = p75 - p25
        mn = float(np.nanmean(vals))
        med = float(np.nanmedian(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        var = float(np.nanvar(vals, ddof=1)) if len(vals) > 1 else 0.0
        skew = self._safe_skew(vals)
        kurt = self._safe_kurtosis(vals)
        cv = (std / mn) if mn != 0 else np.nan

        try:
            mode_val = float(clean.mode().iloc[0])
        except Exception:
            mode_val = med

        lo = p25 - 1.5 * iqr
        hi = p75 + 1.5 * iqr
        n_out = int(((clean < lo) | (clean > hi)).sum())

        dist = self._label_distribution(skew, kurt, cv, n_uniq, len(clean))

        return ColumnProfile(
            name=name, dtype=dtype, n_total=n_total, n_null=n_null,
            n_zero=n_zero, n_unique=n_uniq, null_pct=n_null / n_total,
            mean=mn, median=med, mode=mode_val, std=std, variance=var,
            minimum=float(np.nanmin(vals)), maximum=float(np.nanmax(vals)),
            range_val=float(np.nanmax(vals) - np.nanmin(vals)),
            p1=float(p1), p5=float(p5), p25=float(p25),
            p75=float(p75), p95=float(p95), p99=float(p99),
            iqr=float(iqr), skewness=skew, kurtosis=kurt, cv=cv,
            n_iqr_outliers=n_out, distribution=dist,
        )

    @staticmethod
    def _safe_skew(vals: np.ndarray) -> float:
        if len(vals) < 3:
            return 0.0
        try:
            mu = np.mean(vals)
            sig = np.std(vals, ddof=1)
            if sig == 0:
                return 0.0
            return float(np.mean(((vals - mu) / sig) ** 3))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_kurtosis(vals: np.ndarray) -> float:
        if len(vals) < 4:
            return 0.0
        try:
            mu = np.mean(vals)
            sig = np.std(vals, ddof=1)
            if sig == 0:
                return 0.0
            return float(np.mean(((vals - mu) / sig) ** 4) - 3)
        except Exception:
            return 0.0

    @staticmethod
    def _label_distribution(skew, kurt, cv, n_unique, n) -> str:
        if n_unique <= 2:
            return "binary"
        if n_unique <= 10:
            return "discrete-low-cardinality"
        abs_skew = abs(skew)
        if abs_skew < 0.5 and abs(kurt) < 1.0:
            return _NORMAL
        if skew > 1.0:
            return _RIGHT_SKEWED
        if skew < -1.0:
            return _LEFT_SKEWED
        if kurt < -1.2 and abs_skew < 0.5:
            return _UNIFORM
        if kurt > 3.0 and abs_skew < 0.5:
            return _BIMODAL_HINT
        return "moderate-skew"

    @staticmethod
    def _to_dataframe(profiles: list[ColumnProfile]) -> pd.DataFrame:
        rows = []
        for p in profiles:
            rows.append({
                "column": p.name, "dtype": p.dtype,
                "n_total": p.n_total, "n_null": p.n_null,
                "null_pct": round(p.null_pct, 4),
                "n_unique": p.n_unique, "n_zero": p.n_zero,
                "mean": p.mean, "median": p.median, "std": p.std,
                "min": p.minimum, "max": p.maximum, "range": p.range_val,
                "p25": p.p25, "p75": p.p75, "iqr": p.iqr,
                "skewness": p.skewness, "kurtosis": p.kurtosis,
                "cv": p.cv, "n_iqr_outliers": p.n_iqr_outliers,
                "distribution": p.distribution,
            })
        return pd.DataFrame(rows).set_index("column")

    def _print_report(self, profiles: list[ColumnProfile]) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print("  STATISTICAL PROFILE REPORT")
        print(sep)
        print(
            f"  {'Column':30s}  {'Distribution':16s}  {'Mean':>10s}  "
            f"{'Std':>9s}  {'Skew':>6s}  {'Outliers':>8s}  {'Null%':>6s}"
        )
        print("  " + "-" * 56)
        for p in profiles:
            print(p.summary_line())

        dist_counts: dict = {}
        for p in profiles:
            dist_counts[p.distribution] = dist_counts.get(p.distribution, 0) + 1
        print(f"\n  Distribution breakdown:")
        for dist, cnt in sorted(dist_counts.items(), key=lambda x: -x[1]):
            print(f"    {dist:25s}  {cnt}")

        skewed = sorted(profiles, key=lambda p: abs(p.skewness), reverse=True)[:5]
        if skewed:
            print(f"\n  Most skewed columns (candidates for log transform):")
            for p in skewed:
                if abs(p.skewness) > 0.5:
                    print(f"    {p.name:30s}  skew={p.skewness:+.3f}")

        print(sep + "\n")
