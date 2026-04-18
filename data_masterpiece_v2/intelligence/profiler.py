"""
data_masterpiece_v2.intelligence.profiler - Statistical Profiler

Computes comprehensive statistical profiles for all columns.

Usage:
    >>> profiler = StatisticalProfiler()
    >>> results = profiler.profile(df)
    >>> print(results['summary'])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("StatisticalProfiler")


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""
    name: str
    dtype: str
    count: int
    null_count: int
    null_pct: float
    unique_count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    distribution: str


class StatisticalProfiler:
    """
    Compute statistical profiles for all columns in a DataFrame.

    Provides detailed statistics including:
    - Central tendency (mean, median, mode)
    - Spread (std, variance, range, IQR)
    - Shape (skewness, kurtosis)
    - Distribution classification

    Parameters
    ----------
    percentiles : List[float]
        Percentiles to compute. Default: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    Examples
    --------
    Basic usage:

        >>> profiler = StatisticalProfiler()
        >>> results = profiler.profile(df)
        >>> for col, profile in results['profiles'].items():
        ...     print(f"{col}: mean={profile['mean']:.2f}")
    """

    def __init__(self, percentiles: Optional[List[float]] = None):
        """Initialize the profiler."""
        self.percentiles = percentiles or [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        self.results: Dict[str, ColumnProfile] = {}

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Profile all columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to profile.

        Returns
        -------
        Dict[str, Any]
            Profile results including profiles dict and summary DataFrame.
        """
        logger.info(f"Profiling {len(df.columns)} columns...")

        profiles = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in df.columns:
            if col in numeric_cols:
                profile = self._profile_numeric(df[col])
            else:
                profile = self._profile_categorical(df[col])

            profiles[col] = profile

        # Create summary DataFrame
        summary_df = self._create_summary_df(profiles)

        self.results = {
            'profiles': profiles,
            'summary': summary_df
        }

        return self.results

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a numeric column."""
        clean = series.dropna()

        if len(clean) == 0:
            return self._empty_profile(series.name, 'numeric')

        # Calculate statistics
        mean = clean.mean()
        median = clean.median()
        std = clean.std()
        min_val = clean.min()
        max_val = clean.max()
        q25 = clean.quantile(0.25)
        q75 = clean.quantile(0.75)
        iqr = q75 - q25

        # Calculate skewness and kurtosis
        skewness = self._calculate_skewness(clean)
        kurtosis = self._calculate_kurtosis(clean)

        # Classify distribution
        distribution = self._classify_distribution(skewness, kurtosis, std, mean)

        return {
            'name': series.name,
            'dtype': 'numeric',
            'count': len(series),
            'null_count': series.isna().sum(),
            'null_pct': series.isna().sum() / len(series),
            'unique_count': clean.nunique(),
            'mean': mean,
            'median': median,
            'std': std,
            'min': min_val,
            'max': max_val,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'distribution': distribution
        }

    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a categorical column."""
        clean = series.dropna()

        return {
            'name': series.name,
            'dtype': 'categorical',
            'count': len(series),
            'null_count': series.isna().sum(),
            'null_pct': series.isna().sum() / len(series),
            'unique_count': clean.nunique(),
            'mode': clean.mode().iloc[0] if len(clean) > 0 else None,
            'top_values': clean.value_counts().head(5).to_dict(),
            'distribution': 'categorical'
        }

    def _empty_profile(self, name: str, dtype: str) -> Dict[str, Any]:
        """Create an empty profile for null columns."""
        return {
            'name': name,
            'dtype': dtype,
            'count': 0,
            'null_count': 0,
            'null_pct': 1.0,
            'unique_count': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'iqr': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'distribution': 'empty'
        }

    def _calculate_skewness(self, clean: pd.Series) -> float:
        """Calculate skewness."""
        n = len(clean)
        if n < 3:
            return 0.0

        mean = clean.mean()
        std = clean.std()

        if std == 0:
            return 0.0

        skew = (clean - mean) ** 3
        return skew.sum() / (n * std ** 3)

    def _calculate_kurtosis(self, clean: pd.Series) -> float:
        """Calculate kurtosis."""
        n = len(clean)
        if n < 4:
            return 0.0

        mean = clean.mean()
        std = clean.std()

        if std == 0:
            return 0.0

        kurt = (clean - mean) ** 4
        return kurt.sum() / (n * std ** 4) - 3

    def _classify_distribution(
        self,
        skewness: float,
        kurtosis: float,
        std: float,
        mean: float
    ) -> str:
        """Classify the distribution shape."""
        # Check for binary
        if std < 0.01:
            return 'constant'

        # Classify based on skewness
        if abs(skewness) < 0.5:
            if abs(kurtosis) < 0.5:
                return 'normal'
            elif kurtosis > 0:
                return 'leptokurtic'
            else:
                return 'platykurtic'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        else:
            return 'moderately_skewed'

    def _create_summary_df(self, profiles: Dict[str, Dict]) -> pd.DataFrame:
        """Create a summary DataFrame from profiles."""
        rows = []

        for name, profile in profiles.items():
            row = {
                'column': name,
                'dtype': profile['dtype'],
                'count': profile['count'],
                'null_pct': f"{profile['null_pct']:.1%}",
                'unique': profile['unique_count']
            }

            if profile['dtype'] == 'numeric':
                row.update({
                    'mean': f"{profile['mean']:.2f}",
                    'median': f"{profile['median']:.2f}",
                    'std': f"{profile['std']:.2f}",
                    'min': f"{profile['min']:.2f}",
                    'max': f"{profile['max']:.2f}",
                    'skewness': f"{profile['skewness']:.2f}",
                    'distribution': profile['distribution']
                })
            else:
                row['mode'] = str(profile.get('mode', 'N/A'))[:20]

            rows.append(row)

        return pd.DataFrame(rows)
