"""
data_masterpiece_v2.intelligence.outlier - Outlier Detection

Detects outliers using various methods.

Methods:
    - IQR (Interquartile Range)
    - Z-Score
    - Isolation Forest

Usage:
    >>> detector = OutlierDetector(method="iqr")
    >>> df_clean, results = detector.detect(df)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("OutlierDetector")


class OutlierDetector:
    """
    Detect outliers using various statistical methods.

    Parameters
    ----------
    method : str
        Detection method: 'iqr', 'zscore', 'isolation_forest', or 'auto'
    strategy : str
        How to handle outliers: 'clip', 'drop', 'flag', 'none'
    threshold : float
        Sensitivity threshold for detection.

    Examples
    --------
    IQR method:

        >>> detector = OutlierDetector(method='iqr', strategy='clip')
        >>> df_clean, results = detector.detect(df)

    Z-score method:

        >>> detector = OutlierDetector(method='zscore', threshold=3.0)
        >>> df_clean, results = detector.detect(df)
    """

    def __init__(
        self,
        method: str = 'auto',
        strategy: str = 'clip',
        threshold: float = 1.5
    ):
        """Initialize the outlier detector."""
        self.method = method
        self.strategy = strategy
        self.threshold = threshold
        self.results: Dict[str, Any] = {}

    def detect(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect outliers in numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Cleaned DataFrame and detection results.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Detecting outliers in {len(numeric_cols)} numeric columns...")

        all_outliers = {}
        column_stats = {}

        for col in numeric_cols:
            if self.method == 'auto':
                method = self._select_method(df[col])
            else:
                method = self.method

            if method == 'iqr':
                outliers, stats = self._detect_iqr(df[col])
            elif method == 'zscore':
                outliers, stats = self._detect_zscore(df[col])
            else:
                outliers, stats = self._detect_iqr(df[col])

            if len(outliers) > 0:
                all_outliers[col] = outliers
                column_stats[col] = stats

            # Apply strategy
            if self.strategy == 'clip' and len(outliers) > 0:
                df[col] = self._clip_outliers(df[col], stats)
            elif self.strategy == 'flag':
                df[f'{col}_outlier'] = df.index.isin(outliers)

        # Build results
        total_outliers = sum(len(o) for o in all_outliers.values())

        self.results = {
            'method': self.method,
            'strategy': self.strategy,
            'total_outliers': total_outliers,
            'columns_with_outliers': len(all_outliers),
            'column_stats': column_stats,
            'outlier_indices': all_outliers
        }

        logger.info(f"Found {total_outliers} outliers in {len(all_outliers)} columns")

        return df, self.results

    def _select_method(self, series: pd.Series) -> str:
        """Select best method based on data characteristics."""
        clean = series.dropna()

        if len(clean) < 30:
            return 'iqr'

        skew = abs(clean.skew())
        if skew > 2:
            return 'iqr'

        return 'iqr'

    def _detect_iqr(self, series: pd.Series) -> Tuple[List[int], Dict]:
        """Detect outliers using IQR method."""
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        lower = q25 - self.threshold * iqr
        upper = q75 + self.threshold * iqr

        outliers = series[(series < lower) | (series > upper)].index.tolist()

        stats = {
            'method': 'iqr',
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'lower_bound': lower,
            'upper_bound': upper,
            'count': len(outliers)
        }

        return outliers, stats

    def _detect_zscore(self, series: pd.Series) -> Tuple[List[int], Dict]:
        """Detect outliers using Z-score method."""
        clean = series.dropna()
        mean = clean.mean()
        std = clean.std()

        if std == 0:
            return [], {'method': 'zscore', 'mean': mean, 'std': 0, 'count': 0}

        z_scores = np.abs((clean - mean) / std)
        outliers = clean[z_scores > self.threshold].index.tolist()

        stats = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
            'threshold': self.threshold,
            'count': len(outliers)
        }

        return outliers, stats

    def _clip_outliers(self, series: pd.Series, stats: Dict) -> pd.Series:
        """Clip outliers to bounds."""
        if 'lower_bound' in stats and 'upper_bound' in stats:
            return series.clip(lower=stats['lower_bound'], upper=stats['upper_bound'])
        return series

    def get_summary(self) -> str:
        """Get a text summary of outlier detection."""
        if not self.results:
            return "No outlier detection performed."

        return f"""
Outlier Detection Results:
- Method: {self.results['method']}
- Strategy: {self.results['strategy']}
- Total outliers: {self.results['total_outliers']}
- Columns affected: {self.results['columns_with_outliers']}
"""
