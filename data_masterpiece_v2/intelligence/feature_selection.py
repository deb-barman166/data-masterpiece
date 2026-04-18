"""
data_masterpiece_v2.intelligence.feature_selection - Feature Selection

Selects the most important features for ML.

Methods:
    - Variance threshold
    - Correlation-based
    - Mutual information
    - Random Forest importance

Usage:
    >>> selector = FeatureSelector(method="correlation")
    >>> df_selected, results = selector.select(df, target="survived")
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("FeatureSelector")


class FeatureSelector:
    """
    Select the most important features for machine learning.

    Parameters
    ----------
    method : str
        Selection method: 'auto', 'variance', 'correlation', 'mutual_info', 'random_forest'
    threshold : float
        Threshold for selection.
    top_k : int
        Select top K features (0 = use threshold).

    Examples
    --------
    By correlation:

        >>> selector = FeatureSelector(method='correlation', threshold=0.1)
        >>> df_selected, results = selector.select(df, target='survived')

    Top K features:

        >>> selector = FeatureSelector(method='correlation', top_k=10)
        >>> df_selected, results = selector.select(df, target='survived')
    """

    def __init__(
        self,
        method: str = 'auto',
        threshold: float = 0.01,
        top_k: int = 0
    ):
        """Initialize the feature selector."""
        self.method = method
        self.threshold = threshold
        self.top_k = top_k
        self.results: Dict[str, Any] = {}

    def select(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select features from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Target column name.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Selected DataFrame and selection results.
        """
        logger.info(f"Selecting features for target: {target}")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if len(numeric_cols) == 0:
            logger.warning("No numeric features to select from")
            return df, {'selected_features': df.columns.tolist()}

        # Calculate feature importance
        scores = self._calculate_scores(df[numeric_cols], df[target])

        # Select features
        selected = self._select_features(scores)

        # Create selected DataFrame
        selected_features = [f for f in selected if f in df.columns]
        if target in df.columns:
            selected_features = [target] + selected_features

        df_selected = df[selected_features]

        self.results = {
            'method': self.method,
            'scores': scores,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'n_original': len(numeric_cols)
        }

        logger.info(f"Selected {len(selected_features)} features from {len(numeric_cols)}")

        return df_selected, self.results

    def _calculate_scores(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate importance scores for each feature."""
        scores = {}

        if self.method == 'auto':
            method = 'correlation'
        else:
            method = self.method

        for col in features.columns:
            if method == 'variance':
                scores[col] = features[col].var()

            elif method == 'correlation':
                corr = features[col].corr(target)
                scores[col] = abs(corr) if not np.isnan(corr) else 0

            elif method == 'mutual_info':
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi = mutual_info_regression(
                        features[[col]],
                        target,
                        random_state=42
                    )[0]
                    scores[col] = mi
                except:
                    scores[col] = abs(features[col].corr(target))

            elif method == 'random_forest':
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(features[[col]], target)
                    scores[col] = rf.feature_importances_[0]
                except:
                    scores[col] = abs(features[col].corr(target))

            else:
                scores[col] = abs(features[col].corr(target))

        return scores

    def _select_features(self, scores: Dict[str, float]) -> List[str]:
        """Select features based on scores."""
        if self.top_k > 0:
            # Select top K
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [f[0] for f in sorted_features[:self.top_k]]

        # Select based on threshold
        selected = [f for f, s in scores.items() if s >= self.threshold]

        # Always return at least some features
        if len(selected) == 0:
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in sorted_features[:5]]

        return selected

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        if 'scores' not in self.results:
            return []

        sorted_features = sorted(
            self.results['scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:n]
