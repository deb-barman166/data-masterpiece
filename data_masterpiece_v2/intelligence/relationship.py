"""
data_masterpiece_v2.intelligence.relationship - Relationship Analyzer

Analyzes relationships between features using correlation analysis.

Usage:
    >>> analyzer = RelationshipAnalyzer()
    >>> results = analyzer.analyze(df, target="survived")
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("RelationshipAnalyzer")


class RelationshipAnalyzer:
    """
    Analyze relationships between features using various methods.

    Provides:
    - Correlation matrix
    - Strong correlation pairs
    - Multicollinearity detection
    - Target correlations

    Examples
    --------
    Basic usage:

        >>> analyzer = RelationshipAnalyzer()
        >>> results = analyzer.analyze(df, target='survived')
        >>> print(results['correlation_matrix'])
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize the relationship analyzer."""
        self.threshold = threshold
        self.results: Dict[str, Any] = {}

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Analyze relationships in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Target column for correlation analysis.
        method : str
            Correlation method: 'pearson', 'spearman', 'kendall'

        Returns
        -------
        Dict[str, Any]
            Analysis results including correlation matrix and insights.
        """
        logger.info(f"Analyzing relationships with target: {target}")

        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)

        # Find strong pairs
        strong_pairs = self._find_strong_pairs(corr_matrix, target)

        # Find multicollinearity
        multicollinear = self._find_multicollinearity(corr_matrix)

        # Target correlations
        target_corr = self._get_target_correlations(corr_matrix, target)

        self.results = {
            'correlation_matrix': corr_matrix,
            'strong_pairs': strong_pairs,
            'multicollinear_pairs': multicollinear,
            'target_correlations': target_corr,
            'method': method,
            'threshold': self.threshold
        }

        logger.info(f"Found {len(strong_pairs)} strong correlations")
        logger.info(f"Found {len(multicollinear)} multicollinear pairs")

        return self.results

    def _find_strong_pairs(
        self,
        corr_matrix: pd.DataFrame,
        target: str
    ) -> List[Dict[str, Any]]:
        """Find feature pairs with strong correlations."""
        pairs = []

        if target not in corr_matrix.columns:
            return pairs

        # Get correlations with target
        for col in corr_matrix.columns:
            if col == target:
                continue

            corr = corr_matrix.loc[col, target]

            if abs(corr) >= self.threshold:
                pairs.append({
                    'feature_a': col,
                    'feature_b': target,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'direction': 'positive' if corr > 0 else 'negative'
                })

        # Sort by absolute correlation
        pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

        return pairs

    def _find_multicollinearity(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.85
    ) -> List[Tuple[str, str, float]]:
        """Find multicollinear feature pairs."""
        pairs = []

        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[col1, col2]

                if abs(corr) >= threshold:
                    pairs.append((col1, col2, corr))

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def _get_target_correlations(
        self,
        corr_matrix: pd.DataFrame,
        target: str
    ) -> Dict[str, float]:
        """Get correlations of all features with target."""
        if target not in corr_matrix.columns:
            return {}

        correlations = {}

        for col in corr_matrix.columns:
            if col != target:
                correlations[col] = corr_matrix.loc[col, target]

        # Sort by absolute value
        sorted_corr = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        return sorted_corr

    def get_heatmap_data(self) -> Dict[str, Any]:
        """Get data for correlation heatmap visualization."""
        if 'correlation_matrix' not in self.results:
            return {}

        corr = self.results['correlation_matrix']

        return {
            'data': corr.values.tolist(),
            'labels': corr.columns.tolist(),
            'title': 'Correlation Heatmap'
        }
