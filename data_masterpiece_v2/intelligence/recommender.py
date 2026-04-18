"""
data_masterpiece_v2.intelligence.recommender - Model Recommender

Recommends appropriate ML models based on data characteristics.

Usage:
    >>> recommender = ModelRecommender()
    >>> results = recommender.recommend(df, target="survived")
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from data_masterpiece_v2.utils.logger import get_logger
from data_masterpiece_v2.utils.helpers import detect_task_type

logger = get_logger("ModelRecommender")


class ModelRecommender:
    """
    Recommend appropriate ML models based on data characteristics.

    Analyzes:
    - Problem type (classification/regression)
    - Data size
    - Feature types
    - Complexity requirements

    Examples
    --------
    Basic usage:

        >>> recommender = ModelRecommender()
        >>> results = recommender.recommend(df, target='survived')
        >>> print(results['models'][0]['name'])
    """

    # Model library with metadata
    MODEL_LIBRARY = {
        'classification': {
            'binary': [
                {'name': 'Logistic Regression', 'category': 'linear', 'complexity': 'low', 'pros': ['Fast', 'Interpretable'], 'cons': ['Limited to linear']},
                {'name': 'Random Forest', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['Robust', 'Handles non-linear'], 'cons': ['Can overfit']},
                {'name': 'Gradient Boosting', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['High accuracy', 'Feature importance'], 'cons': ['Slower training']},
                {'name': 'XGBoost', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['State-of-art', 'Handles missing'], 'cons': ['Requires tuning']},
                {'name': 'LightGBM', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['Fast', 'Efficient'], 'cons': ['Less interpretable']},
                {'name': 'Support Vector Machine', 'category': 'svm', 'complexity': 'high', 'pros': ['Good for small data'], 'cons': ['Slow for large data']},
                {'name': 'Neural Network', 'category': 'neural', 'complexity': 'high', 'pros': ['Handles complex patterns'], 'cons': ['Needs tuning', 'Black box']},
                {'name': 'K-Nearest Neighbors', 'category': 'neighbors', 'complexity': 'low', 'pros': ['Simple', 'No training'], 'cons': ['Slow prediction']},
            ],
            'multiclass': [
                {'name': 'Logistic Regression (Multinomial)', 'category': 'linear', 'complexity': 'low', 'pros': ['Fast', 'Multiclass ready'], 'cons': ['Linear only']},
                {'name': 'Random Forest', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['Handles multiclass', 'Robust'], 'cons': ['Can overfit']},
                {'name': 'Gradient Boosting', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['High accuracy'], 'cons': ['Slower']},
                {'name': 'XGBoost', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['State-of-art'], 'cons': ['Complex tuning']},
                {'name': 'Neural Network', 'category': 'neural', 'complexity': 'high', 'pros': ['Complex patterns'], 'cons': ['Needs data']},
            ]
        },
        'regression': [
            {'name': 'Linear Regression', 'category': 'linear', 'complexity': 'low', 'pros': ['Interpretable', 'Fast'], 'cons': ['Linear only']},
            {'name': 'Ridge Regression', 'category': 'linear', 'complexity': 'low', 'pros': ['Regularized', 'Handles multicollinearity'], 'cons': ['Limited flexibility']},
            {'name': 'Lasso Regression', 'category': 'linear', 'complexity': 'low', 'pros': ['Feature selection'], 'cons': ['May exclude useful features']},
            {'name': 'ElasticNet', 'category': 'linear', 'complexity': 'low', 'pros': ['Combines Ridge and Lasso'], 'cons': ['More hyperparameters']},
            {'name': 'Random Forest Regressor', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['Non-linear', 'Robust'], 'cons': ['Less interpretable']},
            {'name': 'Gradient Boosting Regressor', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['High accuracy'], 'cons': ['Can overfit']},
            {'name': 'XGBoost Regressor', 'category': 'ensemble', 'complexity': 'medium', 'pros': ['State-of-art'], 'cons': ['Requires tuning']},
            {'name': 'Support Vector Regressor', 'category': 'svm', 'complexity': 'high', 'pros': ['Good for small data'], 'cons': ['Slow for large data']},
            {'name': 'Neural Network Regressor', 'category': 'neural', 'complexity': 'high', 'pros': ['Complex patterns'], 'cons': ['Needs tuning']},
        ]
    }

    def __init__(self):
        """Initialize the model recommender."""
        self.results: Dict[str, Any] = {}

    def recommend(
        self,
        df: pd.DataFrame,
        target: str,
        problem_info: Optional[Dict] = None,
        n_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Recommend ML models based on data characteristics.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Target column name.
        problem_info : Dict, optional
            Additional problem information from other analysis.
        n_recommendations : int
            Number of models to recommend.

        Returns
        -------
        Dict[str, Any]
            Recommendations including problem type, models, and tips.
        """
        logger.info(f"Recommending models for target: {target}")

        # Detect problem type
        target_series = df[target]
        problem_type = self._detect_problem_type(target_series)
        severity = self._detect_severity(target_series, problem_type)

        # Select models based on problem type
        if problem_type == 'classification':
            if severity == 'binary':
                models = self.MODEL_LIBRARY['classification']['binary']
            else:
                models = self.MODEL_LIBRARY['classification']['multiclass']
        else:
            models = self.MODEL_LIBRARY['regression']

        # Score and rank models
        scored_models = self._score_models(models, df, target)

        # Select top recommendations
        top_models = scored_models[:n_recommendations]

        self.results = {
            'problem_type': problem_type,
            'severity': severity,
            'models': top_models,
            'n_rows': len(df),
            'n_features': len(df.columns) - 1,
            'target_distribution': target_series.value_counts().to_dict()
        }

        logger.info(f"Recommended {len(top_models)} models for {problem_type} ({severity})")

        return self.results

    def _detect_problem_type(self, target: pd.Series) -> str:
        """Detect classification vs regression."""
        return detect_task_type(target)

    def _detect_severity(self, target: pd.Series, problem_type: str) -> str:
        """Detect binary vs multiclass."""
        if problem_type == 'regression':
            return 'continuous'

        unique = target.nunique()

        if unique == 2:
            return 'binary'
        elif unique <= 10:
            return 'multiclass'
        else:
            return 'high_cardinality'

    def _score_models(
        self,
        models: List[Dict],
        df: pd.DataFrame,
        target: str
    ) -> List[Dict[str, Any]]:
        """Score models based on data characteristics."""
        n_rows = len(df)
        n_features = len(df.columns) - 1

        scored = []

        for model in models:
            score = {
                **model,
                'score': 0.0,
                'reasons': []
            }

            # Score based on data size
            if model['complexity'] == 'low':
                if n_rows < 1000:
                    score['score'] += 1.0
                    score['reasons'].append('Good for small data')
            elif model['complexity'] == 'medium':
                if 1000 <= n_rows < 100000:
                    score['score'] += 1.0
                    score['reasons'].append('Good for medium data')
                elif n_rows >= 100000:
                    score['score'] += 0.8
                    score['reasons'].append('Handles large data')
            elif model['complexity'] == 'high':
                if n_rows >= 10000:
                    score['score'] += 1.0
                    score['reasons'].append('Benefits from more data')

            # Score based on feature count
            if n_features < 20 and model['category'] in ['linear', 'neighbors']:
                score['score'] += 0.5
                score['reasons'].append('Good for low-dimensional data')

            scored.append(score)

        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)

        # Add rank
        for i, model in enumerate(scored):
            model['rank'] = i + 1

        return scored

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed info about a specific model."""
        all_models = (
            self.MODEL_LIBRARY['classification']['binary'] +
            self.MODEL_LIBRARY['classification']['multiclass'] +
            self.MODEL_LIBRARY['regression']
        )

        for model in all_models:
            if model['name'].lower() in model_name.lower():
                return model

        return None
