"""
data_masterpiece.intelligence.recommender  --  ModelRecommender

Rule-based ML model recommendation engine.  No training, no ML --
purely heuristic based on data characteristics.

Considers: row count, feature count, feature/target types,
correlation strength, dimensionality, and multicollinearity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger


@dataclass
class ModelRecommendation:
    """A single model recommendation."""
    priority: int
    model_name: str
    category: str           # e.g. "ensemble", "linear", "tree", "neural"
    reasoning: list         # list of pro-reason strings
    caveats: list           # list of warning strings
    library_hint: str       # e.g. "from sklearn.ensemble import RandomForestClassifier"


@dataclass
class RecommendationReport:
    """Full set of recommendations with metadata."""
    target_type: str = ""         # "binary", "multiclass", "continuous"
    recommendations: list = field(default_factory=list)
    preprocessing_notes: list = field(default_factory=list)
    n_rows: int = 0
    n_features: int = 0
    mean_abs_corr: float = 0.0

    def print_report(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print("  MODEL RECOMMENDATION REPORT")
        print(sep)
        print(f"  Target type     : {self.target_type}")
        print(f"  Rows            : {self.n_rows:,}")
        print(f"  Features        : {self.n_features}")
        print(f"  Mean |corr|     : {self.mean_abs_corr:.3f}")
        print()
        for rec in self.recommendations:
            print(f"  #{rec.priority} {rec.model_name}  [{rec.category}]")
            for r in rec.reasoning:
                print(f"     + {r}")
            for c in rec.caveats:
                print(f"     ! {c}")
            print(f"     > {rec.library_hint}")
            print()

        if self.preprocessing_notes:
            print("  Preprocessing suggestions:")
            for n in self.preprocessing_notes:
                print(f"    ! {n}")

        print(sep + "\n")


class ModelRecommender:
    """
    Recommend ML models based on data characteristics.

    Parameters
    ----------
    max_recommendations : Max number of models to recommend (default 5).
    """

    def __init__(self, max_recommendations: int = 5):
        self.max_recs = max_recommendations
        self.log = get_logger("ModelRecommender")

    def recommend(
        self,
        df: pd.DataFrame,
        target: str,
        mean_abs_corr: float = 0.0,
    ) -> RecommendationReport:
        """
        Generate model recommendations.

        Parameters
        ----------
        df              : Clean numeric DataFrame.
        target          : Target column name.
        mean_abs_corr   : Mean absolute correlation among features.

        Returns
        -------
        RecommendationReport
        """
        n_rows = len(df)
        n_features = len([c for c in df.columns if c != target])

        # -- detect target type --
        target_series = df[target]
        unique = target_series.nunique()

        if unique <= 2:
            target_type = "binary"
        elif unique <= 20 and n_rows / unique > 20:
            target_type = "multiclass"
        else:
            target_type = "continuous"

        # -- build recommendations --
        recs: list = []
        notes: list = []

        if target_type == "binary":
            recs.extend(self._binary_recs(n_rows, n_features, mean_abs_corr))
        elif target_type == "multiclass":
            recs.extend(self._multiclass_recs(n_rows, n_features, mean_abs_corr))
        else:
            recs.extend(self._regression_recs(n_rows, n_features, mean_abs_corr))

        # -- preprocessing notes --
        if mean_abs_corr > 0.7:
            notes.append("High multicollinearity detected -- consider PCA or feature pruning.")
        if n_features > 100:
            notes.append("High dimensionality -- consider feature selection or regularization.")
        if n_rows < 500:
            notes.append("Small dataset -- prefer simpler models to avoid overfitting.")
        if n_rows > 100000:
            notes.append("Large dataset -- tree-based ensembles and SGD-based models recommended.")
        if n_rows / n_features < 10:
            notes.append("Low sample-to-feature ratio -- risk of overfitting. Consider regularization.")

        report = RecommendationReport(
            target_type=target_type,
            recommendations=recs[:self.max_recs],
            preprocessing_notes=notes,
            n_rows=n_rows,
            n_features=n_features,
            mean_abs_corr=mean_abs_corr,
        )
        report.print_report()
        return report

    # -- recommendation builders -----------------------------------------------

    def _binary_recs(self, n, f, corr) -> list:
        recs = []
        recs.append(ModelRecommendation(
            priority=1, model_name="XGBoost Classifier", category="ensemble",
            reasoning=[
                "Excellent performance on tabular binary classification",
                "Handles non-linear relationships natively",
                "Built-in feature importance and regularization",
            ],
            caveats=["Requires careful hyperparameter tuning"],
            library_hint="from xgboost import XGBClassifier",
        ))
        recs.append(ModelRecommendation(
            priority=2, model_name="Random Forest Classifier", category="ensemble",
            reasoning=[
                "Robust baseline for binary classification",
                "Resistant to overfitting via bagging",
                "Works well with mixed feature scales",
            ],
            caveats=["May underperform vs gradient boosting on large data"],
            library_hint="from sklearn.ensemble import RandomForestClassifier",
        ))
        recs.append(ModelRecommendation(
            priority=3, model_name="Logistic Regression", category="linear",
            reasoning=[
                "Fast, interpretable baseline",
                "Provides probability estimates natively",
                "Low memory footprint",
            ],
            caveats=["Assumes linear decision boundary", "Sensitive to feature scaling"],
            library_hint="from sklearn.linear_model import LogisticRegression",
        ))
        if n < 10000:
            recs.append(ModelRecommendation(
                priority=4, model_name="Support Vector Machine (RBF)", category="kernel",
                reasoning=[
                    "Effective in high-dimensional spaces",
                    "Strong with proper kernel choice",
                ],
                caveats=["O(n^2) or O(n^3) scaling -- slow for large datasets", "Requires feature scaling"],
                library_hint="from sklearn.svm import SVC",
            ))
        recs.append(ModelRecommendation(
            priority=5, model_name="LightGBM Classifier", category="ensemble",
            reasoning=[
                "Fast training, low memory usage",
                "Native handling of categorical features",
                "State-of-the-art on many tabular benchmarks",
            ],
            caveats=["Can overfit on very small datasets"],
            library_hint="from lightgbm import LGBMClassifier",
        ))
        return recs

    def _multiclass_recs(self, n, f, corr) -> list:
        recs = []
        recs.append(ModelRecommendation(
            priority=1, model_name="XGBoost Classifier", category="ensemble",
            reasoning=[
                "Handles multi-class natively (multi:softmax)",
                "Strong performance on structured data",
                "Built-in regularization",
            ],
            caveats=["May need class-weight balancing for imbalanced data"],
            library_hint="from xgboost import XGBClassifier",
        ))
        recs.append(ModelRecommendation(
            priority=2, model_name="Random Forest Classifier", category="ensemble",
            reasoning=[
                "Natural multi-class support",
                "Handles class imbalance via class_weight",
                "No feature scaling required",
            ],
            caveats=["Memory-intensive with many trees"],
            library_hint="from sklearn.ensemble import RandomForestClassifier",
        ))
        recs.append(ModelRecommendation(
            priority=3, model_name="LightGBM Classifier", category="ensemble",
            reasoning=[
                "Fast multi-class training",
                "Excellent accuracy on tabular data",
            ],
            caveats=["Requires careful hyperparameter search"],
            library_hint="from lightgbm import LGBMClassifier",
        ))
        recs.append(ModelRecommendation(
            priority=4, model_name="Logistic Regression (OvR)", category="linear",
            reasoning=[
                "Fast multi-class baseline (One-vs-Rest)",
                "Interpretable coefficients",
            ],
            caveats=["Linear boundary assumption", "May struggle with complex patterns"],
            library_hint="from sklearn.linear_model import LogisticRegression (multi_class='ovr')",
        ))
        return recs

    def _regression_recs(self, n, f, corr) -> list:
        recs = []
        recs.append(ModelRecommendation(
            priority=1, model_name="XGBoost Regressor", category="ensemble",
            reasoning=[
                "Top performer on structured regression tasks",
                "Handles non-linear patterns and interactions",
                "Built-in regularization (L1/L2)",
            ],
            caveats=["Sensitive to outliers in target"],
            library_hint="from xgboost import XGBRegressor",
        ))
        recs.append(ModelRecommendation(
            priority=2, model_name="Random Forest Regressor", category="ensemble",
            reasoning=[
                "Robust to outliers and noise",
                "No feature scaling needed",
                "Built-in feature importance",
            ],
            caveats=["Cannot extrapolate beyond training range"],
            library_hint="from sklearn.ensemble import RandomForestRegressor",
        ))
        recs.append(ModelRecommendation(
            priority=3, model_name="Ridge Regression", category="linear",
            reasoning=[
                "Strong linear baseline with L2 regularization",
                "Fast training and prediction",
                "Handles multicollinearity well",
            ],
            caveats=["Assumes linear relationship"],
            library_hint="from sklearn.linear_model import Ridge",
        ))
        recs.append(ModelRecommendation(
            priority=4, model_name="LightGBM Regressor", category="ensemble",
            reasoning=[
                "Fast training on large datasets",
                "Often matches or beats XGBoost",
                "Low memory usage",
            ],
            caveats=["Requires hyperparameter tuning for best results"],
            library_hint="from lightgbm import LGBMRegressor",
        ))
        recs.append(ModelRecommendation(
            priority=5, model_name="Gradient Boosting Regressor", category="ensemble",
            reasoning=[
                "Sequential boosting for regression",
                "Good default choice from sklearn",
                "Handles heterogeneous features",
            ],
            caveats=["Slower than parallel ensemble methods"],
            library_hint="from sklearn.ensemble import GradientBoostingRegressor",
        ))
        return recs
