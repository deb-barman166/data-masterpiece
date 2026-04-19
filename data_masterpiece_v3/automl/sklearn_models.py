"""
data_masterpiece_v3.automl.sklearn_models
──────────────────────────────────────────
Scikit-learn AutoML — automatically trains and evaluates multiple ML models.
No AI or pre-trained models used — we BUILD from scratch every time!

Supported models:
  CLASSIFICATION: LogisticRegression, RandomForest, GradientBoosting,
                  SVM, KNN, DecisionTree, ExtraTrees, AdaBoost, BaggingClassifier
  REGRESSION:     LinearRegression, Ridge, Lasso, ElasticNet, RandomForest,
                  GradientBoosting, SVR, KNN, DecisionTree, ExtraTrees
"""

from __future__ import annotations

import time
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, classification_report,
    confusion_matrix,
)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeClassifier,
)
from sklearn.naive_bayes import GaussianNB
from ..utils.logger import get_logger

log = get_logger("SklearnAutoML")

# ─── Model registries ─────────────────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "LogisticRegression":       LogisticRegression(max_iter=500, random_state=42),
    "RandomForestClassifier":   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting":         GradientBoostingClassifier(n_estimators=100, random_state=42),
    "ExtraTreesClassifier":     ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVC":                      SVC(probability=True, random_state=42),
    "KNeighbors":               KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "DecisionTree":             DecisionTreeClassifier(random_state=42),
    "GaussianNB":               GaussianNB(),
    "AdaBoost":                 AdaBoostClassifier(n_estimators=50, random_state=42),
}

REGRESSION_MODELS = {
    "LinearRegression":         LinearRegression(),
    "Ridge":                    Ridge(alpha=1.0),
    "Lasso":                    Lasso(alpha=0.01),
    "ElasticNet":               ElasticNet(alpha=0.01, l1_ratio=0.5),
    "RandomForestRegressor":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoostingRegressor":GradientBoostingRegressor(n_estimators=100, random_state=42),
    "ExtraTreesRegressor":      ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "SVR":                      SVR(),
    "KNeighborsRegressor":      KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "DecisionTreeRegressor":    DecisionTreeRegressor(random_state=42),
    "AdaBoostRegressor":        AdaBoostRegressor(n_estimators=50, random_state=42),
}


# ─── Main AutoML class ────────────────────────────────────────────────────────

class SklearnAutoML:
    """
    Trains ALL sklearn models and ranks them by performance.
    Returns full detailed results including cross-validation, feature importance, etc.
    """

    def __init__(self, task_type="auto", cv_folds=5, max_models=8,
                 time_limit=300.0, output_dir="output/models"):
        self.task_type   = task_type
        self.cv_folds    = cv_folds
        self.max_models  = max_models
        self.time_limit  = time_limit
        self.output_dir  = output_dir
        self.results_: Dict[str, Any] = {}
        self.best_model_ = None
        self.best_name_  = ""

    def run(self, X_train, X_test, y_train, y_test, feature_names=None) -> dict:
        """Train all models and return comprehensive results."""
        task = self._detect_task(y_train) if self.task_type == "auto" else self.task_type
        log.info(f"🤖 SklearnAutoML: task={task}, cv={self.cv_folds} folds")

        models = (
            CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        )
        # Limit models
        model_items = list(models.items())[:self.max_models]

        model_results = []
        t_global = time.time()

        for name, model in model_items:
            if time.time() - t_global > self.time_limit:
                log.warning(f"  Time limit reached! Stopping after {len(model_results)} models.")
                break

            result = self._train_evaluate(
                name, model, task,
                X_train, X_test, y_train, y_test, feature_names
            )
            model_results.append(result)
            score_key = "test_accuracy" if task == "classification" else "test_r2"
            score_val = result.get(score_key, 0)
            log.info(f"  ✓ {name:<35} {score_key}={score_val:.4f}")

        # Rank models
        score_key = "test_accuracy" if task == "classification" else "test_r2"
        model_results.sort(key=lambda x: x.get(score_key, -999), reverse=True)

        best = model_results[0] if model_results else {}
        self.best_name_ = best.get("name", "")

        summary = {
            "task_type": task,
            "n_samples": int(len(X_train) + len(X_test)),
            "n_features": int(X_train.shape[1]),
            "total_build_time": round(time.time() - t_global, 2),
            "best_model": {
                "name": self.best_name_,
                "score": best.get(score_key, 0),
                "score_metric": score_key,
            },
            "models": model_results,
            "backend": "sklearn",
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(f"{self.output_dir}/sklearn_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.results_ = summary
        log.info(f"  🏆 Best: {self.best_name_} ({score_key}={best.get(score_key,0):.4f})")
        return summary

    # ─────────────────────────────────────────────────────────────────────────

    def _train_evaluate(self, name, model, task, X_train, X_test, y_train, y_test, feature_names):
        result = {"name": name, "backend": "sklearn", "status": "success"}
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            result["training_time"] = round(time.time() - t0, 4)

            if task == "classification":
                y_pred      = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                    except Exception:
                        pass

                result["test_accuracy"]  = round(float(accuracy_score(y_test, y_pred)), 4)
                result["test_f1_macro"]  = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
                result["test_precision"] = round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4)
                result["test_recall"]    = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4)

                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    try:
                        result["test_roc_auc"] = round(float(roc_auc_score(y_test, y_pred_proba[:, 1])), 4)
                    except Exception:
                        pass

                # Cross-validation
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                            scoring="accuracy", n_jobs=-1)
                result["cv_mean"] = round(float(cv_scores.mean()), 4)
                result["cv_std"]  = round(float(cv_scores.std()), 4)
                result["cv_scores"] = [round(float(s), 4) for s in cv_scores]
                result["train_accuracy"] = round(float(accuracy_score(y_train, model.predict(X_train))), 4)

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                result["confusion_matrix"] = cm.tolist()

                # Overfit check
                gap = result["train_accuracy"] - result["test_accuracy"]
                result["overfit_gap"] = round(gap, 4)
                result["overfit_risk"] = "high" if gap > 0.15 else ("medium" if gap > 0.05 else "low")

            else:  # regression
                y_pred = model.predict(X_test)
                result["test_r2"]   = round(float(r2_score(y_test, y_pred)), 4)
                result["test_mae"]  = round(float(mean_absolute_error(y_test, y_pred)), 4)
                result["test_rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                result["test_mse"]  = round(float(mean_squared_error(y_test, y_pred)), 4)

                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                            scoring="r2", n_jobs=-1)
                result["cv_mean"]  = round(float(cv_scores.mean()), 4)
                result["cv_std"]   = round(float(cv_scores.std()), 4)
                result["cv_scores"]= [round(float(s), 4) for s in cv_scores]
                result["train_r2"] = round(float(r2_score(y_train, model.predict(X_train))), 4)

                gap = result["train_r2"] - result["test_r2"]
                result["overfit_gap"] = round(gap, 4)
                result["overfit_risk"] = "high" if gap > 0.20 else ("medium" if gap > 0.10 else "low")

            # Feature importance
            if feature_names and hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fi_dict = dict(sorted(
                    zip(feature_names, fi.tolist()),
                    key=lambda x: x[1], reverse=True
                ))
                result["feature_importance"] = {k: round(v, 6) for k, v in list(fi_dict.items())[:20]}
            elif feature_names and hasattr(model, "coef_"):
                coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                fi_dict = dict(zip(feature_names, [abs(float(c)) for c in coef]))
                fi_dict = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
                result["feature_importance"] = {k: round(v, 6) for k, v in list(fi_dict.items())[:20]}

            # Recommendation
            result["recommendation"] = self._recommend(name, result)

        except Exception as e:
            result["status"] = "failed"
            result["error"]  = str(e)
            log.warning(f"  ✗ {name} failed: {e}")

        return result

    def _detect_task(self, y) -> str:
        n_unique = len(np.unique(y))
        if n_unique <= 20 or str(y.dtype) in ("object", "bool", "category"):
            return "classification"
        return "regression"

    def _recommend(self, name, result) -> str:
        tips = {
            "LogisticRegression":        "Great baseline. Fast and interpretable. Check for linear separability.",
            "RandomForestClassifier":    "Robust and powerful. Works well out-of-the-box. Good for feature importance.",
            "RandomForestRegressor":     "Strong ensemble model. Handles non-linearity well.",
            "GradientBoosting":          "Often top performer. Try tuning n_estimators and learning_rate.",
            "GradientBoostingRegressor": "Powerful for regression. Tune learning_rate and max_depth.",
            "Ridge":                     "L2 regularization. Good when features are correlated.",
            "Lasso":                     "L1 regularization. Good for feature selection (zeroes out weak features).",
            "SVC":                       "Powerful with small datasets. Slow on large data. Try RBF kernel.",
            "KNeighbors":                "Simple and effective. Sensitive to feature scaling.",
            "DecisionTree":              "Highly interpretable. Prone to overfitting — consider pruning.",
            "ExtraTreesClassifier":      "Faster than RandomForest. Often comparable performance.",
            "GaussianNB":                "Very fast. Assumes feature independence. Good for text-like data.",
            "AdaBoost":                  "Boosts weak learners. Sensitive to noise and outliers.",
            "ElasticNet":                "Mix of L1+L2. Good when you have many correlated features.",
        }
        return tips.get(name, "Train and validate carefully. Check for overfitting.")
