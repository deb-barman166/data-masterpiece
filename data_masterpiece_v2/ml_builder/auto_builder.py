"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AUTO ML BUILDER - Build Models Automatically!            ║
║                                                                            ║
║  This module automatically builds, trains, and evaluates ML models.        ║
║  Just provide your data and let the magic happen!                           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
    🤖 Auto-detects problem type (classification/regression)
    📊 Builds multiple models automatically
    🔧 Tries different hyperparameters
    📈 Compares and ranks models
    💾 Saves best model
    📋 Generates detailed reports

Quick Start:
    >>> builder = AutoMLBuilder()
    >>> results = builder.build(df, target="survived")
    >>> print(results['best_model']['name'])

For Beginners:
    >>> # It's this easy!
    >>> builder = AutoMLBuilder()  # Just create it
    >>> results = builder.build(df, target="survived")  # Build models
    >>> print(results['best_model'])  # See the best one!
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_masterpiece_v2.config import MLBuilderConfig
from data_masterpiece_v2.utils.logger import get_logger, format_duration
from data_masterpiece_v2.utils.helpers import detect_task_type

logger = get_logger("AutoMLBuilder")

# Try to import ML libraries
try:
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        AdaBoostClassifier, AdaBoostRegressor
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Auto ML disabled.")


@dataclass
class ModelResult:
    """Container for a single model's results."""
    name: str
    model: Any
    train_score: float
    test_score: float
    cv_score: float
    cv_std: float
    training_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class AutoMLBuilder:
    """
    ═══════════════════════════════════════════════════════════════════════════
    AUTO ML BUILDER - Build ML Models Automatically!
    ═══════════════════════════════════════════════════════════════════════════

    This is the heart of automatic machine learning in Data Masterpiece V2!

    It will:
        1. Detect your problem type (classification or regression)
        2. Build multiple models automatically
        3. Evaluate each model
        4. Compare and rank them
        5. Return the best model(s)

    Parameters
    ----------
    config : MLBuilderConfig
        Configuration for the Auto ML process.
    task_type : str
        Override auto-detection: 'classification', 'regression', or 'auto'

    Examples
    --------
    The EASIEST way to build ML models:

        >>> builder = AutoMLBuilder()
        >>> results = builder.build(df, target="survived")

    With custom settings:

        >>> config = MLBuilderConfig()
        >>> config.max_models = 5
        >>> config.search_strategy = "random"
        >>> builder = AutoMLBuilder(config=config)
        >>> results = builder.build(df, target="price")

    Get your best model:

        >>> results = builder.build(df, target="survived")
        >>> best = results['best_model']
        >>> print(f"Best: {best['name']} with score {best['score']}")

    ═══════════════════════════════════════════════════════════════════════════
    """

    # Model registry
    CLASSIFICATION_MODELS = [
        ('Logistic Regression', LogisticRegression, {
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [200]
        }),
        ('Decision Tree', DecisionTreeClassifier, {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10]
        }),
        ('Random Forest', RandomForestClassifier, {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }),
        ('Gradient Boosting', GradientBoostingClassifier, {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }),
        ('K-Nearest Neighbors', KNeighborsClassifier, {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }),
        ('Naive Bayes', GaussianNB, {}),
        ('AdaBoost', AdaBoostClassifier, {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        }),
        ('Neural Network', MLPClassifier, {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'alpha': [0.0001, 0.001],
            'max_iter': [500]
        }),
    ]

    REGRESSION_MODELS = [
        ('Linear Regression', LinearRegression, {}),
        ('Ridge Regression', Ridge, {
            'alpha': [0.1, 1.0, 10.0]
        }),
        ('Lasso Regression', Lasso, {
            'alpha': [0.1, 1.0, 10.0]
        }),
        ('ElasticNet', ElasticNet, {
            'alpha': [0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }),
        ('Decision Tree', DecisionTreeRegressor, {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5]
        }),
        ('Random Forest', RandomForestRegressor, {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        }),
        ('Gradient Boosting', GradientBoostingRegressor, {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }),
        ('K-Nearest Neighbors', KNeighborsRegressor, {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        ('Neural Network', MLPRegressor, {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'alpha': [0.0001, 0.001],
            'max_iter': [500]
        }),
    ]

    def __init__(
        self,
        config: Optional[MLBuilderConfig] = None,
        task_type: str = "auto"
    ):
        """Initialize the Auto ML Builder."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Auto ML. Install with: pip install scikit-learn")

        self.config = config or MLBuilderConfig()
        self.task_type = task_type

        self.results: Dict[str, Any] = {}
        self.trained_models: List[ModelResult] = []
        self._scaler = None

    def build(
        self,
        df: pd.DataFrame,
        target: str,
        split_result: Optional[Dict] = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        ═══════════════════════════════════════════════════════════════════════════
        BUILD MODELS - The main entry point!
        ═══════════════════════════════════════════════════════════════════════════

        This is where the magic happens! Provide your data and target column,
        and this method will:

            1. Detect problem type
            2. Build multiple models
            3. Evaluate each model
            4. Return best models ranked

        Parameters
        ----------
        df : pd.DataFrame
            Your processed DataFrame ready for ML.
        target : str
            The column you want to predict.
        split_result : Dict, optional
            Pre-computed train/test split from IntelligenceController.
        X_train, X_test, y_train, y_test : optional
            Manual train/test split if split_result not provided.

        Returns
        -------
        Dict[str, Any]
            Complete results including:
            - models: List of all trained models with scores
            - best_model: The best model with full details
            - comparison: Model comparison table
            - problem_type: 'classification' or 'regression'
            - training_time: Total time taken

        Examples
        --------
        Basic usage:

            >>> results = builder.build(df, target="survived")
            >>> print(results['best_model']['name'])

        Using pre-split data:

            >>> results = builder.build(
            ...     df,
            ...     target="price",
            ...     X_train=X_train,
            ...     X_test=X_test,
            ...     y_train=y_train,
            ...     y_test=y_test
            ... )

        ═══════════════════════════════════════════════════════════════════════════
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("  🔥 AUTO ML BUILDER - Starting Model Building!")
        logger.info("=" * 60)

        # Get train/test data
        if split_result is not None:
            X_train = split_result['X_train']
            X_test = split_result['X_test']
            y_train = split_result['y_train']
            y_test = split_result['y_test']
        elif X_train is None:
            raise ValueError("Either split_result or X_train/X_test must be provided")

        # Ensure numeric data
        X_train = self._ensure_numeric(X_train)
        X_test = self._ensure_numeric(X_test)

        # Scale features
        X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)

        # Detect problem type
        problem_type = self._detect_problem_type(y_train)
        logger.info(f"  Problem Type: {problem_type.upper()}")

        # Select models to try
        model_registry = (
            self.CLASSIFICATION_MODELS
            if problem_type == 'classification'
            else self.REGRESSION_MODELS
        )

        # Limit models if configured
        if self.config.max_models > 0:
            model_registry = model_registry[:self.config.max_models]

        # Train and evaluate each model
        logger.info(f"  Training {len(model_registry)} models...")
        print(f"\n  🚀 Training {len(model_registry)} models...\n")

        self.trained_models = []

        for i, (name, model_class, param_grid) in enumerate(model_registry):
            print(f"  [{i+1}/{len(model_registry)}] Training {name}...", end=" ")

            result = self._train_model(
                name, model_class, param_grid,
                X_train_scaled, X_test_scaled, y_train, y_test,
                problem_type
            )

            self.trained_models.append(result)
            print(f"✅ Score: {result.test_score:.4f}")

        # Sort by test score
        self.trained_models.sort(key=lambda x: x.test_score, reverse=True)

        # Build results
        elapsed = time.time() - start_time

        best = self.trained_models[0] if self.trained_models else None

        self.results = {
            'models': [self._model_result_to_dict(m) for m in self.trained_models],
            'best_model': self._model_result_to_dict(best) if best else None,
            'problem_type': problem_type,
            'training_time': elapsed,
            'training_time_formatted': format_duration(elapsed),
            'n_models_trained': len(self.trained_models)
        }

        # Print summary
        self._print_summary()

        logger.info("=" * 60)
        logger.info(f"  ✅ AUTO ML COMPLETE! ({format_duration(elapsed)})")
        logger.info("=" * 60)

        return self.results

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric."""
        df = df.copy()

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any remaining NaN
        df = df.fillna(0)

        return df

    def _scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features for model training."""
        self._scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            self._scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect classification vs regression."""
        if self.task_type != "auto":
            return self.task_type

        return detect_task_type(y)

    def _train_model(
        self,
        name: str,
        model_class,
        param_grid: Dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str
    ) -> ModelResult:
        """Train and evaluate a single model."""
        t0 = time.time()

        # Get best hyperparameters (simple grid search)
        best_score = 0
        best_params = {}

        if param_grid:
            # Try a few combinations
            from sklearn.model_selection import GridSearchCV

            try:
                grid = GridSearchCV(
                    model_class(),
                    param_grid,
                    cv=min(3, self.config.cv_folds),
                    scoring='accuracy' if problem_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
                train_score = grid.best_score_
            except:
                model = model_class()
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
        else:
            model = model_class()
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            best_params = model.get_params()

        # Calculate metrics
        y_pred = model.predict(X_test)
        test_score = model.score(X_test, y_test)

        # Cross-validation score
        try:
            cv = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            cv_score = cv.mean()
            cv_std = cv.std()
        except:
            cv_score = test_score
            cv_std = 0

        # Calculate additional metrics
        metrics = self._calculate_metrics(y_test, y_pred, problem_type)

        training_time = time.time() - t0

        return ModelResult(
            name=name,
            model=model,
            train_score=train_score,
            test_score=test_score,
            cv_score=cv_score,
            cv_std=cv_std,
            training_time=training_time,
            metrics=metrics,
            hyperparameters=best_params
        )

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        problem_type: str
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}

        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # ROC AUC for binary
            if len(np.unique(y_true)) == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    if hasattr(y_pred, 'predict_proba'):
                        y_prob = y_pred.predict_proba(y_true)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                except:
                    pass

        else:  # regression
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)

        return metrics

    def _model_result_to_dict(self, result: ModelResult) -> Dict[str, Any]:
        """Convert ModelResult to dictionary."""
        return {
            'name': result.name,
            'train_score': result.train_score,
            'score': result.test_score,
            'cv_score': result.cv_score,
            'cv_std': result.cv_std,
            'training_time': result.training_time,
            'metrics': result.metrics,
            'hyperparameters': result.hyperparameters
        }

    def _print_summary(self) -> None:
        """Print a summary of the training results."""
        print("\n" + "=" * 70)
        print("  🏆 MODEL RANKING")
        print("=" * 70)
        print(f"  {'Rank':<5} {'Model':<25} {'Test Score':<12} {'CV Score':<12} {'Time':<8}")
        print("-" * 70)

        for i, model in enumerate(self.trained_models[:10]):
            print(
                f"  {i+1:<5} {model.name:<25} "
                f"{model.test_score:<12.4f} {model.cv_score:<12.4f} "
                f"{format_duration(model.training_time):<8}"
            )

        print("=" * 70)

        if self.trained_models:
            best = self.trained_models[0]
            print(f"\n  🌟 BEST MODEL: {best.name}")
            print(f"     Test Score: {best.test_score:.4f}")
            print(f"     CV Score: {best.cv_score:.4f} (±{best.cv_std:.4f})")
            print(f"\n  📊 Metrics:")
            for metric, value in best.metrics.items():
                print(f"     - {metric}: {value:.4f}")

    def save_models(
        self,
        models: List[Dict],
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save trained models to disk.

        Parameters
        ----------
        models : List[Dict]
            Models to save (from results['models']).
        output_dir : str
            Directory to save models.

        Returns
        -------
        Dict[str, str]
            Mapping of model name to file path.
        """
        import joblib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        for model_result in self.trained_models:
            safe_name = model_result.name.replace(' ', '_').lower()
            filepath = output_dir / f"{safe_name}.joblib"

            joblib.dump(model_result.model, filepath)
            paths[model_result.name] = str(filepath)

        # Save scaler
        if self._scaler:
            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(self._scaler, scaler_path)
            paths['scaler'] = str(scaler_path)

        # Save results summary
        summary_path = output_dir / "results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        paths['summary'] = str(summary_path)

        logger.info(f"Saved {len(paths)} models to {output_dir}")

        return paths

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.
        model_name : str, optional
            Name of model to use. Uses best model if None.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if model_name:
            model = next(
                (m.model for m in self.trained_models if m.name == model_name),
                None
            )
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
        else:
            if not self.trained_models:
                raise ValueError("No models trained yet")
            model = self.trained_models[0].model

        # Scale features
        X = self._ensure_numeric(X)
        if self._scaler:
            X = pd.DataFrame(
                self._scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return model.predict(X)
