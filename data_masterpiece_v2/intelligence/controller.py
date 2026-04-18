"""
data_masterpiece_v2.intelligence.controller - Intelligence Controller

Orchestrates all intelligence and analysis operations.

Operations:
    1. Statistical Profiling
    2. Outlier Detection
    3. Feature Selection
    4. Relationship Analysis
    5. Model Recommendations
    6. Data Splitting

Usage:
    >>> from data_masterpiece_v2.intelligence import IntelligenceController
    >>> controller = IntelligenceController()
    >>> results = controller.run(df, target="survived")
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from data_masterpiece_v2.config import IntelligenceConfig
from data_masterpiece_v2.utils.logger import get_logger, format_duration
from data_masterpiece_v2.utils.helpers import detect_task_type

# Import intelligence modules
from data_masterpiece_v2.intelligence.profiler import StatisticalProfiler
from data_masterpiece_v2.intelligence.outlier import OutlierDetector
from data_masterpiece_v2.intelligence.feature_selection import FeatureSelector
from data_masterpiece_v2.intelligence.relationship import RelationshipAnalyzer
from data_masterpiece_v2.intelligence.recommender import ModelRecommender
from data_masterpiece_v2.intelligence.splitter import DataSplitter

logger = get_logger("IntelligenceController")


class IntelligenceController:
    """
    Master controller for all intelligence and analysis operations.

    This controller orchestrates a complete data analysis pipeline including
    statistical profiling, outlier detection, feature selection, relationship
    analysis, and model recommendations.

    Parameters
    ----------
    config : IntelligenceConfig
        Configuration for intelligence operations.
    output_dir : str
        Directory for saving outputs.
    show_plots : bool
        Whether to display plots.

    Examples
    --------
    Basic usage:

        >>> controller = IntelligenceController()
        >>> results = controller.run(df, target="survived")
        >>> print(results.keys())
        dict_keys(['profile', 'outliers', 'features', 'relationships', ...])

    With custom config:

        >>> config = IntelligenceConfig()
        >>> config.outlier_method = "iqr"
        >>> config.feature_selection_method = "correlation"
        >>> controller = IntelligenceController(config=config)
        >>> results = controller.run(df, target="price")
    """

    def __init__(
        self,
        config: Optional[IntelligenceConfig] = None,
        output_dir: str = "output/plots",
        show_plots: bool = False
    ):
        """Initialize the intelligence controller."""
        self.config = config or IntelligenceConfig()
        self.output_dir = output_dir
        self.show_plots = show_plots

        # Initialize components
        self._profiler = StatisticalProfiler()
        self._outlier = OutlierDetector(
            method=self.config.outlier_method,
            strategy=self.config.outlier_strategy
        )
        self._selector = FeatureSelector(
            method=self.config.feature_selection_method,
            threshold=self.config.variance_threshold
        )
        self._rel_analyzer = RelationshipAnalyzer()
        self._recommender = ModelRecommender()
        self._splitter = DataSplitter(random_state=self.config.random_state)

        self.results: Dict[str, Any] = {}

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        skip_outlier: bool = False,
        skip_selection: bool = False,
        skip_recommendations: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete intelligence analysis pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame.
        target : str
            Target column name.

        Returns
        -------
        Dict[str, Any]
            Complete analysis results including:
            - profile: Statistical profiles
            - outliers: Outlier detection results
            - features: Feature selection results
            - relationships: Correlation analysis
            - recommendations: Model recommendations
            - split: Train/test split
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("  🧠 INTELLIGENCE PIPELINE STARTING")
        logger.info("=" * 60)

        # Validate input
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        # Ensure numeric data
        df = self._ensure_numeric(df)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: STATISTICAL PROFILING
        # ═══════════════════════════════════════════════════════════════════════
        logger.info("📊 Step 1/6: Statistical Profiling...")
        profile_results = self._profiler.profile(df)
        self.results['profile'] = profile_results
        logger.info(f"  Profiled {len(profile_results['profiles'])} columns")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: OUTLIER DETECTION
        # ═══════════════════════════════════════════════════════════════════════
        if not skip_outlier:
            logger.info("🎯 Step 2/6: Outlier Detection...")
            df, outlier_results = self._outlier.detect(df)
            self.results['outliers'] = outlier_results
            logger.info(f"  Found {outlier_results['total_outliers']} outliers")
        else:
            logger.info("⏭️  Step 2/6: Outlier Detection SKIPPED")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: FEATURE SELECTION
        # ═══════════════════════════════════════════════════════════════════════
        if not skip_selection:
            logger.info("🔍 Step 3/6: Feature Selection...")
            df_selected, feature_results = self._selector.select(df, target)
            self.results['features'] = feature_results
            self.results['df_selected'] = df_selected
            logger.info(f"  Selected {len(feature_results['selected_features'])} features")
        else:
            logger.info("⏭️  Step 3/6: Feature Selection SKIPPED")
            self.results['df_selected'] = df

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: RELATIONSHIP ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        logger.info("🔗 Step 4/6: Relationship Analysis...")
        rel_results = self._rel_analyzer.analyze(self.results['df_selected'], target)
        self.results['relationships'] = rel_results
        self.results['n_strong_correlations'] = len(rel_results.get('strong_pairs', []))
        logger.info(f"  Found {self.results['n_strong_correlations']} strong correlations")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: MODEL RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════════════
        if not skip_recommendations:
            logger.info("🤖 Step 5/6: Model Recommendations...")
            rec_results = self._recommender.recommend(
                self.results['df_selected'],
                target,
                problem_info=rel_results
            )
            self.results['recommendations'] = rec_results
            self.results['n_recommended_models'] = len(rec_results.get('models', []))
            logger.info(f"  Recommended {self.results['n_recommended_models']} models")
        else:
            logger.info("⏭️  Step 5/6: Model Recommendations SKIPPED")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: DATA SPLITTING
        # ═══════════════════════════════════════════════════════════════════════
        logger.info("✂️  Step 6/6: Data Splitting...")
        split_results = self._splitter.split(
            self.results['df_selected'],
            target,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            stratify=self.config.stratify
        )
        self.results['split'] = split_results
        logger.info(f"  Split: {split_results['train_rows']} train / {split_results['test_rows']} test")

        # Calculate elapsed time
        elapsed = time.time() - start_time
        self.results['elapsed_time'] = elapsed

        logger.info("=" * 60)
        logger.info(f"  ✅ INTELLIGENCE PIPELINE COMPLETE ({format_duration(elapsed)})")
        logger.info("=" * 60)

        return self.results

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric for analysis."""
        df = df.copy()

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis."""
        summary = {
            'n_columns_analyzed': len(self.results.get('profile', {}).get('profiles', [])),
            'n_outliers_found': self.results.get('outliers', {}).get('total_outliers', 0),
            'n_features_selected': len(self.results.get('features', {}).get('selected_features', [])),
            'n_strong_correlations': self.results.get('n_strong_correlations', 0),
            'n_recommended_models': self.results.get('n_recommended_models', 0),
            'problem_type': self._detect_problem_type()
        }

        return summary

    def _detect_problem_type(self) -> str:
        """Detect the type of ML problem."""
        if 'recommendations' in self.results:
            return self.results['recommendations'].get('problem_type', 'unknown')
        return 'unknown'
