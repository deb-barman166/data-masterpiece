"""
data_masterpiece  --  The Masterpiece Python Module
====================================================

A production-grade, end-to-end data science pipeline that combines:

  PREPROCESSING ENGINE (6 agents):
    - Cleaning (duplicates, whitespace, constant/null drops)
    - Type Conversion (numeric strings, booleans, datetime extraction)
    - Missing Value Imputation (median/mean/mode/ffill/bfill/constant)
    - Encoding (one-hot / label / frequency / multi-hot / bool-to-int)
    - Feature Engineering (scaling, log transform, derived features)
    - Validation (zero NaN/Inf guarantee, shape sanity)

  INTELLIGENCE ENGINE (9 modules):
    - Statistical Profiling (deep per-column stats, distribution labels)
    - Outlier Detection (IQR/Z-score with clip/drop/flag/impute)
    - Feature Selection (variance/correlation/top-K cascade)
    - Visualization (histograms, box plots, scatter, heatmap, pair plots)
    - Relationship Analysis (correlation matrix, multicollinearity)
    - Model Recommendation (rule-based ML model suggestions)
    - Data Splitting (train/val/test with stratification)
    - HTML Report Generation (self-contained, publication-ready)
    - Master Intelligence Controller (8-step orchestrator)

Quick Start
-----------
    from data_masterpiece import MasterPipeline

    pipeline = MasterPipeline()
    results = pipeline.run("data.csv", target="label")

    # Access results
    print(results["df_clean"].head())
    print(results["split"].X_train.shape)
    print(results["report_path"])

    # Or use individually:
    from data_masterpiece import PipelineController, DataIntelligenceController
    from data_masterpiece import load_data, generate_auto_config
    from data_masterpiece import StatisticalProfiler, OutlierDetectionEngine
    from data_masterpiece import FeatureSelectionEngine, VisualizationEngine
    from data_masterpiece import RelationshipAnalyzer, ModelRecommender
    from data_masterpiece import DataSplitter, ReportGenerator, Config
"""

__version__ = "1.0.0"
__author__ = "Data Masterpiece"

# -- master entry point --
from data_masterpiece.master import MasterPipeline

# -- config --
from data_masterpiece.config import Config, build_auto_config, build_manual_config

# -- preprocessing --
from data_masterpiece.preprocessing import PipelineController, load_data, generate_auto_config as _gac

# -- intelligence --
from data_masterpiece.intelligence import (
    DataIntelligenceController,
    VisualizationEngine,
    RelationshipAnalyzer,
    RelationshipReport,
    ModelRecommender,
    RecommendationReport,
    DataSplitter,
    SplitResult,
    OutlierDetectionEngine,
    OutlierReport,
    FeatureSelectionEngine,
    SelectionReport,
    StatisticalProfiler,
    ColumnProfile,
    ReportGenerator,
)

# -- utils --
from data_masterpiece.utils import get_logger, get_file_logger

__all__ = [
    # Master
    "MasterPipeline",
    # Config
    "Config", "build_auto_config", "build_manual_config",
    # Preprocessing
    "PipelineController", "load_data", "generate_auto_config",
    # Intelligence
    "DataIntelligenceController",
    "VisualizationEngine",
    "RelationshipAnalyzer", "RelationshipReport",
    "ModelRecommender", "RecommendationReport",
    "DataSplitter", "SplitResult",
    "OutlierDetectionEngine", "OutlierReport",
    "FeatureSelectionEngine", "SelectionReport",
    "StatisticalProfiler", "ColumnProfile",
    "ReportGenerator",
    # Utils
    "get_logger", "get_file_logger",
]
