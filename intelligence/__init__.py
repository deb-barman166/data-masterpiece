"""
data_masterpiece.intelligence  --  Data intelligence & analysis engine.

A 9-module analytical engine that takes a clean DataFrame + target column
and produces statistical profiles, outlier reports, feature selection,
correlation analysis, model recommendations, data splits, visualizations,
and a full HTML report.

Modules:
    StatisticalProfiler     - Deep per-column statistical profiling
    OutlierDetectionEngine   - IQR / Z-score outlier detection & treatment
    FeatureSelectionEngine   - Variance / correlation / top-K feature filtering
    VisualizationEngine      - Auto & manual chart generation
    RelationshipAnalyzer     - Correlation & multicollinearity analysis
    ModelRecommender         - Rule-based ML model recommendations
    DataSplitter             - Train / val / test splitting
    ReportGenerator          - Self-contained HTML report assembly
    DataIntelligenceController - Master 8-step orchestrator
"""

from data_masterpiece.intelligence.controller import DataIntelligenceController
from data_masterpiece.intelligence.visualization import VisualizationEngine
from data_masterpiece.intelligence.relationship import RelationshipAnalyzer, RelationshipReport
from data_masterpiece.intelligence.recommender import ModelRecommender, RecommendationReport
from data_masterpiece.intelligence.splitter import DataSplitter, SplitResult
from data_masterpiece.intelligence.outliers import OutlierDetectionEngine, OutlierReport
from data_masterpiece.intelligence.feature_selection import FeatureSelectionEngine, SelectionReport
from data_masterpiece.intelligence.profiler import StatisticalProfiler, ColumnProfile
from data_masterpiece.intelligence.reporter import ReportGenerator

__all__ = [
    "DataIntelligenceController",
    "VisualizationEngine",
    "RelationshipAnalyzer",
    "RelationshipReport",
    "ModelRecommender",
    "RecommendationReport",
    "DataSplitter",
    "SplitResult",
    "OutlierDetectionEngine",
    "OutlierReport",
    "FeatureSelectionEngine",
    "SelectionReport",
    "StatisticalProfiler",
    "ColumnProfile",
    "ReportGenerator",
]
