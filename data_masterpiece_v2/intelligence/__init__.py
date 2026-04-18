"""
data_masterpiece_v2.intelligence - Intelligence and Analysis Module

This module provides intelligent data analysis capabilities:
- Statistical profiling
- Outlier detection
- Feature selection
- Relationship analysis
- Model recommendations
- Data splitting

Quick Start:
    >>> from data_masterpiece_v2.intelligence import IntelligenceController
    >>> controller = IntelligenceController()
    >>> results = controller.run(df, target="survived")
"""

from data_masterpiece_v2.intelligence.controller import IntelligenceController
from data_masterpiece_v2.intelligence.profiler import StatisticalProfiler
from data_masterpiece_v2.intelligence.relationship import RelationshipAnalyzer
from data_masterpiece_v2.intelligence.splitter import DataSplitter

__all__ = [
    "IntelligenceController",
    "StatisticalProfiler",
    "RelationshipAnalyzer",
    "DataSplitter",
]
