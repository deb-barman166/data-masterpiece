"""
data_masterpiece_v2.preprocessing - Data Preprocessing Pipeline

This module handles all data preprocessing tasks including:
- Data loading from various formats
- Missing value handling
- Outlier detection
- Feature encoding
- Feature scaling
- Data cleaning

Quick Start:
    >>> from data_masterpiece_v2.preprocessing import PreprocessingController
    >>> controller = PreprocessingController()
    >>> df_clean = controller.run(df, target="survived")
"""

from data_masterpiece_v2.preprocessing.controller import PreprocessingController
from data_masterpiece_v2.preprocessing.core.loader import DataLoader

__all__ = [
    "PreprocessingController",
    "DataLoader",
]
