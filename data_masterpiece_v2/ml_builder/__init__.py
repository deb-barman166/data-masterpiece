"""
data_masterpiece_v2.ml_builder - Auto ML/DL Model Builder

Automatically builds and evaluates ML/DL models!

Features:
    - Auto ML for classification and regression
    - Multiple model families
    - Hyperparameter tuning
    - Model comparison
    - Easy-to-use interface

Quick Start:
    >>> from data_masterpiece_v2.ml_builder import AutoMLBuilder
    >>> builder = AutoMLBuilder()
    >>> results = builder.build(df, target="survived")
    >>> print(results['best_model'])
"""

from data_masterpiece_v2.ml_builder.auto_builder import AutoMLBuilder

__all__ = ["AutoMLBuilder"]
