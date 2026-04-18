"""
data_masterpiece_v2.intelligence.splitter - Data Splitter

Splits data into training, validation, and test sets.

Usage:
    >>> splitter = DataSplitter(random_state=42)
    >>> result = splitter.split(df, target="survived", test_size=0.2)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("DataSplitter")


@dataclass
class SplitResult:
    """Container for split results."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None


class DataSplitter:
    """
    Split data into training, validation, and test sets.

    Features:
    - Stratified splitting for classification
    - Configurable split ratios
    - Random or ordered splitting
    - Preserves data types

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.
    test_size : float
        Proportion for test set (0-1).
    val_size : float
        Proportion for validation set (0-1).

    Examples
    --------
    Basic usage:

        >>> splitter = DataSplitter(random_state=42)
        >>> result = splitter.split(df, target='survived')
        >>> X_train = result.X_train
        >>> y_test = result.y_test

    With validation set:

        >>> splitter = DataSplitter(random_state=42, test_size=0.2, val_size=0.1)
        >>> result = splitter.split(df, target='survived')
        >>> X_val = result.X_val
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.0
    ):
        """Initialize the data splitter."""
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.result: Optional[Dict] = None

    def split(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        stratify: bool = True
    ) -> Dict[str, Any]:
        """
        Split DataFrame into train/test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Target column name.
        test_size : float, optional
            Override default test size.
        val_size : float, optional
            Override default validation size.
        stratify : bool
            Use stratified splitting for classification.

        Returns
        -------
        Dict[str, Any]
            Split results with DataFrames and metadata.
        """
        if test_size is not None:
            self.test_size = test_size
        if val_size is not None:
            self.val_size = val_size

        logger.info(f"Splitting data with test_size={self.test_size}, val_size={self.val_size}")

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Determine if classification
        is_classification = y.dtype in ['object', 'int'] or y.nunique() < 20
        stratify_param = y if (stratify and is_classification) else None

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        # Split into train and validation
        X_train_final = X_train
        y_train_final = y_train
        X_val = None
        y_val = None

        if self.val_size > 0:
            val_proportion = self.val_size / (1 - self.test_size)
            stratify_train = y_train if (stratify and is_classification) else None

            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train,
                test_size=val_proportion,
                random_state=self.random_state,
                stratify=stratify_train
            )

        # Build result
        self.result = {
            'X_train': X_train_final,
            'X_test': X_test,
            'y_train': y_train_final,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'train_rows': len(X_train_final),
            'test_rows': len(X_test),
            'val_rows': len(X_val) if X_val is not None else 0,
            'n_features': X_train_final.shape[1],
            'stratified': stratify_param is not None,
            'random_state': self.random_state
        }

        logger.info(f"Train: {self.result['train_rows']}, Test: {self.result['test_rows']}, Val: {self.result['val_rows']}")

        return self.result

    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the split."""
        if self.result is None:
            return {}

        return {
            'train_rows': self.result['train_rows'],
            'test_rows': self.result['test_rows'],
            'val_rows': self.result['val_rows'],
            'train_pct': self.result['train_rows'] / (
                self.result['train_rows'] + self.result['test_rows'] + self.result['val_rows']
            ) * 100,
            'test_pct': self.result['test_rows'] / (
                self.result['train_rows'] + self.result['test_rows'] + self.result['val_rows']
            ) * 100,
            'val_pct': self.result['val_rows'] / (
                self.result['train_rows'] + self.result['test_rows'] + self.result['val_rows']
            ) * 100 if self.result['val_rows'] > 0 else 0
        }
