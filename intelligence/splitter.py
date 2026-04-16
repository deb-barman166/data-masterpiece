"""
data_masterpiece.intelligence.splitter  --  DataSplitter

Split a DataFrame into train / validation / test sets with optional
stratification for classification targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger


@dataclass
class SplitResult:
    """Container for train/val/test splits."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    split_info: dict = field(default_factory=dict)

    @property
    def has_val(self) -> bool:
        return self.X_val is not None


class DataSplitter:
    """
    Split a DataFrame into train / (validation) / test sets.

    Parameters
    ----------
    random_state : Seed for reproducibility.
    shuffle      : Whether to shuffle before splitting.
    """

    def __init__(self, random_state: int = 42, shuffle: bool = True):
        self.random_state = random_state
        self.shuffle = shuffle
        self.log = get_logger("DataSplitter")

    def split(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.20,
        val_size: float = 0.0,
        stratify: bool = True,
    ) -> SplitResult:
        """
        Split the DataFrame.

        Parameters
        ----------
        df        : Input DataFrame (must contain the target column).
        target    : Name of the target column.
        test_size : Fraction for test set (default 0.20).
        val_size  : Fraction for validation set (default 0.0 = no val).
        stratify  : Use stratified split for classification (default True).

        Returns
        -------
        SplitResult
        """
        self.log.info(
            f"Splitting: test={test_size}, val={val_size}, stratify={stratify}"
        )

        X = df.drop(columns=[target])
        y = df[target]

        strat_col = y if stratify else None
        # stratification only works well for classification
        if strat_col is not None and y.dtype in (float, "float64"):
            unique_floats = y.nunique()
            if unique_floats > 20:
                strat_col = None
                self.log.info(
                    "Target appears continuous -- disabling stratification."
                )

        # -- compute sizes --
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size) if val_size > 0 else 0
        n_train = n - n_test - n_val

        if n_train <= 0:
            raise ValueError(
                f"Dataset too small for requested split: "
                f"n={n}, test={test_size}, val={val_size}"
            )

        # -- perform split --
        if n_val > 0:
            # train+val  |  test
            X_tv, X_test, y_tv, y_test = self._sklearn_split(
                X, y, test_size=n_test / n, stratify=strat_col,
            )
            strat_tv = y_tv if strat_col is not None else None
            # train  |  val
            X_train, X_val, y_train, y_val = self._sklearn_split(
                X_tv, y_tv, test_size=n_val / (n_train + n_val),
                stratify=strat_tv,
            )
        else:
            X_train, X_test, y_train, y_test = self._sklearn_split(
                X, y, test_size=test_size, stratify=strat_col,
            )
            X_val, y_val = None, None

        split_info = {
            "strategy": f"train={1 - test_size - val_size:.2f} / val={val_size:.2f} / test={test_size:.2f}",
            "train_rows": len(X_train),
            "val_rows": len(X_val) if X_val is not None else 0,
            "test_rows": len(X_test),
            "total_rows": n,
            "stratified": strat_col is not None,
            "random_state": self.random_state,
            "n_features": X_train.shape[1],
        }

        result = SplitResult(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val,
            split_info=split_info,
        )

        self.log.info(
            f"Split complete: train={split_info['train_rows']} / "
            f"val={split_info['val_rows']} / test={split_info['test_rows']}"
        )
        return result

    def _sklearn_split(
        self, X, y, test_size, stratify=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split using sklearn-like logic with numpy for zero external deps."""
        n = len(X)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        n_test = int(n * test_size)

        if stratify is not None:
            # stratified split
            test_idx = self._stratified_indices(y.iloc[indices], n_test)
            test_idx_sorted = sorted(test_idx)
        else:
            test_idx_sorted = list(range(n - n_test, n))

        train_idx_sorted = sorted(set(range(n)) - set(test_idx_sorted))

        X_train = X.iloc[train_idx_sorted].reset_index(drop=True)
        X_test = X.iloc[test_idx_sorted].reset_index(drop=True)
        y_train = y.iloc[train_idx_sorted].reset_index(drop=True)
        y_test = y.iloc[test_idx_sorted].reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _stratified_indices(y: pd.Series, n_test: int) -> list:
        """Select approximately n_test indices preserving class proportions."""
        class_counts = y.value_counts()
        test_indices: list = []

        for cls, count in class_counts.items():
            cls_indices = np.where(y.values == cls)[0].tolist()
            n_cls_test = max(1, round(len(cls_indices) * n_test / len(y)))
            n_cls_test = min(n_cls_test, len(cls_indices))
            test_indices.extend(cls_indices[:n_cls_test])

        return test_indices[:n_test]
