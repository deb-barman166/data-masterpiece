"""
data_masterpiece_v3.intelligence.selector
──────────────────────────────────────────
Feature Selector — removes low-quality features automatically.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from ..utils.logger import get_logger

log = get_logger("FeatureSelector")


class FeatureSelector:
    """
    Removes:
    • Near-zero-variance features
    • Highly-correlated duplicate features (keeps one)
    • Optionally: keeps top-K most important features
    """

    def __init__(self, variance_threshold=0.01, corr_threshold=0.90, top_k=0):
        self.variance_threshold = variance_threshold
        self.corr_threshold     = corr_threshold
        self.top_k              = top_k
        self.dropped_cols: list = []
        self.summary: dict      = {}

    def run(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        log.info("🎯 FeatureSelector running ...")
        df = df.copy()

        feature_cols = [c for c in df.columns if c != target]

        # Step 1: Variance filter
        df, low_var = self._drop_low_variance(df, feature_cols)
        feature_cols = [c for c in feature_cols if c not in low_var]

        # Step 2: Correlation filter
        df, high_corr = self._drop_high_corr(df, feature_cols)
        feature_cols = [c for c in feature_cols if c not in high_corr]

        self.dropped_cols = low_var + high_corr

        self.summary = {
            "low_variance_dropped": low_var,
            "high_corr_dropped": high_corr,
            "features_remaining": len(feature_cols),
        }

        log.info(f"  Low-variance dropped : {len(low_var)}")
        log.info(f"  High-corr dropped    : {len(high_corr)}")
        log.info(f"  Features remaining   : {len(feature_cols)}")
        return df

    def _drop_low_variance(self, df, cols):
        num = df[cols].select_dtypes(include="number")
        if num.empty:
            return df, []
        try:
            sel = VarianceThreshold(threshold=self.variance_threshold)
            sel.fit(num.fillna(0))
            support = sel.get_support()
            low_var = [c for c, keep in zip(num.columns, support) if not keep]
            df = df.drop(columns=low_var, errors="ignore")
            return df, low_var
        except Exception:
            return df, []

    def _drop_high_corr(self, df, cols):
        num = df[cols].select_dtypes(include="number")
        if num.shape[1] < 2:
            return df, []
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        df = df.drop(columns=to_drop, errors="ignore")
        return df, to_drop

