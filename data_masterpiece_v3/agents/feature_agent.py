"""
data_masterpiece_v3.agents.feature_agent
──────────────────────────────────────────
Feature Engineering Agent — creates powerful new features from existing ones.
Think of this as "discovering hidden patterns" in your data!

Supported feature types:
  ratio     : col_a / col_b
  diff      : col_a - col_b
  product   : col_a * col_b
  agg_mean  : mean of multiple columns
  agg_sum   : sum of multiple columns
  agg_max   : max of multiple columns
  agg_min   : min of multiple columns
  log1p     : log(1 + col)  → good for skewed data
  square    : col²
  sqrt      : √col
  zscore    : (col - mean) / std
  polynomial: col² and col³ together
  interaction: col_a * col_b  (same as product)
  bin       : cut col into N equal-width bins
  rank      : percentile rank of values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..utils.logger import get_logger

log = get_logger("FeatureAgent")


class FeatureAgent:
    """Creates new derived features based on config."""

    def __init__(self, cfg: dict):
        self.features_cfg: dict = cfg.get("features", {})
        self.mode: str          = cfg.get("mode", "auto")
        g                       = cfg.get("global", {})
        self.log_transform_skewed = g.get("log_transform_skewed", False)
        self.skew_threshold     = 2.0
        self.summary: dict      = {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("⚙️  FeatureAgent started")
        df = df.copy()
        transforms = []

        # Apply derived features from config
        derived = self.features_cfg.get("derived", [])
        for feat_def in derived:
            df, name = self._apply_feature(df, feat_def)
            if name:
                transforms.append(name)

        # Auto log-transform skewed columns
        if self.log_transform_skewed:
            df, log_cols = self._auto_log_transform(df)
            transforms.extend(log_cols)

        self.summary = {"feature_transforms": transforms}
        log.info(f"  Created {len(transforms)} features")
        return df

    def _apply_feature(self, df, feat_def):
        feat_type = feat_def.get("type", "")
        name = feat_def.get("name", "")

        try:
            if feat_type == "ratio":
                a, b = feat_def["col_a"], feat_def["col_b"]
                if a in df.columns and b in df.columns:
                    n = name or f"{a}_div_{b}"
                    df[n] = df[a] / (df[b].replace(0, np.nan))
                    log.info(f"  ratio: {n} = {a}/{b}")
                    return df, n

            elif feat_type == "diff":
                a, b = feat_def["col_a"], feat_def["col_b"]
                if a in df.columns and b in df.columns:
                    n = name or f"{a}_minus_{b}"
                    df[n] = df[a] - df[b]
                    log.info(f"  diff: {n}")
                    return df, n

            elif feat_type in ("product", "interaction"):
                a, b = feat_def["col_a"], feat_def["col_b"]
                if a in df.columns and b in df.columns:
                    n = name or f"{a}_x_{b}"
                    df[n] = df[a] * df[b]
                    log.info(f"  product: {n}")
                    return df, n

            elif feat_type == "agg_mean":
                cols = [c for c in feat_def.get("cols", []) if c in df.columns]
                if cols:
                    n = name or "agg_mean"
                    df[n] = df[cols].mean(axis=1)
                    log.info(f"  agg_mean: {n}")
                    return df, n

            elif feat_type == "agg_sum":
                cols = [c for c in feat_def.get("cols", []) if c in df.columns]
                if cols:
                    n = name or "agg_sum"
                    df[n] = df[cols].sum(axis=1)
                    log.info(f"  agg_sum: {n}")
                    return df, n

            elif feat_type == "agg_max":
                cols = [c for c in feat_def.get("cols", []) if c in df.columns]
                if cols:
                    n = name or "agg_max"
                    df[n] = df[cols].max(axis=1)
                    return df, n

            elif feat_type == "agg_min":
                cols = [c for c in feat_def.get("cols", []) if c in df.columns]
                if cols:
                    n = name or "agg_min"
                    df[n] = df[cols].min(axis=1)
                    return df, n

            elif feat_type == "log1p":
                col = feat_def.get("col", "")
                if col in df.columns:
                    n = name or f"{col}_log1p"
                    df[n] = np.log1p(df[col].clip(lower=0))
                    log.info(f"  log1p: {n}")
                    return df, n

            elif feat_type == "square":
                col = feat_def.get("col", "")
                if col in df.columns:
                    n = name or f"{col}_sq"
                    df[n] = df[col] ** 2
                    log.info(f"  square: {n}")
                    return df, n

            elif feat_type == "sqrt":
                col = feat_def.get("col", "")
                if col in df.columns:
                    n = name or f"{col}_sqrt"
                    df[n] = np.sqrt(df[col].clip(lower=0))
                    return df, n

            elif feat_type == "zscore":
                col = feat_def.get("col", "")
                if col in df.columns:
                    n = name or f"{col}_zscore"
                    df[n] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
                    return df, n

            elif feat_type == "bin":
                col = feat_def.get("col", "")
                n_bins = feat_def.get("n_bins", 5)
                if col in df.columns:
                    n = name or f"{col}_bin"
                    df[n] = pd.cut(df[col], bins=n_bins, labels=False)
                    return df, n

            elif feat_type == "rank":
                col = feat_def.get("col", "")
                if col in df.columns:
                    n = name or f"{col}_rank"
                    df[n] = df[col].rank(pct=True)
                    return df, n

            elif feat_type == "polynomial":
                col = feat_def.get("col", "")
                if col in df.columns:
                    df[f"{col}_sq"]  = df[col] ** 2
                    df[f"{col}_cub"] = df[col] ** 3
                    log.info(f"  polynomial: {col}_sq, {col}_cub")
                    return df, f"{col}_poly"

        except Exception as e:
            log.warning(f"  Feature creation failed ({feat_type}): {e}")

        return df, ""

    def _auto_log_transform(self, df):
        log_cols = []
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            skew = abs(df[col].skew())
            if skew > self.skew_threshold and df[col].min() >= 0:
                df[f"{col}_log1p"] = np.log1p(df[col])
                log_cols.append(f"{col}_log1p")
                log.info(f"  Auto log1p: {col} (skew={skew:.2f})")
        return df, log_cols
