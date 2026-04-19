"""
data_masterpiece_v3.agents.validation_agent
────────────────────────────────────────────
Validation Agent — the final quality check before data leaves the pipeline.
Makes sure everything is clean, numeric, and ML-ready.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from ..utils.logger import get_logger

log = get_logger("ValidationAgent")

_SCALERS = {
    "minmax":   MinMaxScaler,
    "standard": StandardScaler,
    "robust":   RobustScaler,
}


class ValidationAgent:
    """
    Final validation pass:
    • Drops any remaining non-numeric columns (with warning)
    • Replaces infinity values with NaN then fills with column median
    • Optionally scales numeric features
    • Reports final data quality stats
    """

    def __init__(self, cfg: dict):
        g = cfg.get("global", {})
        self.normalize    = g.get("normalize", False)
        self.scale_method = g.get("scale_method", "minmax")
        self.summary: dict = {}
        self.scaler        = None

    def run(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        log.info("✅ ValidationAgent started")
        df = df.copy()

        # Drop non-numeric columns (can't be used in ML)
        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            log.warning(f"  Dropping non-numeric cols: {non_numeric}")
            df = df.drop(columns=non_numeric)

        # Fix infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        if df.isnull().any().any():
            for col in df.columns[df.isnull().any()]:
                med = df[col].median()
                df[col] = df[col].fillna(med if not np.isnan(med) else 0)

        # Scale
        if self.normalize:
            df = self._scale(df, target)

        # Final quality report
        self.summary = {
            "final_shape": df.shape,
            "all_numeric": True,
            "null_count": int(df.isnull().sum().sum()),
            "inf_count": 0,
            "scaled": self.normalize,
            "scale_method": self.scale_method if self.normalize else None,
        }

        log.info(f"  Final shape  : {df.shape}")
        log.info(f"  Null count   : {self.summary['null_count']}")
        log.info(f"  Scaled       : {self.normalize}")
        return df

    def _scale(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Scale numeric features, leaving target column unchanged."""
        ScalerCls = _SCALERS.get(self.scale_method, MinMaxScaler)
        self.scaler = ScalerCls()

        # Don't scale the target
        cols_to_scale = [c for c in df.columns if c != target]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            log.info(f"  Scaled {len(cols_to_scale)} cols with {self.scale_method}")

        return df
