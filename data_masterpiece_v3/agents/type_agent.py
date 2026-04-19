"""
data_masterpiece_v3.agents.type_agent
──────────────────────────────────────
Type Conversion Agent — makes sure every column has the right data type.
Automatically detects and converts dates, booleans, numbers, and text.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from ..utils.logger import get_logger

log = get_logger("TypeAgent")


class TypeAgent:
    """
    Smart type conversion:
    AUTO: Detects boolean-like, date-like, numeric-like columns automatically.
    MANUAL: User specifies exact types in "type_conversion" section of JSON.
    """

    def __init__(self, cfg: dict):
        self.per_col: dict           = cfg.get("type_conversion", {})
        self.datetime_include_time   = cfg.get("global", {}).get("datetime_include_time", False)
        self.summary: dict           = {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("🔄 TypeAgent started")
        df = df.copy()
        conversions = {}

        # Apply manual conversions first
        for col, dtype in self.per_col.items():
            if col not in df.columns:
                continue
            df, ok = self._convert(df, col, dtype)
            if ok:
                conversions[col] = dtype

        # Auto-detect remaining object columns
        obj_cols = df.select_dtypes(include="object").columns
        for col in obj_cols:
            if col in conversions:
                continue
            df, detected = self._auto_detect(df, col)
            if detected:
                conversions[col] = detected

        self.summary = {"type_conversions": conversions}
        log.info(f"  Converted {len(conversions)} columns")
        return df

    def _convert(self, df, col, dtype):
        try:
            if dtype in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype in ("float", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif dtype == "bool":
                df[col] = df[col].map({"true": True, "false": False, "1": True, "0": False,
                                        "yes": True, "no": False, True: True, False: False})
            elif dtype in ("datetime", "date"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if not self.datetime_include_time:
                    df[col] = df[col].dt.date
            elif dtype == "str":
                df[col] = df[col].astype(str)
            else:
                log.warning(f"  [{col}] Unknown type '{dtype}'")
                return df, False
            log.info(f"  [{col}] → {dtype}")
            return df, True
        except Exception as e:
            log.warning(f"  [{col}] Conversion failed: {e}")
            return df, False

    def _auto_detect(self, df, col):
        series = df[col].dropna()
        if len(series) == 0:
            return df, None

        # Boolean detection
        unique_lower = set(series.astype(str).str.lower().unique())
        bool_vals = {"true", "false", "yes", "no", "0", "1"}
        if unique_lower.issubset(bool_vals):
            df[col] = series.astype(str).str.lower().map(
                {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
            )
            log.info(f"  [{col}] auto → bool(int)")
            return df, "bool→int"

        # Numeric detection
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            null_rate = converted.isnull().mean()
            if null_rate < 0.1:
                df[col] = converted
                log.info(f"  [{col}] auto → numeric")
                return df, "numeric"
        except Exception:
            pass

        # Datetime detection
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            null_rate = converted.isnull().mean()
            if null_rate < 0.1 and converted.dtype != "object":
                # Convert datetime to numeric features
                df[col + "_year"]  = converted.dt.year
                df[col + "_month"] = converted.dt.month
                df[col + "_day"]   = converted.dt.day
                if self.datetime_include_time:
                    df[col + "_hour"] = converted.dt.hour
                df = df.drop(columns=[col])
                log.info(f"  [{col}] auto → datetime (expanded to year/month/day)")
                return df, "datetime→expanded"
        except Exception:
            pass

        return df, None
