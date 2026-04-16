"""
data_masterpiece.preprocessing.agents.validation  --  ValidationAgent

Final quality gate -- ensures output is ML-ready.
Checks: zero NaN, zero +/-Inf, no non-numeric columns, shape sanity,
duplicate column names, memory usage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from data_masterpiece.preprocessing.agents.base import BaseAgent
from data_masterpiece.utils.helpers import memory_usage_mb


class ValidationAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()
        errors: list = []
        warns: list = []

        # -- 1. no non-numeric columns --
        bad_dtypes = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if bad_dtypes:
            errors.append(f"Non-numeric columns remain: {bad_dtypes}")

        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            self.log.warning(f"Bool columns found -- converting to int: {bool_cols}")
            for c in bool_cols:
                df[c] = df[c].astype(int)

        # -- 2. NaN check --
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        if nan_cols:
            errors.append(f"NaN values remain: {nan_cols}")
            self.log.warning("Auto-fixing remaining NaNs -> 0")
            df.fillna(0, inplace=True)

        # -- 3. Inf check --
        num_df = df.select_dtypes(include=[np.number]).astype(float)
        inf_mask = np.isinf(num_df).any()
        inf_cols = inf_mask[inf_mask].index.tolist()
        if inf_cols:
            warns.append(f"Inf values in: {inf_cols} -- replacing with 0")
            df[inf_cols] = (
                df[inf_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            )

        # -- 4. duplicate column names --
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            warns.append(f"Duplicate column names: {dup_cols}")
            seen: set = set()
            new_cols: list = []
            for c in df.columns:
                if c in seen:
                    new_cols.append(f"{c}_dup")
                else:
                    new_cols.append(c)
                seen.add(c)
            df.columns = new_cols

        # -- 5. all-zero columns --
        zero_cols = [c for c in df.columns if (df[c] == 0).all()]
        if zero_cols:
            warns.append(f"All-zero columns (may be noise): {zero_cols}")

        # -- 6. shape sanity --
        if df.shape[0] == 0:
            errors.append("DataFrame has 0 rows after processing!")
        if df.shape[1] == 0:
            errors.append("DataFrame has 0 columns after processing!")

        # -- 7. log summary --
        mem = memory_usage_mb(df)
        self.log.info(f"Memory usage: {mem:.2f} MB")
        self.log.info(f"Final shape: {df.shape}")
        self.log.info(f"Dtypes: {df.dtypes.value_counts().to_dict()}")

        for w in warns:
            self.log.warning(w)

        if errors:
            for e in errors:
                self.log.error(f"VALIDATION FAILED: {e}")
        else:
            self.log.info("Validation PASSED -- dataset is fully ML-ready.")

        self.report.update({
            "validation_errors": errors,
            "validation_warnings": warns,
            "final_shape": df.shape,
            "memory_mb": round(mem, 3),
            "dtypes_summary": df.dtypes.value_counts().to_dict(),
        })
        return df
