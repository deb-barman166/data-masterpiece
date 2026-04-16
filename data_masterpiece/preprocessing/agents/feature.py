"""
data_masterpiece.preprocessing.agents.feature  --  FeatureTransformationAgent

Handles:
  - Normalization (min-max) / Standardization (z-score) -- optional
  - Log transform for skewed numeric columns -- optional
  - User-defined derived features (ratio / difference / aggregation / etc.)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from data_masterpiece.preprocessing.agents.base import BaseAgent


class FeatureTransformationAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()
        gcfg = config.get("global", {})
        fcfg = config.get("features", {})
        applied: list = []

        # -- 1. log transform skewed columns --
        if gcfg.get("log_transform_skewed", False):
            df = self._log_transform(df, applied)

        # -- 2. normalise / standardise --
        scale_mode = gcfg.get("normalize", False)
        if scale_mode:
            method = gcfg.get("scale_method", "minmax")
            df = self._scale(df, method, applied)

        # -- 3. user-defined derived features --
        for rule in fcfg.get("derived", []):
            df = self._apply_rule(df, rule, applied)

        self.report["feature_transforms"] = applied
        self.log.info(f"Feature transforms applied: {len(applied)}")
        return df

    # -- helpers ----------------------------------------------------------------

    def _log_transform(
        self, df: pd.DataFrame, log_list: list,
    ) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = df[col].dropna()
            if (s > 0).all():
                try:
                    skew_val = float(s.skew())
                except Exception:
                    continue
                if abs(skew_val) > 1.0:
                    df[col] = np.log1p(df[col])
                    log_list.append(f"log1p({col})")
                    self.log.info(
                        f"log1p applied to [{col}] (skew={skew_val:.2f})"
                    )
        return df

    def _scale(
        self, df: pd.DataFrame, method: str, log_list: list,
    ) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = df[col]
            if method == "minmax":
                mn, mx = s.min(), s.max()
                rng = mx - mn
                if rng != 0:
                    df[col] = (s - mn) / rng
                    log_list.append(f"minmax({col})")
            elif method in ("zscore", "standard"):
                mu, sigma = s.mean(), s.std()
                if sigma and sigma != 0:
                    df[col] = (s - mu) / sigma
                    log_list.append(f"zscore({col})")
        self.log.info(
            f"Scaling ({method}) applied to {len(num_cols)} numeric columns."
        )
        return df

    def _apply_rule(
        self, df: pd.DataFrame, rule: dict, log_list: list,
    ) -> pd.DataFrame:
        t = rule.get("type", "")
        try:
            if t == "ratio":
                a = rule["col_a"]
                b = rule["col_b"]
                name = rule.get("name", f"{a}_div_{b}")
                df[name] = df[a] / df[b].replace(0, np.nan).fillna(1)
                df[name] = df[name].fillna(0)
            elif t == "diff":
                a = rule["col_a"]
                b = rule["col_b"]
                name = rule.get("name", f"{a}_minus_{b}")
                df[name] = df[a] - df[b]
            elif t == "product":
                a = rule["col_a"]
                b = rule["col_b"]
                name = rule.get("name", f"{a}_mul_{b}")
                df[name] = df[a] * df[b]
            elif t == "agg_mean":
                cols = rule["cols"]
                name = rule["name"]
                df[name] = df[cols].mean(axis=1)
            elif t == "agg_sum":
                cols = rule["cols"]
                name = rule["name"]
                df[name] = df[cols].sum(axis=1)
            elif t == "log1p":
                col = rule["col"]
                df[col] = np.log1p(df[col])
                name = f"log1p({col})"
            elif t == "square":
                col = rule["col"]
                name = f"{col}_sq"
                df[name] = df[col] ** 2
            elif t == "sqrt":
                col = rule["col"]
                name = f"{col}_sqrt"
                df[name] = np.sqrt(df[col].clip(lower=0))
            elif t == "interaction":
                a = rule["col_a"]
                b = rule["col_b"]
                name = rule.get("name", f"{a}_x_{b}")
                df[name] = df[a] * df[b]
            elif t == "binning":
                col = rule["col"]
                name = rule.get("name", f"{col}_binned")
                bins = rule.get("bins", 5)
                labels = rule.get("labels", None)
                df[name] = pd.cut(
                    df[col], bins=bins, labels=labels, duplicates="drop",
                ).cat.codes.astype(float)
            elif t == "polynomial":
                col = rule["col"]
                degree = rule.get("degree", 2)
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d
                name = f"{col}_pow(2..{degree})"
            else:
                self.log.warning(f"Unknown rule type: {t}")
                return df
            log_list.append(name)
            self.log.info(f"Derived feature created: {name}")
        except KeyError as e:
            self.log.error(f"Rule {rule} failed -- missing column: {e}")
        except Exception as e:
            self.log.error(f"Rule {rule} failed: {e}")
        return df
