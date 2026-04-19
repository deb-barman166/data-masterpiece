"""
data_masterpiece_v3.agents.encoding_agent
──────────────────────────────────────────
Encoding Agent — converts text/category columns into numbers.
Machines only understand numbers, so this agent does the translation!

Supported encodings:
  label     → simple 0, 1, 2, 3 ...
  onehot    → creates a new column for each category
  ordinal   → user provides an order [low, medium, high]
  frequency → replace with how often each value appears
  binary    → 2-category → 0/1
  multihot  → comma-separated multi-labels → multiple 0/1 cols
  target    → mean-encode using target column (advanced)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..utils.logger import get_logger

log = get_logger("EncodingAgent")


class EncodingAgent:
    """Encodes categorical columns into numeric representations."""

    def __init__(self, cfg: dict):
        g = cfg.get("global", {})
        self.per_col: dict         = cfg.get("encoding", {})
        self.low_card              = g.get("low_card_threshold", 10)
        self.med_card              = g.get("med_card_threshold", 50)
        self.mode: str             = cfg.get("mode", "auto")
        self.encoding_log: dict    = {}
        self.summary: dict         = {}

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("🔡 EncodingAgent started")
        df = df.copy()

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            enc_type = self._choose_encoding(df, col)
            df = self._apply_encoding(df, col, enc_type)

        self.summary = {"encoding_log": self.encoding_log}
        log.info(f"  Encoded {len(self.encoding_log)} columns")
        return df

    # ─────────────────────────────────────────────────────────────────────────

    def _choose_encoding(self, df: pd.DataFrame, col: str) -> str:
        """Auto-choose encoding based on cardinality."""
        if col in self.per_col:
            enc = self.per_col[col]
            # Handle ordinal dict: {"type": "ordinal", "order": [...]}
            if isinstance(enc, dict):
                return enc
            return enc

        n_unique = df[col].nunique()

        if n_unique == 2:
            return "binary"
        elif n_unique <= self.low_card:
            return "onehot"
        elif n_unique <= self.med_card:
            return "label"
        else:
            return "frequency"

    def _apply_encoding(self, df: pd.DataFrame, col: str, enc) -> pd.DataFrame:
        """Apply the chosen encoding to a column."""
        try:
            if isinstance(enc, dict):
                enc_type = enc.get("type", "label")
            else:
                enc_type = enc

            if enc_type == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoding_log[col] = "label"
                log.info(f"  [{col}] label encoding")

            elif enc_type == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.encoding_log[col] = "onehot"
                log.info(f"  [{col}] one-hot → {dummies.shape[1]} cols")

            elif enc_type == "binary":
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) >= 2:
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    df[col] = df[col].map(mapping).fillna(0).astype(int)
                    self.encoding_log[col] = "binary"
                    log.info(f"  [{col}] binary: {mapping}")

            elif enc_type == "frequency":
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map).fillna(0)
                self.encoding_log[col] = "frequency"
                log.info(f"  [{col}] frequency encoding")

            elif enc_type == "ordinal":
                order = enc.get("order", []) if isinstance(enc, dict) else []
                if order:
                    mapping = {v: i for i, v in enumerate(order)}
                    df[col] = df[col].map(mapping).fillna(-1).astype(int)
                    self.encoding_log[col] = "ordinal"
                    log.info(f"  [{col}] ordinal: {order}")
                else:
                    # fallback to label
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoding_log[col] = "label(ordinal-fallback)"

            elif enc_type == "multihot":
                # comma-separated multi-label encoding
                all_tags = set()
                df[col].dropna().apply(
                    lambda x: all_tags.update([t.strip() for t in str(x).split(",")])
                )
                for tag in sorted(all_tags):
                    safe_tag = tag.replace(" ", "_")
                    df[f"{col}_{safe_tag}"] = df[col].apply(
                        lambda x: int(tag in str(x).split(",")) if pd.notna(x) else 0
                    )
                df = df.drop(columns=[col])
                self.encoding_log[col] = f"multihot({len(all_tags)} tags)"
                log.info(f"  [{col}] multi-hot → {len(all_tags)} tag cols")

            else:
                log.warning(f"  [{col}] Unknown encoding '{enc_type}' — using label")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoding_log[col] = "label(fallback)"

        except Exception as e:
            log.warning(f"  [{col}] Encoding failed: {e} — skipping")

        return df
