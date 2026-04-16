"""
data_masterpiece.intelligence.feature_selection  --  FeatureSelectionEngine

Rule-based feature selection -- no ML, no training, fully deterministic.

Three cascading filters:
  1. VARIANCE FILTER      -- Remove near-constant columns.
  2. CORRELATION FILTER   -- Remove multicollinear features (keep target-relevant).
  3. TARGET RELEVANCE     -- Keep top-K features by target correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger


@dataclass
class SelectionReport:
    """Full audit of every feature-selection decision."""
    original_features: list = field(default_factory=list)
    selected_features: list = field(default_factory=list)
    dropped_features: dict = field(default_factory=dict)
    variance_dropped: list = field(default_factory=list)
    corr_dropped: list = field(default_factory=list)
    topk_dropped: list = field(default_factory=list)
    feature_scores: pd.Series = field(default_factory=pd.Series)

    def print_report(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print("  FEATURE SELECTION REPORT")
        print(sep)
        print(f"  Original features : {len(self.original_features)}")
        print(f"  Selected features : {len(self.selected_features)}")
        print(f"  Dropped total     : {len(self.dropped_features)}")
        print()

        if self.variance_dropped:
            print(f"  [Variance filter]  dropped {len(self.variance_dropped)} cols:")
            for c in self.variance_dropped:
                print(f"    X  {c}")

        if self.corr_dropped:
            print(f"  [Corr filter]      dropped {len(self.corr_dropped)} redundant cols:")
            for c in self.corr_dropped:
                print(f"    X  {c}  --  {self.dropped_features[c]}")

        if self.topk_dropped:
            print(f"  [Top-K filter]     dropped {len(self.topk_dropped)} low-relevance cols:")
            for c in self.topk_dropped:
                print(f"    X  {c}")

        if not self.feature_scores.empty:
            print(f"\n  Top features by |corr| with target:")
            for feat, score in self.feature_scores.head(10).items():
                bar = "#" * int(score * 20)
                print(f"    {feat:30s}  {score:.4f}  {bar}")

        print(f"\n  Selected: {self.selected_features}")
        print(sep + "\n")


class FeatureSelectionEngine:
    """
    Cascading rule-based feature selection.

    Parameters
    ----------
    variance_threshold : Drop columns with variance below this.
    corr_threshold     : Drop one of each pair with |r| >= this.
    top_k              : Keep only top-K features by target correlation. 0 = disabled.
    target             : Target column name.
    always_keep        : Columns never to drop.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        corr_threshold: float = 0.90,
        top_k: int = 0,
        target: str = None,
        always_keep: list = None,
    ):
        self.var_thr = variance_threshold
        self.corr_thr = corr_threshold
        self.top_k = top_k
        self.target = target
        self.always_keep = set(always_keep or [])
        if target:
            self.always_keep.add(target)
        self.log = get_logger("FeatureSelectionEngine")

    def run(
        self, df: pd.DataFrame, target: str = None,
    ) -> tuple[pd.DataFrame, SelectionReport]:
        """
        Apply all three filters in cascade.

        Returns (filtered_df, SelectionReport).
        """
        tgt = target or self.target
        report = SelectionReport(original_features=df.columns.tolist())
        df = df.copy()
        dropped: dict = {}

        # 1. variance filter
        df, v_dropped = self._variance_filter(df, dropped)
        report.variance_dropped = v_dropped
        self.log.info(f"Variance filter: dropped {len(v_dropped)} columns.")

        # 2. multicollinearity filter
        df, c_dropped = self._correlation_filter(df, tgt, dropped)
        report.corr_dropped = c_dropped
        self.log.info(f"Correlation filter: dropped {len(c_dropped)} redundant columns.")

        # 3. top-K filter
        feature_scores = pd.Series(dtype=float)
        k_dropped = []
        if self.top_k > 0 and tgt and tgt in df.columns:
            df, k_dropped, feature_scores = self._topk_filter(df, tgt, dropped)
            self.log.info(f"Top-K filter: kept top {self.top_k}, dropped {len(k_dropped)}.")
        elif tgt and tgt in df.columns:
            feat_cols = [c for c in df.columns if c != tgt]
            if feat_cols:
                feature_scores = (
                    df[feat_cols].corrwith(df[tgt])
                    .abs()
                    .sort_values(ascending=False)
                )

        report.topk_dropped = k_dropped
        report.dropped_features = dropped
        report.selected_features = df.columns.tolist()
        report.feature_scores = feature_scores
        report.print_report()
        return df, report

    def _variance_filter(self, df, dropped):
        v_dropped = []
        num_df = df.select_dtypes(include=[np.number])
        for col in num_df.columns:
            if col in self.always_keep:
                continue
            var = float(df[col].var())
            if var < self.var_thr:
                df = df.drop(columns=[col])
                reason = f"variance={var:.6f} < threshold={self.var_thr}"
                dropped[col] = reason
                v_dropped.append(col)
        return df, v_dropped

    def _correlation_filter(self, df, target, dropped):
        c_dropped = []
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.always_keep
        ]
        if len(num_cols) < 2:
            return df, c_dropped

        target_corr: dict = {}
        if target and target in df.columns:
            for col in num_cols:
                target_corr[col] = abs(float(df[col].corr(df[target])))

        corr_matrix = df[num_cols].corr(numeric_only=True)
        to_drop: set = set()

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                a, b = num_cols[i], num_cols[j]
                if a in to_drop or b in to_drop:
                    continue
                r = abs(float(corr_matrix.loc[a, b]))
                if r >= self.corr_thr:
                    score_a = target_corr.get(a, 0.0)
                    score_b = target_corr.get(b, 0.0)
                    loser = b if score_a >= score_b else a
                    winner = a if loser == b else b
                    to_drop.add(loser)
                    reason = (
                        f"multicollinear with '{winner}' "
                        f"(|r|={r:.3f} >= {self.corr_thr}); "
                        f"'{winner}' kept (higher target relevance)"
                    )
                    dropped[loser] = reason
                    c_dropped.append(loser)

        if to_drop:
            df = df.drop(columns=list(to_drop))
        return df, c_dropped

    def _topk_filter(self, df, target, dropped):
        feat_cols = [c for c in df.columns if c != target and c not in self.always_keep]
        if not feat_cols:
            return df, [], pd.Series(dtype=float)

        scores = df[feat_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        keep_set = set(scores.head(self.top_k).index.tolist()) | self.always_keep
        k_dropped = []

        for col in feat_cols:
            if col not in keep_set:
                df = df.drop(columns=[col])
                score = scores.get(col, 0.0)
                reason = (
                    f"top-K filter: target correlation |r|={score:.4f} "
                    f"below top-{self.top_k} threshold"
                )
                dropped[col] = reason
                k_dropped.append(col)

        return df, k_dropped, scores
