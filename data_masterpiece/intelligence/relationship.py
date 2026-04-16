"""
data_masterpiece.intelligence.relationship  --  RelationshipAnalyzer

Computes and reports on pairwise relationships:
  - Pearson correlation matrix
  - Strong pairs (|r| >= 0.5)
  - Multicollinear pairs (|r| >= 0.85)
  - Target-feature correlation ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger


@dataclass
class RelationshipReport:
    """Complete relationship analysis output."""
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_correlations: pd.Series = field(default_factory=pd.Series)
    strong_pairs: list = field(default_factory=list)
    multicollinear_pairs: list = field(default_factory=list)
    n_features: int = 0

    def print_report(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print("  RELATIONSHIP ANALYSIS REPORT")
        print(sep)
        print(f"  Features analysed: {self.n_features}")

        if not self.target_correlations.empty:
            print(f"\n  Target correlation ranking:")
            for feat, r in self.target_correlations.head(10).items():
                bar = "#" * int(abs(r) * 40)
                sign = "+" if r >= 0 else "-"
                print(f"    {feat:30s}  r={r:+.4f}  {sign}{bar}")

        print(f"\n  Strong pairs (|r| >= 0.5): {len(self.strong_pairs)}")
        for p in self.strong_pairs[:10]:
            print(f"    {p['feature_a']} x {p['feature_b']}  r={p['correlation']:.4f}  ({p['direction']})")

        print(f"\n  Multicollinear pairs (|r| >= 0.85): {len(self.multicollinear_pairs)}")
        for a, b, r in self.multicollinear_pairs[:10]:
            print(f"    {a} x {b}  r={r:.4f}")

        print(sep + "\n")


class RelationshipAnalyzer:
    """
    Analyze pairwise feature relationships and correlations.

    Parameters
    ----------
    strong_threshold    : Minimum |r| for a "strong" pair (default 0.5).
    multicollinear_threshold : Minimum |r| for multicollinearity flag (default 0.85).
    """

    def __init__(
        self,
        strong_threshold: float = 0.5,
        multicollinear_threshold: float = 0.85,
    ):
        self.strong_thr = strong_threshold
        self.mc_thr = multicollinear_threshold
        self.log = get_logger("RelationshipAnalyzer")

    def analyze(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> RelationshipReport:
        """
        Compute full relationship analysis.

        Parameters
        ----------
        df     : Fully numeric DataFrame.
        target : Optional target column name for ranking.

        Returns
        -------
        RelationshipReport
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        report = RelationshipReport(n_features=len(num_cols))

        if len(num_cols) < 2:
            self.log.warning("Fewer than 2 numeric columns -- skipping.")
            return report

        # -- correlation matrix --
        corr = df[num_cols].corr(numeric_only=True)
        report.correlation_matrix = corr

        # -- target correlation ranking --
        if target and target in corr.columns:
            tc = corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False)
            report.target_correlations = tc

        # -- strong pairs --
        strong = []
        mc_pairs = []
        cols = corr.columns.tolist()
        seen: set = set()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                r = corr.loc[a, b]
                if np.isnan(r):
                    continue
                ar = abs(r)

                if ar >= self.strong_thr:
                    strong.append({
                        "feature_a": a,
                        "feature_b": b,
                        "correlation": float(r),
                        "direction": "positive" if r >= 0 else "negative",
                    })

                if ar >= self.mc_thr and (b, a) not in seen:
                    mc_pairs.append((a, b, float(r)))
                    seen.add((a, b))

        # sort by absolute correlation descending
        strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        mc_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        report.strong_pairs = strong
        report.multicollinear_pairs = mc_pairs

        report.print_report()
        return report
