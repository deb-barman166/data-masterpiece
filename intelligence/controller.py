"""
data_masterpiece.intelligence.controller  --  DataIntelligenceController

Master controller wiring all 8 intelligence modules into an 8-step pipeline:
  1. Statistical Profiling
  2. Outlier Detection
  3. Feature Selection
  4. Visualisation
  5. Relationship Analysis
  6. Model Recommendation
  7. Data Splitting
  8. HTML Report
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger
from data_masterpiece.intelligence.visualization import VisualizationEngine
from data_masterpiece.intelligence.relationship import RelationshipAnalyzer, RelationshipReport
from data_masterpiece.intelligence.recommender import ModelRecommender, RecommendationReport
from data_masterpiece.intelligence.splitter import DataSplitter, SplitResult
from data_masterpiece.intelligence.outliers import OutlierDetectionEngine, OutlierReport
from data_masterpiece.intelligence.feature_selection import FeatureSelectionEngine, SelectionReport
from data_masterpiece.intelligence.profiler import StatisticalProfiler, ColumnProfile
from data_masterpiece.intelligence.reporter import ReportGenerator


class DataIntelligenceController:
    """
    Master intelligence controller orchestrating all 8 analysis modules.

    Parameters
    ----------
    output_dir    : Directory for chart PNGs.
    report_path   : Where to save the HTML report.
    show_plots    : Whether to display plots (notebook use).
    random_state  : Seed for splitting.
    outlier_method     : "iqr" | "zscore" | "both" | "auto"
    outlier_strategy   : "clip" | "drop" | "flag" | "impute"
    iqr_factor         : IQR multiplier.
    zscore_thresh      : Z-score threshold.
    variance_threshold : Min variance for feature retention.
    corr_threshold     : Max correlation for multicollinearity.
    top_k              : Keep top-K features (0 = disabled).
    """

    def __init__(
        self,
        output_dir: str = "output/plots",
        report_path: str = "output/report.html",
        show_plots: bool = False,
        random_state: int = 42,
        outlier_method: str = "auto",
        outlier_strategy: str = "clip",
        iqr_factor: float = 1.5,
        zscore_thresh: float = 3.0,
        variance_threshold: float = 0.01,
        corr_threshold: float = 0.90,
        top_k: int = 0,
    ):
        self.log = get_logger("DataIntelligenceController")
        self.output_dir = output_dir
        self.report_path = report_path
        self.show_plots = show_plots
        self.random_state = random_state
        self._profiler = StatisticalProfiler()
        self._outlier = OutlierDetectionEngine(
            method=outlier_method, strategy=outlier_strategy,
            iqr_factor=iqr_factor, zscore_thresh=zscore_thresh,
        )
        self._selector = FeatureSelectionEngine(
            variance_threshold=variance_threshold,
            corr_threshold=corr_threshold, top_k=top_k,
        )
        self._viz = VisualizationEngine(output_dir=output_dir, show=show_plots)
        self._rel = RelationshipAnalyzer()
        self._rec = ModelRecommender()
        self._split = DataSplitter(random_state=random_state)
        self._reporter = ReportGenerator(output_path=report_path)

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        mode: str = "auto",
        graph_config: dict = None,
        test_size: float = 0.20,
        val_size: float = 0.0,
        stratify: bool = True,
        max_viz_cols: int = 20,
        skip_outlier: bool = False,
        skip_selection: bool = False,
        skip_report: bool = False,
        outlier_skip_cols: list = None,
        always_keep_cols: list = None,
    ) -> dict:
        """
        Run the full 8-step intelligence pipeline.

        Returns a dict with all intermediate and final results.
        """
        self._validate_input(df, target)
        t0 = time.perf_counter()
        self.log.info("=" * 62)
        self.log.info(
            f"  Data Intelligence Engine  |  {df.shape}  |  target='{target}'"
        )
        self.log.info("=" * 62)

        # Step 1: Profile
        self.log.info("Step 1/8 -- Statistical Profiling ...")
        profile_df, profiles = self._profiler.profile(df)

        # Step 2: Outlier Detection
        outlier_report, df_clean = None, df.copy()
        if not skip_outlier:
            self.log.info("Step 2/8 -- Outlier Detection ...")
            if outlier_skip_cols:
                self._outlier.skip_cols = set(outlier_skip_cols)
            df_clean, outlier_report = self._outlier.run(df_clean)
        else:
            self.log.info("Step 2/8 -- Outlier Detection SKIPPED.")

        # Step 3: Feature Selection
        selection_report, df_selected = None, df_clean.copy()
        if not skip_selection:
            self.log.info("Step 3/8 -- Feature Selection ...")
            self._selector.target = target
            self._selector.always_keep = set(always_keep_cols or []) | {target}
            df_selected, selection_report = self._selector.run(df_clean, target=target)
        else:
            self.log.info("Step 3/8 -- Feature Selection SKIPPED.")

        # Step 4: Visualisation
        self.log.info("Step 4/8 -- Visualisation ...")
        plots = (
            self._viz.run_manual(df_selected, graph_config)
            if mode == "manual" and graph_config
            else self._viz.run_auto(df_selected, target=target, max_cols=max_viz_cols)
        )

        # Step 5: Relationships
        self.log.info("Step 5/8 -- Relationship Analysis ...")
        rel_report = self._rel.analyze(df_selected, target=target)
        mean_abs_corr = self._mean_corr(rel_report)

        # Step 6: Recommendations
        self.log.info("Step 6/8 -- Model Recommendation ...")
        rec_report = self._rec.recommend(
            df_selected, target=target, mean_abs_corr=mean_abs_corr,
        )

        # Step 7: Split
        self.log.info("Step 7/8 -- Data Splitting ...")
        split_result = self._split.split(
            df_selected, target=target,
            test_size=test_size, val_size=val_size, stratify=stratify,
        )

        # Step 8: HTML Report
        report_path = ""
        if not skip_report:
            self.log.info("Step 8/8 -- Report Generation ...")
            report_path = self._reporter.generate(
                df=df_selected, target=target, profiles=profiles,
                outlier_report=outlier_report,
                selection_report=selection_report,
                rel_report=rel_report, rec_report=rec_report,
                split_result=split_result, plot_dir=self.output_dir,
            )

        elapsed = round(time.perf_counter() - t0, 3)
        self._summary(plots, rec_report, split_result, elapsed, report_path)

        return {
            "df_profiled": df, "df_clean": df_clean,
            "df_selected": df_selected,
            "profiles": profiles, "profile_df": profile_df,
            "outlier_report": outlier_report,
            "selection_report": selection_report,
            "plots": plots, "relationship": rel_report,
            "recommendation": rec_report,
            "split": split_result,
            "report_path": report_path,
            "elapsed_s": elapsed,
        }

    @staticmethod
    def _mean_corr(rel_report: RelationshipReport) -> float:
        corr = rel_report.correlation_matrix
        if corr.empty or len(corr) < 2:
            return 0.0
        arr = corr.values
        n = len(arr)
        vals = [
            abs(arr[i][j])
            for i in range(n) for j in range(i + 1, n)
            if not np.isnan(arr[i][j])
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _summary(self, plots, rec, split, elapsed, report_path):
        sep = "=" * 62
        info = split.split_info
        print(f"\n{sep}\n  DATA INTELLIGENCE ENGINE  --  COMPLETE\n{sep}")
        print(f"  Charts : {len(plots)} saved")
        print(
            "  Models : "
            + " | ".join(
                f"[{r.priority}] {r.model_name}" for r in rec.recommendations[:3]
            )
        )
        print(
            f"  Split  : train={info['train_rows']} / "
            f"val={info.get('val_rows', 0)} / test={info['test_rows']}"
        )
        if report_path:
            print(f"  Report : {report_path}")
        print(f"  Time   : {elapsed}s\n{sep}\n")

    @staticmethod
    def _validate_input(df, target):
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")
        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' not found. Available: {df.columns.tolist()}"
            )
        if df.isna().any(axis=None):
            raise ValueError("DataFrame contains NaN. Run preprocessing first.")
