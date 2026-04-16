"""
data_masterpiece.master  --  MasterPipeline

The crown jewel.  A single class that runs the entire Data Masterpiece
pipeline from raw data to ML-ready output with full analysis:

    Raw Data -> Preprocess -> Intelligence -> Report -> ML-Ready Output
"""

from __future__ import annotations

import os
import time

import pandas as pd

from data_masterpiece.utils.logger import get_logger
from data_masterpiece.preprocessing.core.loader import load_data
from data_masterpiece.preprocessing.core.auto_config import generate_auto_config
from data_masterpiece.preprocessing.controller import PipelineController
from data_masterpiece.intelligence.controller import DataIntelligenceController
from data_masterpiece.config import Config


class MasterPipeline:
    """
    End-to-end data pipeline: Preprocess -> Analyze -> Split -> Report.

    This is the single entry point that combines both engines into one
    seamless workflow.  Feed it raw data, get back everything you need.

    Parameters
    ----------
    config : Optional Config object.  If None, uses AUTO defaults.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.log = get_logger("MasterPipeline")
        self._preprocess_summary: dict = {}
        self._intelligence_results: dict = {}

    # -- public API ------------------------------------------------------------

    def run(
        self,
        source,
        target: str,
        save_csv: bool = True,
        save_report: bool = True,
    ) -> dict:
        """
        Run the full end-to-end pipeline.

        Parameters
        ----------
        source    : Path, URL, or DataFrame -- the raw input data.
        target    : Target column name for ML analysis.
        save_csv  : Whether to save the preprocessed CSV.
        save_report : Whether to generate the HTML report.

        Returns
        -------
        dict with keys:
            df_raw           : Original raw DataFrame
            df_clean         : Preprocessed, fully numeric DataFrame
            df_selected      : After feature selection (if intelligence enabled)
            preprocess_summary : Preprocessing pipeline summary
            intelligence_results : Full intelligence analysis results
            split            : Train/val/test split result
            report_path      : Path to HTML report (if generated)
            csv_path         : Path to saved CSV (if saved)
            elapsed_s        : Total pipeline time
        """
        t0 = time.perf_counter()

        self.log.info("=" * 70)
        self.log.info("  DATA MASTERPIECE PIPELINE  --  v1.0")
        self.log.info("  End-to-end: Raw Data -> Clean -> Analyze -> Split -> Report")
        self.log.info("=" * 70)

        # -- Stage 1: Load --
        self.log.info("[Stage 1/4] Loading raw data ...")
        df_raw = load_data(source)
        self.log.info(f"  Raw data: {df_raw.shape}")

        # -- Stage 2: Preprocess --
        self.log.info("[Stage 2/4] Running preprocessing pipeline ...")
        df_clean, preprocess_summary = self._run_preprocess(df_raw)
        self._preprocess_summary = preprocess_summary

        # -- Stage 3: Intelligence --
        intelligence_results = {}
        df_selected = df_clean
        split_result = None
        report_path = ""

        if self.config.run_intelligence and target in df_clean.columns:
            self.log.info("[Stage 3/4] Running intelligence engine ...")
            intelligence_results, df_selected, split_result, report_path = (
                self._run_intelligence(df_clean, target)
            )
            self._intelligence_results = intelligence_results
        else:
            self.log.info("[Stage 3/4] Intelligence engine SKIPPED.")

        # -- Stage 4: Save --
        self.log.info("[Stage 4/4] Saving outputs ...")
        csv_path = ""
        if save_csv:
            csv_path = self.config.output_path
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            df_selected.to_csv(csv_path, index=False)
            self.log.info(f"  CSV saved: {csv_path}")

        elapsed = round(time.perf_counter() - t0, 3)

        # -- final summary --
        self._print_final_summary(
            df_raw, df_clean, df_selected,
            preprocess_summary, intelligence_results,
            split_result, csv_path, report_path, elapsed,
        )

        return {
            "df_raw": df_raw,
            "df_clean": df_clean,
            "df_selected": df_selected,
            "preprocess_summary": preprocess_summary,
            "intelligence_results": intelligence_results,
            "split": split_result,
            "report_path": report_path,
            "csv_path": csv_path,
            "elapsed_s": elapsed,
        }

    # -- convenience methods ---------------------------------------------------

    def preprocess_only(self, source, save_csv: bool = True) -> tuple:
        """Run only the preprocessing stage.  Returns (df_clean, summary)."""
        df_raw = load_data(source)
        df_clean, summary = self._run_preprocess(df_raw)
        if save_csv:
            os.makedirs(os.path.dirname(self.config.output_path) or ".", exist_ok=True)
            df_clean.to_csv(self.config.output_path, index=False)
        return df_clean, summary

    def analyze(
        self, df: pd.DataFrame, target: str,
    ) -> dict:
        """Run only the intelligence stage on an already-clean DataFrame."""
        return self._run_intelligence(df, target)

    # -- private helpers -------------------------------------------------------

    def _run_preprocess(self, df_raw: pd.DataFrame) -> tuple:
        """Run the preprocessing pipeline and return (df_clean, summary)."""
        auto_cfg = generate_auto_config(df_raw)

        # merge user config
        user_dict = self.config.to_dict()
        auto_cfg["global"].update(user_dict.get("global", {}))
        for key in ("cleaning", "missing", "encoding", "type_conversion", "features"):
            if key in user_dict and user_dict[key]:
                auto_cfg[key] = user_dict[key]
        auto_cfg["active_agents"] = self.config.active_agents
        auto_cfg["mode"] = self.config.mode
        auto_cfg["output_path"] = self.config.output_path

        controller = PipelineController(auto_cfg)
        df_clean = controller.run(df_raw)
        return df_clean, controller.summary

    def _run_intelligence(self, df: pd.DataFrame, target: str) -> tuple:
        """Run the intelligence engine.  Returns (results, df_selected, split, report_path)."""
        ctrl = DataIntelligenceController(
            output_dir=self.config.plot_dir,
            report_path=self.config.report_path,
            outlier_method=self.config.outlier_method,
            outlier_strategy=self.config.outlier_strategy,
            iqr_factor=self.config.iqr_factor,
            zscore_thresh=self.config.zscore_thresh,
            variance_threshold=self.config.intelligence_variance_threshold,
            corr_threshold=self.config.intelligence_corr_threshold,
            top_k=self.config.intelligence_top_k,
        )

        results = ctrl.run(
            df, target=target,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            stratify=self.config.stratify,
            max_viz_cols=self.config.max_viz_cols,
            skip_outlier=self.config.skip_outlier,
            skip_selection=self.config.skip_selection,
            skip_report=self.config.skip_report,
        )

        return (
            results,
            results.get("df_selected", df),
            results.get("split"),
            results.get("report_path", ""),
        )

    def _print_final_summary(self, *args):
        df_raw, df_clean, df_selected, pp, intel, split, csv, report, elapsed = args
        sep = "=" * 70
        print(f"\n{sep}")
        print("  DATA MASTERPIECE  --  PIPELINE COMPLETE")
        print(sep)
        print(f"  Raw input       : {df_raw.shape}")
        print(f"  After preprocess : {df_clean.shape}")
        print(f"  After selection  : {df_selected.shape}")
        print(f"  Rows removed     : {pp.get('rows_removed', 0)}")
        print(f"  Cols dropped     : {len(pp.get('dropped_cols', []))}")
        print(f"  Columns imputed  : {len(pp.get('columns_imputed', []))}")
        print(f"  Encodings        : {len(pp.get('encoding_log', {}))}")
        print(f"  Derived features : {pp.get('feature_transforms', [])}")
        if split:
            info = split.split_info
            print(
                f"  Data split       : "
                f"train={info['train_rows']} / "
                f"val={info.get('val_rows', 0)} / "
                f"test={info['test_rows']}"
            )
        if csv:
            print(f"  CSV output       : {csv}")
        if report:
            print(f"  HTML report      : {report}")
        print(f"  Total time       : {elapsed}s")
        print(f"{sep}\n")
