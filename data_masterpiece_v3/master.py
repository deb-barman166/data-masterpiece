"""
data_masterpiece_v3.master
───────────────────────────
MasterPipeline v3 — The Crown Jewel.

ONE class. ONE call. EVERYTHING done.

    from data_masterpiece_v3 import MasterPipeline, Config

    # AUTO mode (simplest possible)
    result = MasterPipeline().run("data.csv", target="price")

    # Manual mode (full control via JSON)
    cfg    = Config.from_json("my_config.json")
    result = MasterPipeline(cfg).run("data.csv", target="price")

Pipeline stages:
  1. LOAD         → smart multi-format data loader
  2. CLEAN        → remove duplicates, nulls, garbage
  3. TYPE         → smart type conversion (dates, booleans, numerics)
  4. MISSING      → intelligent imputation (auto or per-column)
  5. ENCODE       → categorical → numeric
  6. FEATURES     → derived feature engineering
  7. VALIDATE     → quality check + optional scaling
  8. STATS        → deep statistical analysis
  9. OUTLIERS     → detect and handle outliers
  10. SELECT      → drop low-quality features
  11. SPLIT       → train/val/test → numpy arrays
  12. CHARTS      → 13+ professional dark-theme charts
  13. AUTOML      → (optional) train all sklearn + PyTorch models
  14. REPORT      → animated Legend-level HTML report
"""

from __future__ import annotations

import os
import time
from typing import Optional, Union

import pandas as pd

from .config import Config
from .utils.logger import get_logger, section_banner
from .utils.loader import load_data
from .agents.cleaning_agent import CleaningAgent
from .agents.type_agent import TypeAgent
from .agents.missing_agent import MissingAgent
from .agents.encoding_agent import EncodingAgent
from .agents.feature_agent import FeatureAgent
from .agents.validation_agent import ValidationAgent
from .intelligence.stats import StatsEngine
from .intelligence.outlier import OutlierEngine
from .intelligence.selector import FeatureSelector
from .intelligence.splitter import DataSplitter
from .visualization.charts import ChartEngine
from .reporting.report_builder import build_report

log = get_logger("MasterPipeline")


class MasterPipeline:
    """
    Data Masterpiece v3 — End-to-end data science pipeline.

    ╔════════════════════════════════════════════════════════╗
    ║  Raw Data → Clean → Transform → Analyze → ML-Ready   ║
    ╚════════════════════════════════════════════════════════╝

    Parameters
    ----------
    config : Config object (optional)
        Pass a Config to customize behaviour.
        If None, AUTO mode is used (pipeline decides everything).

    Examples
    --------
    # Easiest way (AUTO mode):
    from data_masterpiece_v3 import MasterPipeline
    result = MasterPipeline().run("titanic.csv", target="survived")

    # Manual mode via JSON:
    from data_masterpiece_v3 import MasterPipeline, Config
    cfg = Config.from_json("my_config.json")
    result = MasterPipeline(cfg).run("titanic.csv", target="survived")

    # Generate starter config:
    Config().save_json("starter_config.json")  # then edit it!
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.cfg_dict = self.config.to_dict()
        self._results: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        source,
        target: str = None,
        ask_automl: bool = True,
    ) -> dict:
        """
        Run the complete end-to-end pipeline.

        Parameters
        ----------
        source    : file path, URL, or DataFrame
        target    : column name to predict (ML target)
        ask_automl: if True and run_automl=True in config,
                    confirms with user before training models

        Returns
        -------
        dict with all results, paths, and data splits
        """
        t0 = time.perf_counter()

        section_banner("DATA MASTERPIECE v3  —  PIPELINE START", width=62)

        # ── Stage 1: Load ───────────────────────────────────────────────────
        log.info("📂 [1/9] Loading data ...")
        df_raw = load_data(source)
        log.info(f"  Raw shape: {df_raw.shape}")

        # ── Stage 2: Preprocess ─────────────────────────────────────────────
        log.info("🔧 [2/9] Preprocessing ...")
        df_clean, pp_summary = self._run_preprocess(df_raw, target)

        # ── Stage 3: Statistics ─────────────────────────────────────────────
        log.info("📊 [3/9] Computing statistics ...")
        stats_engine = StatsEngine()
        stats = stats_engine.run(df_clean, target=target)

        # ── Stage 4: Outlier treatment ──────────────────────────────────────
        df_proc = df_clean
        if not self.config.skip_outlier and target and target in df_clean.columns:
            log.info("🔭 [4/9] Handling outliers ...")
            outlier_eng = OutlierEngine(
                method   = self.config.outlier_method,
                strategy = self.config.outlier_strategy,
                iqr_factor    = self.config.iqr_factor,
                zscore_thresh  = self.config.zscore_thresh,
            )
            df_proc = outlier_eng.run(df_clean)
        else:
            log.info("🔭 [4/9] Outlier handling SKIPPED")

        # ── Stage 5: Feature selection ──────────────────────────────────────
        if not self.config.skip_selection:
            log.info("🎯 [5/9] Feature selection ...")
            selector = FeatureSelector(
                variance_threshold = self.config.intelligence_variance_threshold,
                corr_threshold     = self.config.intelligence_corr_threshold,
                top_k              = self.config.intelligence_top_k,
            )
            df_selected = selector.run(df_proc, target=target)
        else:
            log.info("🎯 [5/9] Feature selection SKIPPED")
            df_selected = df_proc

        # ── Stage 6: Charts ─────────────────────────────────────────────────
        log.info("🎨 [6/9] Generating charts ...")
        fi = {}
        chart_engine = ChartEngine(
            output_dir = self.config.plot_dir,
            dpi        = self.config.chart_dpi,
            max_cols   = self.config.max_viz_cols,
        )
        charts = chart_engine.generate_all(
            df_selected,
            target               = target,
            feature_importance   = fi,
            relationship_groups  = self.config.relationship_columns,
        )

        # ── Stage 7: Train/Val/Test split ───────────────────────────────────
        split_result = None
        split_info   = {}
        if target and target in df_selected.columns:
            log.info("✂️  [7/9] Creating train/val/test split ...")
            splitter = DataSplitter(
                test_size  = self.config.test_size,
                val_size   = self.config.val_size,
                stratify   = self.config.stratify,
                output_dir = self.config.ml_ready_dir,
            )
            try:
                split_result = splitter.run(df_selected, target)
                split_info   = split_result.split_info
                # Now update feature importance using a quick RF
                fi = self._quick_feature_importance(split_result)
            except Exception as e:
                log.warning(f"  Split failed: {e}")
        else:
            log.info("✂️  [7/9] Split SKIPPED (no valid target)")

        # ── Stage 8: AutoML ─────────────────────────────────────────────────
        automl_results = {}
        if self.config.run_automl and split_result is not None:
            if ask_automl:
                print("\n" + "="*55)
                print("  🤖 AutoML is enabled in your config.")
                print("  This will train MULTIPLE ML models on your data.")
                ans = input("  Proceed? [y/N]: ").strip().lower()
                print("="*55 + "\n")
                do_automl = ans in ("y", "yes")
            else:
                do_automl = True

            if do_automl:
                log.info("🤖 [8/9] Running AutoML ...")
                from .automl.builder import AutoMLBuilder
                builder = AutoMLBuilder(self.config)
                automl_results = builder.run(split_result)
            else:
                log.info("🤖 [8/9] AutoML skipped by user.")
        else:
            log.info("🤖 [8/9] AutoML DISABLED (set run_automl=True to enable)")

        # ── Stage 9: Report ─────────────────────────────────────────────────
        report_path = ""
        if not self.config.skip_report:
            log.info("📄 [9/9] Building Legend HTML report ...")
            report_path = build_report(
                output_path    = self.config.report_path,
                stats          = stats,
                charts         = charts,
                split_info     = split_info,
                automl_results = automl_results,
                preprocess_summary = pp_summary,
                target         = target,
                config_dict    = self.config.to_dict(),
            )
            log.info(f"  Report saved → {report_path}")

        # ── Save processed CSV ───────────────────────────────────────────────
        csv_path = self.config.output_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df_selected.to_csv(csv_path, index=False)
        log.info(f"  Processed CSV → {csv_path}")

        elapsed = round(time.perf_counter() - t0, 3)
        self._print_summary(df_raw, df_clean, df_selected, pp_summary,
                            split_info, automl_results, csv_path, report_path, elapsed)

        self._results = {
            "df_raw":           df_raw,
            "df_clean":         df_clean,
            "df_processed":     df_selected,
            "stats":            stats,
            "preprocess_summary": pp_summary,
            "split":            split_result,
            "split_info":       split_info,
            "automl_results":   automl_results,
            "charts":           charts,
            "report_path":      report_path,
            "csv_path":         csv_path,
            "elapsed_s":        elapsed,
        }
        return self._results

    def preprocess_only(self, source, target: str = None) -> tuple:
        """
        Run only the preprocessing pipeline.
        Returns (df_clean, summary_dict)
        """
        df_raw = load_data(source)
        df_clean, summary = self._run_preprocess(df_raw, target)
        os.makedirs(os.path.dirname(self.config.output_path) or ".", exist_ok=True)
        df_clean.to_csv(self.config.output_path, index=False)
        return df_clean, summary

    def generate_starter_config(self, path: str = "starter_config.json") -> str:
        """Save a full starter config JSON that you can edit."""
        self.config.save_json(path)
        log.info(f"  Starter config saved → {path}")
        return path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_preprocess(self, df_raw: pd.DataFrame, target: str = None) -> tuple:
        cfg = self.cfg_dict
        active = self.config.active_agents
        summary = {}

        df = df_raw.copy()

        if "cleaning" in active:
            agent = CleaningAgent(cfg)
            df = agent.run(df)
            summary.update(agent.summary)

        if "type_conversion" in active:
            agent = TypeAgent(cfg)
            df = agent.run(df)
            summary.update(agent.summary)

        if "missing" in active:
            agent = MissingAgent(cfg)
            df = agent.run(df)
            summary.update(agent.summary)

        if "encoding" in active:
            agent = EncodingAgent(cfg)
            df = agent.run(df)
            summary.update(agent.summary)
            summary["encoding_log"] = agent.encoding_log

        if "feature" in active:
            agent = FeatureAgent(cfg)
            df = agent.run(df)
            summary.update(agent.summary)

        if "validation" in active:
            agent = ValidationAgent(cfg)
            df = agent.run(df, target=target)
            summary.update(agent.summary)

        return df, summary

    def _quick_feature_importance(self, split_result) -> dict:
        """Run a fast RandomForest to get feature importance scores."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            import numpy as np

            X, y = split_result.X_train, split_result.y_train
            n_unique = len(set(y.tolist()))
            task = "clf" if n_unique <= 20 else "reg"

            if task == "clf":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            model.fit(X, y)
            fi = model.feature_importances_
            return dict(sorted(
                zip(split_result.feature_names, fi.tolist()),
                key=lambda x: x[1], reverse=True
            ))
        except Exception:
            return {}

    def _print_summary(self, df_raw, df_clean, df_sel, pp, split_info,
                        automl, csv, report, elapsed):
        sep = "═" * 65
        print(f"\n\033[96m{sep}")
        print(f"  ⚡ DATA MASTERPIECE v3 — PIPELINE COMPLETE")
        print(f"{sep}\033[0m")
        print(f"  \033[2mRaw input       :\033[0m  {df_raw.shape}")
        print(f"  \033[2mAfter preprocess:\033[0m  {df_clean.shape}")
        print(f"  \033[2mAfter selection :\033[0m  {df_sel.shape}")
        print(f"  \033[2mRows removed    :\033[0m  {pp.get('rows_removed', 0)}")
        print(f"  \033[2mCols removed    :\033[0m  {pp.get('cols_removed', 0)}")
        print(f"  \033[2mImputed cols    :\033[0m  {len(pp.get('columns_imputed', []))}")
        print(f"  \033[2mEncodings done  :\033[0m  {len(pp.get('encoding_log', {}))}")
        print(f"  \033[2mNew features    :\033[0m  {len(pp.get('feature_transforms', []))}")
        if split_info:
            print(f"  \033[2mData split      :\033[0m  "
                  f"train={split_info.get('train_rows',0):,} / "
                  f"val={split_info.get('val_rows',0):,} / "
                  f"test={split_info.get('test_rows',0):,}")
        if automl:
            best = automl.get("best_model", {})
            print(f"  \033[2mBest ML model   :\033[0m  {best.get('name','N/A')} "
                  f"(score={best.get('score',0):.4f})")
        print(f"  \033[2mCSV output      :\033[0m  {csv}")
        if report:
            print(f"  \033[2mHTML report     :\033[0m  {report}")
        print(f"  \033[93mTotal time      :  {elapsed}s\033[0m")
        print(f"\033[96m{sep}\033[0m\n")
