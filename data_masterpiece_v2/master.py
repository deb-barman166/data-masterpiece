"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATA MASTERPIECE V2 - MASTER PIPELINE                     ║
║                                                                            ║
║         Your Complete Journey from Raw Data to ML-Ready Masterpiece!        ║
║                                                                            ║
║   This module is the heart of Data Masterpiece V2! It orchestrates the     ║
║   entire pipeline from loading your data to generating insights and         ║
║   even building ML models automatically!                                    ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Quick Start (For Beginners!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1. The EASIEST way - Just 3 lines!
    >>> from data_masterpiece_v2 import MasterPipeline
    >>> pipeline = MasterPipeline()
    >>> result = pipeline.run("your_data.csv", target="your_target")

    # 2. AUTO mode with automatic ML model building!
    >>> pipeline = MasterPipeline(mode="auto")
    >>> pipeline.config.ml_builder.enable_auto_ml = True
    >>> result = pipeline.run("data.csv", target="survived")

    # 3. MANUAL mode for full control!
    >>> from data_masterpiece_v2 import Config
    >>> config = Config.create_manual_config()
    >>> config.preprocessing.missing_strategy = "median"
    >>> pipeline = MasterPipeline(config=config)
    >>> result = pipeline.run("data.csv", target="price")

What You Get:
━━━━━━━━━━━━━━
    ✓ Clean, ML-ready data
    ✓ Beautiful animated HTML report
    ✓ Professional visualizations
    ✓ Auto ML model (optional)
    ✓ Data insights and recommendations

"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data_masterpiece_v2.config import Config, PreprocessingConfig, IntelligenceConfig, MLBuilderConfig, OutputConfig
from data_masterpiece_v2.utils.logger import get_logger, setup_logging, format_duration
from data_masterpiece_v2.utils.helpers import ensure_dir, format_bytes, validate_dataframe

# Import preprocessing components
from data_masterpiece_v2.preprocessing.controller import PreprocessingController
from data_masterpiece_v2.preprocessing.core.loader import DataLoader

# Import intelligence components
from data_masterpiece_v2.intelligence.controller import IntelligenceController

# Import ML builder
from data_masterpiece_v2.ml_builder.auto_builder import AutoMLBuilder

# Import reporter
from data_masterpiece_v2.reports.animated_reporter import AnimatedReportGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logger = get_logger("MasterPipeline")


class MasterPipeline:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    MASTER PIPELINE - The Heart of Data Masterpiece V2!
    ═══════════════════════════════════════════════════════════════════════════════

    This is your main gateway to everything Data Masterpiece V2 can do!

    Features:
        • Auto Mode - Let the magic happen automatically!
        • Manual Mode - Full control over every setting
        • Auto ML - Builds ML models for you automatically
        • Animated Reports - Beautiful, interactive dark theme reports
        • Professional Visualizations - Charts that tell stories

    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, uses AUTO mode with intelligent defaults.
    mode : str, optional
        Operating mode: "auto" or "manual". Overrides config if provided.

    Attributes
    ----------
    config : Config
        The configuration object used by this pipeline.
    data : Dict
        Cached data from the last run (df_raw, df_processed, etc.)

    Examples
    --------
    Basic Usage (Easiest Way!):

        >>> from data_masterpiece_v2 import MasterPipeline
        >>> pipeline = MasterPipeline()
        >>> result = pipeline.run("your_data.csv", target="survived")

    With Custom Config:

        >>> from data_masterpiece_v2 import Config, MasterPipeline
        >>> config = Config.create_manual_config()
        >>> config.preprocessing.drop_duplicates = False
        >>> pipeline = MasterPipeline(config=config)
        >>> result = pipeline.run("data.csv", target="price")

    With Auto ML:

        >>> pipeline = MasterPipeline()
        >>> pipeline.config.ml_builder.enable_auto_ml = True
        >>> result = pipeline.run("data.csv", target="survived")

    ═════════════════════════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Master Pipeline.

        Parameters
        ----------
        config : Config, optional
            Pre-configured Config object.
        mode : str, optional
            Quick way to set mode: "auto" or "manual".
        **kwargs
            Additional configuration options.
        """
        # Set up configuration
        if config is not None:
            self.config = config
        elif mode is not None:
            self.config = Config(mode=mode)
        else:
            self.config = Config(mode="auto")

        # Override config with any kwargs
        if kwargs:
            self._apply_kwargs(kwargs)

        # Initialize components
        self._initialize_components()

        # Data storage
        self.data: Dict[str, Any] = {}
        self._is_fitted = False

        # Banner
        self._print_banner()

    def _apply_kwargs(self, kwargs: Dict) -> None:
        """Apply keyword arguments to config."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # Ensure output directories exist
        ensure_dir(self.config.output.output_dir)
        ensure_dir(self.config.output.plots_dir)
        ensure_dir(self.config.output.models_dir)
        ensure_dir(self.config.output.logs_dir)

        # Set up logging
        setup_logging(
            log_level=self.config.log_level,
            log_to_file=self.config.log_to_file,
            log_dir=self.config.output.logs_dir
        )

        logger.info("Initializing Master Pipeline V2...")
        logger.info(f"Mode: {self.config.mode.upper()}")
        logger.info(f"Output directory: {self.config.output.output_dir}")

    def _print_banner(self) -> None:
        """Print welcome banner."""
        banner = f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║           ███████╗ ██████╗ ██████╗ ███████╗███████╗████████╗             ║
    ║           ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝             ║
    ║           ███████╗██║   ██║██████╔╝█████╗  ███████╗   ██║                ║
    ║           ╚════██║██║   ██║██╔══██╗██╔══╝  ╚════██║   ██║                ║
    ║           ███████║╚██████╔╝██████╔╝███████╗███████║   ██║                ║
    ║           ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝                ║
    ║                                                                          ║
    ║                    ★ VERSION 2 - LEGEND LEVEL ★                           ║
    ║                                                                          ║
    ║              From Raw Data to ML Masterpiece - Made Easy!                ║
    ║                                                                          ║
    ║   Mode: {self.config.mode.upper():<12}  |  Auto ML: {'ON 🔥' if self.config.ml_builder.enable_auto_ml else 'OFF'}                         ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def run(
        self,
        source: Union[str, pd.DataFrame, Path],
        target: str,
        save_csv: Optional[bool] = None,
        save_report: Optional[bool] = None,
        save_models: Optional[bool] = None,
        build_models: Optional[bool] = None,
        show_plots: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        ═══════════════════════════════════════════════════════════════════════════
        RUN THE PIPELINE - Main entry point!
        ═══════════════════════════════════════════════════════════════════════════

        This is where the magic happens! Run your entire data analysis pipeline
        with just one call!

        Parameters
        ----------
        source : str, Path, or pd.DataFrame
            Your data source!
            • CSV, Excel, JSON, Parquet file path
            • URL to download data
            • pandas DataFrame directly

        target : str
            The column you want to predict (your "target" variable).
            Examples: "survived", "price", "sales", "churned"

        save_csv : bool, optional
            Save the processed data as CSV? Default: True

        save_report : bool, optional
            Generate a beautiful HTML report? Default: True

        save_models : bool, optional
            Save trained ML models? Default: True (if Auto ML enabled)

        build_models : bool, optional
            Build ML models automatically? Default: from config

        show_plots : bool
            Display plots during processing? Default: False

        verbose : bool
            Print detailed progress? Default: True

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all results:
            {
                'df_raw': raw data,
                'df_processed': cleaned data,
                'df_train': training data,
                'df_test': test data,
                'target': target column name,
                'features': feature column names,
                'preprocessing_summary': what was done to your data,
                'profile': statistical profile,
                'models': trained models (if Auto ML enabled),
                'best_model': best model with scores,
                'report_path': path to HTML report,
                'csv_path': path to processed CSV,
                'plots_paths': paths to generated plots,
                'elapsed_time': total time taken
            }

        Examples
        --------
        Basic usage:

            >>> result = pipeline.run("data.csv", target="survived")
            >>> print(result.keys())
            dict_keys(['df_processed', 'report_path', 'csv_path', ...])

        With DataFrame:

            >>> import pandas as pd
            >>> df = pd.read_csv("data.csv")
            >>> result = pipeline.run(df, target="survived")

        With custom settings:

            >>> result = pipeline.run(
            ...     "data.csv",
            ...     target="price",
            ...     build_models=True,
            ...     save_report=True
            ... )

        ═══════════════════════════════════════════════════════════════════════════
        """
        start_time = time.time()

        # Override config with function parameters
        if save_csv is not None:
            self.config.output.save_csv = save_csv
        if save_report is not None:
            self.config.output.save_report = save_report
        if save_models is not None:
            self.config.output.save_models = save_models
        if build_models is not None:
            self.config.ml_builder.enable_auto_ml = build_models

        logger.info("=" * 70)
        logger.info("  🎯 DATA MASTERPIECE V2 - PIPELINE STARTING")
        logger.info("=" * 70)
        logger.info(f"  Source: {source if isinstance(source, str) else 'DataFrame'}")
        logger.info(f"  Target: {target}")
        logger.info(f"  Mode: {self.config.mode.upper()}")
        logger.info(f"  Auto ML: {'ON 🔥' if self.config.ml_builder.enable_auto_ml else 'OFF'}")
        logger.info("=" * 70)

        # Initialize results dictionary
        results: Dict[str, Any] = {
            'target': target,
            'config': self.config.to_dict(),
            'stages_completed': []
        }

        try:
            # ═══════════════════════════════════════════════════════════════════
            # STAGE 1: LOAD DATA
            # ═══════════════════════════════════════════════════════════════════
            if verbose:
                print("\n" + "─" * 60)
                print("  📂 STAGE 1: Loading Your Data")
                print("─" * 60)

            df_raw = self._load_data(source)
            results['df_raw'] = df_raw
            results['n_rows_raw'] = len(df_raw)
            results['n_cols_raw'] = len(df_raw.columns)

            if verbose:
                print(f"  ✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
                print(f"  📊 Memory: {format_bytes(df_raw.memory_usage(deep=True).sum())}")

            # ═══════════════════════════════════════════════════════════════════
            # STAGE 2: PREPROCESSING
            # ═══════════════════════════════════════════════════════════════════
            if verbose:
                print("\n" + "─" * 60)
                print("  🔧 STAGE 2: Preprocessing Your Data")
                print("─" * 60)

            df_processed, preprocess_summary = self._preprocess_data(df_raw, target)
            results['df_processed'] = df_processed
            results['preprocessing_summary'] = preprocess_summary
            results['stages_completed'].append('preprocessing')

            if verbose:
                print(f"  ✅ Preprocessed: {len(df_processed):,} rows × {len(df_processed.columns)} columns")
                print(f"  📝 Operations: {preprocess_summary.get('total_operations', 0)}")

            # ═══════════════════════════════════════════════════════════════════
            # STAGE 3: INTELLIGENCE & ANALYSIS
            # ═══════════════════════════════════════════════════════════════════
            if verbose:
                print("\n" + "─" * 60)
                print("  🧠 STAGE 3: Analyzing Your Data")
                print("─" * 60)

            intelligence_results = self._run_intelligence(df_processed, target, show_plots)
            results.update(intelligence_results)
            results['stages_completed'].append('intelligence')

            if verbose:
                print(f"  ✅ Analysis complete!")
                print(f"  📊 Strong correlations found: {intelligence_results.get('n_strong_correlations', 0)}")
                print(f"  🤖 Recommended models: {intelligence_results.get('n_recommended_models', 0)}")

            # ═══════════════════════════════════════════════════════════════════
            # STAGE 4: AUTO ML (If Enabled)
            # ═══════════════════════════════════════════════════════════════════
            if self.config.ml_builder.enable_auto_ml:
                if verbose:
                    print("\n" + "─" * 60)
                    print("  🔥 STAGE 4: Building ML Models (Auto ML)")
                    print("─" * 60)

                ml_results = self._build_models(df_processed, target)
                results.update(ml_results)
                results['stages_completed'].append('ml_building')

                if verbose:
                    best = ml_results.get('best_model', {})
                    print(f"  ✅ Best Model: {best.get('name', 'N/A')}")
                    print(f"  📈 Score: {best.get('score', 'N/A')}")
            else:
                # Still prepare train/test split
                if 'split' not in results:
                    from data_masterpiece_v2.intelligence.splitter import DataSplitter
                    splitter = DataSplitter(random_state=self.config.intelligence.random_state)
                    results['split'] = splitter.split(
                        df_processed,
                        target=target,
                        test_size=self.config.intelligence.test_size,
                        val_size=self.config.intelligence.val_size,
                        stratify=self.config.intelligence.stratify
                    )

            # ═══════════════════════════════════════════════════════════════════
            # STAGE 5: GENERATE REPORT
            # ═══════════════════════════════════════════════════════════════════
            if self.config.output.save_report:
                if verbose:
                    print("\n" + "─" * 60)
                    print("  📄 STAGE 5: Generating Beautiful Report")
                    print("─" * 60)

                report_path = self._generate_report(results)
                results['report_path'] = report_path
                results['stages_completed'].append('report')

                if verbose:
                    print(f"  ✅ Report saved: {report_path}")

            # ═══════════════════════════════════════════════════════════════════
            # STAGE 6: SAVE OUTPUTS
            # ═══════════════════════════════════════════════════════════════════
            if verbose:
                print("\n" + "─" * 60)
                print("  💾 STAGE 6: Saving Outputs")
                print("─" * 60)

            csv_path = self._save_outputs(df_processed, results)
            results['csv_path'] = csv_path

            if verbose and csv_path:
                print(f"  ✅ CSV saved: {csv_path}")

            # Calculate elapsed time
            elapsed = time.time() - start_time
            results['elapsed_time'] = elapsed
            results['elapsed_formatted'] = format_duration(elapsed)

            # ═══════════════════════════════════════════════════════════════════
            # FINAL SUMMARY
            # ═══════════════════════════════════════════════════════════════════
            self._print_final_summary(results)

            self._is_fitted = True
            self.data = results

            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            if verbose:
                import traceback
                print(f"\n  ❌ ERROR: {str(e)}")
                print(f"\n  Full traceback:")
                traceback.print_exc()
            results['error'] = str(e)
            results['stages_completed'].append('FAILED')
            return results

    def _load_data(self, source: Union[str, pd.DataFrame, Path]) -> pd.DataFrame:
        """Load data from various sources."""
        loader = DataLoader()

        if isinstance(source, pd.DataFrame):
            logger.info(f"Using provided DataFrame: {source.shape}")
            return source.copy()

        elif isinstance(source, (str, Path)):
            source = str(source)

            # Check if it's a URL
            if source.startswith('http://') or source.startswith('https://'):
                logger.info(f"Loading from URL: {source}")
                return loader.load_from_url(source)

            # Check if it's a file path
            elif os.path.isfile(source):
                logger.info(f"Loading from file: {source}")
                return loader.load(source)

            else:
                raise FileNotFoundError(f"File not found: {source}")

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target: str
    ) -> tuple:
        """Run preprocessing pipeline."""
        controller = PreprocessingController(config=self.config.preprocessing)
        df_processed = controller.run(df, target=target)
        summary = controller.get_summary()
        return df_processed, summary

    def _run_intelligence(
        self,
        df: pd.DataFrame,
        target: str,
        show_plots: bool = False
    ) -> Dict[str, Any]:
        """Run intelligence/analysis pipeline."""
        controller = IntelligenceController(
            config=self.config.intelligence,
            output_dir=self.config.output.plots_dir,
            show_plots=show_plots
        )
        results = controller.run(df, target=target)
        return results

    def _build_models(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Build ML models automatically."""
        builder = AutoMLBuilder(config=self.config.ml_builder)

        # Get train/test split from results if available
        split_result = self.data.get('split')

        results = builder.build(
            df=df,
            target=target,
            split_result=split_result
        )

        # Save models if configured
        if self.config.output.save_models and results.get('models'):
            model_dir = ensure_dir(self.config.output.models_dir)
            builder.save_models(results['models'], model_dir)
            results['models_dir'] = str(model_dir)

        return results

    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate animated HTML report."""
        generator = AnimatedReportGenerator(
            output_path=os.path.join(
                self.config.output.output_dir,
                self.config.output.report_filename
            )
        )

        report_path = generator.generate(results)
        return report_path

    def _save_outputs(
        self,
        df: pd.DataFrame,
        results: Dict[str, Any]
    ) -> Optional[str]:
        """Save processed data and models."""
        if not self.config.output.save_csv:
            return None

        csv_path = os.path.join(
            self.config.output.output_dir,
            self.config.output.csv_filename
        )

        ensure_dir(os.path.dirname(csv_path))
        df.to_csv(csv_path, index=False)

        return csv_path

    def _print_final_summary(self, results: Dict[str, Any]) -> None:
        """Print final pipeline summary."""
        summary = f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    🎉 PIPELINE COMPLETED! 🎉                              ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║   📊 DATA SUMMARY                                                         ║
    ║   ─────────────────────────────────────────────────────────────────────   ║
    ║   Raw Data:       {results.get('n_rows_raw', 'N/A'):>10,} rows × {results.get('n_cols_raw', 'N/A')} columns          ║
    ║   Processed:      {len(results.get('df_processed', pd.DataFrame())):>10,} rows × {len(results.get('df_processed', pd.DataFrame()).columns)} columns          ║
    ║   Target:         {results.get('target', 'N/A'):>50}  ║
    ║                                                                          ║
    ║   ⏱️  TIMING                                                                ║
    ║   ─────────────────────────────────────────────────────────────────────   ║
    ║   Total Time:     {results.get('elapsed_formatted', 'N/A'):>50}  ║
    ║                                                                          ║
    ║   📁 OUTPUT FILES                                                         ║
    ║   ─────────────────────────────────────────────────────────────────────   ║"""

        if results.get('csv_path'):
            summary += f"""
    ║   CSV:            {results.get('csv_path', 'N/A'):<50}  ║"""

        if results.get('report_path'):
            summary += f"""
    ║   Report:         {results.get('report_path', 'N/A'):<50}  ║"""

        if results.get('best_model'):
            best = results['best_model']
            summary += f"""
    ║                                                                          ║
    ║   🤖 BEST MODEL (Auto ML)                                                 ║
    ║   ─────────────────────────────────────────────────────────────────────   ║
    ║   Name:           {best.get('name', 'N/A'):<50}  ║
    ║   Score:          {best.get('score', 'N/A'):<50}  ║"""

        summary += """
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
        """

        print(summary)

    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def preprocess_only(
        self,
        source: Union[str, pd.DataFrame],
        target: str,
        save_csv: bool = True
    ) -> tuple:
        """
        Run only the preprocessing stage.

        Parameters
        ----------
        source : str or pd.DataFrame
            Data source.
        target : str
            Target column.
        save_csv : bool
            Save processed data?

        Returns
        -------
        tuple
            (df_processed, summary)
        """
        df_raw = self._load_data(source)
        df_processed, summary = self._preprocess_data(df_raw, target)

        if save_csv:
            self._save_outputs(df_processed, {'csv_path': None})

        return df_processed, summary

    def analyze_only(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """
        Run only the intelligence/analysis stage.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame.
        target : str
            Target column.

        Returns
        -------
        Dict
            Analysis results.
        """
        return self._run_intelligence(df, target, show_plots=False)

    def build_models_only(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """
        Run only the Auto ML stage.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame.
        target : str
            Target column.

        Returns
        -------
        Dict
            ML results with models and scores.
        """
        return self._build_models(df, target)

    def get_data(self) -> Dict[str, Any]:
        """
        Get cached data from the last run.

        Returns
        -------
        Dict
            Cached data dictionary.
        """
        if not self._is_fitted:
            logger.warning("Pipeline not yet run. No data to return.")
        return self.data

    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame from the last run.

        Returns
        -------
        pd.DataFrame
            Processed data.
        """
        if 'df_processed' in self.data:
            return self.data['df_processed']
        elif 'df_raw' in self.data:
            logger.warning("Only raw data available. Run pipeline first.")
            return self.data['df_raw']
        else:
            raise ValueError("No data available. Run pipeline first.")

    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the best trained model from the last run.

        Returns
        -------
        Dict or None
            Best model info and scores.
        """
        return self.data.get('best_model')

    def summary(self) -> None:
        """Print a summary of the last run."""
        if not self._is_fitted:
            print("Pipeline not yet run.")
            return

        results = self.data
        print("\n" + "=" * 60)
        print("  PIPELINE SUMMARY")
        print("=" * 60)
        print(f"\n  Target: {results.get('target', 'N/A')}")
        print(f"  Rows: {len(results.get('df_processed', pd.DataFrame())):,}")
        print(f"  Columns: {len(results.get('df_processed', pd.DataFrame()).columns)}")
        print(f"  Time: {results.get('elapsed_formatted', 'N/A')}")
        print(f"  Report: {results.get('report_path', 'N/A')}")

        if results.get('best_model'):
            print(f"\n  Best Model: {results['best_model'].get('name', 'N/A')}")
            print(f"  Score: {results['best_model'].get('score', 'N/A')}")

        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_analyze(
    source: Union[str, pd.DataFrame],
    target: str,
    auto_ml: bool = True
) -> Dict[str, Any]:
    """
    Quick analysis with minimal setup!

    This is the EASIEST way to use Data Masterpiece V2!

    Parameters
    ----------
    source : str or pd.DataFrame
        Your data file path or DataFrame.
    target : str
        Target column to predict.
    auto_ml : bool
        Build ML models automatically? Default: True

    Returns
    -------
    Dict
        Complete results dictionary.

    Examples
    --------
    >>> result = quick_analyze("titanic.csv", target="Survived")
    >>> print(result['best_model'])

    >>> result = quick_analyze(my_dataframe, target="price", auto_ml=True)
    """
    pipeline = MasterPipeline()
    pipeline.config.ml_builder.enable_auto_ml = auto_ml

    return pipeline.run(
        source=source,
        target=target,
        save_report=True,
        save_csv=True,
        build_models=auto_ml
    )


def load_and_preprocess(
    source: Union[str, pd.DataFrame],
    target: str,
    mode: str = "auto"
) -> pd.DataFrame:
    """
    Load and preprocess data, returning only the clean DataFrame.

    Perfect for when you want to do your own analysis!

    Parameters
    ----------
    source : str or pd.DataFrame
        Data source.
    target : str
        Target column.
    mode : str
        Preprocessing mode: "auto" or "manual".

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for ML!

    Examples
    --------
    >>> df_clean = load_and_preprocess("data.csv", target="survived")
    >>> # Now use df_clean with sklearn or your favorite ML library!
    """
    config = Config(mode=mode)
    pipeline = MasterPipeline(config=config)

    df_processed, _ = pipeline.preprocess_only(source, target)
    return df_processed


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """
    Run a demo of Data Masterpiece V2!

    This will create sample data and run it through the pipeline.
    """
    print("\n" + "=" * 60)
    print("  🧪 DATA MASTERPIECE V2 - DEMO")
    print("=" * 60 + "\n")

    # Create sample data
    np.random.seed(42)
    n = 1000

    data = {
        'id': range(1, n + 1),
        'age': np.random.randint(18, 80, n),
        'income': np.random.randint(20000, 150000, n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'married': np.random.choice(['Yes', 'No'], n),
        'purchased': (np.random.rand(n) * (data.get('age', np.zeros(n)) / 50 +
              data.get('income', np.zeros(n)) / 50000) > 0.8).astype(int)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Run pipeline
    print("Running pipeline on sample data...")
    print(f"Data shape: {df.shape}\n")

    result = quick_analyze(df, target="purchased", auto_ml=True)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print(f"\n  Check the output folder for your report and data!")
    print(f"  Report: {result.get('report_path', 'N/A')}")

    return result


if __name__ == "__main__":
    # Run demo
    demo()
