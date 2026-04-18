"""
data_masterpiece_v2.config - Unified Configuration System

This module provides the configuration system for Data Masterpiece V2.
Supports both AUTO mode (intelligent defaults) and MANUAL mode (full control).

Quick Start:
    # AUTO MODE - Everything is done automatically!
    config = Config(mode="auto")

    # MANUAL MODE - You control everything!
    config = Config(mode="manual")
    config.missing = {"strategy": "mean", "columns": {"age": "median"}}
    config.encoding = {"strategy": "onehot", "columns": ["category"]}

    # Load from JSON (Manual mode)
    config = Config.from_json("my_config.json")

JSON Configuration Example:
    {
        "mode": "manual",
        "preprocessing": {
            "drop_duplicates": true,
            "missing": {
                "strategy": "auto",
                "columns": {}
            },
            "encoding": {
                "strategy": "auto",
                "columns": {}
            }
        },
        "intelligence": {
            "outlier_method": "iqr",
            "outlier_strategy": "clip"
        },
        "ml_builder": {
            "enable_auto_ml": true,
            "task_type": "auto",
            "max_models": 5
        },
        "output": {
            "save_csv": true,
            "save_report": true,
            "save_models": true,
            "output_dir": "output"
        }
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    # Data cleaning
    drop_duplicates: bool = True
    drop_constant_columns: bool = True
    drop_id_columns: bool = True
    id_column_patterns: List[str] = field(default_factory=lambda: ["id", "uuid", "key", "index"])

    # Missing values
    null_drop_threshold: float = 0.6  # Drop columns with >60% nulls
    missing_strategy: str = "auto"  # auto, mean, median, mode, knn, drop
    fill_value: Any = None

    # Encoding
    encoding_strategy: str = "auto"  # auto, label, onehot, target, ordinal
    high_cardinality_threshold: int = 50
    low_cardinality_threshold: int = 10

    # Feature engineering
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    create_interaction_features: bool = False
    create_statistical_features: bool = True

    # Scaling
    scale_method: str = "auto"  # auto, standard, minmax, robust, none
    normalize: bool = False

    # Text processing
    normalize_text: bool = True
    text_lowercase: bool = True

    # DateTime features
    extract_datetime_features: bool = True
    datetime_include_time: bool = True

    # Column-specific overrides (for manual mode)
    column_configs: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PreprocessingConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IntelligenceConfig:
    """Configuration for intelligence (analysis) pipeline."""

    # Outlier detection
    outlier_method: str = "auto"  # auto, iqr, zscore, isolation_forest, none
    outlier_strategy: str = "clip"  # clip, drop, flag, impute, none
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    isolation_contamination: float = 0.05

    # Feature selection
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.9
    feature_selection_method: str = "auto"  # auto, variance, correlation, mutual_info, random_forest
    top_k_features: int = 0  # 0 = keep all
    always_keep_columns: List[str] = field(default_factory=list)

    # Visualization
    max_viz_columns: int = 20
    chart_types: List[str] = field(default_factory=lambda: ["histogram", "boxplot", "scatter", "heatmap", "pairplot"])
    figure_dpi: int = 150

    # Data splitting
    test_size: float = 0.2
    val_size: float = 0.0
    stratify: bool = True
    random_state: int = 42

    # Report
    generate_report: bool = True
    report_style: str = "animated"  # animated, simple, detailed

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IntelligenceConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MLBuilderConfig:
    """Configuration for Auto ML/DL Builder."""

    # General settings
    enable_auto_ml: bool = True
    task_type: str = "auto"  # auto, classification, regression, clustering
    max_models: int = 10
    max_trials: int = 50
    timeout_seconds: int = 600

    # Classification specific
    classification_type: str = "auto"  # auto, binary, multiclass
    cv_folds: int = 5

    # Model families to try
    model_families: List[str] = field(default_factory=lambda: [
        "linear", "tree", "ensemble", "neighbors", "svm", "neural_network"
    ])

    # Hyperparameter search
    search_strategy: str = "random"  # random, grid, bayesian
    optimization_metric: str = "auto"  # auto, accuracy, f1, precision, recall, roc_auc, rmse, mae

    # Model saving
    save_models: bool = True
    save_best_model: bool = True
    model_format: str = "joblib"  # joblib, pickle, onnx

    # Deep Learning (requires tensorflow/pytorch)
    enable_deep_learning: bool = False
    dl_epochs: int = 100
    dl_batch_size: int = 32
    dl_early_stopping: bool = True
    dl_validation_split: float = 0.2
    dl_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MLBuilderConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OutputConfig:
    """Configuration for output files."""

    # Paths
    output_dir: str = "output"
    csv_filename: str = "processed_data.csv"
    report_filename: str = "report.html"
    models_dir: str = "models"
    plots_dir: str = "plots"
    logs_dir: str = "logs"

    # What to save
    save_csv: bool = True
    save_report: bool = True
    save_plots: bool = True
    save_models: bool = True
    save_preprocessing_pipeline: bool = True

    # Report settings
    report_title: str = "Data Masterpiece Analysis Report"
    include_executive_summary: bool = True
    include_statistical_profile: bool = True
    include_correlation_analysis: bool = True
    include_model_recommendations: bool = True
    include_ml_results: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "OutputConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """
    Main configuration class for Data Masterpiece V2.

    This class holds all configuration for the entire pipeline including:
    - Preprocessing settings
    - Intelligence/Analysis settings
    - Auto ML/DL builder settings
    - Output settings

    Parameters
    ----------
    mode : str
        Operating mode: "auto" (intelligent defaults) or "manual" (full control)

    Examples
    --------
    >>> # Auto mode - everything is automatic!
    >>> config = Config(mode="auto")
    >>> config.preprocessing.drop_duplicates = True

    >>> # Manual mode - full control!
    >>> config = Config(mode="manual")
    >>> config.preprocessing.missing_strategy = "median"
    >>> config.ml_builder.enable_auto_ml = True
    >>> config.to_json("my_config.json")
    """

    # Operating mode
    mode: str = "auto"  # "auto" or "manual"

    # Sub-configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    ml_builder: MLBuilderConfig = field(default_factory=MLBuilderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_filename: str = "data_masterpiece.log"

    # Version
    version: str = "2.0.0"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode not in ["auto", "manual"]:
            raise ValueError(f"Mode must be 'auto' or 'manual', got '{self.mode}'")

        if self.mode == "auto":
            # In auto mode, enable ML builder by default
            self.ml_builder.enable_auto_ml = True
            self.intelligence.generate_report = True

    def to_dict(self) -> Dict:
        """Convert entire config to dictionary."""
        return {
            "mode": self.mode,
            "preprocessing": self.preprocessing.to_dict(),
            "intelligence": self.intelligence.to_dict(),
            "ml_builder": self.ml_builder.to_dict(),
            "output": self.output.to_dict(),
            "log_level": self.log_level,
            "version": self.version
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert config to JSON string and optionally save to file.

        Parameters
        ----------
        filepath : str, optional
            Path to save JSON file. If None, returns JSON string.

        Returns
        -------
        str
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=4, default=str)

        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        config.mode = data.get("mode", "auto")

        if "preprocessing" in data:
            config.preprocessing = PreprocessingConfig.from_dict(data["preprocessing"])
        if "intelligence" in data:
            config.intelligence = IntelligenceConfig.from_dict(data["intelligence"])
        if "ml_builder" in data:
            config.ml_builder = MLBuilderConfig.from_dict(data["ml_builder"])
        if "output" in data:
            config.output = OutputConfig.from_dict(data["output"])

        config.log_level = data.get("log_level", "INFO")
        config.version = data.get("version", "2.0.0")

        return config

    @classmethod
    def from_json(cls, filepath: str) -> "Config":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON configuration file.

        Returns
        -------
        Config
            Configuration object.

        Examples
        --------
        >>> config = Config.from_json("config.json")
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_auto_config(cls) -> "Config":
        """
        Create an auto-mode configuration with intelligent defaults.

        This is perfect for beginners - just load your data and go!

        Returns
        -------
        Config
            Auto-configured object.
        """
        config = cls(mode="auto")
        return config

    @classmethod
    def create_manual_config(
        cls,
        preprocessing_overrides: Optional[Dict] = None,
        intelligence_overrides: Optional[Dict] = None,
        ml_overrides: Optional[Dict] = None,
        output_overrides: Optional[Dict] = None
    ) -> "Config":
        """
        Create a manual-mode configuration with custom settings.

        Parameters
        ----------
        preprocessing_overrides : dict, optional
            Override preprocessing settings.
        intelligence_overrides : dict, optional
            Override intelligence settings.
        ml_overrides : dict, optional
            Override ML builder settings.
        output_overrides : dict, optional
            Override output settings.

        Returns
        -------
        Config
            Manual-configured object.

        Examples
        --------
        >>> config = Config.create_manual_config(
        ...     preprocessing_overrides={"drop_duplicates": False},
        ...     ml_overrides={"enable_auto_ml": True}
        ... )
        """
        config = cls(mode="manual")

        if preprocessing_overrides:
            for key, value in preprocessing_overrides.items():
                if hasattr(config.preprocessing, key):
                    setattr(config.preprocessing, key, value)

        if intelligence_overrides:
            for key, value in intelligence_overrides.items():
                if hasattr(config.intelligence, key):
                    setattr(config.intelligence, key, value)

        if ml_overrides:
            for key, value in ml_overrides.items():
                if hasattr(config.ml_builder, key):
                    setattr(config.ml_builder, key, value)

        if output_overrides:
            for key, value in output_overrides.items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)

        return config

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.

        Returns
        -------
        List[str]
            List of validation warnings (empty if all good).
        """
        warnings = []

        if self.mode == "manual":
            # Manual mode specific validations
            if self.preprocessing.missing_strategy == "auto":
                warnings.append("In manual mode, consider specifying explicit missing value strategy")

        if self.ml_builder.enable_auto_ml:
            if self.ml_builder.timeout_seconds < 60:
                warnings.append("ML timeout is very short, consider increasing to at least 60 seconds")

        return warnings

    def print_summary(self):
        """Print a nice summary of the configuration."""
        print("\n" + "=" * 60)
        print("  DATA MASTERPIECE V2 - CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"\n  Mode: {'AUTO ✨' if self.mode == 'auto' else 'MANUAL 🎛️'}")
        print(f"\n  Preprocessing:")
        print(f"    - Drop duplicates: {self.preprocessing.drop_duplicates}")
        print(f"    - Missing strategy: {self.preprocessing.missing_strategy}")
        print(f"    - Encoding strategy: {self.preprocessing.encoding_strategy}")
        print(f"    - Scale method: {self.preprocessing.scale_method}")

        print(f"\n  Intelligence:")
        print(f"    - Outlier method: {self.intelligence.outlier_method}")
        print(f"    - Feature selection: {self.intelligence.feature_selection_method}")
        print(f"    - Max viz columns: {self.intelligence.max_viz_columns}")

        print(f"\n  ML Builder:")
        print(f"    - Enable Auto ML: {self.ml_builder.enable_auto_ml} {'🔥' if self.ml_builder.enable_auto_ml else ''}")
        print(f"    - Task type: {self.ml_builder.task_type}")
        print(f"    - Max models: {self.ml_builder.max_models}")
        print(f"    - Deep Learning: {self.ml_builder.enable_deep_learning}")

        print(f"\n  Output:")
        print(f"    - Save CSV: {self.output.save_csv}")
        print(f"    - Save Report: {self.output.save_report}")
        print(f"    - Save Models: {self.output.save_models}")
        print(f"    - Output dir: {self.output.output_dir}")

        print("\n" + "=" * 60 + "\n")


# Convenience function for loading config
def load_config_from_json(filepath: str) -> Config:
    """
    Load configuration from JSON file.

    This is a convenience function that wraps Config.from_json().

    Parameters
    ----------
    filepath : str
        Path to JSON configuration file.

    Returns
    -------
    Config
        Configuration object.

    Examples
    --------
    >>> config = load_config_from_json("my_config.json")
    """
    return Config.from_json(filepath)
