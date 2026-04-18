"""
data_masterpiece_v2.preprocessing.controller - Preprocessing Controller

This module orchestrates all preprocessing operations in sequence.

Operations:
    1. Data Cleaning - Remove duplicates, constant columns, etc.
    2. Type Conversion - Convert column types appropriately
    3. Missing Values - Handle missing data
    4. Feature Encoding - Encode categorical variables
    5. Feature Scaling - Scale numerical features
    6. Feature Engineering - Create new features

Usage:
    >>> from data_masterpiece_v2.preprocessing import PreprocessingController
    >>> from data_masterpiece_v2.config import PreprocessingConfig
    >>> config = PreprocessingConfig()
    >>> controller = PreprocessingController(config)
    >>> df_clean = controller.run(df, target="survived")
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from data_masterpiece_v2.config import PreprocessingConfig
from data_masterpiece_v2.utils.logger import get_logger, format_duration
from data_masterpiece_v2.utils.helpers import (
    detect_column_type,
    is_likely_id_column,
    clean_column_name,
    ensure_dir
)

logger = get_logger("PreprocessingController")


class PreprocessingController:
    """
    Orchestrates all preprocessing operations in a configurable pipeline.

    This controller manages the sequence of preprocessing operations and
    tracks statistics for each operation.

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object with preprocessing options.

    Attributes
    ----------
    summary : Dict
        Summary statistics of preprocessing operations.

    Examples
    --------
    Basic usage:

        >>> controller = PreprocessingController()
        >>> df_clean = controller.run(df, target="survived")
        >>> print(controller.summary)

    With custom config:

        >>> config = PreprocessingConfig()
        >>> config.drop_duplicates = True
        >>> config.missing_strategy = "median"
        >>> controller = PreprocessingController(config)
        >>> df_clean = controller.run(df, target="survived")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessing controller."""
        self.config = config or PreprocessingConfig()
        self.summary: Dict[str, Any] = {}
        self._operation_logs: List[Dict] = []

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        column_configs: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to preprocess.
        target : str
            Target column name (won't be modified).
        column_configs : Dict, optional
            Column-specific configurations for manual mode.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        start_time = time.time()
        original_shape = df.shape

        logger.info(f"Starting preprocessing pipeline...")
        logger.info(f"Input shape: {df.shape}")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Clean column names
        df = self._clean_column_names(df)

        # Verify target exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        # Log column types
        self._log_column_types(df)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: DATA CLEANING
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_clean(df, target)
        self._log_operation("cleaning", stats)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: TYPE CONVERSION
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_type_conversion(df)
        self._log_operation("type_conversion", stats)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: MISSING VALUES
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_missing_values(df, target)
        self._log_operation("missing_values", stats)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: FEATURE ENCODING
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_encoding(df, target, column_configs)
        self._log_operation("encoding", stats)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: FEATURE SCALING
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_scaling(df, target)
        self._log_operation("scaling", stats)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: FEATURE ENGINEERING
        # ═══════════════════════════════════════════════════════════════════════
        df, stats = self._step_feature_engineering(df, target)
        self._log_operation("feature_engineering", stats)

        # Build summary
        elapsed = time.time() - start_time
        self._build_summary(original_shape, df.shape, elapsed)

        logger.info(f"Preprocessing complete! Shape: {df.shape}")
        logger.info(f"Total time: {format_duration(elapsed)}")

        return df

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for ML compatibility."""
        new_columns = [clean_column_name(col) for col in df.columns]

        # Handle duplicate column names
        seen = {}
        for i, col in enumerate(new_columns):
            if col in seen:
                seen[col] += 1
                new_columns[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0

        df.columns = new_columns
        return df

    def _log_column_types(self, df: pd.DataFrame) -> None:
        """Log the types of columns detected."""
        type_counts = {}
        for col in df.columns:
            col_type = detect_column_type(df[col])
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        logger.info(f"Column types detected: {type_counts}")

    def _log_operation(self, operation: str, stats: Dict) -> None:
        """Log an operation's statistics."""
        self._operation_logs.append({
            'operation': operation,
            'stats': stats
        })

    def _build_summary(
        self,
        original_shape: Tuple[int, int],
        final_shape: Tuple[int, int],
        elapsed: float
    ) -> None:
        """Build the preprocessing summary."""
        self.summary = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'rows_removed': original_shape[0] - final_shape[0],
            'cols_removed': original_shape[1] - final_shape[1],
            'total_operations': len(self._operation_logs),
            'operations': self._operation_logs,
            'elapsed_seconds': elapsed
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get the preprocessing summary."""
        return self.summary

    # ═══════════════════════════════════════════════════════════════════════════
    # PREPROCESSING STEPS
    # ═══════════════════════════════════════════════════════════════════════════

    def _step_clean(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict]:
        """Step 1: Data cleaning."""
        stats = {
            'duplicates_removed': 0,
            'constant_cols_removed': [],
            'id_cols_removed': [],
            'cols_dropped': []
        }

        original_len = len(df)

        # Remove duplicates
        if self.config.drop_duplicates:
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                df = df.drop_duplicates()
                stats['duplicates_removed'] = n_duplicates
                logger.info(f"Removed {n_duplicates} duplicate rows")

        # Remove constant columns
        if self.config.drop_constant_columns:
            for col in df.columns:
                if col == target:
                    continue
                if df[col].nunique() <= 1:
                    stats['constant_cols_removed'].append(col)
                    df = df.drop(columns=[col])

        # Remove ID columns
        if self.config.drop_id_columns:
            for col in df.columns:
                if col == target:
                    continue
                if is_likely_id_column(df[col]):
                    stats['id_cols_removed'].append(col)
                    df = df.drop(columns=[col])
                    logger.info(f"Dropped ID column: {col}")

        # Remove high-null columns
        null_threshold = self.config.null_drop_threshold
        for col in df.columns:
            if col == target:
                continue
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > null_threshold:
                stats['cols_dropped'].append(col)
                df = df.drop(columns=[col])
                logger.info(f"Dropped column with {null_pct:.1%} nulls: {col}")

        stats['rows_removed'] = original_len - len(df)

        return df, stats

    def _step_type_conversion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Step 2: Type conversion."""
        stats = {
            'conversions': [],
            'datetime_extracted': []
        }

        for col in df.columns:
            col_type = detect_column_type(df[col])

            if col_type == 'datetime':
                try:
                    # Extract datetime features
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        stats['datetime_extracted'].append(col)
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")

        return df, stats

    def _step_missing_values(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Step 3: Handle missing values."""
        stats = {
            'strategy': self.config.missing_strategy,
            'columns_imputed': {},
            'rows_with_missing': int(df.isna().any(axis=1).sum())
        }

        if df.isna().sum().sum() == 0:
            return df, stats

        # Auto strategy selection
        if self.config.missing_strategy == 'auto':
            strategy = self._select_missing_strategy(df)
        else:
            strategy = self.config.missing_strategy

        # Apply imputation
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            null_count = df[col].isna().sum()

            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    stats['columns_imputed'][col] = {'strategy': 'mean', 'count': null_count}

            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    stats['columns_imputed'][col] = {'strategy': 'median', 'count': null_count}

            elif strategy == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    stats['columns_imputed'][col] = {'strategy': 'mode', 'count': null_count}

            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                stats['columns_imputed'][col] = {'strategy': 'drop', 'count': null_count}

            elif strategy == 'forward_fill':
                df[col] = df[col].ffill().bfill()
                stats['columns_imputed'][col] = {'strategy': 'ffill', 'count': null_count}

            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].interpolate(method='linear')
                    stats['columns_imputed'][col] = {'strategy': 'interpolate', 'count': null_count}

        logger.info(f"Missing value strategy: {strategy}")
        logger.info(f"Imputed {len(stats['columns_imputed'])} columns")

        return df, stats

    def _select_missing_strategy(self, df: pd.DataFrame) -> str:
        """Select the best missing value strategy based on data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # If most columns are numeric, use median
        if len(numeric_cols) > len(categorical_cols):
            return 'median'
        else:
            return 'mode'

    def _step_encoding(
        self,
        df: pd.DataFrame,
        target: str,
        column_configs: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Step 4: Feature encoding."""
        stats = {
            'columns_encoded': [],
            'encoding_method': {}
        }

        column_configs = column_configs or {}

        for col in df.columns:
            if col == target:
                continue

            # Check for column-specific config
            col_config = column_configs.get(col, {})
            encoding_strategy = col_config.get('encoding', self.config.encoding_strategy)

            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Get column info
            unique_count = df[col].nunique()
            total_count = len(df)

            # Auto-select encoding strategy
            if encoding_strategy == 'auto':
                if unique_count <= 2:
                    encoding_strategy = 'binary'
                elif unique_count <= self.config.low_cardinality_threshold:
                    encoding_strategy = 'onehot'
                elif unique_count <= self.config.high_cardinality_threshold:
                    encoding_strategy = 'label'
                else:
                    encoding_strategy = 'target'

            # Apply encoding
            if encoding_strategy == 'binary':
                df[col] = df[col].astype('category').cat.codes
                stats['columns_encoded'].append(col)
                stats['encoding_method'][col] = 'binary'

            elif encoding_strategy == 'onehot':
                if unique_count <= self.config.high_cardinality_threshold:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    stats['columns_encoded'].extend(dummies.columns.tolist())
                    stats['encoding_method'][col] = 'onehot'

            elif encoding_strategy == 'label':
                df[col] = df[col].astype('category').cat.codes
                stats['columns_encoded'].append(col)
                stats['encoding_method'][col] = 'label'

        return df, stats

    def _step_scaling(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict]:
        """Step 5: Feature scaling."""
        stats = {
            'columns_scaled': [],
            'method': self.config.scale_method
        }

        if not self.config.normalize and self.config.scale_method == 'auto':
            return df, stats

        # Get numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if not numeric_cols:
            return df, stats

        # Auto-select scaling method
        scale_method = self.config.scale_method
        if scale_method == 'auto':
            # Use robust scaling for data with outliers
            has_outliers = any(
                (df[col].std() / (df[col].mean() + 1e-10)) > 0.5
                for col in numeric_cols
            )
            scale_method = 'robust' if has_outliers else 'standard'

        # Apply scaling
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if scale_method == 'standard':
            scaler = StandardScaler()
        elif scale_method == 'minmax':
            scaler = MinMaxScaler()
        elif scale_method == 'robust':
            scaler = RobustScaler()
        else:
            return df, stats

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        stats['columns_scaled'] = numeric_cols
        stats['method'] = scale_method

        logger.info(f"Scaled {len(numeric_cols)} columns using {scale_method}")

        return df, stats

    def _step_feature_engineering(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Step 6: Feature engineering."""
        stats = {
            'features_created': [],
            'operations': []
        }

        # Statistical features
        if self.config.create_statistical_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if len(numeric_cols) >= 2:
                # Row-wise statistics
                df['_row_mean'] = df[numeric_cols].mean(axis=1)
                df['_row_std'] = df[numeric_cols].std(axis=1)
                df['_row_min'] = df[numeric_cols].min(axis=1)
                df['_row_max'] = df[numeric_cols].max(axis=1)

                stats['features_created'].extend([
                    '_row_mean', '_row_std', '_row_min', '_row_max'
                ])
                stats['operations'].append('row_statistics')

        # Interaction features
        if self.config.create_interaction_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if len(numeric_cols) >= 2:
                # Create a few interaction features
                for i, col1 in enumerate(numeric_cols[:5]):
                    for col2 in numeric_cols[i+1:6]:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        stats['features_created'].append(f'{col1}_x_{col2}')

                stats['operations'].append('interaction_features')

        # Polynomial features (limited)
        if self.config.create_polynomial_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)

            if len(numeric_cols) >= 1:
                # Only create squared features for first few columns
                for col in numeric_cols[:3]:
                    df[f'{col}_squared'] = df[col] ** 2
                    stats['features_created'].append(f'{col}_squared')

                stats['operations'].append('polynomial_features')

        logger.info(f"Created {len(stats['features_created'])} new features")

        return df, stats
