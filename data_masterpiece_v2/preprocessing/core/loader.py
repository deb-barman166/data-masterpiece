"""
data_masterpiece_v2.preprocessing.core.loader - Data Loader

Handles loading data from various file formats and sources.

Supported formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    - URL (CSV download)
    - Clipboard

Usage:
    >>> from data_masterpiece_v2.preprocessing import DataLoader
    >>> loader = DataLoader()
    >>> df = loader.load("data.csv")
    >>> df = loader.load_from_url("https://example.com/data.csv")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests

from data_masterpiece_v2.utils.logger import get_logger

logger = get_logger("DataLoader")


class DataLoader:
    """
    Universal data loader supporting multiple file formats and sources.

    Features:
        • Auto-detect file format
        • Handle various encodings
        • Download from URL
        • Read from clipboard
        • Sample large datasets

    Parameters
    ----------
    encoding : str
        File encoding. Default: 'utf-8'
    low_memory : bool
        Use low memory mode for large files. Default: True
    sample_size : int, optional
        Sample N rows for initial exploration. None = load all.

    Examples
    --------
    Load CSV:

        >>> loader = DataLoader()
        >>> df = loader.load("data.csv")

    Load Excel:

        >>> df = loader.load("data.xlsx", sheet_name="Sheet1")

    Load from URL:

        >>> df = loader.load_from_url("https://example.com/data.csv")

    ═══════════════════════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        low_memory: bool = True,
        sample_size: Optional[int] = None
    ):
        """Initialize the data loader."""
        self.encoding = encoding
        self.low_memory = low_memory
        self.sample_size = sample_size
        self._file_info: Dict = {}

    def load(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file.

        Parameters
        ----------
        filepath : str
            Path to the data file.
        **kwargs
            Additional arguments passed to pandas read function.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file format is not supported.
        """
        filepath = str(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Detect file type
        ext = Path(filepath).suffix.lower()

        logger.info(f"Loading file: {filepath}")
        logger.info(f"Detected format: {ext}")

        # Load based on file type
        if ext == '.csv':
            df = self._load_csv(filepath, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            df = self._load_excel(filepath, **kwargs)
        elif ext == '.json':
            df = self._load_json(filepath, **kwargs)
        elif ext == '.parquet':
            df = self._load_parquet(filepath, **kwargs)
        elif ext in ['.tsv', '.txt']:
            df = self._load_tsv(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Store file info
        self._file_info = {
            'filepath': filepath,
            'format': ext,
            'rows': len(df),
            'columns': len(df.columns)
        }

        # Sample if requested
        if self.sample_size and len(df) > self.sample_size:
            logger.info(f"Sampling {self.sample_size} rows from {len(df)} total")
            df = df.sample(n=self.sample_size, random_state=42)

        logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        return df

    def _load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        try:
            # Try common encodings
            for enc in [self.encoding, 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=enc, low_memory=self.low_memory, **kwargs)
                    return df
                except UnicodeDecodeError:
                    continue

            # Fallback to reading with errors='ignore'
            df = pd.read_csv(filepath, encoding=self.encoding, low_memory=self.low_memory,
                           on_bad_lines='skip', **kwargs)
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def _load_excel(self, filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl', **kwargs)
            logger.info(f"Loaded sheet: {sheet_name if isinstance(sheet_name, str) else f'Sheet{sheet_name + 1}'}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise

    def _load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        try:
            df = pd.read_json(filepath, **kwargs)
            return df
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise

    def _load_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(filepath, **kwargs)
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise

    def _load_tsv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load TSV file."""
        try:
            df = pd.read_csv(filepath, sep='\t', encoding=self.encoding, low_memory=self.low_memory, **kwargs)
            return df
        except Exception as e:
            logger.error(f"Error loading TSV: {e}")
            raise

    def load_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a URL.

        Parameters
        ----------
        url : str
            URL to download data from.
        **kwargs
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            Downloaded and parsed DataFrame.
        """
        logger.info(f"Downloading from URL: {url}")

        try:
            # Send request
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine format from URL
            if 'csv' in url.lower() or url.endswith('.csv'):
                from io import StringIO
                df = pd.read_csv(StringIO(response.text), **kwargs)
            elif 'excel' in url.lower() or url.endswith(('.xlsx', '.xls')):
                from io import BytesIO
                df = pd.read_excel(BytesIO(response.content), **kwargs)
            elif url.endswith('.json'):
                from io import StringIO
                df = pd.read_json(StringIO(response.text), **kwargs)
            else:
                # Try CSV as default
                from io import StringIO
                df = pd.read_csv(StringIO(response.text), **kwargs)

            logger.info(f"Downloaded: {df.shape[0]} rows × {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"Error downloading from URL: {e}")
            raise

    def load_from_clipboard(self, **kwargs) -> pd.DataFrame:
        """
        Load data from clipboard.

        Expects tab-separated or comma-separated data.

        Returns
        -------
        pd.DataFrame
            Data from clipboard.
        """
        try:
            df = pd.read_clipboard(**kwargs)
            logger.info(f"Loaded from clipboard: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading from clipboard: {e}")
            raise ValueError("Could not read data from clipboard")

    def load_multiple(
        self,
        filepaths: List[str],
        concat_axis: int = 0
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple files.

        Parameters
        ----------
        filepaths : List[str]
            List of file paths.
        concat_axis : int
            Axis to concatenate along (0 = rows, 1 = columns).

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame.
        """
        dfs = []
        for filepath in filepaths:
            df = self.load(filepath)
            dfs.append(df)

        result = pd.concat(dfs, axis=concat_axis, ignore_index=(concat_axis == 0))
        logger.info(f"Concatenated {len(filepaths)} files: {result.shape[0]} rows × {result.shape[1]} columns")

        return result

    def get_file_info(self) -> Dict:
        """Get information about the last loaded file."""
        return self._file_info

    def peek(self, filepath: str, n: int = 5) -> pd.DataFrame:
        """
        Peek at the first n rows without loading the full file.

        Parameters
        ----------
        filepath : str
            Path to the file.
        n : int
            Number of rows to show.

        Returns
        -------
        pd.DataFrame
            First n rows.
        """
        ext = Path(filepath).suffix.lower()

        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=n, encoding=self.encoding)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, nrows=n)
        elif ext == '.json':
            df = pd.read_json(filepath, nrows=n)
        else:
            df = self.load(filepath).head(n)

        return df

    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed information about DataFrame columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.

        Returns
        -------
        pd.DataFrame
            Column information including type, nulls, unique values.
        """
        info = pd.DataFrame({
            'column': df.columns,
            'dtype': df.dtypes.values,
            'null_count': df.isnull().sum().values,
            'null_pct': (df.isnull().sum().values / len(df) * 100).round(2),
            'unique_count': df.nunique().values,
            'unique_pct': (df.nunique().values / len(df) * 100).round(2),
            'sample': [df[col].dropna().iloc[0] if df[col].notna().any() else None for col in df.columns]
        })

        return info
