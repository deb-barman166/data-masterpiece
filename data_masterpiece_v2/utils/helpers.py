"""
data_masterpiece_v2.utils.helpers - Helper Functions

This module provides utility functions used throughout Data Masterpiece V2.
Includes file operations, data manipulation helpers, formatting functions, and more.

Functions:
    ensure_dir() - Ensure a directory exists
    format_bytes() - Format bytes to human readable string
    format_duration() - Format seconds to human readable string
    print_progress() - Print a progress bar
    safe_divide() - Safe division with default value
    detect_task_type() - Detect ML task type from target column
    infer_problem_type() - Infer problem type from data
"""

from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ==============================================================================
# File System Helpers
# ==============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists.

    Returns
    -------
    Path
        Path object of the directory.

    Examples
    --------
    >>> ensure_dir("output/reports")
    PosixPath('output/reports')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    algorithm : str
        Hash algorithm to use (md5, sha1, sha256).

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    hash_func = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str
        Glob pattern to match.
    recursive : bool
        Whether to search recursively.

    Returns
    -------
    List[Path]
        List of matching file paths.
    """
    directory = Path(directory)

    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


def get_relative_path(filepath: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Get relative path from a base directory.

    Parameters
    ----------
    filepath : str or Path
        File path.
    base : str or Path
        Base directory.

    Returns
    -------
    Path
        Relative path.
    """
    return Path(filepath).relative_to(base)


# ==============================================================================
# DataFrame Helpers
# ==============================================================================

def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero or invalid.

    Parameters
    ----------
    numerator : Any
        Numerator value.
    denominator : Any
        Denominator value.
    default : float
        Default value to return on error.

    Returns
    -------
    float
        Result of division or default value.
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the type of a pandas Series column.

    Parameters
    ----------
    series : pd.Series
        Data series to analyze.

    Returns
    -------
    str
        Column type: 'numeric', 'categorical', 'datetime', 'boolean', 'text', 'id', 'unknown'
    """
    # Check for missing values
    non_null = series.dropna()

    if len(non_null) == 0:
        return 'empty'

    # Check dtype
    dtype = series.dtype

    # Numeric
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if it's likely an ID column
        if is_likely_id_column(series):
            return 'id'
        return 'numeric'

    # Boolean
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'

    # String/Object
    if dtype == 'object' or dtype.name == 'string':

        # Check for datetime strings
        if looks_like_datetime(non_null):
            return 'datetime'

        # Check cardinality for categorical vs text
        unique_ratio = non_null.nunique() / len(non_null)

        if unique_ratio < 0.5:  # Low unique ratio suggests categorical
            return 'categorical'
        elif non_null.str.len().mean() > 100:  # Long strings suggest text
            return 'text'
        else:
            return 'categorical'

    # Check for categorical (low cardinality)
    if hasattr(dtype, 'categories'):
        return 'categorical'

    return 'unknown'


def is_likely_id_column(series: pd.Series) -> bool:
    """
    Check if a numeric column is likely an ID column.

    Parameters
    ----------
    series : pd.Series
        Data series to check.

    Returns
    -------
    bool
        True if column appears to be an ID column.
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    # Check for sequential patterns
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Check if values are sequential
        unique_vals = sorted(non_null.unique())

        # ID columns often have high cardinality
        if len(unique_vals) < len(non_null) * 0.1:
            return False  # Too few unique values

        # Check for patterns like 1, 2, 3, ... or UUID patterns
        if len(unique_vals) > 10:
            # Check for roughly sequential
            differences = np.diff(unique_vals[:100])
            if len(differences) > 0 and np.std(differences) / (np.mean(differences) + 1e-10) < 0.1:
                return True

            # Check for large gaps (random IDs)
            if unique_vals[-1] > len(unique_vals) * 2:
                return True

    # Check column name
    name_lower = str(series.name).lower()
    id_patterns = ['id', 'uuid', 'guid', 'key', 'index', 'idx', 'pk', 'sk']

    for pattern in id_patterns:
        if pattern in name_lower:
            return True

    return False


def looks_like_datetime(values: pd.Series) -> bool:
    """
    Check if string values look like dates or times.

    Parameters
    ----------
    values : pd.Series
        String values to check.

    Returns
    -------
    bool
        True if values look like dates/times.
    """
    # Common date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
        r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
        r'\d{4}/\d{2}/\d{2}',  # 2024/01/15
        r'\d{2}-\d{2}-\d{4}',  # 15-01-2024
    ]

    sample = values.astype(str).head(10)

    for pattern in date_patterns:
        if any(re.match(pattern, str(v)) for v in sample):
            return True

    return False


def detect_task_type(target: pd.Series) -> str:
    """
    Detect the ML task type from a target column.

    Parameters
    ----------
    target : pd.Series
        Target column data.

    Returns
    -------
    str
        Task type: 'classification_binary', 'classification_multiclass', 'regression', 'unknown'
    """
    dtype = target.dtype

    # Numeric target
    if pd.api.types.is_numeric_dtype(dtype):
        unique_count = target.nunique()

        if unique_count == 2:
            return 'classification_binary'
        elif unique_count <= 10:
            return 'classification_multiclass'
        else:
            return 'regression'

    # Boolean target
    if pd.api.types.is_bool_dtype(dtype):
        return 'classification_binary'

    # Object/Categorical target
    if dtype == 'object' or hasattr(dtype, 'categories'):
        unique_count = target.nunique()

        if unique_count == 2:
            return 'classification_binary'
        elif unique_count <= 10:
            return 'classification_multiclass'
        else:
            # High cardinality categorical - treat as regression or use encoding
            return 'regression'

    return 'unknown'


def infer_problem_type(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Infer the type of ML problem from dataframe and target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target : str
        Target column name.

    Returns
    -------
    Dict
        Dictionary with problem type and metadata.
    """
    result = {
        'task_type': 'unknown',
        'is_time_series': False,
        'has_text': False,
        'has_images': False,
        'severity': 'unknown',
        'recommendations': []
    }

    if target not in df.columns:
        return result

    target_series = df[target]
    task_type = detect_task_type(target_series)

    if task_type == 'classification_binary':
        result['task_type'] = 'classification'
        result['severity'] = 'binary'
    elif task_type == 'classification_multiclass':
        result['task_type'] = 'classification'
        result['severity'] = 'multiclass'
    else:
        result['task_type'] = 'regression'
        result['severity'] = 'continuous'

    # Check for time series indicators
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0 or 'date' in target.lower() or 'time' in target.lower():
        result['is_time_series'] = True
        result['recommendations'].append("Consider time series specific models")

    # Check for text columns
    text_cols = [col for col in df.columns if detect_column_type(df[col]) == 'text']
    if len(text_cols) > 0:
        result['has_text'] = True
        result['recommendations'].append("Consider NLP preprocessing for text columns")

    return result


# ==============================================================================
# Formatting Helpers
# ==============================================================================

def format_bytes(bytes_num: Union[int, float]) -> str:
    """
    Format bytes to human readable string.

    Parameters
    ----------
    bytes_num : int or float
        Number of bytes.

    Returns
    -------
    str
        Formatted string (e.g., "1.5 GB").

    Examples
    --------
    >>> format_bytes(1536)
    '1.5 KB'
    >>> format_bytes(1073741824)
    '1.0 GB'
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_num)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"

    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float, detailed: bool = False) -> str:
    """
    Format duration in seconds to human readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.
    detailed : bool
        Whether to include detailed breakdown.

    Returns
    -------
    str
        Formatted duration string.

    Examples
    --------
    >>> format_duration(65)
    '1m 5s'
    >>> format_duration(3665)
    '1h 1m 5s'
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if detailed:
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or len(parts) == 0:
            parts.append(f"{secs}s")
        return " ".join(parts)
    else:
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def format_number(num: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number with appropriate precision.

    Parameters
    ----------
    num : int or float
        Number to format.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted number string.
    """
    if isinstance(num, int):
        return f"{num:,}"

    if abs(num) >= 1e6:
        return f"{num / 1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage.

    Parameters
    ----------
    value : float
        Value between 0 and 1 (or can be > 1 for raw percentages).
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted percentage string.
    """
    # If value is already a percentage (e.g., 50 instead of 0.5)
    if abs(value) > 1:
        return f"{value:.{decimals}f}%"

    return f"{value * 100:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Parameters
    ----------
    text : str
        String to truncate.
    max_length : int
        Maximum length.
    suffix : str
        Suffix to add when truncated.

    Returns
    -------
    str
        Truncated string.
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


# ==============================================================================
# Progress and Display Helpers
# ==============================================================================

def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
    fill: str = "█",
    empty: str = "░"
) -> None:
    """
    Print a progress bar to the console.

    Parameters
    ----------
    current : int
        Current progress.
    total : int
        Total steps.
    prefix : str
        Text before the progress bar.
    suffix : str
        Text after the progress bar.
    length : int
        Width of the progress bar.
    fill : str
        Character for filled portion.
    empty : str
        Character for empty portion.

    Examples
    --------
    >>> for i in range(100):
    ...     print_progress(i + 1, 100, prefix="Progress:", suffix="Complete")
    ...     time.sleep(0.1)
    """
    if total == 0:
        return

    percent = current / total
    filled = int(length * percent)
    bar = fill * filled + empty * (length - filled)

    print(f"\r{prefix} |{bar}| {percent:.1%} {suffix}", end="", flush=True)

    if current >= total:
        print()  # New line when complete


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    max_col_width: int = 20
) -> None:
    """
    Print a nicely formatted table to the console.

    Parameters
    ----------
    headers : List[str]
        Column headers.
    rows : List[List[Any]]
        Table rows.
    max_col_width : int
        Maximum column width.
    """
    # Truncate headers
    headers = [str(h)[:max_col_width] for h in headers]

    # Calculate column widths
    col_widths = [len(h) for h in headers]

    for row in rows:
        row = [str(cell)[:max_col_width] for cell in row]
        col_widths = [max(w, len(cell)) for w, cell in zip(col_widths, row)]

    # Create format string
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    # Print table
    print()
    print(fmt.format(*headers))
    print(sep)

    for row in rows:
        row = [str(cell)[:max_col_width] for cell in row]
        print(fmt.format(*row))


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a section header.

    Parameters
    ----------
    title : str
        Section title.
    char : str
        Character to use for border.
    width : int
        Total width.
    """
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


# ==============================================================================
# Data Validation Helpers
# ==============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate a dataframe for common issues.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    required_columns : List[str], optional
        List of required column names.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of issues)
    """
    issues = []

    if df is None or df.empty:
        issues.append("Dataframe is empty")
        return False, issues

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            issues.append(f"Missing required columns: {', '.join(missing)}")

    # Check for all-null columns
    all_null = df.columns[df.isna().all()].tolist()
    if all_null:
        issues.append(f"All-null columns: {', '.join(all_null)}")

    # Check for constant columns
    constant = df.columns[df.nunique() <= 1].tolist()
    if constant:
        issues.append(f"Constant columns: {', '.join(constant)}")

    return len(issues) == 0, issues


def clean_column_name(name: str) -> str:
    """
    Clean a column name for use in ML pipelines.

    Parameters
    ----------
    name : str
        Original column name.

    Returns
    -------
    str
        Cleaned column name.
    """
    # Replace special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    # Replace multiple underscores with single
    name = re.sub(r'_+', '_', name)

    # Make lowercase
    name = name.lower()

    # Ensure not empty
    if not name:
        name = "column"

    return name


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names.
    """
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df
