"""
data_masterpiece_v3.utils.loader
──────────────────────────────────
Smart data loader — handles CSV, Excel, JSON, Parquet, URL, and DataFrame.
So you can just say: load_data("my_file.csv") and it works!
"""

from __future__ import annotations

import io
import pathlib
from typing import Union

import pandas as pd


def load_data(source: Union[str, pathlib.Path, pd.DataFrame, bytes]) -> pd.DataFrame:
    """
    Load data from any common source.

    Parameters
    ----------
    source : str, Path, DataFrame, or bytes
        - CSV / Excel / JSON / Parquet file path
        - HTTP/HTTPS URL pointing to any of the above
        - An already-loaded pandas DataFrame (returned as-is)
        - Raw bytes (CSV content)

    Returns
    -------
    pd.DataFrame
    """
    # Already a DataFrame → pass-through
    if isinstance(source, pd.DataFrame):
        return source.reset_index(drop=True)

    # Raw bytes → try CSV
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source))

    source = str(source)

    # URL
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)

    # Local file
    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif suffix == ".json":
        return pd.read_json(path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    else:
        # Try CSV as fallback
        try:
            return pd.read_csv(path)
        except Exception:
            raise ValueError(f"Unsupported file type: {suffix}")


def _load_url(url: str) -> pd.DataFrame:
    """Load a CSV / Excel / JSON / Parquet from a URL."""
    if ".csv" in url or "csv" in url.lower():
        return pd.read_csv(url)
    elif ".xls" in url:
        return pd.read_excel(url)
    elif ".json" in url:
        return pd.read_json(url)
    elif ".parquet" in url or ".pq" in url:
        return pd.read_parquet(url)
    else:
        return pd.read_csv(url)  # best-effort
