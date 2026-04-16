"""
data_masterpiece.preprocessing.core.loader  --  Universal data loader.

Supports: CSV, TSV, Excel, JSON, Parquet, local path, URL, DataFrame.
"""

from __future__ import annotations

import os
from typing import Union

import pandas as pd

from data_masterpiece.utils.logger import get_logger

log = get_logger("Loader")

_EXT_MAP = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet",
}


def load_data(
    source: Union[str, pd.DataFrame],
    sep: str = None,
    sheet_name=0,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load data from a DataFrame, local file path, or URL."""
    if isinstance(source, pd.DataFrame):
        log.info("Source is already a DataFrame -- pass-through.")
        return source.copy()

    if not isinstance(source, str):
        raise TypeError(f"source must be str or DataFrame, got {type(source)}")

    src = source.strip()
    log.info(f"Loading data from: {src}")

    _, ext = os.path.splitext(src.split("?")[0].lower())
    fmt = _EXT_MAP.get(ext, "csv")

    try:
        if fmt == "csv":
            actual_sep = sep if sep else ("\t" if ext == ".tsv" else ",")
            df = pd.read_csv(
                src, sep=actual_sep, encoding=encoding,
                low_memory=False, on_bad_lines="warn",
            )
        elif fmt == "excel":
            df = pd.read_excel(src, sheet_name=sheet_name)
        elif fmt == "json":
            df = pd.read_json(src, encoding=encoding)
        elif fmt == "parquet":
            df = pd.read_parquet(src)
        else:
            df = pd.read_csv(src, low_memory=False, on_bad_lines="warn")

        log.info(f"Loaded successfully -- shape: {df.shape}")
        return df
    except Exception as exc:
        log.error(f"Failed to load data: {exc}")
        raise
