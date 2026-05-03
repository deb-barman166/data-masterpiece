"""
loader.py — Smart Data Loader
==============================
Bedrock Truth: "Data comes in many formats and sources.
A good loader NEVER fails silently — it always tells you what went wrong and why."

Supports: CSV, Excel (.xlsx/.xls), JSON, TSV
Sources:  Local file path, HTTP/HTTPS URL
"""

import os
import io
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse


class DataLoader:
    """
    Universal data loader with automatic format detection.

    Strategy: Try the most likely format first based on extension,
    fall back gracefully through alternatives, never silently return wrong data.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".parquet": "parquet",
    }

    def __init__(self, source: str, sample_size: int = None):
        """
        Args:
            source: File path or URL string
            sample_size: If set, randomly sample this many rows after loading
        """
        self.source = source.strip()
        self.sample_size = sample_size
        self.metadata = {}
        self._df = None

    def load(self) -> pd.DataFrame:
        """Load data from source. Returns a clean DataFrame."""
        is_url = self._is_url(self.source)

        if is_url:
            df = self._load_from_url(self.source)
        else:
            df = self._load_from_file(self.source)

        # ── Basic cleaning ────────────────────────────────────
        df = self._clean(df)

        # ── Optional sampling ─────────────────────────────────
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

        # ── Store metadata ────────────────────────────────────
        self.metadata = self._compute_metadata(df, is_url)
        self._df = df
        return df

    # ── URL Detection ─────────────────────────────────────────

    def _is_url(self, source: str) -> bool:
        try:
            parsed = urlparse(source)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    # ── Load from URL ─────────────────────────────────────────

    def _load_from_url(self, url: str) -> pd.DataFrame:
        """Download file from URL and load into DataFrame."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; DataRelationshipPipeline/1.0)"
                )
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content = response.content
            content_type = response.headers.get("Content-Type", "")

            # Determine format from URL path or Content-Type
            url_path = urlparse(url).path.lower()
            ext = Path(url_path).suffix.lower()

            if ext in (".csv", ".tsv") or "text/csv" in content_type:
                sep = "\t" if ext == ".tsv" else ","
                return pd.read_csv(io.BytesIO(content), sep=sep)

            elif ext in (".xlsx", ".xls") or "spreadsheet" in content_type:
                return pd.read_excel(io.BytesIO(content))

            elif ext == ".json" or "json" in content_type:
                return pd.read_json(io.BytesIO(content))

            elif ext == ".parquet":
                return pd.read_parquet(io.BytesIO(content))

            else:
                # Attempt CSV (most common open-data format)
                try:
                    return pd.read_csv(io.BytesIO(content))
                except Exception:
                    raise ValueError(
                        f"Cannot determine file format from URL: {url}\n"
                        f"Content-Type: {content_type}\n"
                        f"Hint: Try providing a direct link to CSV/Excel/JSON file."
                    )

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to: {url}\nCheck your internet connection."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out for: {url}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP Error {e.response.status_code} for: {url}")

    # ── Load from File ────────────────────────────────────────

    def _load_from_file(self, path_str: str) -> pd.DataFrame:
        """Load from local file with automatic format detection."""
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path_str}\n"
                f"Current directory: {os.getcwd()}\n"
                f"Hint: Use absolute path or check the filename."
            )

        ext = path.suffix.lower()
        fmt = self.SUPPORTED_EXTENSIONS.get(ext)

        if fmt is None:
            raise ValueError(
                f"Unsupported file format: '{ext}'\n"
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        try:
            if fmt == "csv":
                # Auto-detect separator
                return self._read_csv_smart(path)
            elif fmt == "tsv":
                return pd.read_csv(path, sep="\t")
            elif fmt == "excel":
                return pd.read_excel(path)
            elif fmt == "json":
                return self._read_json_smart(path)
            elif fmt == "parquet":
                return pd.read_parquet(path)
        except Exception as e:
            raise ValueError(f"Error reading {path.name}: {e}")

    def _read_csv_smart(self, path: Path) -> pd.DataFrame:
        """
        Smart CSV reader: auto-detects separator and encoding.
        First Principle: CSV files are not always comma-separated.
        """
        # Sniff the separator
        with open(path, "r", errors="replace") as f:
            sample = f.read(2048)

        # Count candidate separators
        candidates = {",": sample.count(","), ";": sample.count(";"),
                      "\t": sample.count("\t"), "|": sample.count("|")}
        sep = max(candidates, key=candidates.get)

        # Try encodings in order
        for encoding in ["utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                return pd.read_csv(path, sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Cannot decode file: {path}. Try converting to UTF-8.")

    def _read_json_smart(self, path: Path) -> pd.DataFrame:
        """Try different JSON orientations."""
        for orient in [None, "records", "index", "columns", "values"]:
            try:
                return pd.read_json(path, orient=orient)
            except Exception:
                continue
        raise ValueError("Cannot parse JSON file. Ensure it is a valid JSON structure.")

    # ── Data Cleaning ─────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal, non-destructive cleaning.
        First Principle: Never alter data meaning. Only fix obvious structural issues.
        """
        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            # Replace "nan", "None", "null", "NA" strings with actual NaN
            df[col] = df[col].replace(["nan", "None", "null", "NULL", "NA", "N/A", ""], pd.NA)

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Drop completely empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Attempt numeric conversion for object columns that look numeric
        for col in df.select_dtypes(include="object").columns:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only convert if >70% of values are numeric
                if converted.notna().mean() > 0.70:
                    df[col] = converted
            except Exception:
                pass

        return df.reset_index(drop=True)

    # ── Metadata ──────────────────────────────────────────────

    def _compute_metadata(self, df: pd.DataFrame, is_url: bool) -> dict:
        """Compute metadata about the loaded dataset."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

        # Classify column types for analysis routing
        col_types = {}
        for col in df.columns:
            if col in numeric_cols:
                col_types[col] = "numeric"
            elif col in categorical_cols:
                unique_ratio = df[col].nunique() / max(len(df), 1)
                if df[col].nunique() <= 20:
                    col_types[col] = "categorical"
                else:
                    col_types[col] = "high_cardinality"
            elif col in datetime_cols:
                col_types[col] = "datetime"
            else:
                col_types[col] = "unknown"

        return {
            "source": self.source,
            "is_url": is_url,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "col_types": col_types,
            "missing_pct": (
                df.isnull().sum().sum() / max(df.size, 1) * 100
            ),
            "duplicated_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        }
