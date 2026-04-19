"""data_masterpiece_v3.intelligence.splitter
──────────────────────────────────────────
Data Splitter — creates Train / Validation / Test sets.
Saves everything as numpy arrays and CSV files for instant ML use.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ..utils.logger import get_logger

log = get_logger("DataSplitter")


class SplitResult:
    """Container for train/val/test split arrays."""
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 feature_names, scaler, split_info):
        self.X_train = X_train
        self.X_val   = X_val
        self.X_test  = X_test
        self.y_train = y_train
        self.y_val   = y_val
        self.y_test  = y_test
        self.feature_names = feature_names
        self.scaler  = scaler
        self.split_info = split_info


class DataSplitter:
    """Splits clean data into train/val/test and saves ML-ready files."""

    def __init__(self, test_size=0.20, val_size=0.10, stratify=True, output_dir="output/ml_ready"):
        self.test_size  = test_size
        self.val_size   = val_size
        self.stratify   = stratify
        self.output_dir = output_dir

    def run(self, df: pd.DataFrame, target: str) -> SplitResult:
        log.info(f"✂️  DataSplitter: target='{target}'")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        X = df.drop(columns=[target])
        y = df[target]

        feature_names = list(X.columns)

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratify only for classification
        strat_y = y if (self.stratify and y.nunique() < 20) else None

        # First split: train+val vs test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X_scaled, y.values,
            test_size=self.test_size,
            random_state=42,
            stratify=strat_y,
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = X_tv, np.array([]), y_tv, np.array([])
        if self.val_size > 0:
            val_ratio = self.val_size / (1 - self.test_size)
            strat_tv = y_tv if (self.stratify and pd.Series(y_tv).nunique() < 20) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv,
                test_size=val_ratio,
                random_state=42,
                stratify=strat_tv,
            )

        split_info = {
            "train_rows": len(X_train),
            "val_rows":   len(X_val),
            "test_rows":  len(X_test),
            "n_features": len(feature_names),
            "target":     target,
        }

        result = SplitResult(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names, scaler, split_info
        )

        self._save(result, df, target)
        log.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return result

    def _save(self, result: SplitResult, df: pd.DataFrame, target: str):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(f"{self.output_dir}/X_train.npy", result.X_train)
        np.save(f"{self.output_dir}/X_test.npy",  result.X_test)
        np.save(f"{self.output_dir}/y_train.npy", result.y_train)
        np.save(f"{self.output_dir}/y_test.npy",  result.y_test)
        if len(result.X_val) > 0:
            np.save(f"{self.output_dir}/X_val.npy", result.X_val)
            np.save(f"{self.output_dir}/y_val.npy", result.y_val)

        with open(f"{self.output_dir}/feature_names.txt", "w") as f:
            f.write("\n".join(result.feature_names))

        with open(f"{self.output_dir}/scaler.pkl", "wb") as f:
            pickle.dump(result.scaler, f)

        meta = {
            **result.split_info,
            "feature_names": result.feature_names,
            "scale_method": "minmax",
        }
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save CSV splits
        feature_df = pd.DataFrame(result.X_train, columns=result.feature_names)
        feature_df[target] = result.y_train
        feature_df.to_csv(f"{self.output_dir}/train.csv", index=False)

        if len(result.X_val) > 0:
            val_df = pd.DataFrame(result.X_val, columns=result.feature_names)
            val_df[target] = result.y_val
            val_df.to_csv(f"{self.output_dir}/val.csv", index=False)

        test_df = pd.DataFrame(result.X_test, columns=result.feature_names)
        test_df[target] = result.y_test
        test_df.to_csv(f"{self.output_dir}/test.csv", index=False)

        # PyTorch dataset helper
        self._write_pytorch_dataset(result.feature_names, target)

        log.info(f"  ML-ready files saved → {self.output_dir}/")

    def _write_pytorch_dataset(self, feature_names, target):
        code = f'''"""
Auto-generated PyTorch Dataset for data_masterpiece_v3 output.
Usage:
    from ml_ready.pytorch_dataset import MasterDataset
    train_ds = MasterDataset("X_train.npy", "y_train.npy")
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class MasterDataset(Dataset):
    def __init__(self, X_path: str, y_path: str):
        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Feature names: {feature_names}
# Target column: {target}
# Number of features: {len(feature_names)}
'''
        with open(f"{self.output_dir}/pytorch_dataset.py", "w") as f:
            f.write(code)
