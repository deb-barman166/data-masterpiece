"""
data_masterpiece_v3.automl.pytorch_models
──────────────────────────────────────────
PyTorch AutoML — builds and trains Neural Networks from scratch.
No pre-trained weights. No black-box AI. Pure math!

Architecture:
  Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
"""

from __future__ import annotations

import os
import json
import time
from typing import List

import numpy as np
from ..utils.logger import get_logger

log = get_logger("PyTorchAutoML")


class PyTorchAutoML:
    """
    Builds and trains a fully-connected neural network using PyTorch.
    Supports classification and regression tasks.
    """

    def __init__(
        self,
        task_type: str = "classification",
        hidden_sizes: List[int] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        output_dir: str = "output/models",
        dropout: float = 0.3,
    ):
        self.task_type   = task_type
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.output_dir  = output_dir
        self.dropout     = dropout
        self.history_    = []
        self.result_     = {}

    def run(self, X_train, X_test, y_train, y_test, X_val=None, y_val=None) -> dict:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            log.warning("PyTorch not installed. Skipping PyTorch AutoML.")
            return {"status": "skipped", "reason": "torch not installed"}

        log.info(f"🔥 PyTorchAutoML: task={self.task_type}, epochs={self.epochs}")
        log.info(f"   Architecture: {X_train.shape[1]} → {self.hidden_sizes} → output")

        t0 = time.time()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr_raw = y_train

        # Encode labels for classification
        if self.task_type == "classification":
            classes = np.unique(y_tr_raw)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_tr_enc = np.array([class_to_idx[c] for c in y_tr_raw])
            y_tr = torch.tensor(y_tr_enc, dtype=torch.long)
            n_out = len(classes)
        else:
            y_tr = torch.tensor(y_tr_raw, dtype=torch.float32).unsqueeze(1)
            n_out = 1
            classes = None

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Build model
        model = self._build_model(X_train.shape[1], self.hidden_sizes, n_out, nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        loss_fn = nn.CrossEntropyLoss() if self.task_type == "classification" else nn.MSELoss()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"   Trainable parameters: {n_params:,}")

        # Training loop
        history = []
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = model(X_batch)
                loss = loss_fn(out, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

            if (epoch + 1) % 10 == 0:
                log.info(f"   Epoch {epoch+1:3d}/{self.epochs} | Loss: {avg_loss:.6f}")

        self.history_ = history

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(X_test, dtype=torch.float32)
            out_te = model(X_te)

            if self.task_type == "classification":
                y_pred = out_te.argmax(dim=1).numpy()
                y_test_enc = np.array([class_to_idx.get(c, 0) for c in y_test])
                from sklearn.metrics import accuracy_score, f1_score
                test_acc = float(accuracy_score(y_test_enc, y_pred))
                test_f1  = float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0))
                metrics = {
                    "test_accuracy": round(test_acc, 4),
                    "test_f1_macro": round(test_f1, 4),
                }
            else:
                y_pred = out_te.squeeze().numpy()
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                metrics = {
                    "test_r2":   round(float(r2_score(y_test, y_pred)), 4),
                    "test_mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                }

        training_time = round(time.time() - t0, 2)
        log.info(f"   Training time: {training_time}s | Metrics: {metrics}")

        # Save model
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{self.output_dir}/pytorch_model.pt")
        model_info = {
            "name": "PyTorchMLP",
            "backend": "pytorch",
            "status": "success",
            "task_type": self.task_type,
            "architecture": {
                "input_size":    int(X_train.shape[1]),
                "hidden_sizes":  self.hidden_sizes,
                "output_size":   int(n_out),
                "n_params":      int(n_params),
                "dropout":       self.dropout,
            },
            "training_config": {
                "epochs":       self.epochs,
                "lr":           self.lr,
                "batch_size":   self.batch_size,
                "optimizer":    "Adam",
                "scheduler":    "CosineAnnealing",
                "loss_fn":      "CrossEntropy" if self.task_type == "classification" else "MSE",
            },
            "training_time": training_time,
            "final_loss": history[-1]["loss"] if history else None,
            "training_history": history,
            **metrics,
        }

        with open(f"{self.output_dir}/pytorch_results.json", "w") as f:
            json.dump(model_info, f, indent=2)

        self.result_ = model_info
        return model_info

    def _build_model(self, in_size, hidden_sizes, out_size, nn):
        layers = []
        prev_size = in_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, out_size))
        return nn.Sequential(*layers)
