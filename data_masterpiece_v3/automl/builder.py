"""
data_masterpiece_v3.automl.builder
────────────────────────────────────
AutoML Orchestrator — runs all backends and consolidates results.
Asks for user permission before building any model.
"""

from __future__ import annotations

import json
import os
from typing import List

from .sklearn_models import SklearnAutoML
from .pytorch_models import PyTorchAutoML
from ..utils.logger import get_logger

log = get_logger("AutoMLBuilder")


class AutoMLBuilder:
    """
    Orchestrates the full AutoML pipeline:
    1. Asks user permission (if run_automl=True it's already granted)
    2. Runs sklearn models
    3. Optionally runs PyTorch MLP
    4. Combines and ranks all results
    """

    def __init__(self, config):
        self.config = config
        self.all_results: list = []
        self.combined_summary: dict = {}

    def run(self, split_result) -> dict:
        """
        Run AutoML on pre-split data.

        Parameters
        ----------
        split_result : SplitResult from DataSplitter
        """
        log.info("=" * 60)
        log.info("  🚀 AUTO ML BUILDER — STARTING")
        log.info("=" * 60)

        X_train = split_result.X_train
        X_test  = split_result.X_test
        y_train = split_result.y_train
        y_test  = split_result.y_test
        X_val   = split_result.X_val
        y_val   = split_result.y_val
        feature_names = split_result.feature_names

        backends: List[str] = self.config.automl_backends
        task = self.config.automl_task

        # ── Sklearn ──────────────────────────────────────────────────────────
        if "sklearn" in backends:
            log.info("  Running sklearn models ...")
            sklearn_aml = SklearnAutoML(
                task_type   = task,
                cv_folds    = self.config.automl_cv_folds,
                max_models  = self.config.automl_max_models,
                time_limit  = self.config.automl_time_limit,
                output_dir  = self.config.automl_output_dir,
            )
            sklearn_result = sklearn_aml.run(X_train, X_test, y_train, y_test, feature_names)
            self.all_results.extend(sklearn_result.get("models", []))
            detected_task = sklearn_result.get("task_type", "regression")
        else:
            detected_task = task if task != "auto" else "regression"
            sklearn_result = {}

        # ── PyTorch ──────────────────────────────────────────────────────────
        pytorch_result = {}
        if "pytorch" in backends:
            log.info("  Running PyTorch MLP ...")
            pt_aml = PyTorchAutoML(
                task_type    = detected_task,
                hidden_sizes = self.config.pytorch_hidden_sizes,
                epochs       = self.config.pytorch_epochs,
                lr           = self.config.pytorch_lr,
                batch_size   = self.config.pytorch_batch_size,
                output_dir   = self.config.automl_output_dir,
            )
            pytorch_result = pt_aml.run(X_train, X_test, y_train, y_test, X_val, y_val)
            if pytorch_result.get("status") == "success":
                self.all_results.append(pytorch_result)

        # ── Rank all models ───────────────────────────────────────────────────
        score_key = "test_accuracy" if detected_task == "classification" else "test_r2"
        ranked = sorted(
            [r for r in self.all_results if r.get("status") == "success"],
            key=lambda x: x.get(score_key, -999),
            reverse=True,
        )

        best = ranked[0] if ranked else {}

        self.combined_summary = {
            "task_type": detected_task,
            "score_metric": score_key,
            "total_models_trained": len(ranked),
            "best_model": {
                "name":    best.get("name", "N/A"),
                "backend": best.get("backend", "N/A"),
                "score":   round(best.get(score_key, 0), 4),
            },
            "leaderboard": [
                {
                    "rank":    i + 1,
                    "name":    r["name"],
                    "backend": r.get("backend", "sklearn"),
                    score_key: r.get(score_key, 0),
                    "training_time": r.get("training_time", 0),
                    "overfit_risk": r.get("overfit_risk", "unknown"),
                }
                for i, r in enumerate(ranked)
            ],
            "all_models": ranked,
            "sklearn_summary": sklearn_result,
            "pytorch_summary": pytorch_result,
        }

        # Save combined results
        os.makedirs(self.config.automl_output_dir, exist_ok=True)
        out_path = os.path.join(self.config.automl_output_dir, "automl_results.json")
        with open(out_path, "w") as f:
            json.dump(self.combined_summary, f, indent=2, default=str)

        log.info(f"  🏆 Best Model : {best.get('name','N/A')} ({score_key}={best.get(score_key,0):.4f})")
        log.info(f"  Results saved : {out_path}")
        return self.combined_summary
