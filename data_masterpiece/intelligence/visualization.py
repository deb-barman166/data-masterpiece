"""
data_masterpiece.intelligence.visualization  --  VisualizationEngine

Generates publication-quality charts automatically:
  - Distribution histograms
  - Box plots
  - Scatter plots (feature vs target)
  - Correlation heatmap
  - Pair plots (top features)
  - Bar charts for categorical-like features

Uses only matplotlib (no seaborn dependency).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from data_masterpiece.utils.logger import get_logger

# -- font setup --
#fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/SimHei.ttf")
#fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
#plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -- colour palette --
_PALETTE = [
    "#6366f1", "#06b6d4", "#10b981", "#f59e0b",
    "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
    "#f97316", "#3b82f6",
]


class VisualizationEngine:
    """
    Auto-generate or manually generate charts for a dataset.

    Parameters
    ----------
    output_dir : Directory to save PNG chart files.
    show       : If True, call plt.show() after each chart (useful in notebooks).
    dpi        : Resolution for saved figures.
    """

    def __init__(
        self, output_dir: str = "output/plots", show: bool = False, dpi: int = 150,
    ):
        self.output_dir = output_dir
        self.show = show
        self.dpi = dpi
        self.log = get_logger("VisualizationEngine")
        os.makedirs(output_dir, exist_ok=True)

    # -- public API ------------------------------------------------------------

    def run_auto(
        self,
        df: pd.DataFrame,
        target: str,
        max_cols: int = 20,
    ) -> list[str]:
        """
        Auto-generate a comprehensive set of charts.

        Returns
        -------
        List of saved file paths.
        """
        saved: list[str] = []
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target
        ][:max_cols]

        if not num_cols:
            self.log.warning("No numeric columns to visualise.")
            return saved

        # 1. Correlation heatmap
        path = self._save_heatmap(df[num_cols + ([target] if target in df.columns else [])])
        if path:
            saved.append(path)

        # 2. Per-feature: distribution + scatter
        for i, col in enumerate(num_cols):
            color = _PALETTE[i % len(_PALETTE)]

            # histogram
            path = self._save_histogram(df[col], col, color)
            if path:
                saved.append(path)

            # box plot
            path = self._save_boxplot(df[col], col, color)
            if path:
                saved.append(path)

            # scatter vs target
            if target in df.columns:
                path = self._save_scatter(df, col, target, color)
                if path:
                    saved.append(path)

        # 3. Pair plot of top-5 features
        if len(num_cols) >= 2:
            top = num_cols[:5]
            path = self._save_pairplot(df, top, target)
            if path:
                saved.append(path)

        self.log.info(f"Generated {len(saved)} charts -> {self.output_dir}")
        return saved

    def run_manual(
        self,
        df: pd.DataFrame,
        graph_config: dict,
    ) -> list[str]:
        """
        Generate charts from a manual config dict.

        graph_config format:
            {
                "histograms": ["col1", "col2"],
                "boxplots": ["col1"],
                "scatter": [{"x": "feat1", "y": "feat2"}],
                "heatmap": true,
            }
        """
        saved: list[str] = []

        for col in graph_config.get("histograms", []):
            if col in df.columns:
                path = self._save_histogram(df[col], col, _PALETTE[0])
                if path:
                    saved.append(path)

        for col in graph_config.get("boxplots", []):
            if col in df.columns:
                path = self._save_boxplot(df[col], col, _PALETTE[1])
                if path:
                    saved.append(path)

        for sc in graph_config.get("scatter", []):
            x, y = sc["x"], sc["y"]
            if x in df.columns and y in df.columns:
                path = self._save_scatter(df, x, y, _PALETTE[2])
                if path:
                    saved.append(path)

        if graph_config.get("heatmap"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            path = self._save_heatmap(df[num_cols])
            if path:
                saved.append(path)

        self.log.info(f"Manual mode: {len(saved)} charts saved.")
        return saved

    # -- chart generators ------------------------------------------------------

    def _save_histogram(
        self, series: pd.Series, name: str, color: str,
    ) -> Optional[str]:
        clean = series.dropna()
        if len(clean) == 0:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        bins = min(50, max(10, int(np.sqrt(len(clean)))))
        ax.hist(clean, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution: {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")

        mean_val = clean.mean()
        ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Mean={mean_val:.2f}")
        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        return self._save(fig, f"hist_{name}.png")

    def _save_boxplot(
        self, series: pd.Series, name: str, color: str,
    ) -> Optional[str]:
        clean = series.dropna()
        if len(clean) == 0:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(clean, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_title(f"Box Plot: {name}", fontsize=13, fontweight="bold")
        ax.set_ylabel(name)
        plt.tight_layout()
        return self._save(fig, f"box_{name}.png")

    def _save_scatter(
        self, df: pd.DataFrame, x_col: str, y_col: str, color: str,
    ) -> Optional[str]:
        mask = df[x_col].notna() & df[y_col].notna()
        if mask.sum() < 5:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   c=color, alpha=0.4, s=15, edgecolors="none")
        ax.set_title(f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        return self._save(fig, f"scatter_{x_col}_vs_{y_col}.png")

    def _save_heatmap(self, df: pd.DataFrame) -> Optional[str]:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return None

        corr = num_df.corr()
        # limit to 30 cols max for readability
        if corr.shape[0] > 30:
            corr = corr.iloc[:30, :30]

        fig, ax = plt.subplots(figsize=(max(8, corr.shape[0] * 0.4), max(6, corr.shape[1] * 0.35)))
        cax = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        return self._save(fig, "correlation_heatmap.png")

    def _save_pairplot(
        self, df: pd.DataFrame, cols: list, target: str = None,
    ) -> Optional[str]:
        cols_present = [c for c in cols if c in df.columns]
        if len(cols_present) < 2:
            return None

        all_cols = cols_present
        if target and target in df.columns and target not in all_cols:
            all_cols = all_cols + [target]

        n = len(all_cols)
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        fig.suptitle("Pair Plot (Top Features)", fontsize=14, fontweight="bold", y=1.01)

        for i in range(n):
            for j in range(n):
                ax = axes[i][j] if n > 1 else axes
                x_c, y_c = all_cols[j], all_cols[i]

                if x_c == y_c:
                    ax.hist(df[x_c].dropna(), bins=20,
                            color=_PALETTE[0], alpha=0.7, edgecolor="white")
                else:
                    ax.scatter(df[x_c], df[y_c], s=8, alpha=0.3,
                               c=_PALETTE[(i + j) % len(_PALETTE)], edgecolors="none")

                if i == n - 1:
                    ax.set_xlabel(x_c, fontsize=8)
                if j == 0:
                    ax.set_ylabel(y_c, fontsize=8)
                ax.tick_params(labelsize=6)

        plt.tight_layout()
        return self._save(fig, "pairplot.png")

    # -- helpers ----------------------------------------------------------------

    def _save(self, fig, filename: str) -> Optional[str]:
        try:
            fpath = os.path.join(self.output_dir, filename)
            fig.savefig(fpath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return fpath
        except Exception as e:
            self.log.error(f"Failed to save {filename}: {e}")
            plt.close(fig)
            return None
