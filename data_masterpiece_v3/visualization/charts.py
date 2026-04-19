"""
data_masterpiece_v3.visualization.charts
──────────────────────────────────────────
Professional Chart Engine — creates beautiful, publication-quality plots.

Charts generated:
  01 - Correlation Heatmap (full + annotated)
  02 - Target Distribution (histogram + KDE)
  03 - Feature Distributions (grid of histograms)
  04 - Box Plots (outlier view)
  05 - Scatter: each feature vs target
  06 - Violin Plots (distribution shape)
  07 - Feature Importance Bar Chart
  08 - Pair Plot (top features)
  09 - Missing Values Heatmap
  10 - Categorical Count Plots
  11 - Pie Charts (categorical breakdown)
  12 - Class Balance (for classification)
  13 - Relationship Matrix (multi-column)
  14 - Skewness Bar Chart
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ..utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("ChartEngine")

# ── Dark neon theme ────────────────────────────────────────────────────────────
DARK_BG   = "#0a0a0f"
CARD_BG   = "#12121f"
BORDER    = "#2a2a3e"
TEXT_PRI  = "#e0e0ff"
TEXT_SEC  = "#a0a0c0"
CYAN      = "#00f5ff"
PINK      = "#ff00a0"
GREEN     = "#39ff14"
ORANGE    = "#ff6b35"
PURPLE    = "#b967ff"
GOLD      = "#ffd700"
RED       = "#ff3366"
BLUE      = "#00bfff"

NEON_PALETTE = [CYAN, PINK, GREEN, ORANGE, PURPLE, GOLD, RED, BLUE,
                "#ff9f43", "#a29bfe", "#fd79a8", "#55efc4"]

NEON_CMAP = LinearSegmentedColormap.from_list(
    "neon_dark", [DARK_BG, PURPLE, CYAN, GREEN], N=256
)

def _apply_dark_style(fig, ax_list):
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, (list, tuple, np.ndarray)) else [ax_list]):
        if ax is None:
            continue
        try:
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=TEXT_SEC, labelsize=8)
            ax.xaxis.label.set_color(TEXT_PRI)
            ax.yaxis.label.set_color(TEXT_PRI)
            ax.title.set_color(CYAN)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
        except Exception:
            pass


class ChartEngine:
    """Generates all charts and saves them to output_dir."""

    def __init__(self, output_dir: str = "output/plots", dpi: int = 150, max_cols: int = 25):
        self.output_dir = output_dir
        self.dpi        = dpi
        self.max_cols   = max_cols
        os.makedirs(output_dir, exist_ok=True)
        self.generated: List[str] = []

    # ── Main entry ─────────────────────────────────────────────────────────────

    def generate_all(
        self,
        df: pd.DataFrame,
        target: str = None,
        feature_importance: dict = None,
        relationship_groups: List[List[str]] = None,
    ) -> List[str]:
        """Generate all charts. Returns list of saved file paths."""
        log.info("🎨 ChartEngine generating all charts ...")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target and target in num_cols:
            feature_cols = [c for c in num_cols if c != target]
        else:
            feature_cols = num_cols

        # Limit columns for performance
        feature_cols = feature_cols[:self.max_cols]

        self._chart_missing_heatmap(df)
        if len(num_cols) >= 2:
            self._chart_correlation_heatmap(df, num_cols[:self.max_cols])
        if target and target in df.columns:
            self._chart_target_distribution(df, target)
            self._chart_scatter_vs_target(df, feature_cols[:12], target)
            self._chart_class_balance(df, target)
        self._chart_feature_distributions(df, feature_cols[:16])
        self._chart_boxplots(df, feature_cols[:16])
        self._chart_violin_plots(df, feature_cols[:10], target)
        self._chart_skewness(df, feature_cols)
        if feature_importance:
            self._chart_feature_importance(feature_importance)
        if len(feature_cols) >= 2:
            self._chart_pairplot(df, feature_cols[:6], target)
        self._chart_categorical_bars(df, target)
        self._chart_pie_charts(df, target)
        if relationship_groups:
            for group in relationship_groups:
                self._chart_relationship_matrix(df, group)

        log.info(f"  Generated {len(self.generated)} charts → {self.output_dir}/")
        return self.generated

    # ─────────────────────────────────────────────────────────────────────────
    #  Individual chart methods
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        self.generated.append(path)
        return path

    def _chart_missing_heatmap(self, df: pd.DataFrame):
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if null_cols.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        _apply_dark_style(fig, ax)
        null_pct = (null_cols / len(df) * 100).sort_values(ascending=False)
        bars = ax.barh(null_pct.index, null_pct.values,
                       color=[PINK if v > 50 else ORANGE if v > 20 else CYAN
                              for v in null_pct.values])
        for bar, val in zip(bars, null_pct.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color=TEXT_PRI, fontsize=9)
        ax.set_xlabel("Missing %", color=TEXT_PRI)
        ax.set_title("🔍 Missing Values by Column", color=CYAN, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 110)
        self._save(fig, "00_missing_values.png")

    def _chart_correlation_heatmap(self, df: pd.DataFrame, cols: List[str]):
        num_df = df[cols].select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return
        corr = num_df.corr()
        n = len(corr)
        fig_size = max(10, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        _apply_dark_style(fig, ax)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(250, 15, as_cmap=True)
        sns.heatmap(
            corr, ax=ax, mask=mask,
            cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=(n <= 20), fmt=".2f", linewidths=0.5,
            linecolor=BORDER,
            annot_kws={"size": 7, "color": TEXT_PRI},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("🔗 Correlation Heatmap", color=CYAN, fontsize=14, fontweight="bold", pad=15)
        ax.tick_params(axis="both", colors=TEXT_SEC, labelsize=8)
        self._save(fig, "01_correlation_heatmap.png")

    def _chart_target_distribution(self, df: pd.DataFrame, target: str):
        series = df[target].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _apply_dark_style(fig, axes)

        # Histogram + KDE
        ax = axes[0]
        ax.hist(series, bins=40, color=CYAN, alpha=0.7, edgecolor=DARK_BG, density=True)
        series.plot.kde(ax=ax, color=PINK, linewidth=2.5)
        ax.axvline(series.mean(), color=GOLD, linestyle="--", linewidth=1.5, label=f"Mean={series.mean():.2f}")
        ax.axvline(series.median(), color=GREEN, linestyle="--", linewidth=1.5, label=f"Median={series.median():.2f}")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_PRI, fontsize=9)
        ax.set_title(f"📊 Target Distribution: {target}", color=CYAN, fontsize=12)

        # Box plot
        ax2 = axes[1]
        bp = ax2.boxplot(series.values, patch_artist=True, vert=True,
                         boxprops=dict(facecolor=PURPLE, color=CYAN),
                         whiskerprops=dict(color=CYAN),
                         capprops=dict(color=CYAN),
                         medianprops=dict(color=GOLD, linewidth=2),
                         flierprops=dict(marker="o", color=PINK, alpha=0.5))
        ax2.set_title(f"📦 Target Box Plot: {target}", color=CYAN, fontsize=12)
        ax2.set_xticklabels([target], color=TEXT_PRI)
        self._save(fig, "02_target_distribution.png")

    def _chart_feature_distributions(self, df: pd.DataFrame, cols: List[str]):
        if not cols:
            return
        n_cols = min(4, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.patch.set_facecolor(DARK_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            _apply_dark_style(fig, ax)
            series = df[col].dropna()
            skew = series.skew()
            color = PINK if abs(skew) > 2 else (ORANGE if abs(skew) > 1 else CYAN)
            ax.hist(series, bins=30, color=color, alpha=0.8, edgecolor=DARK_BG)
            try:
                series.plot.kde(ax=ax, secondary_y=False, color=GREEN, linewidth=1.5)
            except Exception:
                pass
            ax.set_title(f"{col}\nskew={skew:.2f}", color=CYAN, fontsize=9)
            ax.set_xlabel("")

        for j in range(len(cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("📈 Feature Distributions", color=GOLD, fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, "03_feature_distributions.png")

    def _chart_boxplots(self, df: pd.DataFrame, cols: List[str]):
        if not cols:
            return
        num_df = df[cols].select_dtypes(include="number")
        if num_df.empty:
            return
        n = len(num_df.columns)
        fig, ax = plt.subplots(figsize=(max(12, n * 0.9), 7))
        _apply_dark_style(fig, ax)
        bp = ax.boxplot(
            [num_df[c].dropna().values for c in num_df.columns],
            patch_artist=True, vert=True,
            boxprops=dict(color=CYAN),
            whiskerprops=dict(color=TEXT_SEC),
            capprops=dict(color=TEXT_SEC),
            medianprops=dict(color=GOLD, linewidth=2),
            flierprops=dict(marker=".", color=PINK, alpha=0.3, markersize=4),
        )
        colors = NEON_PALETTE * ((n // len(NEON_PALETTE)) + 1)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(num_df.columns, rotation=45, ha="right", color=TEXT_SEC, fontsize=9)
        ax.set_title("📦 Box Plots — Outlier View", color=CYAN, fontsize=13, fontweight="bold")
        self._save(fig, "04_boxplot_grid.png")

    def _chart_scatter_vs_target(self, df: pd.DataFrame, feature_cols: List[str], target: str):
        if not feature_cols or target not in df.columns:
            return
        n = min(len(feature_cols), 12)
        cols_per_row = 4
        n_rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
        fig.patch.set_facecolor(DARK_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(feature_cols[:n]):
            ax = axes_flat[i]
            _apply_dark_style(fig, ax)
            x = df[col].dropna()
            y = df.loc[x.index, target]
            ax.scatter(x, y, alpha=0.4, s=12, color=CYAN, edgecolors="none")
            # Trend line
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color=PINK, linewidth=1.5, linestyle="--")
            except Exception:
                pass
            corr = x.corr(y)
            ax.set_title(f"{col} vs {target}\nr={corr:.2f}", color=CYAN, fontsize=9)
            ax.set_xlabel(col, color=TEXT_SEC, fontsize=8)
            ax.set_ylabel(target, color=TEXT_SEC, fontsize=8)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"🎯 Feature vs Target: {target}", color=GOLD, fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, "05_scatter_vs_target.png")

    def _chart_violin_plots(self, df: pd.DataFrame, cols: List[str], target: str = None):
        num_df = df[cols].select_dtypes(include="number")
        if num_df.empty or num_df.shape[1] < 2:
            return
        fig, ax = plt.subplots(figsize=(max(12, len(num_df.columns)), 7))
        _apply_dark_style(fig, ax)
        data_to_plot = [num_df[c].dropna().values for c in num_df.columns]
        parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(NEON_PALETTE[i % len(NEON_PALETTE)])
            pc.set_alpha(0.7)
            pc.set_edgecolor(DARK_BG)
        parts["cmeans"].set_color(GOLD)
        parts["cmedians"].set_color(CYAN)
        ax.set_xticks(range(1, len(num_df.columns) + 1))
        ax.set_xticklabels(num_df.columns, rotation=45, ha="right", color=TEXT_SEC, fontsize=9)
        ax.set_title("🎻 Violin Plots — Distribution Shape", color=CYAN, fontsize=13, fontweight="bold")
        self._save(fig, "06_violin_plots.png")

    def _chart_skewness(self, df: pd.DataFrame, cols: List[str]):
        num_df = df[cols].select_dtypes(include="number")
        if num_df.empty:
            return
        skews = num_df.skew().sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(12, max(5, len(skews) * 0.4)))
        _apply_dark_style(fig, ax)
        colors_bar = [PINK if abs(s) > 2 else ORANGE if abs(s) > 1 else CYAN for s in skews.values]
        bars = ax.barh(skews.index, skews.values, color=colors_bar, alpha=0.85)
        ax.axvline(0, color=TEXT_SEC, linewidth=1)
        ax.axvline(1, color=ORANGE, linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(-1, color=ORANGE, linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(2, color=PINK, linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(-2, color=PINK, linewidth=1, linestyle="--", alpha=0.5)
        for bar, val in zip(bars, skews.values):
            ax.text(val + (0.02 if val >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center",
                    ha="left" if val >= 0 else "right",
                    color=TEXT_PRI, fontsize=8)
        ax.set_xlabel("Skewness", color=TEXT_PRI)
        ax.set_title("📐 Skewness by Feature\n(|skew|>1=skewed, |skew|>2=highly skewed)",
                     color=CYAN, fontsize=12, fontweight="bold")
        self._save(fig, "07_skewness.png")

    def _chart_feature_importance(self, fi: dict):
        if not fi:
            return
        items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20]
        names, values = zip(*items)
        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.45)))
        _apply_dark_style(fig, ax)
        colors_bar = NEON_PALETTE * ((len(names) // len(NEON_PALETTE)) + 1)
        bars = ax.barh(names, values, color=colors_bar[:len(names)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color=TEXT_PRI, fontsize=9)
        ax.set_xlabel("Importance Score", color=TEXT_PRI)
        ax.set_title("⭐ Feature Importance (Top 20)", color=CYAN, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        self._save(fig, "08_feature_importance.png")

    def _chart_pairplot(self, df: pd.DataFrame, cols: List[str], target: str = None):
        num_df = df[cols].select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return
        try:
            with plt.style.context("dark_background"):
                if target and target in df.columns and df[target].nunique() < 10:
                    hue_data = df.loc[num_df.index, target]
                    plot_df = num_df.copy()
                    plot_df[target] = hue_data
                    g = sns.pairplot(plot_df, hue=target,
                                     plot_kws={"alpha": 0.5, "s": 15},
                                     diag_kws={"alpha": 0.7},
                                     palette=NEON_PALETTE[:df[target].nunique()])
                else:
                    g = sns.pairplot(num_df, corner=True,
                                     plot_kws={"alpha": 0.4, "color": CYAN, "s": 12},
                                     diag_kws={"color": PURPLE, "alpha": 0.7})
                g.figure.patch.set_facecolor(DARK_BG)
                for ax in g.axes.flatten():
                    if ax:
                        ax.set_facecolor(CARD_BG)
                g.figure.suptitle("🔵 Pair Plot", color=GOLD, fontsize=14, y=1.02)
                path = os.path.join(self.output_dir, "09_pairplot.png")
                g.savefig(path, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
                plt.close()
                self.generated.append(path)
        except Exception as e:
            log.warning(f"  Pairplot failed: {e}")

    def _chart_categorical_bars(self, df: pd.DataFrame, target: str = None):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return
        n = min(len(cat_cols), 6)
        cols_per_row = 3
        n_rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 5 * n_rows))
        fig.patch.set_facecolor(DARK_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(cat_cols[:n]):
            ax = axes_flat[i]
            _apply_dark_style(fig, ax)
            vc = df[col].value_counts().head(15)
            colors_bar = NEON_PALETTE * ((len(vc) // len(NEON_PALETTE)) + 1)
            ax.bar(range(len(vc)), vc.values, color=colors_bar[:len(vc)], alpha=0.85)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha="right", color=TEXT_SEC, fontsize=8)
            ax.set_title(f"📊 {col}", color=CYAN, fontsize=10)
            for j, val in enumerate(vc.values):
                ax.text(j, val + max(vc.values) * 0.01, str(val),
                        ha="center", color=TEXT_PRI, fontsize=7)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("📊 Categorical Feature Counts", color=GOLD, fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, "10_categorical_bars.png")

    def _chart_pie_charts(self, df: pd.DataFrame, target: str = None):
        """Pie chart for columns with few unique values (≤10)."""
        candidates = []
        for col in df.columns:
            if df[col].nunique() <= 10 and df[col].nunique() >= 2:
                candidates.append(col)
        candidates = candidates[:6]
        if not candidates:
            return

        n = len(candidates)
        cols_per_row = min(3, n)
        n_rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row,
                                 figsize=(5 * cols_per_row, 5 * n_rows))
        fig.patch.set_facecolor(DARK_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(candidates):
            ax = axes_flat[i]
            ax.set_facecolor(DARK_BG)
            vc = df[col].value_counts()
            colors_pie = NEON_PALETTE[:len(vc)]
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=vc.index,
                colors=colors_pie, autopct="%1.1f%%",
                startangle=90, pctdistance=0.8,
                wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
            )
            for text in texts:
                text.set_color(TEXT_SEC)
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_color(DARK_BG)
                autotext.set_fontsize(7)
                autotext.set_fontweight("bold")
            ax.set_title(f"🥧 {col}", color=CYAN, fontsize=11)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("🥧 Category Breakdowns", color=GOLD, fontsize=14, fontweight="bold", y=1.01)
        self._save(fig, "11_pie_charts.png")

    def _chart_class_balance(self, df: pd.DataFrame, target: str):
        """Class balance chart (for classification targets)."""
        series = df[target]
        if series.nunique() > 20:
            return  # continuous target → skip
        vc = series.value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(DARK_BG)
        _apply_dark_style(fig, axes)

        colors_pie = NEON_PALETTE[:len(vc)]
        # Bar
        axes[0].bar(vc.index.astype(str), vc.values, color=colors_pie, alpha=0.85)
        axes[0].set_title(f"⚖️ Class Balance: {target}", color=CYAN, fontsize=12)
        for j, (idx, val) in enumerate(vc.items()):
            axes[0].text(j, val + max(vc.values) * 0.01, str(val),
                         ha="center", color=TEXT_PRI, fontsize=9)
        # Pie
        axes[1].pie(vc.values, labels=vc.index.astype(str), colors=colors_pie,
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor": DARK_BG, "linewidth": 2})
        axes[1].set_title(f"🥧 Class Proportion", color=CYAN, fontsize=12)
        self._save(fig, "12_class_balance.png")

    def _chart_relationship_matrix(self, df: pd.DataFrame, cols: List[str]):
        """Multi-column relationship scatter matrix."""
        valid = [c for c in cols if c in df.columns]
        if len(valid) < 2:
            return
        n = len(valid)
        fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
        fig.patch.set_facecolor(DARK_BG)

        for i, col_y in enumerate(valid):
            for j, col_x in enumerate(valid):
                ax = axes[i][j] if n > 1 else axes
                _apply_dark_style(fig, ax)
                if i == j:
                    # Diagonal: histogram
                    ax.hist(df[col_x].dropna(), bins=25,
                            color=NEON_PALETTE[i % len(NEON_PALETTE)], alpha=0.8)
                    ax.set_title(col_x, color=CYAN, fontsize=8)
                else:
                    # Off-diagonal: scatter
                    x = df[col_x].dropna()
                    y = df.loc[x.index, col_y]
                    corr = x.corr(y)
                    color = PINK if abs(corr) > 0.7 else (ORANGE if abs(corr) > 0.4 else CYAN)
                    ax.scatter(x, y, alpha=0.3, s=8, color=color)
                    ax.set_title(f"r={corr:.2f}", color=color, fontsize=8)
                if i == n - 1:
                    ax.set_xlabel(col_x, color=TEXT_SEC, fontsize=7)
                if j == 0:
                    ax.set_ylabel(col_y, color=TEXT_SEC, fontsize=7)

        name = "_".join(valid[:3])
        fig.suptitle(f"🔗 Relationship Matrix: {', '.join(valid)}",
                     color=GOLD, fontsize=13, fontweight="bold", y=1.01)
        self._save(fig, f"13_relationship_{'_'.join(valid[:2])}.png")
