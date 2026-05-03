"""
analyzer.py — Relationship Analyzer
=====================================
Bedrock Truth: "Relationships between variables are either:
  1. Numeric ↔ Numeric  → Correlation (Pearson/Spearman), Regression
  2. Numeric ↔ Categorical → ANOVA, Effect Size, Box Distribution
  3. Categorical ↔ Categorical → Chi-Square, Cramér's V, Contingency

Every relationship has a test, every test has an interpretation."
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Color palette for the report ─────────────────────────────────────
COLORS = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "tertiary":   "#43B89C",
    "quaternary": "#F7971E",
    "bg":         "#0F0F1A",
    "card":       "#1A1A2E",
    "text":       "#E0E0FF",
    "grid":       "#2A2A3E",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["card"],
        "plot_bgcolor":  COLORS["card"],
        "font":          {"color": COLORS["text"], "family": "Inter, sans-serif"},
        "title":         {"font": {"size": 16, "color": COLORS["text"]}},
        "xaxis":         {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis":         {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "colorway":      [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                          COLORS["quaternary"], "#A8EDEA", "#FED6E3", "#89F7FE"],
    }
}


def apply_theme(fig):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        title_font=dict(size=15, color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=COLORS["grid"]),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                     linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                     linecolor=COLORS["grid"])
    return fig


def fig_to_html(fig) -> str:
    """Convert Plotly figure to self-contained HTML div."""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["sendDataToCloud"],
        "responsive": True,
    })


class RelationshipAnalyzer:
    """
    Analyzes pairwise relationships between all selected columns.
    Routes each pair to the correct statistical test automatically.
    """

    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.df = df.copy()
        self.meta = metadata
        self.results = {}
        self.charts = {}
        self._col_types = metadata.get("col_types", {})

    # ═════════════════════════════════════════════════════════════
    # PUBLIC: Master analysis function
    # ═════════════════════════════════════════════════════════════

    def analyze_all(self) -> dict:
        """Run all analyses and return consolidated results dict."""
        results = {}

        # ── 1. Dataset overview stats ─────────────────────────
        results["overview"] = self._compute_overview()

        # ── 2. Per-column descriptive statistics ──────────────
        results["descriptive"] = self._compute_descriptive()

        # ── 3. Missing value analysis ─────────────────────────
        results["missing"] = self._compute_missing()

        # ── 4. Correlation matrix (numeric only) ──────────────
        results["correlation"] = self._compute_correlation()

        # ── 5. Pairwise relationship tests ────────────────────
        results["pairwise"] = self._compute_pairwise_relationships()

        # ── 6. Mutual information (all types) ─────────────────
        results["mutual_info"] = self._compute_mutual_information()

        # ── 7. Outlier analysis ───────────────────────────────
        results["outliers"] = self._compute_outliers()

        # ── 8. Relationship count (for summary) ───────────────
        results["relationship_count"] = len(results["pairwise"].get("pairs", []))

        self.results = results
        return results

    # ═════════════════════════════════════════════════════════════
    # ANALYSIS FUNCTIONS
    # ═════════════════════════════════════════════════════════════

    def _compute_overview(self) -> dict:
        df = self.df
        return {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_count": len(self.meta.get("numeric_cols", [])),
            "categorical_count": len(self.meta.get("categorical_cols", [])),
            "total_missing": int(df.isnull().sum().sum()),
            "total_cells": int(df.size),
            "duplicate_rows": self.meta.get("duplicated_rows", 0),
            "memory_mb": self.meta.get("memory_mb", 0),
        }

    def _compute_descriptive(self) -> dict:
        """Extended describe() with skewness and kurtosis."""
        stats_dict = {}
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            dtype = self._col_types.get(col, "unknown")

            if dtype == "numeric":
                stats_dict[col] = {
                    "type": "numeric",
                    "count": int(col_data.count()),
                    "mean": round(float(col_data.mean()), 4),
                    "median": round(float(col_data.median()), 4),
                    "std": round(float(col_data.std()), 4),
                    "min": round(float(col_data.min()), 4),
                    "max": round(float(col_data.max()), 4),
                    "q25": round(float(col_data.quantile(0.25)), 4),
                    "q75": round(float(col_data.quantile(0.75)), 4),
                    "skewness": round(float(col_data.skew()), 4),
                    "kurtosis": round(float(col_data.kurt()), 4),
                    "missing": int(self.df[col].isnull().sum()),
                    "unique": int(col_data.nunique()),
                }
            else:
                top_vals = col_data.value_counts().head(5)
                stats_dict[col] = {
                    "type": "categorical",
                    "count": int(col_data.count()),
                    "unique": int(col_data.nunique()),
                    "top": str(top_vals.index[0]) if len(top_vals) > 0 else "N/A",
                    "top_freq": int(top_vals.iloc[0]) if len(top_vals) > 0 else 0,
                    "missing": int(self.df[col].isnull().sum()),
                    "top_values": top_vals.to_dict(),
                }
        return stats_dict

    def _compute_missing(self) -> dict:
        missing = {}
        for col in self.df.columns:
            n_missing = int(self.df[col].isnull().sum())
            pct = round(n_missing / len(self.df) * 100, 2)
            missing[col] = {"count": n_missing, "pct": pct}
        return missing

    def _compute_correlation(self) -> dict:
        """Pearson and Spearman correlation matrices."""
        num_cols = self.meta.get("numeric_cols", [])
        # Only use columns actually in self.df
        num_cols = [c for c in num_cols if c in self.df.columns]

        if len(num_cols) < 2:
            return {"pearson": {}, "spearman": {}, "cols": []}

        sub = self.df[num_cols].dropna(how="all")

        pearson = sub.corr(method="pearson").round(4)
        spearman = sub.corr(method="spearman").round(4)

        return {
            "pearson": pearson.to_dict(),
            "spearman": spearman.to_dict(),
            "cols": num_cols,
        }

    def _compute_pairwise_relationships(self) -> dict:
        """
        For every pair of columns, run the appropriate statistical test.
        Returns structured results with strength, p-value, and interpretation.
        """
        cols = self.df.columns.tolist()
        pairs = []

        for col_a, col_b in combinations(cols, 2):
            type_a = self._col_types.get(col_a, "unknown")
            type_b = self._col_types.get(col_b, "unknown")

            # Normalize types for routing
            def is_num(t): return t == "numeric"
            def is_cat(t): return t in ("categorical", "high_cardinality")

            try:
                if is_num(type_a) and is_num(type_b):
                    result = self._test_num_num(col_a, col_b)
                elif is_num(type_a) and is_cat(type_b):
                    result = self._test_num_cat(col_a, col_b)
                elif is_cat(type_a) and is_num(type_b):
                    result = self._test_num_cat(col_b, col_a)
                    result["col_a"], result["col_b"] = col_a, col_b
                elif is_cat(type_a) and is_cat(type_b):
                    result = self._test_cat_cat(col_a, col_b)
                else:
                    continue

                result["col_a"] = col_a
                result["col_b"] = col_b
                pairs.append(result)

            except Exception as e:
                pairs.append({
                    "col_a": col_a, "col_b": col_b,
                    "test": "error", "error": str(e),
                    "strength": "unknown", "significant": False
                })

        # Sort by strength
        strength_order = {"very strong": 0, "strong": 1, "moderate": 2,
                          "weak": 3, "very weak": 4, "unknown": 5}
        pairs.sort(key=lambda x: strength_order.get(x.get("strength", "unknown"), 5))

        return {"pairs": pairs}

    def _test_num_num(self, col_a: str, col_b: str) -> dict:
        """Pearson + Spearman correlation for numeric-numeric pairs."""
        data = self.df[[col_a, col_b]].dropna()
        x, y = data[col_a], data[col_b]

        if len(x) < 3:
            return {"test": "insufficient_data", "strength": "unknown", "significant": False}

        # Pearson
        pearson_r, pearson_p = stats.pearsonr(x, y)

        # Spearman (handles non-linear monotonic relationships)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Determine dominant test (use Spearman if data is skewed)
        skew_a = abs(x.skew())
        skew_b = abs(y.skew())
        use_spearman = (skew_a > 1.5 or skew_b > 1.5)

        primary_r = spearman_r if use_spearman else pearson_r
        primary_p = spearman_p if use_spearman else pearson_p

        strength = self._correlation_strength(abs(primary_r))
        direction = "positive" if primary_r > 0 else "negative"
        significant = primary_p < 0.05

        interpretation = (
            f"{strength.title()} {direction} correlation "
            f"(r={primary_r:.3f}, p={primary_p:.4f}). "
            f"{'Statistically significant.' if significant else 'Not significant (p≥0.05).'}"
        )

        return {
            "test": "spearman" if use_spearman else "pearson",
            "r": round(float(primary_r), 4),
            "p_value": round(float(primary_p), 6),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "strength": strength,
            "direction": direction,
            "significant": significant,
            "n": len(x),
            "interpretation": interpretation,
            "pair_type": "numeric-numeric",
        }

    def _test_num_cat(self, num_col: str, cat_col: str) -> dict:
        """ANOVA + eta-squared for numeric-categorical pairs."""
        data = self.df[[num_col, cat_col]].dropna()
        groups = data.groupby(cat_col)[num_col].apply(list)

        # Need at least 2 groups with ≥2 observations
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(valid_groups) < 2:
            return {"test": "insufficient_groups", "strength": "unknown", "significant": False}

        group_arrays = [np.array(v) for v in valid_groups.values()]

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_arrays)

        # Eta-squared (effect size): SS_between / SS_total
        grand_mean = data[num_col].mean()
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_arrays
        )
        ss_total = sum((x - grand_mean) ** 2 for g in group_arrays for x in g)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        strength = self._eta_strength(eta_sq)
        significant = p_value < 0.05

        interpretation = (
            f"{strength.title()} relationship between '{num_col}' and '{cat_col}'. "
            f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}, η²={eta_sq:.3f}. "
            f"{'Significant difference between groups.' if significant else 'No significant group difference.'}"
        )

        return {
            "test": "anova",
            "f_stat": round(float(f_stat), 4),
            "p_value": round(float(p_value), 6),
            "eta_squared": round(float(eta_sq), 4),
            "strength": strength,
            "significant": significant,
            "n_groups": len(valid_groups),
            "group_means": {
                str(k): round(float(np.mean(v)), 4)
                for k, v in zip(valid_groups.keys(), group_arrays)
            },
            "interpretation": interpretation,
            "pair_type": "numeric-categorical",
        }

    def _test_cat_cat(self, col_a: str, col_b: str) -> dict:
        """Chi-square + Cramér's V for categorical-categorical pairs."""
        data = self.df[[col_a, col_b]].dropna()

        contingency = pd.crosstab(data[col_a], data[col_b])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {"test": "insufficient_categories", "strength": "unknown", "significant": False}

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V (normalized effect size [0,1])
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0

        strength = self._cramers_strength(cramers_v)
        significant = p_value < 0.05

        interpretation = (
            f"{strength.title()} association between '{col_a}' and '{col_b}'. "
            f"Chi-square={chi2:.3f}, p={p_value:.4f}, Cramér's V={cramers_v:.3f}. "
            f"{'Statistically significant association.' if significant else 'No significant association.'}"
        )

        return {
            "test": "chi_square",
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "dof": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "strength": strength,
            "significant": significant,
            "n": int(n),
            "contingency_shape": list(contingency.shape),
            "interpretation": interpretation,
            "pair_type": "categorical-categorical",
            "contingency": contingency.to_dict(),
        }

    def _compute_mutual_information(self) -> dict:
        """Mutual information between all column pairs."""
        results = {}
        cols = self.df.columns.tolist()
        df_clean = self.df.copy()

        # Encode categoricals for MI calculation
        label_encoders = {}
        for col in cols:
            if self._col_types.get(col) in ("categorical", "high_cardinality"):
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(
                    df_clean[col].fillna("__missing__").astype(str)
                )
                label_encoders[col] = le
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        mi_matrix = {}
        for col_a in cols:
            mi_matrix[col_a] = {}
            X = df_clean.drop(columns=[col_a])
            y = df_clean[col_a]

            # Use regression MI for numeric, classification MI for categorical
            if self._col_types.get(col_a) == "numeric":
                try:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                except Exception:
                    mi_scores = [0] * len(X.columns)
            else:
                try:
                    mi_scores = mutual_info_classif(X, y, random_state=42)
                except Exception:
                    mi_scores = [0] * len(X.columns)

            for col_b, score in zip(X.columns, mi_scores):
                mi_matrix[col_a][col_b] = round(float(score), 4)
            mi_matrix[col_a][col_a] = 0.0  # Self = 0

        return mi_matrix

    def _compute_outliers(self) -> dict:
        """IQR-based outlier detection for numeric columns."""
        outliers = {}
        for col in self.meta.get("numeric_cols", []):
            if col not in self.df.columns:
                continue
            data = self.df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (data < lower) | (data > upper)
            outlier_count = int(mask.sum())
            outliers[col] = {
                "count": outlier_count,
                "pct": round(outlier_count / max(len(data), 1) * 100, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
        return outliers

    # ═════════════════════════════════════════════════════════════
    # CHART GENERATION
    # ═════════════════════════════════════════════════════════════

    def generate_all_charts(self) -> dict:
        """Generate all Plotly charts. Returns dict of chart_name → HTML."""
        charts = {}

        charts["missing_values"] = self._chart_missing_values()
        charts["distributions"] = self._chart_distributions()
        charts["correlation_heatmap"] = self._chart_correlation_heatmap()
        charts["pairplot"] = self._chart_pairplot()
        charts["categorical_bars"] = self._chart_categorical_bars()
        charts["scatter_matrix"] = self._chart_scatter_relationships()
        charts["outlier_box"] = self._chart_outlier_box()
        charts["mutual_info"] = self._chart_mutual_info()

        self.charts = {k: v for k, v in charts.items() if v is not None}
        return self.charts

    def _chart_missing_values(self) -> str:
        """Bar chart showing missing values per column."""
        missing = self.results.get("missing", {})
        if not missing:
            return None

        cols = list(missing.keys())
        pcts = [missing[c]["pct"] for c in cols]
        counts = [missing[c]["count"] for c in cols]

        colors = [
            COLORS["secondary"] if p > 20 else
            (COLORS["quaternary"] if p > 5 else COLORS["tertiary"])
            for p in pcts
        ]

        fig = go.Figure(go.Bar(
            x=cols,
            y=pcts,
            text=[f"{p}%<br>({n} rows)" for p, n in zip(pcts, counts)],
            textposition="auto",
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Column",
            yaxis_title="Missing %",
            height=350,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_distributions(self) -> str:
        """Histogram + KDE for all numeric columns."""
        num_cols = [c for c in self.meta.get("numeric_cols", []) if c in self.df.columns]
        if not num_cols:
            return None

        n = len(num_cols)
        cols_grid = min(n, 3)
        rows_grid = (n + cols_grid - 1) // cols_grid

        fig = make_subplots(
            rows=rows_grid, cols=cols_grid,
            subplot_titles=num_cols,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["quaternary"], "#89F7FE", "#FED6E3", "#A8EDEA"]

        for i, col in enumerate(num_cols):
            row = i // cols_grid + 1
            col_pos = i % cols_grid + 1
            data = self.df[col].dropna()
            color = palette[i % len(palette)]

            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=col,
                    marker_color=color,
                    opacity=0.75,
                    nbinsx=30,
                    showlegend=False,
                    hovertemplate=f"<b>{col}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=row, col=col_pos
            )

        fig.update_layout(
            title="Distribution of Numeric Columns",
            height=280 * rows_grid,
            bargap=0.05,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_correlation_heatmap(self) -> str:
        """Correlation heatmap for numeric columns."""
        corr_data = self.results.get("correlation", {})
        cols = corr_data.get("cols", [])
        if len(cols) < 2:
            return None

        pearson = corr_data["pearson"]
        matrix = pd.DataFrame(pearson)

        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, "#FF6584"],   # Strong negative → red
                [0.5, COLORS["card"]],  # No correlation → dark
                [1.0, "#6C63FF"],   # Strong positive → purple
            ],
            zmin=-1, zmax=1,
            text=matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Pearson r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["−1.0", "−0.5", "0", "+0.5", "+1.0"],
                tickfont=dict(color=COLORS["text"]),
                title_font=dict(color=COLORS["text"]),
            )
        ))

        fig.update_layout(
            title="Pearson Correlation Heatmap",
            height=max(400, len(cols) * 60),
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_pairplot(self) -> str:
        """Scatter matrix (pairplot) for numeric columns."""
        num_cols = [c for c in self.meta.get("numeric_cols", []) if c in self.df.columns]
        if len(num_cols) < 2:
            return None

        # Limit to 6 columns for readability
        plot_cols = num_cols[:6]
        df_sub = self.df[plot_cols].dropna()

        # Color by first categorical column if available
        cat_cols = [c for c in self.meta.get("categorical_cols", []) if c in self.df.columns]
        color_col = None
        if cat_cols:
            # Only color if cardinality ≤ 10
            for cc in cat_cols:
                if self.df[cc].nunique() <= 10:
                    df_sub = df_sub.copy()
                    df_sub[cc] = self.df.loc[df_sub.index, cc]
                    color_col = cc
                    break

        fig = px.scatter_matrix(
            df_sub,
            dimensions=plot_cols,
            color=color_col,
            opacity=0.6,
            color_discrete_sequence=[
                COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                COLORS["quaternary"], "#89F7FE",
            ],
        )
        fig.update_traces(
            diagonal_visible=True,
            showupperhalf=False,
            marker=dict(size=4),
        )
        fig.update_layout(
            title="Scatter Matrix (Pairplot)",
            height=600,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_categorical_bars(self) -> str:
        """Bar charts for categorical column value counts."""
        cat_cols = [
            c for c in self.meta.get("categorical_cols", [])
            if c in self.df.columns and self.df[c].nunique() <= 25
        ]
        if not cat_cols:
            return None

        n = len(cat_cols)
        cols_grid = min(n, 2)
        rows_grid = (n + cols_grid - 1) // cols_grid

        fig = make_subplots(
            rows=rows_grid, cols=cols_grid,
            subplot_titles=cat_cols,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["quaternary"]]

        for i, col in enumerate(cat_cols):
            row = i // cols_grid + 1
            col_pos = i % cols_grid + 1
            vc = self.df[col].value_counts().head(15)

            fig.add_trace(
                go.Bar(
                    x=vc.index.astype(str),
                    y=vc.values,
                    name=col,
                    marker_color=palette[i % len(palette)],
                    showlegend=False,
                    hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y}}<extra></extra>",
                ),
                row=row, col=col_pos
            )

        fig.update_layout(
            title="Categorical Column Distributions",
            height=320 * rows_grid,
            bargap=0.15,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_scatter_relationships(self) -> str:
        """Top scatter plots for strongest numeric-numeric relationships."""
        pairs = self.results.get("pairwise", {}).get("pairs", [])
        nn_pairs = [
            p for p in pairs
            if p.get("pair_type") == "numeric-numeric" and "r" in p
        ]
        nn_pairs = sorted(nn_pairs, key=lambda x: abs(x.get("r", 0)), reverse=True)[:6]

        if not nn_pairs:
            return None

        n = len(nn_pairs)
        cols_grid = min(n, 3)
        rows_grid = (n + cols_grid - 1) // cols_grid

        titles = [
            f"{p['col_a']} vs {p['col_b']} (r={p['r']:.3f})"
            for p in nn_pairs
        ]

        fig = make_subplots(
            rows=rows_grid, cols=cols_grid,
            subplot_titles=titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        for i, pair in enumerate(nn_pairs):
            row = i // cols_grid + 1
            col_pos = i % cols_grid + 1
            data = self.df[[pair["col_a"], pair["col_b"]]].dropna()
            x = data[pair["col_a"]]
            y = data[pair["col_b"]]

            # Scatter
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode="markers",
                    marker=dict(
                        color=COLORS["primary"],
                        opacity=0.5,
                        size=5,
                    ),
                    name=f"{pair['col_a']} vs {pair['col_b']}",
                    showlegend=False,
                    hovertemplate=(
                        f"{pair['col_a']}: %{{x}}<br>"
                        f"{pair['col_b']}: %{{y}}<extra></extra>"
                    ),
                ),
                row=row, col=col_pos
            )

            # Trendline (OLS)
            if len(x) > 2:
                try:
                    m, b = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = m * x_line + b
                    fig.add_trace(
                        go.Scatter(
                            x=x_line, y=y_line,
                            mode="lines",
                            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                            showlegend=False,
                        ),
                        row=row, col=col_pos
                    )
                except Exception:
                    pass

        fig.update_layout(
            title="Top Correlated Numeric Pairs (with Trendlines)",
            height=320 * rows_grid,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_outlier_box(self) -> str:
        """Box plots for outlier visualization in numeric columns."""
        num_cols = [c for c in self.meta.get("numeric_cols", []) if c in self.df.columns]
        if not num_cols:
            return None

        fig = go.Figure()
        palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["quaternary"], "#89F7FE"]

        for i, col in enumerate(num_cols[:10]):  # Limit to 10
            data = self.df[col].dropna()
            fig.add_trace(go.Box(
                y=data,
                name=col,
                marker_color=palette[i % len(palette)],
                boxpoints="outliers",
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate=f"<b>{col}</b><br>Value: %{{y}}<extra></extra>",
            ))

        fig.update_layout(
            title="Box Plots — Outlier Detection",
            yaxis_title="Value",
            height=450,
            showlegend=False,
        )
        apply_theme(fig)
        return fig_to_html(fig)

    def _chart_mutual_info(self) -> str:
        """Heatmap of mutual information scores."""
        mi_data = self.results.get("mutual_info", {})
        if not mi_data:
            return None

        cols = list(mi_data.keys())
        if len(cols) < 2:
            return None

        matrix = pd.DataFrame(mi_data).reindex(index=cols, columns=cols).fillna(0)

        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, COLORS["card"]],
                [0.5, COLORS["primary"]],
                [1.0, COLORS["secondary"]],
            ],
            text=matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>MI Score: %{z:.4f}<extra></extra>",
            colorbar=dict(
                title="MI Score",
                tickfont=dict(color=COLORS["text"]),
                title_font=dict(color=COLORS["text"]),
            )
        ))

        fig.update_layout(
            title="Mutual Information Matrix (All Column Pairs)",
            height=max(400, len(cols) * 60),
        )
        apply_theme(fig)
        return fig_to_html(fig)

    # ═════════════════════════════════════════════════════════════
    # STRENGTH CLASSIFIERS
    # ═════════════════════════════════════════════════════════════

    def _correlation_strength(self, abs_r: float) -> str:
        if abs_r >= 0.80: return "very strong"
        if abs_r >= 0.60: return "strong"
        if abs_r >= 0.40: return "moderate"
        if abs_r >= 0.20: return "weak"
        return "very weak"

    def _eta_strength(self, eta_sq: float) -> str:
        """Cohen's benchmark for eta-squared."""
        if eta_sq >= 0.14: return "strong"
        if eta_sq >= 0.06: return "moderate"
        if eta_sq >= 0.01: return "weak"
        return "very weak"

    def _cramers_strength(self, v: float) -> str:
        if v >= 0.50: return "very strong"
        if v >= 0.30: return "strong"
        if v >= 0.10: return "moderate"
        if v >= 0.05: return "weak"
        return "very weak"
