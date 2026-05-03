"""
reporter.py — HTML Report Generator
=====================================
Bedrock Truth: "A report is useless if it can't be READ and UNDERSTOOD.
Design for clarity first, information density second, aesthetics third."

Generates a self-contained, beautiful HTML report with:
- Dark glassmorphism UI
- Interactive Plotly charts
- Statistical tables
- AI-quality interpretation text
- No external dependencies (all inline)
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd


COLORS = {
    "bg":         "#0A0A1A",
    "card":       "#1A1A2E",
    "card2":      "#16213E",
    "border":     "#2A2A4E",
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "tertiary":   "#43B89C",
    "quaternary": "#F7971E",
    "text":       "#E0E0FF",
    "text_dim":   "#8888AA",
    "success":    "#43B89C",
    "warning":    "#F7971E",
    "danger":     "#FF6584",
}


def strength_color(strength: str) -> str:
    return {
        "very strong": "#6C63FF",
        "strong":      "#43B89C",
        "moderate":    "#F7971E",
        "weak":        "#FF8C61",
        "very weak":   "#FF6584",
    }.get(strength, "#8888AA")


def significance_badge(significant: bool, p_value: float = None) -> str:
    if significant:
        return '<span class="badge badge-sig">✓ Significant</span>'
    return '<span class="badge badge-nonsig">✗ Not Significant</span>'


def pair_type_icon(pair_type: str) -> str:
    return {
        "numeric-numeric":      "📈",
        "numeric-categorical":  "📊",
        "categorical-categorical": "🔲",
    }.get(pair_type, "🔗")


class HTMLReporter:
    """Generates a professional, self-contained HTML report."""

    def __init__(self, df: pd.DataFrame, results: dict, charts: dict, metadata: dict):
        self.df = df
        self.results = results
        self.charts = charts
        self.meta = metadata
        self.timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
        self.source_name = Path(str(metadata.get("source", "dataset"))).name

    def generate(self, output_path: str):
        """Render and write the HTML report."""
        html = self._build_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _build_html(self) -> str:
        overview    = self.results.get("overview", {})
        descriptive = self.results.get("descriptive", {})
        missing     = self.results.get("missing", {})
        pairwise    = self.results.get("pairwise", {}).get("pairs", [])
        outliers    = self.results.get("outliers", {})

        sig_pairs   = [p for p in pairwise if p.get("significant")]
        n_sig       = len(sig_pairs)
        n_total     = len(pairwise)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Data Relationship Report — {self.source_name}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
{self._css()}
</style>
</head>
<body>

<!-- ══════════════ HEADER ══════════════ -->
<header class="hero">
  <div class="hero-content">
    <div class="hero-badge">🔬 DATA RELATIONSHIP PIPELINE</div>
    <h1 class="hero-title">Relationship Analysis Report</h1>
    <p class="hero-subtitle">
      <span class="highlight">{self.source_name}</span> ·
      Generated {self.timestamp}
    </p>

    <!-- KPI Cards -->
    <div class="kpi-grid">
      {self._kpi("📋", "Rows", f"{overview.get('shape', [0,0])[0]:,}")}
      {self._kpi("🔢", "Columns", str(overview.get("shape", [0,0])[1]))}
      {self._kpi("🔗", "Relationships Tested", str(n_total))}
      {self._kpi("⚡", "Significant", str(n_sig))}
      {self._kpi("📊", "Numeric Cols", str(overview.get("numeric_count", 0)))}
      {self._kpi("🏷️", "Categorical Cols", str(overview.get("categorical_count", 0)))}
    </div>
  </div>
</header>

<!-- ══════════════ NAV ══════════════ -->
<nav class="sidenav">
  <div class="nav-logo">📊</div>
  <a href="#overview" class="nav-link" title="Overview">📋</a>
  <a href="#descriptive" class="nav-link" title="Statistics">📈</a>
  <a href="#missing" class="nav-link" title="Missing Data">❓</a>
  <a href="#distributions" class="nav-link" title="Distributions">📊</a>
  <a href="#correlation" class="nav-link" title="Correlation">🌡️</a>
  <a href="#pairwise" class="nav-link" title="Relationships">🔗</a>
  <a href="#scatter" class="nav-link" title="Scatter Plots">🔵</a>
  <a href="#categorical" class="nav-link" title="Categorical">🏷️</a>
  <a href="#outliers" class="nav-link" title="Outliers">⚠️</a>
  <a href="#mutual-info" class="nav-link" title="Mutual Info">🧠</a>
</nav>

<main class="main-content">

<!-- ══════════════ SECTION 1: OVERVIEW ══════════════ -->
<section id="overview" class="section">
  <div class="section-header">
    <h2>📋 Dataset Overview</h2>
    <p>High-level summary of your dataset's structure and health.</p>
  </div>
  <div class="grid-2">
    {self._overview_table(overview)}
    {self._health_card(overview, missing)}
  </div>
</section>

<!-- ══════════════ SECTION 2: DESCRIPTIVE STATS ══════════════ -->
<section id="descriptive" class="section">
  <div class="section-header">
    <h2>📈 Descriptive Statistics</h2>
    <p>Statistical summary for every column in your dataset.</p>
  </div>
  {self._descriptive_table(descriptive)}
</section>

<!-- ══════════════ SECTION 3: MISSING VALUES ══════════════ -->
<section id="missing" class="section">
  <div class="section-header">
    <h2>❓ Missing Value Analysis</h2>
    <p>Completeness of each column — missing data impacts relationship validity.</p>
  </div>
  {self._missing_chart_section()}
  {self._missing_table(missing)}
</section>

<!-- ══════════════ SECTION 4: DISTRIBUTIONS ══════════════ -->
<section id="distributions" class="section">
  <div class="section-header">
    <h2>📊 Column Distributions</h2>
    <p>Histograms reveal the shape of each variable's distribution.</p>
  </div>
  {self._chart_section("distributions")}
</section>

<!-- ══════════════ SECTION 5: CORRELATION ══════════════ -->
<section id="correlation" class="section">
  <div class="section-header">
    <h2>🌡️ Correlation Heatmap</h2>
    <p>Pearson correlation between all numeric pairs. 
       <strong>Purple = strong positive, Red = strong negative, Dark = no relationship.</strong>
    </p>
  </div>
  {self._chart_section("correlation_heatmap")}
  {self._chart_section("pairplot")}
</section>

<!-- ══════════════ SECTION 6: PAIRWISE RELATIONSHIPS ══════════════ -->
<section id="pairwise" class="section">
  <div class="section-header">
    <h2>🔗 Pairwise Relationship Analysis</h2>
    <p>Every column pair analyzed with the appropriate statistical test.</p>
  </div>
  {self._pairwise_filter_bar()}
  {self._pairwise_cards(pairwise)}
</section>

<!-- ══════════════ SECTION 7: SCATTER PLOTS ══════════════ -->
<section id="scatter" class="section">
  <div class="section-header">
    <h2>🔵 Scatter Plots — Top Correlations</h2>
    <p>Visual inspection of the strongest numeric-numeric relationships with trendlines.</p>
  </div>
  {self._chart_section("scatter_matrix")}
</section>

<!-- ══════════════ SECTION 8: CATEGORICAL ══════════════ -->
<section id="categorical" class="section">
  <div class="section-header">
    <h2>🏷️ Categorical Distributions</h2>
    <p>Value counts for each categorical column.</p>
  </div>
  {self._chart_section("categorical_bars")}
</section>

<!-- ══════════════ SECTION 9: OUTLIERS ══════════════ -->
<section id="outliers" class="section">
  <div class="section-header">
    <h2>⚠️ Outlier Detection</h2>
    <p>IQR-based outlier detection. Points outside the whiskers are potential outliers.</p>
  </div>
  {self._chart_section("outlier_box")}
  {self._outlier_table(outliers)}
</section>

<!-- ══════════════ SECTION 10: MUTUAL INFORMATION ══════════════ -->
<section id="mutual-info" class="section">
  <div class="section-header">
    <h2>🧠 Mutual Information Matrix</h2>
    <p>Mutual Information captures <strong>non-linear</strong> relationships that correlation misses.
       Higher score = stronger dependency between columns.
    </p>
  </div>
  {self._chart_section("mutual_info")}
</section>

<!-- ══════════════ FOOTER ══════════════ -->
<footer class="footer">
  <p>
    <strong>Data Relationship Pipeline v1.0</strong> ·
    Built with First Principles Architecture ·
    Report generated {self.timestamp}
  </p>
  <p class="footer-dim">
    Statistical tests: Pearson/Spearman Correlation · One-way ANOVA · Chi-Square · Cramér's V · Mutual Information · IQR Outlier Detection
  </p>
</footer>

</main>

<script>
{self._javascript()}
</script>

</body>
</html>"""

    # ══════════════════════════════════════════════════════════════
    # HTML COMPONENT BUILDERS
    # ══════════════════════════════════════════════════════════════

    def _kpi(self, icon: str, label: str, value: str) -> str:
        return f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-label">{label}</div>
        </div>"""

    def _overview_table(self, overview: dict) -> str:
        rows = [
            ("Rows", f"{overview.get('shape', [0,0])[0]:,}"),
            ("Columns", str(overview.get('shape', [0,0])[1])),
            ("Numeric Columns", str(overview.get('numeric_count', 0))),
            ("Categorical Columns", str(overview.get('categorical_count', 0))),
            ("Total Cells", f"{overview.get('total_cells', 0):,}"),
            ("Total Missing", f"{overview.get('total_missing', 0):,}"),
            ("Duplicate Rows", str(overview.get('duplicate_rows', 0))),
            ("Memory Usage", f"{self.meta.get('memory_mb', 0):.3f} MB"),
            ("Source", f"{self.meta.get('source', 'N/A')[:60]}..."),
        ]
        rows_html = "".join(
            f"<tr><td class='td-label'>{r[0]}</td><td class='td-value'>{r[1]}</td></tr>"
            for r in rows
        )
        return f"""
        <div class="card">
          <h3 class="card-title">📋 Dataset Info</h3>
          <table class="info-table">
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    def _health_card(self, overview: dict, missing: dict) -> str:
        total_missing = overview.get("total_missing", 0)
        total_cells = max(overview.get("total_cells", 1), 1)
        missing_pct = round(total_missing / total_cells * 100, 2)
        dup = overview.get("duplicate_rows", 0)
        n_rows = overview.get("shape", [0, 0])[0]
        dup_pct = round(dup / max(n_rows, 1) * 100, 2)

        def health_bar(pct, invert=True):
            score = 100 - pct if invert else pct
            color = (COLORS["success"] if score >= 80 else
                     COLORS["warning"] if score >= 50 else COLORS["danger"])
            return f"""
            <div class="health-bar-wrap">
              <div class="health-bar" style="width:{min(score,100):.0f}%;background:{color}"></div>
            </div>
            <span class="health-pct">{score:.0f}/100</span>"""

        return f"""
        <div class="card">
          <h3 class="card-title">🏥 Data Health</h3>
          <div class="health-item">
            <span class="health-label">Completeness</span>
            {health_bar(missing_pct)}
            <div class="health-note">{missing_pct:.1f}% missing data</div>
          </div>
          <div class="health-item">
            <span class="health-label">Uniqueness</span>
            {health_bar(dup_pct)}
            <div class="health-note">{dup_pct:.1f}% duplicate rows</div>
          </div>
          <div class="health-item">
            <span class="health-label">Columns with No Missing</span>
            <div class="health-bar-wrap">
              <div class="health-bar" style="width:{sum(1 for m in missing.values() if m['count']==0)/max(len(missing),1)*100:.0f}%;background:{COLORS['success']}"></div>
            </div>
            <span class="health-pct">{sum(1 for m in missing.values() if m['count']==0)}/{len(missing)}</span>
          </div>
        </div>"""

    def _descriptive_table(self, descriptive: dict) -> str:
        if not descriptive:
            return "<p class='dim'>No descriptive statistics available.</p>"

        num_items = {k: v for k, v in descriptive.items() if v.get("type") == "numeric"}
        cat_items = {k: v for k, v in descriptive.items() if v.get("type") == "categorical"}

        html = ""

        if num_items:
            rows = ""
            for col, s in num_items.items():
                skew_color = (COLORS["danger"] if abs(s.get("skewness", 0)) > 2 else
                              COLORS["warning"] if abs(s.get("skewness", 0)) > 1 else COLORS["success"])
                rows += f"""
                <tr>
                  <td class='td-col'>{col}</td>
                  <td>{s.get('count',''):,}</td>
                  <td>{s.get('mean','')}</td>
                  <td>{s.get('median','')}</td>
                  <td>{s.get('std','')}</td>
                  <td>{s.get('min','')}</td>
                  <td>{s.get('max','')}</td>
                  <td>{s.get('q25','')}</td>
                  <td>{s.get('q75','')}</td>
                  <td style="color:{skew_color}">{s.get('skewness','')}</td>
                  <td>{s.get('missing',0)}</td>
                </tr>"""

            html += f"""
            <div class="card overflow-x">
              <h3 class="card-title">🔢 Numeric Columns</h3>
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Column</th><th>Count</th><th>Mean</th><th>Median</th>
                    <th>Std Dev</th><th>Min</th><th>Max</th><th>Q1</th><th>Q3</th>
                    <th>Skewness</th><th>Missing</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""

        if cat_items:
            rows = ""
            for col, s in cat_items.items():
                top_vals = s.get("top_values", {})
                top_str = ", ".join(
                    f"<code>{str(k)[:20]}</code> ({v})"
                    for k, v in list(top_vals.items())[:3]
                )
                rows += f"""
                <tr>
                  <td class='td-col'>{col}</td>
                  <td>{s.get('count',''):,}</td>
                  <td>{s.get('unique','')}</td>
                  <td><code>{str(s.get('top',''))[:30]}</code></td>
                  <td>{s.get('top_freq','')}</td>
                  <td>{s.get('missing',0)}</td>
                  <td class="top-vals">{top_str}</td>
                </tr>"""

            html += f"""
            <div class="card overflow-x" style="margin-top:1.5rem">
              <h3 class="card-title">🏷️ Categorical Columns</h3>
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Column</th><th>Count</th><th>Unique Values</th>
                    <th>Most Frequent</th><th>Top Freq</th><th>Missing</th>
                    <th>Top 3 Values</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""

        return html

    def _missing_chart_section(self) -> str:
        chart = self.charts.get("missing_values")
        if not chart:
            return ""
        return f'<div class="card">{chart}</div>'

    def _missing_table(self, missing: dict) -> str:
        rows = ""
        for col, m in sorted(missing.items(), key=lambda x: -x[1]["pct"]):
            pct = m["pct"]
            color = (COLORS["danger"] if pct > 20 else
                     COLORS["warning"] if pct > 5 else COLORS["success"])
            bar = f'<div class="mini-bar" style="width:{min(pct,100):.0f}%;background:{color}"></div>'
            rows += f"""
            <tr>
              <td class='td-col'>{col}</td>
              <td>{m['count']:,}</td>
              <td>
                <div class="mini-bar-wrap">{bar}</div>
                <span style="color:{color}">{pct:.1f}%</span>
              </td>
              <td><span style="color:{color}">{'⛔ High' if pct>20 else '⚠️ Medium' if pct>5 else '✅ Low'}</span></td>
            </tr>"""

        return f"""
        <div class="card overflow-x" style="margin-top:1rem">
          <table class="data-table">
            <thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Severity</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    def _pairwise_filter_bar(self) -> str:
        return """
        <div class="filter-bar">
          <button class="filter-btn active" onclick="filterPairs('all')">All</button>
          <button class="filter-btn" onclick="filterPairs('numeric-numeric')">📈 Numeric↔Numeric</button>
          <button class="filter-btn" onclick="filterPairs('numeric-categorical')">📊 Numeric↔Categorical</button>
          <button class="filter-btn" onclick="filterPairs('categorical-categorical')">🔲 Categorical↔Categorical</button>
          <button class="filter-btn sig-btn" onclick="filterPairs('significant')">⚡ Significant Only</button>
        </div>"""

    def _pairwise_cards(self, pairs: list) -> str:
        if not pairs:
            return '<div class="card"><p class="dim">No relationships computed (need ≥2 columns).</p></div>'

        cards = ""
        for pair in pairs:
            pair_type = pair.get("pair_type", "unknown")
            strength = pair.get("strength", "unknown")
            sig = pair.get("significant", False)
            col_a = pair.get("col_a", "?")
            col_b = pair.get("col_b", "?")
            test = pair.get("test", "?")
            p_val = pair.get("p_value", None)
            interp = pair.get("interpretation", "")
            sig_class = "significant" if sig else ""

            # Build stats block
            stats_html = ""
            if pair_type == "numeric-numeric":
                stats_html = f"""
                <div class="stat-row">
                  <span class="stat">r = <strong>{pair.get('r','N/A')}</strong></span>
                  <span class="stat">p = <strong>{pair.get('p_value','N/A')}</strong></span>
                  <span class="stat">Pearson r = <strong>{pair.get('pearson_r','N/A')}</strong></span>
                  <span class="stat">Spearman ρ = <strong>{pair.get('spearman_r','N/A')}</strong></span>
                  <span class="stat">n = <strong>{pair.get('n','N/A')}</strong></span>
                </div>"""
            elif pair_type == "numeric-categorical":
                stats_html = f"""
                <div class="stat-row">
                  <span class="stat">F = <strong>{pair.get('f_stat','N/A')}</strong></span>
                  <span class="stat">p = <strong>{pair.get('p_value','N/A')}</strong></span>
                  <span class="stat">η² = <strong>{pair.get('eta_squared','N/A')}</strong></span>
                  <span class="stat">Groups = <strong>{pair.get('n_groups','N/A')}</strong></span>
                </div>"""
            elif pair_type == "categorical-categorical":
                stats_html = f"""
                <div class="stat-row">
                  <span class="stat">χ² = <strong>{pair.get('chi2','N/A')}</strong></span>
                  <span class="stat">p = <strong>{pair.get('p_value','N/A')}</strong></span>
                  <span class="stat">Cramér's V = <strong>{pair.get('cramers_v','N/A')}</strong></span>
                  <span class="stat">df = <strong>{pair.get('dof','N/A')}</strong></span>
                </div>"""

            strength_c = strength_color(strength)
            cards += f"""
            <div class="pair-card {sig_class}" data-type="{pair_type}" data-sig="{str(sig).lower()}">
              <div class="pair-header">
                <div class="pair-title">
                  <span class="pair-icon">{pair_type_icon(pair_type)}</span>
                  <span class="pair-cols">
                    <strong>{col_a}</strong>
                    <span class="pair-arrow"> ↔ </span>
                    <strong>{col_b}</strong>
                  </span>
                </div>
                <div class="pair-badges">
                  <span class="badge" style="background:{strength_c}22;color:{strength_c};border:1px solid {strength_c}">
                    {strength.title()}
                  </span>
                  {significance_badge(sig)}
                  <span class="badge badge-test">{test.replace('_',' ').title()}</span>
                </div>
              </div>
              {stats_html}
              <div class="pair-interp">{interp}</div>
            </div>"""

        return f'<div class="pairs-grid" id="pairs-grid">{cards}</div>'

    def _chart_section(self, chart_key: str) -> str:
        chart = self.charts.get(chart_key)
        if not chart:
            return f'<div class="card"><p class="dim">No data available for this chart.</p></div>'
        return f'<div class="card chart-card">{chart}</div>'

    def _outlier_table(self, outliers: dict) -> str:
        if not outliers:
            return ""
        rows = ""
        for col, o in sorted(outliers.items(), key=lambda x: -x[1]["pct"]):
            pct = o["pct"]
            color = (COLORS["danger"] if pct > 10 else
                     COLORS["warning"] if pct > 5 else COLORS["success"])
            rows += f"""
            <tr>
              <td class='td-col'>{col}</td>
              <td>{o['count']}</td>
              <td style="color:{color}">{pct:.1f}%</td>
              <td>{o['lower_bound']}</td>
              <td>{o['upper_bound']}</td>
            </tr>"""

        return f"""
        <div class="card overflow-x" style="margin-top:1rem">
          <h3 class="card-title">IQR Outlier Summary</h3>
          <table class="data-table">
            <thead>
              <tr><th>Column</th><th>Outlier Count</th><th>Outlier %</th>
              <th>Lower Fence (Q1−1.5×IQR)</th><th>Upper Fence (Q3+1.5×IQR)</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # ══════════════════════════════════════════════════════════════
    # CSS
    # ══════════════════════════════════════════════════════════════

    def _css(self) -> str:
        return f"""
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: {COLORS['bg']};
  --card: {COLORS['card']};
  --card2: {COLORS['card2']};
  --border: {COLORS['border']};
  --primary: {COLORS['primary']};
  --secondary: {COLORS['secondary']};
  --tertiary: {COLORS['tertiary']};
  --text: {COLORS['text']};
  --text-dim: {COLORS['text_dim']};
}}

html {{ scroll-behavior: smooth; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  line-height: 1.6;
  min-height: 100vh;
}}

/* ── Hero ── */
.hero {{
  background: linear-gradient(135deg, #0F0F1A 0%, #1A0A2E 50%, #0A1A2E 100%);
  border-bottom: 1px solid var(--border);
  padding: 3rem 2rem 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 0%, rgba(108,99,255,0.15) 0%, transparent 70%);
  pointer-events: none;
}}
.hero-content {{ position: relative; max-width: 1200px; margin: 0 auto; }}
.hero-badge {{
  display: inline-block;
  background: rgba(108,99,255,0.15);
  border: 1px solid rgba(108,99,255,0.4);
  color: var(--primary);
  padding: 0.3rem 1rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 2px;
  margin-bottom: 1rem;
}}
.hero-title {{
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 800;
  background: linear-gradient(135deg, #fff 0%, var(--primary) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}}
.hero-subtitle {{ color: var(--text-dim); font-size: 1rem; margin-bottom: 2rem; }}
.highlight {{ color: var(--primary); font-weight: 600; }}

/* ── KPI Grid ── */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1rem;
  max-width: 900px;
  margin: 0 auto;
}}
.kpi-card {{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: 16px;
  padding: 1rem;
  transition: transform 0.2s, border-color 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); border-color: var(--primary); }}
.kpi-icon {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 800; color: var(--primary); }}
.kpi-label {{ font-size: 0.72rem; color: var(--text-dim); font-weight: 500; }}

/* ── Side Nav ── */
.sidenav {{
  position: fixed; top: 50%; left: 0;
  transform: translateY(-50%);
  background: rgba(26,26,46,0.9);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 0 16px 16px 0;
  padding: 0.8rem 0.5rem;
  display: flex; flex-direction: column; align-items: center; gap: 0.4rem;
  z-index: 100;
}}
.nav-logo {{ font-size: 1.2rem; margin-bottom: 0.5rem; }}
.nav-link {{
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 8px;
  text-decoration: none;
  font-size: 1rem;
  transition: background 0.2s;
}}
.nav-link:hover {{ background: rgba(108,99,255,0.2); }}

/* ── Main Layout ── */
.main-content {{
  margin-left: 52px;
  max-width: 1300px;
  padding: 2rem 2rem 4rem;
}}

/* ── Sections ── */
.section {{ margin-bottom: 3rem; }}
.section-header {{ margin-bottom: 1.5rem; }}
.section-header h2 {{
  font-size: 1.5rem; font-weight: 700;
  color: var(--text);
  border-left: 4px solid var(--primary);
  padding-left: 0.8rem;
  margin-bottom: 0.3rem;
}}
.section-header p {{ color: var(--text-dim); font-size: 0.9rem; padding-left: 1.2rem; }}

/* ── Cards ── */
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}}
.card-title {{ font-size: 1rem; font-weight: 600; color: var(--primary); margin-bottom: 1rem; }}
.chart-card {{ padding: 1rem; }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
@media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

.overflow-x {{ overflow-x: auto; }}

/* ── Tables ── */
.data-table, .info-table {{
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}}
.data-table th {{
  background: rgba(108,99,255,0.15);
  color: var(--primary);
  font-weight: 600;
  padding: 0.6rem 0.8rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
.data-table td, .info-table td {{
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  vertical-align: middle;
}}
.data-table tr:hover td {{ background: rgba(108,99,255,0.05); }}
.td-label {{ color: var(--text-dim); font-weight: 500; width: 40%; }}
.td-value {{ font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
.td-col {{ font-weight: 600; color: var(--primary); font-family: 'JetBrains Mono', monospace; }}
.top-vals {{ font-size: 0.78rem; }}
code {{
  background: rgba(108,99,255,0.1);
  color: var(--primary);
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
}}

/* ── Health bars ── */
.health-item {{ margin-bottom: 1rem; }}
.health-label {{ display: block; font-size: 0.82rem; color: var(--text-dim); margin-bottom: 0.3rem; }}
.health-bar-wrap {{
  height: 8px;
  background: rgba(255,255,255,0.08);
  border-radius: 4px;
  overflow: hidden;
  display: inline-block; width: calc(100% - 50px);
}}
.health-bar {{ height: 100%; border-radius: 4px; transition: width 0.8s ease; }}
.health-pct {{ margin-left: 0.5rem; font-size: 0.8rem; font-weight: 600; }}
.health-note {{ font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem; }}

/* ── Mini bars (missing table) ── */
.mini-bar-wrap {{ display: inline-block; width: 100px; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; vertical-align: middle; margin-right: 0.5rem; }}
.mini-bar {{ height: 100%; border-radius: 3px; }}

/* ── Filter bar ── */
.filter-bar {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }}
.filter-btn {{
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 0.4rem 0.9rem;
  border-radius: 20px;
  cursor: pointer;
  font-size: 0.82rem;
  transition: all 0.2s;
  font-family: 'Inter', sans-serif;
}}
.filter-btn:hover, .filter-btn.active {{
  background: rgba(108,99,255,0.2);
  border-color: var(--primary);
  color: var(--primary);
}}
.sig-btn {{ border-color: {COLORS['tertiary']}44; color: {COLORS['tertiary']}; }}
.sig-btn:hover, .sig-btn.active {{
  background: {COLORS['tertiary']}22;
  border-color: {COLORS['tertiary']};
  color: {COLORS['tertiary']};
}}

/* ── Pair Cards ── */
.pairs-grid {{ display: flex; flex-direction: column; gap: 0.8rem; }}
.pair-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  transition: border-color 0.2s, transform 0.1s;
}}
.pair-card:hover {{ border-color: var(--primary); transform: translateX(3px); }}
.pair-card.significant {{ border-left: 3px solid {COLORS['tertiary']}; }}
.pair-header {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.7rem; }}
.pair-title {{ display: flex; align-items: center; gap: 0.5rem; }}
.pair-icon {{ font-size: 1.2rem; }}
.pair-cols {{ font-size: 0.95rem; }}
.pair-arrow {{ color: var(--primary); font-weight: 700; }}
.pair-badges {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }}

/* ── Badges ── */
.badge {{
  display: inline-flex; align-items: center;
  padding: 0.2rem 0.6rem;
  border-radius: 12px;
  font-size: 0.72rem;
  font-weight: 600;
}}
.badge-sig {{ background: {COLORS['tertiary']}22; color: {COLORS['tertiary']}; border: 1px solid {COLORS['tertiary']}44; }}
.badge-nonsig {{ background: {COLORS['danger']}22; color: {COLORS['danger']}; border: 1px solid {COLORS['danger']}44; }}
.badge-test {{ background: rgba(108,99,255,0.1); color: var(--primary); border: 1px solid rgba(108,99,255,0.2); }}

.stat-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin-bottom: 0.5rem; }}
.stat {{ font-size: 0.83rem; color: var(--text-dim); }}
.stat strong {{ color: var(--text); font-family: 'JetBrains Mono', monospace; }}
.pair-interp {{
  font-size: 0.84rem;
  color: var(--text-dim);
  border-top: 1px solid rgba(255,255,255,0.06);
  padding-top: 0.6rem;
  margin-top: 0.4rem;
  line-height: 1.5;
}}
.pair-card[style*="display: none"] {{ display: none !important; }}

/* ── Footer ── */
.footer {{
  text-align: center;
  padding: 2rem;
  border-top: 1px solid var(--border);
  color: var(--text-dim);
  font-size: 0.82rem;
  margin-top: 3rem;
}}
.footer-dim {{ font-size: 0.73rem; margin-top: 0.4rem; opacity: 0.6; }}

.dim {{ color: var(--text-dim); font-style: italic; }}
"""

    # ══════════════════════════════════════════════════════════════
    # JAVASCRIPT
    # ══════════════════════════════════════════════════════════════

    def _javascript(self) -> str:
        return """
// ── Pair card filtering ─────────────────────────────────
function filterPairs(filter) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');

  document.querySelectorAll('.pair-card').forEach(card => {
    const type = card.dataset.type;
    const sig  = card.dataset.sig === 'true';

    let show = false;
    if (filter === 'all') show = true;
    else if (filter === 'significant') show = sig;
    else show = type === filter;

    card.style.display = show ? '' : 'none';
  });
}

// ── Active nav highlighting ──────────────────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const id = e.target.id;
      document.querySelectorAll('.nav-link').forEach(a => {
        a.style.background = a.getAttribute('href') === '#' + id
          ? 'rgba(108,99,255,0.25)' : '';
      });
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('section[id]').forEach(s => observer.observe(s));

// ── Animate health bars on load ──────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.health-bar').forEach(bar => {
    const w = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = w; }, 200);
  });
});
"""
