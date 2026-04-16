"""
data_masterpiece.intelligence.reporter  --  ReportGenerator

Generates a self-contained, styled HTML report from all pipeline outputs.
Includes: executive summary, statistical profile, outlier detection,
feature selection, correlation analysis, model recommendations,
data splits, and embedded chart thumbnails.
"""

from __future__ import annotations

import base64
import datetime
import os

import numpy as np
import pandas as pd

from data_masterpiece.utils.logger import get_logger
from data_masterpiece.intelligence.profiler import ColumnProfile
from data_masterpiece.intelligence.relationship import RelationshipReport
from data_masterpiece.intelligence.recommender import RecommendationReport
from data_masterpiece.intelligence.splitter import SplitResult
from data_masterpiece.intelligence.outliers import OutlierReport
from data_masterpiece.intelligence.feature_selection import SelectionReport


class ReportGenerator:
    """
    Assemble a self-contained HTML report from all engine outputs.

    Parameters
    ----------
    output_path : Where to save the HTML file.
    """

    def __init__(self, output_path: str = "output/report.html"):
        self.output_path = output_path
        self.log = get_logger("ReportGenerator")

    def generate(
        self,
        df: pd.DataFrame,
        target: str,
        profiles: list = None,
        outlier_report: OutlierReport = None,
        selection_report: SelectionReport = None,
        rel_report: RelationshipReport = None,
        rec_report: RecommendationReport = None,
        split_result: SplitResult = None,
        plot_dir: str = "output/plots",
        extra_meta: dict = None,
    ) -> str:
        """Render and save the HTML report.  Returns absolute path."""
        self.log.info("Generating HTML report ...")
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        html = self._build_html(
            df=df, target=target,
            profiles=profiles or [],
            outlier_report=outlier_report,
            selection_report=selection_report,
            rel_report=rel_report,
            rec_report=rec_report,
            split_result=split_result,
            plot_dir=plot_dir,
            extra_meta=extra_meta or {},
        )
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        self.log.info(f"Report saved -> {self.output_path}")
        return os.path.abspath(self.output_path)

    def _build_html(self, **kw) -> str:
        df = kw["df"]
        target = kw["target"]
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_rows, n_cols = df.shape
        nan_total = int(df.isna().sum().sum())

        sections = "\n".join([
            self._section_summary(df, target, now, kw.get("extra_meta", {})),
            self._section_profile(kw.get("profiles", [])),
            self._section_outlier(kw.get("outlier_report")),
            self._section_selection(kw.get("selection_report")),
            self._section_correlation(kw.get("rel_report")),
            self._section_models(kw.get("rec_report")),
            self._section_split(kw.get("split_result")),
            self._section_charts(kw.get("plot_dir", "")),
        ])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Masterpiece Report</title>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d2e; --border: #2a2d40;
    --text: #e8eaf0; --muted: #8b8fa8; --accent: #6c63ff;
    --green: #2ecc71; --yellow: #f39c12; --red: #e74c3c;
    --blue: #3498db; --teal: #1abc9c;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.6; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 2rem; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; margin-bottom: 6px; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; color: var(--accent); margin: 0 0 16px; }}
  h3 {{ font-size: 1rem; color: var(--teal); margin-bottom: 10px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 40px; font-size: 0.85rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .metric {{ background: #12152a; border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }}
  .metric .val {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
  .metric .lbl {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #12152a; color: var(--muted); text-align: left; padding: 8px 12px; font-weight: 500; text-transform: uppercase; font-size: 11px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #12152a; }}
  .table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); max-height: 500px; overflow-y: auto; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: #1a3a2a; color: var(--green); }}
  .badge-yellow {{ background: #3a2a1a; color: var(--yellow); }}
  .badge-red {{ background: #3a1a1a; color: var(--red); }}
  .badge-blue {{ background: #1a2a3a; color: var(--blue); }}
  .badge-purple {{ background: #2a1a3a; color: var(--accent); }}
  .model-card {{ background: #12152a; border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .model-card h3 {{ font-size: 0.95rem; color: var(--text); margin-bottom: 8px; }}
  .model-card .priority {{ font-size: 1.2rem; font-weight: 700; color: var(--accent); }}
  ul.reasons {{ list-style: none; padding: 0; }}
  ul.reasons li {{ font-size: 11px; color: var(--muted); padding: 2px 0; }}
  ul.reasons li::before {{ content: "+ "; color: var(--green); }}
  ul.caveats li::before {{ content: "! "; color: var(--yellow); }}
  .hint {{ font-size: 10px; color: var(--accent); margin-top: 6px; font-family: monospace; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
  .chart-card {{ background: #12152a; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  .chart-card img {{ width: 100%; height: auto; display: block; }}
  .chart-card p {{ padding: 8px 12px; font-size: 11px; color: var(--muted); }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 4px 0; }}
  .bar-label {{ width: 160px; font-size: 11px; color: var(--muted); text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-track {{ flex: 1; background: #12152a; border-radius: 4px; height: 8px; }}
  .bar-fill {{ height: 8px; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--teal)); }}
  .bar-val {{ width: 50px; font-size: 11px; }}
  code {{ background: #12152a; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 11px; color: var(--teal); }}
  .split-row {{ display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }}
  .split-label {{ color: var(--muted); font-size: 12px; width: 80px; }}
  .split-bar {{ height: 12px; border-radius: 6px; }}
  .split-count {{ text-align: right; font-size: 12px; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <h1>Data Masterpiece Report</h1>
  <p class="subtitle">Generated {now} | Target: <code>{target}</code> | {n_rows:,} rows x {n_cols} features | NaN remaining: {nan_total}</p>
{sections}
</div>
</body>
</html>"""

    # -- section builders -----------------------------------------------------

    def _section_summary(self, df, target, now, meta):
        n_rows, n_cols = df.shape
        nan_total = int(df.isna().sum().sum())
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        return f"""
<div class="card">
  <h2>Executive Summary</h2>
  <div class="grid-4">
    <div class="metric"><div class="val">{n_rows:,}</div><div class="lbl">Total Rows</div></div>
    <div class="metric"><div class="val">{n_cols}</div><div class="lbl">Features</div></div>
    <div class="metric"><div class="val">{nan_total}</div><div class="lbl">NaN Remaining</div></div>
    <div class="metric"><div class="val">{mem_mb:.1f} MB</div><div class="lbl">Memory Usage</div></div>
  </div>
</div>"""

    def _section_profile(self, profiles):
        if not profiles:
            return ""
        rows = ""
        for p in profiles:
            skew_badge = (
                f'<span class="badge badge-red">{p.skewness:+.2f}</span>'
                if abs(p.skewness) > 1.5 else
                f'<span class="badge badge-yellow">{p.skewness:+.2f}</span>'
                if abs(p.skewness) > 0.5 else
                f'<span class="badge badge-green">{p.skewness:+.2f}</span>'
            )
            null_badge = (
                f'<span class="badge badge-red">{p.null_pct:.1%}</span>'
                if p.null_pct > 0.1 else
                f'<span class="badge badge-green">{p.null_pct:.1%}</span>'
            )
            out_badge = (
                f'<span class="badge badge-yellow">{p.n_iqr_outliers}</span>'
                if p.n_iqr_outliers > 0 else
                f'<span class="badge badge-green">0</span>'
            )
            dist_badge = f'<span class="badge badge-blue">{p.distribution}</span>'
            rows += f"""
<tr><td><strong>{p.name}</strong></td><td>{p.mean:.3f}</td><td>{p.median:.3f}</td>
<td>{p.std:.3f}</td><td>{p.minimum:.2f}</td><td>{p.maximum:.2f}</td>
<td>{p.p25:.2f}</td><td>{p.p75:.2f}</td><td>{skew_badge}</td><td>{p.kurtosis:.2f}</td>
<td>{out_badge}</td><td>{null_badge}</td><td>{dist_badge}</td></tr>"""
        return f"""
<div class="card">
  <h2>Statistical Profile</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Column</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th>
    <th>Max</th><th>P25</th><th>P75</th><th>Skew</th><th>Kurtosis</th>
    <th>Outliers</th><th>Null%</th><th>Distribution</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>"""

    def _section_outlier(self, report):
        if report is None:
            return ""
        rows = ""
        for col, stats in report.column_stats.items():
            n = stats.get("n_outliers", 0)
            if n == 0:
                continue
            badge = f'<span class="badge badge-yellow">{n}</span>'
            rows += f"""<tr><td><strong>{col}</strong></td><td>{stats['method']}</td>
<td>{stats['strategy']}</td><td>{stats['lower']:.3f}</td><td>{stats['upper']:.3f}</td><td>{badge}</td></tr>"""
        if not rows:
            rows = '<tr><td colspan="6" style="text-align:center;color:#2ecc71">No significant outliers detected</td></tr>'
        return f"""
<div class="card">
  <h2>Outlier Detection</h2>
  <div class="grid-3" style="margin-bottom:16px">
    <div class="metric"><div class="val">{report.total_outlier_cells:,}</div><div class="lbl">Outlier Cells</div></div>
    <div class="metric"><div class="val">{report.rows_dropped}</div><div class="lbl">Rows Dropped</div></div>
    <div class="metric"><div class="val">{len(report.columns_flagged)}</div><div class="lbl">Flag Columns Added</div></div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Column</th><th>Method</th><th>Strategy</th><th>Lower</th><th>Upper</th><th>Outliers</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>"""

    def _section_selection(self, report):
        if report is None:
            return ""
        bars = ""
        if not report.feature_scores.empty:
            max_score = float(report.feature_scores.max()) or 1.0
            for feat, score in report.feature_scores.head(15).items():
                pct = score / max_score * 100
                bars += f"""<div class="bar-row">
  <div class="bar-label">{feat}</div>
  <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
  <div class="bar-val">{score:.4f}</div>
</div>"""
        dropped = ""
        for col, reason in report.dropped_features.items():
            dropped += f"<tr><td><strong>{col}</strong></td><td>{reason}</td></tr>"
        if not dropped:
            dropped = '<tr><td colspan="2" style="text-align:center;color:#2ecc71">No features dropped</td></tr>'
        return f"""
<div class="card">
  <h2>Feature Selection</h2>
  <div class="grid-3" style="margin-bottom:16px">
    <div class="metric"><div class="val">{len(report.original_features)}</div><div class="lbl">Original Features</div></div>
    <div class="metric"><div class="val">{len(report.selected_features)}</div><div class="lbl">Selected Features</div></div>
    <div class="metric"><div class="val">{len(report.dropped_features)}</div><div class="lbl">Features Dropped</div></div>
  </div>
  <div class="grid-2">
    <div><h3>Feature Importance by Target Correlation</h3>{bars or "<p style='color:var(--muted)'>No target specified.</p>"}</div>
    <div><h3>Dropped Features &amp; Reasons</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Column</th><th>Reason</th></tr></thead>
        <tbody>{dropped}</tbody>
      </table></div>
    </div>
  </div>
</div>"""

    def _section_correlation(self, report):
        if report is None or report.correlation_matrix.empty:
            return ""
        strong = ""
        for p in report.strong_pairs[:15]:
            d = '<span class="badge badge-green">positive</span>' if p["direction"] == "positive" else '<span class="badge badge-red">negative</span>'
            strong += f"""<tr><td>{p['feature_a']}</td><td>{p['feature_b']}</td><td>{p['correlation']:.4f}</td><td>{d}</td></tr>"""
        if not strong:
            strong = '<tr><td colspan="4" style="text-align:center;color:var(--muted)">No strong pairs found</td></tr>'
        mc = ""
        for a, b, r in report.multicollinear_pairs[:10]:
            mc += f"<tr><td>{a}</td><td>{b}</td><td>{r:.4f}</td></tr>"
        if not mc:
            mc = '<tr><td colspan="3" style="text-align:center;color:#2ecc71">No severe multicollinearity</td></tr>'
        return f"""
<div class="card">
  <h2>Relationship Analysis</h2>
  <div class="grid-2">
    <div><h3>Strong Pairs (|r| >= 0.5)</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Direction</th></tr></thead>
        <tbody>{strong}</tbody>
      </table></div>
    </div>
    <div><h3>Multicollinear Pairs (|r| >= 0.85)</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Feature A</th><th>Feature B</th><th>r</th></tr></thead>
        <tbody>{mc}</tbody>
      </table></div>
    </div>
  </div>
</div>"""

    def _section_models(self, report):
        if report is None:
            return ""
        type_map = {
            "binary": '<span class="badge badge-purple">Binary Classification</span>',
            "multiclass": '<span class="badge badge-blue">Multi-class Classification</span>',
            "continuous": '<span class="badge badge-green">Regression</span>',
        }
        cards = ""
        for rec in report.recommendations:
            reasons = "".join(f"<li>{r}</li>" for r in rec.reasoning)
            caveats = "".join(f'<li style="color:var(--yellow)">{c}</li>' for c in rec.caveats)
            cards += f"""
<div class="model-card">
  <div style="display:flex;justify-content:space-between;align-items:start">
    <h3>#{rec.priority} {rec.model_name}</h3>
    <span class="badge badge-blue">{rec.category}</span>
  </div>
  <ul class="reasons">{reasons}</ul>
  <ul class="reasons caveats">{caveats}</ul>
  <p class="hint">{rec.library_hint}</p>
</div>"""
        notes = "".join(
            f'<li style="color:var(--yellow);font-size:12px;padding:3px 0">! {n}</li>'
            for n in report.preprocessing_notes
        )
        notes_html = f'<ul style="list-style:none;margin-top:16px">{notes}</ul>' if notes else ""
        return f"""
<div class="card">
  <h2>Model Recommendations</h2>
  <p style="color:var(--muted);margin-bottom:16px">
    Target type: {type_map.get(report.target_type, report.target_type)} |
    {report.n_rows:,} rows | {report.n_features} features |
    Mean |corr|: {report.mean_abs_corr:.3f}
  </p>
  <div class="grid-2">{cards}</div>
  {notes_html}
</div>"""

    def _section_split(self, result):
        if result is None:
            return ""
        info = result.split_info
        total = info["train_rows"] + info.get("val_rows", 0) + info["test_rows"]

        def bar(label, n, color):
            pct = n / total * 100 if total else 0
            return f"""<div class="split-row">
  <span class="split-label">{label}</span>
  <div class="bar-track"><div class="split-bar" style="width:{pct:.1f}%;background:{color}"></div></div>
  <span class="split-count">{n:,} rows ({pct:.0f}%)</span>
</div>"""

        bars = bar("Train", info["train_rows"], "var(--green)")
        if info.get("val_rows"):
            bars += bar("Validation", info["val_rows"], "var(--yellow)")
        bars += bar("Test", info["test_rows"], "var(--blue)")
        strat = '<span class="badge badge-green">Yes</span>' if info["stratified"] else '<span class="badge badge-yellow">No</span>'
        return f"""
<div class="card">
  <h2>Data Split</h2>
  <div class="grid-2">
    <div><h3>Split Distribution</h3>{bars}</div>
    <div><h3>Split Details</h3><table>
      <tr><td style="color:var(--muted)">Strategy</td><td>{info['strategy']}</td></tr>
      <tr><td style="color:var(--muted)">Stratified</td><td>{strat}</td></tr>
      <tr><td style="color:var(--muted)">Features (X)</td><td>{result.X_train.shape[1]}</td></tr>
      <tr><td style="color:var(--muted)">Random seed</td><td>{info['random_state']}</td></tr>
    </table></div>
  </div>
</div>"""

    def _section_charts(self, plot_dir):
        if not os.path.isdir(plot_dir):
            return ""
        cards = ""
        for fname in sorted(os.listdir(plot_dir)):
            if not fname.lower().endswith(".png"):
                continue
            fpath = os.path.join(plot_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                title = fname.replace("_", " ").replace(".png", "").title()
                cards += f"""
<div class="chart-card">
  <img src="data:image/png;base64,{b64}" alt="{title}" loading="lazy">
  <p>{title}</p>
</div>"""
            except Exception:
                pass
        if not cards:
            return ""
        return f"""
<div class="card">
  <h2>Visualisations</h2>
  <div class="chart-grid">{cards}</div>
</div>"""
