"""
data_masterpiece_v3.reporting.report_builder
─────────────────────────────────────────────
Legend-Level Animated HTML Report Builder.

Features:
  ✦ Deep dark neon theme with animated particle background
  ✦ Animated counters (numbers count up on load)
  ✦ Animated progress bars for model scores
  ✦ Glowing card borders with hover effects
  ✦ Interactive tabs for different sections
  ✦ Embedded base64 charts (no external dependencies)
  ✦ AutoML leaderboard table with rank badges
  ✦ Full statistical detail for every column
  ✦ Responsive layout
"""

from __future__ import annotations

import base64
import json
import os
import datetime
from typing import List, Optional


def _img_to_b64(path: str) -> str:
    """Convert image file to base64 data URI."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


def build_report(
    output_path: str,
    stats: dict,
    charts: List[str],
    split_info: dict = None,
    automl_results: dict = None,
    preprocess_summary: dict = None,
    target: str = None,
    config_dict: dict = None,
) -> str:
    """Generate the full HTML report and save it to output_path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    overview  = stats.get("overview", {})
    col_stats = stats.get("column_stats", {})
    corr      = stats.get("correlation", {})
    normality = stats.get("normality", {})
    target_corr = corr.get("target_correlation", {})

    # Build sections
    html = _html_shell(
        overview       = overview,
        col_stats      = col_stats,
        corr           = corr,
        normality      = normality,
        target_corr    = target_corr,
        charts         = charts,
        split_info     = split_info or {},
        automl_results = automl_results or {},
        preprocess_summary = preprocess_summary or {},
        target         = target or "N/A",
        config_dict    = config_dict or {},
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  HTML Template
# ─────────────────────────────────────────────────────────────────────────────

def _html_shell(**ctx) -> str:
    overview       = ctx["overview"]
    col_stats      = ctx["col_stats"]
    corr           = ctx["corr"]
    normality      = ctx["normality"]
    target_corr    = ctx["target_corr"]
    charts         = ctx["charts"]
    split_info     = ctx["split_info"]
    automl_results = ctx["automl_results"]
    pp             = ctx["preprocess_summary"]
    target         = ctx["target"]
    config_dict    = ctx["config_dict"]
    ts             = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pre-compute values
    n_rows    = overview.get("n_rows", 0)
    n_cols    = overview.get("n_cols", 0)
    n_nulls   = overview.get("total_nulls", 0)
    n_dups    = overview.get("duplicate_rows", 0)
    mem_mb    = overview.get("memory_mb", 0)
    n_numeric = overview.get("numeric_cols", 0)

    stat_cards = _stat_cards(n_rows, n_cols, n_nulls, n_dups, mem_mb, n_numeric,
                              pp, split_info)
    columns_section = _columns_section(col_stats)
    correlation_section = _correlation_section(target_corr, corr)
    normality_section = _normality_section(normality)
    charts_section = _charts_section(charts)
    automl_section = _automl_section(automl_results)
    split_section = _split_section(split_info)
    config_section = _config_section(config_dict)
    preprocess_section = _preprocess_section(pp)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ Data Masterpiece v3 — Legend Report</title>
<style>
:root {{
  --bg:      #0a0a0f;
  --bg2:     #0f0f1a;
  --card:    #12121f;
  --border:  #2a2a3e;
  --txt:     #e0e0ff;
  --txt2:    #a0a0c0;
  --cyan:    #00f5ff;
  --pink:    #ff00a0;
  --green:   #39ff14;
  --orange:  #ff6b35;
  --purple:  #b967ff;
  --gold:    #ffd700;
  --red:     #ff3366;
  --blue:    #00bfff;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
html {{ scroll-behavior: smooth; }}

body {{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:var(--bg);
  color:var(--txt);
  min-height:100vh;
  overflow-x:hidden;
}}

/* ── Particle canvas ── */
#particles-canvas {{
  position:fixed;top:0;left:0;
  width:100%;height:100%;
  pointer-events:none;z-index:0;opacity:0.25;
}}

/* ── Layout ── */
.page-wrap {{ position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:20px; }}

/* ── Header ── */
.header {{
  text-align:center;padding:50px 20px 30px;
  background:linear-gradient(180deg,rgba(0,245,255,0.05) 0%,transparent 100%);
  border-bottom:1px solid var(--border);margin-bottom:30px;
}}
.header h1 {{
  font-size:clamp(24px,5vw,52px);font-weight:900;letter-spacing:2px;
  background:linear-gradient(135deg,var(--cyan),var(--purple),var(--pink));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 20px rgba(0,245,255,0.4));
  animation:pulse-glow 3s ease-in-out infinite;
}}
.header .sub {{
  color:var(--txt2);margin-top:10px;font-size:14px;letter-spacing:1px;
}}
.header .ts {{
  color:var(--txt2);font-size:12px;margin-top:5px;opacity:0.7;
}}
.header .target-badge {{
  display:inline-block;margin-top:12px;padding:5px 16px;
  background:linear-gradient(135deg,rgba(185,103,255,0.2),rgba(0,245,255,0.1));
  border:1px solid var(--purple);border-radius:20px;
  color:var(--purple);font-size:13px;font-weight:600;
}}

/* ── Tabs ── */
.tabs {{
  display:flex;flex-wrap:wrap;gap:6px;
  margin-bottom:24px;padding:4px;
  background:var(--card);border:1px solid var(--border);border-radius:12px;
}}
.tab-btn {{
  padding:10px 20px;border:none;border-radius:8px;cursor:pointer;
  font-size:13px;font-weight:600;letter-spacing:0.5px;
  background:transparent;color:var(--txt2);
  transition:all 0.25s;
}}
.tab-btn:hover {{ background:rgba(0,245,255,0.08);color:var(--cyan); }}
.tab-btn.active {{
  background:linear-gradient(135deg,rgba(0,245,255,0.15),rgba(185,103,255,0.1));
  color:var(--cyan);border:1px solid rgba(0,245,255,0.3);
  box-shadow:0 0 12px rgba(0,245,255,0.15);
}}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; animation:fadeIn 0.3s ease; }}

/* ── Stat Cards ── */
.stats-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:16px;margin-bottom:28px;
}}
.stat-card {{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:20px;text-align:center;
  position:relative;overflow:hidden;
  transition:transform 0.3s, box-shadow 0.3s;
  animation:slideUp 0.5s ease both;
}}
.stat-card:hover {{
  transform:translateY(-4px);
  box-shadow:0 8px 30px rgba(0,245,255,0.12);
  border-color:rgba(0,245,255,0.3);
}}
.stat-card::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--card-accent,var(--cyan));
  box-shadow:0 0 8px var(--card-accent,var(--cyan));
}}
.stat-card .icon {{ font-size:24px;margin-bottom:8px; }}
.stat-card .val {{
  font-size:clamp(22px,4vw,34px);font-weight:800;
  color:var(--card-accent,var(--cyan));
  font-variant-numeric:tabular-nums;
}}
.stat-card .label {{ font-size:11px;color:var(--txt2);margin-top:4px;letter-spacing:0.5px; }}

/* ── Section title ── */
.section-title {{
  font-size:18px;font-weight:700;color:var(--cyan);
  margin:28px 0 16px;padding-bottom:8px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
}}
.section-title::before {{
  content:'';display:inline-block;width:4px;height:20px;
  background:linear-gradient(180deg,var(--cyan),var(--purple));
  border-radius:4px;
}}

/* ── Table ── */
.data-table {{
  width:100%;border-collapse:collapse;font-size:12px;
  background:var(--card);border-radius:12px;overflow:hidden;
  border:1px solid var(--border);
}}
.data-table th {{
  background:rgba(0,245,255,0.06);color:var(--cyan);
  padding:10px 12px;text-align:left;font-size:11px;
  font-weight:700;letter-spacing:0.5px;border-bottom:1px solid var(--border);
}}
.data-table td {{
  padding:9px 12px;border-bottom:1px solid rgba(42,42,62,0.5);
  color:var(--txt2);vertical-align:middle;
}}
.data-table tr:hover td {{ background:rgba(0,245,255,0.03);color:var(--txt); }}
.data-table tr:last-child td {{ border-bottom:none; }}

/* ── Charts grid ── */
.charts-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(480px,1fr));
  gap:20px;
}}
.chart-card {{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;overflow:hidden;
  transition:transform 0.3s,box-shadow 0.3s;
  animation:fadeIn 0.5s ease both;
}}
.chart-card:hover {{
  transform:translateY(-3px);
  box-shadow:0 8px 30px rgba(185,103,255,0.15);
  border-color:rgba(185,103,255,0.3);
}}
.chart-card img {{
  width:100%;display:block;
  border-bottom:1px solid var(--border);
}}
.chart-card .chart-title {{
  padding:10px 14px;color:var(--txt2);font-size:12px;
  background:rgba(0,0,0,0.2);
}}

/* ── AutoML leaderboard ── */
.leaderboard {{ display:flex;flex-direction:column;gap:10px; }}
.model-card {{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:16px 20px;
  display:grid;gap:12px;
  transition:all 0.3s;
  animation:slideUp 0.4s ease both;
}}
.model-card:hover {{
  border-color:rgba(0,245,255,0.3);
  box-shadow:0 4px 20px rgba(0,245,255,0.08);
}}
.model-card.rank-1 {{
  border-color:rgba(255,215,0,0.4);
  background:linear-gradient(135deg,rgba(255,215,0,0.04),var(--card));
}}
.model-header {{ display:flex;align-items:center;gap:12px;flex-wrap:wrap; }}
.rank-badge {{
  width:30px;height:30px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;
  font-weight:800;font-size:12px;flex-shrink:0;
}}
.rank-1 .rank-badge {{ background:rgba(255,215,0,0.2);color:var(--gold);border:1px solid var(--gold); }}
.rank-2 .rank-badge {{ background:rgba(160,160,192,0.2);color:#c0c0c0;border:1px solid #c0c0c0; }}
.rank-3 .rank-badge {{ background:rgba(255,107,53,0.2);color:var(--orange);border:1px solid var(--orange); }}
.rank-other .rank-badge {{ background:rgba(0,245,255,0.1);color:var(--cyan);border:1px solid var(--border); }}
.model-name {{ font-size:15px;font-weight:700;color:var(--txt); }}
.backend-tag {{
  padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;
  letter-spacing:0.5px;
}}
.sklearn-tag {{ background:rgba(57,255,20,0.1);color:var(--green);border:1px solid rgba(57,255,20,0.3); }}
.pytorch-tag {{ background:rgba(255,107,53,0.1);color:var(--orange);border:1px solid rgba(255,107,53,0.3); }}
.model-metrics {{ display:flex;gap:20px;flex-wrap:wrap; }}
.metric-item {{ text-align:center; }}
.metric-val {{ font-size:18px;font-weight:800;color:var(--cyan); }}
.metric-lab {{ font-size:10px;color:var(--txt2);letter-spacing:0.5px; }}
.score-bar-wrap {{ background:rgba(0,0,0,0.3);border-radius:6px;height:6px;overflow:hidden; }}
.score-bar {{
  height:100%;border-radius:6px;
  background:linear-gradient(90deg,var(--cyan),var(--purple));
  transition:width 1.5s cubic-bezier(0.4,0,0.2,1);
  box-shadow:0 0 8px rgba(0,245,255,0.4);
}}
.overfit-badge {{
  padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;
}}
.overfit-low  {{ background:rgba(57,255,20,0.1);color:var(--green); }}
.overfit-med  {{ background:rgba(255,107,53,0.1);color:var(--orange); }}
.overfit-high {{ background:rgba(255,51,102,0.1);color:var(--red); }}
.model-recommendation {{
  font-size:11px;color:var(--txt2);
  padding:8px 12px;background:rgba(0,0,0,0.2);border-radius:6px;
  border-left:2px solid var(--purple);
}}

/* ── Progress bars ── */
.progress-bar-row {{ display:flex;align-items:center;gap:12px;margin-bottom:8px; }}
.progress-label {{ min-width:160px;font-size:12px;color:var(--txt2); }}
.progress-track {{
  flex:1;height:8px;background:rgba(0,0,0,0.4);
  border-radius:6px;overflow:hidden;
}}
.progress-fill {{
  height:100%;border-radius:6px;
  background:linear-gradient(90deg,var(--cyan),var(--purple));
  box-shadow:0 0 6px rgba(0,245,255,0.3);
  transition:width 1.5s cubic-bezier(0.4,0,0.2,1);width:0;
}}
.progress-val {{ min-width:50px;font-size:12px;color:var(--cyan);text-align:right; }}

/* ── Chips / badges ── */
.chip {{
  display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:10px;font-weight:700;margin:2px;
}}
.chip-cyan  {{ background:rgba(0,245,255,0.1);color:var(--cyan); }}
.chip-pink  {{ background:rgba(255,0,160,0.1);color:var(--pink); }}
.chip-green {{ background:rgba(57,255,20,0.1);color:var(--green); }}
.chip-gold  {{ background:rgba(255,215,0,0.1);color:var(--gold); }}

/* ── Column stats card ── */
.col-stats-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:14px;
}}
.col-card {{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:14px;
  transition:all 0.3s;animation:fadeIn 0.4s ease both;
}}
.col-card:hover {{
  border-color:rgba(0,245,255,0.25);
  box-shadow:0 4px 16px rgba(0,245,255,0.07);
}}
.col-name {{ font-size:13px;font-weight:700;color:var(--cyan);margin-bottom:8px; }}
.col-dtype {{ font-size:10px;color:var(--purple);margin-bottom:8px; }}
.col-row {{
  display:flex;justify-content:space-between;
  font-size:11px;color:var(--txt2);
  padding:4px 0;border-bottom:1px solid rgba(42,42,62,0.5);
}}
.col-row:last-child {{ border-bottom:none; }}
.col-row span:last-child {{ color:var(--txt);font-weight:600; }}

/* ── JSON viewer ── */
.json-block {{
  background:#0d0d18;border:1px solid var(--border);
  border-radius:8px;padding:16px;
  font-family:'Consolas','Courier New',monospace;
  font-size:11px;color:var(--green);
  overflow-x:auto;white-space:pre-wrap;
  max-height:400px;overflow-y:auto;
}}

/* ── Animations ── */
@keyframes pulse-glow {{
  0%,100% {{ filter:drop-shadow(0 0 20px rgba(0,245,255,0.4)); }}
  50%      {{ filter:drop-shadow(0 0 40px rgba(185,103,255,0.6)); }}
}}
@keyframes fadeIn {{
  from {{ opacity:0;transform:translateY(8px); }}
  to   {{ opacity:1;transform:translateY(0); }}
}}
@keyframes slideUp {{
  from {{ opacity:0;transform:translateY(16px); }}
  to   {{ opacity:1;transform:translateY(0); }}
}}
@keyframes countUp {{
  from {{ opacity:0; }}
  to   {{ opacity:1; }}
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:6px;height:6px; }}
::-webkit-scrollbar-track {{ background:var(--bg2); }}
::-webkit-scrollbar-thumb {{ background:var(--border);border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:var(--cyan); }}

/* ── Responsive ── */
@media(max-width:700px) {{
  .charts-grid {{ grid-template-columns:1fr; }}
  .model-metrics {{ gap:12px; }}
  .tabs {{ gap:4px; }}
  .tab-btn {{ padding:8px 12px;font-size:11px; }}
}}
</style>
</head>
<body>

<canvas id="particles-canvas"></canvas>

<div class="page-wrap">

<!-- ═══ HEADER ═══ -->
<div class="header">
  <h1>⚡ DATA MASTERPIECE v3</h1>
  <div class="sub">🔬 LEGEND-LEVEL DATA ANALYSIS &amp; ML PIPELINE</div>
  <div class="ts">Generated: {ts}</div>
  <div class="target-badge">🎯 Target: {target}</div>
</div>

<!-- ═══ TABS ═══ -->
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('overview')">📊 Overview</button>
  <button class="tab-btn" onclick="showTab('columns')">📋 Columns</button>
  <button class="tab-btn" onclick="showTab('charts')">🎨 Charts</button>
  <button class="tab-btn" onclick="showTab('correlation')">🔗 Correlation</button>
  <button class="tab-btn" onclick="showTab('normality')">📐 Statistics</button>
  <button class="tab-btn" onclick="showTab('preprocess')">⚙️ Pipeline</button>
  <button class="tab-btn" onclick="showTab('split')">✂️ Data Split</button>
  <button class="tab-btn" onclick="showTab('automl')">🤖 AutoML</button>
  <button class="tab-btn" onclick="showTab('config')">🔧 Config</button>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: OVERVIEW                                         -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-overview" class="tab-content active">
  <div class="stats-grid">
    {stat_cards}
  </div>

  <div class="section-title">📊 Dataset Overview</div>
  <table class="data-table">
    <thead><tr>
      <th>Metric</th><th>Value</th><th>Detail</th>
    </tr></thead>
    <tbody>
      <tr><td>Total Rows</td><td style="color:var(--cyan);font-weight:700">{n_rows:,}</td><td><span class="chip chip-cyan">Samples</span></td></tr>
      <tr><td>Total Columns</td><td style="color:var(--purple);font-weight:700">{n_cols}</td><td><span class="chip chip-cyan">Features</span></td></tr>
      <tr><td>Numeric Columns</td><td style="color:var(--green);font-weight:700">{n_numeric}</td><td>Used in ML</td></tr>
      <tr><td>Missing Values</td><td style="color:{"var(--red)" if n_nulls>0 else "var(--green)"};font-weight:700">{n_nulls:,}</td><td>{"⚠️ Needs attention" if n_nulls > 0 else "✅ Clean"}</td></tr>
      <tr><td>Duplicate Rows</td><td style="color:{"var(--orange)" if n_dups>0 else "var(--green)"};font-weight:700">{n_dups:,}</td><td>{"Removed" if n_dups > 0 else "None found"}</td></tr>
      <tr><td>Memory Usage</td><td style="color:var(--gold);font-weight:700">{mem_mb} MB</td><td>In-memory size</td></tr>
    </tbody>
  </table>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: COLUMNS                                          -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-columns" class="tab-content">
  <div class="section-title">📋 Column-by-Column Analysis</div>
  <div class="col-stats-grid">
    {columns_section}
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: CHARTS                                           -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-charts" class="tab-content">
  <div class="section-title">🎨 Visual Analytics Gallery ({len(charts)} Charts)</div>
  <div class="charts-grid">
    {charts_section}
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: CORRELATION                                      -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-correlation" class="tab-content">
  {correlation_section}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: NORMALITY / STATS                                -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-normality" class="tab-content">
  {normality_section}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: PREPROCESS                                       -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-preprocess" class="tab-content">
  {preprocess_section}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: DATA SPLIT                                       -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-split" class="tab-content">
  {split_section}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: AUTOML                                           -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-automl" class="tab-content">
  {automl_section}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!--  TAB: CONFIG                                           -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="tab-config" class="tab-content">
  {config_section}
</div>

</div><!-- /page-wrap -->

<script>
// ── Tab system ────────────────────────────────────────────
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  animateProgressBars();
}}

// ── Animated counter ───────────────────────────────────────
function animateCounter(el, end, duration=1500) {{
  const start = 0;
  const step = (end - start) / (duration / 16);
  let current = start;
  const timer = setInterval(() => {{
    current += step;
    if (current >= end) {{ current = end; clearInterval(timer); }}
    el.textContent = Math.round(current).toLocaleString();
  }}, 16);
}}

// ── Progress bar animation ────────────────────────────────
function animateProgressBars() {{
  setTimeout(() => {{
    document.querySelectorAll('.progress-fill[data-width], .score-bar[data-width]').forEach(bar => {{
      bar.style.width = bar.dataset.width + '%';
    }});
  }}, 100);
}}

// ── Particle background ───────────────────────────────────
const canvas = document.getElementById('particles-canvas');
const ctx = canvas.getContext('2d');
let particles = [];

function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}}
resize();
window.addEventListener('resize', resize);

const COLORS = ['#00f5ff','#ff00a0','#b967ff','#39ff14','#ffd700'];

function createParticle() {{
  return {{
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.8 + 0.3,
    dx: (Math.random() - 0.5) * 0.4,
    dy: (Math.random() - 0.5) * 0.4,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
    alpha: Math.random() * 0.6 + 0.1,
    pulse: Math.random() * Math.PI * 2,
    pulseSpeed: Math.random() * 0.02 + 0.005,
  }};
}}

for (let i = 0; i < 120; i++) particles.push(createParticle());

function drawParticles() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => {{
    p.pulse += p.pulseSpeed;
    const alpha = p.alpha * (0.7 + 0.3 * Math.sin(p.pulse));
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = alpha;
    ctx.fill();
    ctx.globalAlpha = 1;
    p.x += p.dx; p.y += p.dy;
    if (p.x < 0 || p.x > canvas.width)  p.dx *= -1;
    if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
  }});

  // Connect nearby particles
  for (let i=0;i<particles.length;i++) {{
    for (let j=i+1;j<particles.length;j++) {{
      const dx=particles[i].x-particles[j].x;
      const dy=particles[i].y-particles[j].y;
      const dist=Math.sqrt(dx*dx+dy*dy);
      if (dist<80) {{
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.strokeStyle='rgba(0,245,255,'+(0.08*(1-dist/80))+')';
        ctx.lineWidth=0.5;
        ctx.stroke();
      }}
    }}
  }}
  requestAnimationFrame(drawParticles);
}}
drawParticles();

// ── On load ───────────────────────────────────────────────
window.addEventListener('load', () => {{
  document.querySelectorAll('[data-counter]').forEach(el => {{
    const val = parseFloat(el.dataset.counter);
    if (!isNaN(val)) animateCounter(el, val);
  }});
  setTimeout(animateProgressBars, 300);
}});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _stat_cards(n_rows, n_cols, n_nulls, n_dups, mem_mb, n_numeric, pp, split):
    cards = [
        ("📦", n_rows, "Total Rows", "--cyan"),
        ("🏷️", n_cols, "Total Columns", "--purple"),
        ("🔢", n_numeric, "Numeric Cols", "--green"),
        ("❓", n_nulls, "Missing Values", "--orange" if n_nulls > 0 else "--green"),
        ("🗂️", n_dups, "Duplicates", "--pink" if n_dups > 0 else "--green"),
        ("💾", mem_mb, "Memory (MB)", "--gold"),
    ]
    if split:
        cards.append(("🚂", split.get("train_rows", 0), "Train Rows", "--cyan"))
        cards.append(("🧪", split.get("test_rows", 0), "Test Rows", "--blue"))

    html = ""
    for i, (icon, val, label, accent) in enumerate(cards):
        delay = i * 0.08
        html += f"""
<div class="stat-card" style="--card-accent:var({accent});animation-delay:{delay:.2f}s">
  <div class="icon">{icon}</div>
  <div class="val" data-counter="{val}">{int(val):,}</div>
  <div class="label">{label}</div>
</div>"""
    return html


def _columns_section(col_stats: dict) -> str:
    html = ""
    for col, s in col_stats.items():
        dtype = s.get("dtype", "?")
        null_pct = s.get("null_pct", 0)
        n_unique = s.get("unique_count", 0)
        is_num = "mean" in s

        html += f"""
<div class="col-card">
  <div class="col-name">📌 {col}</div>
  <div class="col-dtype"><span class="chip chip-purple">{dtype}</span></div>
"""
        rows_data = [
            ("Null %", f"{null_pct}%"),
            ("Unique", f"{n_unique:,}"),
        ]
        if is_num:
            rows_data += [
                ("Mean",   f"{s.get('mean',0):.4f}"),
                ("Median", f"{s.get('median',0):.4f}"),
                ("Std",    f"{s.get('std',0):.4f}"),
                ("Min",    f"{s.get('min',0):.4f}"),
                ("Max",    f"{s.get('max',0):.4f}"),
                ("Skewness", f"{s.get('skewness',0):.4f}"),
                ("IQR",    f"{s.get('iqr',0):.4f}"),
            ]
        else:
            top = s.get("top_values", {})
            rows_data.append(("Mode", str(s.get("mode", "N/A"))))
            for tv, tc in list(top.items())[:3]:
                rows_data.append((f"  '{tv}'", f"{tc:,}"))

        for k, v in rows_data:
            html += f'<div class="col-row"><span>{k}</span><span>{v}</span></div>\n'
        html += "</div>\n"
    return html


def _charts_section(charts: List[str]) -> str:
    html = ""
    for path in charts:
        name = os.path.basename(path)
        b64  = _img_to_b64(path)
        if not b64:
            continue
        html += f"""
<div class="chart-card">
  <img src="{b64}" alt="{name}" loading="lazy">
  <div class="chart-title">📊 {name.replace('_',' ').replace('.png','').title()}</div>
</div>"""
    return html or "<p style='color:var(--txt2)'>No charts generated.</p>"


def _correlation_section(target_corr: dict, corr: dict) -> str:
    html = '<div class="section-title">🔗 Target Correlation</div>'
    if not target_corr:
        html += "<p style='color:var(--txt2)'>No target correlation data.</p>"
    else:
        html += '<table class="data-table"><thead><tr><th>Feature</th><th>Correlation</th><th>Strength</th><th>Bar</th></tr></thead><tbody>'
        for feat, val in list(target_corr.items())[:30]:
            abs_val = abs(val)
            color = "var(--pink)" if abs_val > 0.7 else ("var(--orange)" if abs_val > 0.4 else "var(--cyan)")
            strength = "🔥 Strong" if abs_val > 0.7 else ("⚡ Moderate" if abs_val > 0.4 else "💧 Weak")
            pct = round(abs_val * 100, 1)
            html += f"""<tr>
  <td>{feat}</td>
  <td style="color:{color};font-weight:700">{val:.4f}</td>
  <td>{strength}</td>
  <td style="width:150px">
    <div class="progress-track">
      <div class="progress-fill" data-width="{pct}" style="background:{color};box-shadow:0 0 6px {color}"></div>
    </div>
  </td>
</tr>"""
        html += "</tbody></table>"
    return html


def _normality_section(normality: dict) -> str:
    if not normality:
        return "<p style='color:var(--txt2)'>No normality test data.</p>"
    html = '<div class="section-title">📐 Normality &amp; Distribution Tests</div>'
    html += '<table class="data-table"><thead><tr><th>Column</th><th>Shapiro Stat</th><th>P-Value</th><th>Normal?</th><th>Skewness</th><th>Kurtosis</th></tr></thead><tbody>'
    for col, info in normality.items():
        is_norm = info.get("is_normal", False)
        color = "var(--green)" if is_norm else "var(--pink)"
        label = "✅ Yes" if is_norm else "❌ No"
        skew  = info.get("skewness", 0)
        kurt  = info.get("kurtosis", 0)
        skew_color = "var(--pink)" if abs(skew) > 2 else ("var(--orange)" if abs(skew) > 1 else "var(--txt)")
        html += f"""<tr>
  <td>{col}</td>
  <td>{info.get('shapiro_stat','?')}</td>
  <td>{info.get('shapiro_pval','?')}</td>
  <td style="color:{color};font-weight:700">{label}</td>
  <td style="color:{skew_color}">{skew}</td>
  <td>{kurt}</td>
</tr>"""
    html += "</tbody></table>"
    return html


def _preprocess_section(pp: dict) -> str:
    html = '<div class="section-title">⚙️ Preprocessing Pipeline Summary</div>'
    if not pp:
        return html + "<p style='color:var(--txt2)'>No preprocessing data available.</p>"

    keys_map = {
        "rows_before":       ("📦 Rows Before",   "var(--txt2)"),
        "rows_after":        ("📦 Rows After",    "var(--cyan)"),
        "rows_removed":      ("🗑️ Rows Removed",  "var(--orange)"),
        "cols_before":       ("📋 Cols Before",   "var(--txt2)"),
        "cols_after":        ("📋 Cols After",    "var(--cyan)"),
        "cols_removed":      ("🗑️ Cols Removed",  "var(--orange)"),
    }
    html += '<table class="data-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
    for k, (label, color) in keys_map.items():
        val = pp.get(k, "N/A")
        html += f'<tr><td>{label}</td><td style="color:{color};font-weight:700">{val}</td></tr>'

    imputed = pp.get("columns_imputed", [])
    if imputed:
        html += f'<tr><td>🔧 Imputed Columns</td><td>{"  ".join(f"<span class=chip chip-cyan>{c}</span>" for c in imputed)}</td></tr>'

    enc = pp.get("encoding_log", {})
    if enc:
        chips = "  ".join(f'<span class="chip chip-pink">{c}: {v}</span>' for c, v in list(enc.items())[:15])
        html += f'<tr><td>🔡 Encodings</td><td>{chips}</td></tr>'

    transforms = pp.get("feature_transforms", [])
    if transforms:
        chips = "  ".join(f'<span class="chip chip-green">{t}</span>' for t in transforms[:15])
        html += f'<tr><td>⚙️ New Features</td><td>{chips}</td></tr>'

    html += "</tbody></table>"
    return html


def _split_section(split: dict) -> str:
    html = '<div class="section-title">✂️ Train / Val / Test Split</div>'
    if not split:
        return html + "<p style='color:var(--txt2)'>No split data available.</p>"

    train = split.get("train_rows", 0)
    val   = split.get("val_rows", 0)
    test  = split.get("test_rows", 0)
    total = train + val + test or 1
    n_feat= split.get("n_features", 0)

    html += f"""
<div class="stats-grid" style="max-width:600px">
  <div class="stat-card" style="--card-accent:var(--cyan)">
    <div class="icon">🚂</div>
    <div class="val" data-counter="{train}">{train:,}</div>
    <div class="label">Train Rows ({train/total*100:.1f}%)</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--purple)">
    <div class="icon">🔬</div>
    <div class="val" data-counter="{val}">{val:,}</div>
    <div class="label">Val Rows ({val/total*100:.1f}%)</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--blue)">
    <div class="icon">🧪</div>
    <div class="val" data-counter="{test}">{test:,}</div>
    <div class="label">Test Rows ({test/total*100:.1f}%)</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--green)">
    <div class="icon">🧩</div>
    <div class="val" data-counter="{n_feat}">{n_feat:,}</div>
    <div class="label">Features</div>
  </div>
</div>
<div class="section-title">📁 ML-Ready Files</div>
<table class="data-table">
  <thead><tr><th>File</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td>X_train.npy</td><td>Training feature matrix (numpy array)</td></tr>
    <tr><td>y_train.npy</td><td>Training labels (numpy array)</td></tr>
    <tr><td>X_val.npy</td><td>Validation feature matrix</td></tr>
    <tr><td>y_val.npy</td><td>Validation labels</td></tr>
    <tr><td>X_test.npy</td><td>Test feature matrix</td></tr>
    <tr><td>y_test.npy</td><td>Test labels</td></tr>
    <tr><td>train.csv / val.csv / test.csv</td><td>Human-readable CSV splits</td></tr>
    <tr><td>scaler.pkl</td><td>Fitted scaler (for new data)</td></tr>
    <tr><td>feature_names.txt</td><td>Ordered list of feature columns</td></tr>
    <tr><td>pytorch_dataset.py</td><td>Ready-to-use PyTorch Dataset class</td></tr>
    <tr><td>metadata.json</td><td>Split metadata and feature info</td></tr>
  </tbody>
</table>"""
    return html


def _automl_section(automl: dict) -> str:
    html = '<div class="section-title">🤖 AutoML Results</div>'
    if not automl:
        return html + """
<div style="text-align:center;padding:40px;color:var(--txt2)">
  <div style="font-size:48px">🔒</div>
  <div style="margin-top:12px;font-size:14px">AutoML was not enabled.<br>
  Set <code style="color:var(--cyan)">run_automl=True</code> in your config to build models automatically.</div>
</div>"""

    task       = automl.get("task_type", "unknown")
    score_key  = automl.get("score_metric", "test_accuracy")
    best       = automl.get("best_model", {})
    leaderboard= automl.get("leaderboard", [])
    all_models = automl.get("all_models", [])

    html += f"""
<div class="stats-grid" style="max-width:700px">
  <div class="stat-card" style="--card-accent:var(--gold)">
    <div class="icon">🏆</div>
    <div class="val" style="font-size:18px">{best.get("name","N/A")}</div>
    <div class="label">Best Model</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--green)">
    <div class="icon">⭐</div>
    <div class="val">{best.get("score",0):.4f}</div>
    <div class="label">{score_key.replace("_"," ").title()}</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--cyan)">
    <div class="icon">🤖</div>
    <div class="val" data-counter="{automl.get("total_models_trained",0)}">{automl.get("total_models_trained",0)}</div>
    <div class="label">Models Trained</div>
  </div>
  <div class="stat-card" style="--card-accent:var(--purple)">
    <div class="icon">🧠</div>
    <div class="val">{task.title()}</div>
    <div class="label">Task Type</div>
  </div>
</div>
<div class="section-title">🏅 Model Leaderboard</div>
<div class="leaderboard">"""

    for item in leaderboard:
        rank = item.get("rank", 0)
        name = item.get("name", "?")
        backend = item.get("backend", "sklearn")
        score = item.get(score_key, 0)
        t_time = item.get("training_time", 0)
        overfit = item.get("overfit_risk", "low")
        rank_cls = f"rank-{rank}" if rank <= 3 else "rank-other"
        backend_cls = f"{backend}-tag"
        overfit_cls = f"overfit-{overfit}"
        score_pct = max(0, min(100, round(abs(score) * 100, 1)))

        # Find full model result for extra details
        full = next((m for m in all_models if m.get("name") == name), {})
        rec = full.get("recommendation", "")
        cv_mean = full.get("cv_mean", None)
        cv_std  = full.get("cv_std", None)

        html += f"""
<div class="model-card {rank_cls}">
  <div class="model-header">
    <div class="rank-badge">{rank}</div>
    <div class="model-name">{name}</div>
    <span class="backend-tag {backend_cls}">{backend.upper()}</span>
    <span class="overfit-badge {overfit_cls}">Overfit: {overfit}</span>
  </div>
  <div class="model-metrics">
    <div class="metric-item">
      <div class="metric-val" style="color:var(--gold)">{score:.4f}</div>
      <div class="metric-lab">{score_key.replace("_"," ").upper()}</div>
    </div>"""
        if cv_mean is not None:
            html += f"""
    <div class="metric-item">
      <div class="metric-val" style="color:var(--purple)">{cv_mean:.4f}</div>
      <div class="metric-lab">CV MEAN</div>
    </div>
    <div class="metric-item">
      <div class="metric-val" style="color:var(--orange)">±{cv_std:.4f}</div>
      <div class="metric-lab">CV STD</div>
    </div>"""
        html += f"""
    <div class="metric-item">
      <div class="metric-val" style="color:var(--blue)">{t_time:.3f}s</div>
      <div class="metric-lab">TRAIN TIME</div>
    </div>
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar" data-width="{score_pct}"></div>
  </div>"""
        if rec:
            html += f'<div class="model-recommendation">💡 {rec}</div>'
        html += "</div>"

    html += "</div>"
    return html


def _config_section(cfg: dict) -> str:
    html = '<div class="section-title">🔧 Pipeline Configuration</div>'
    if not cfg:
        return html + "<p style='color:var(--txt2)'>No config data.</p>"
    pretty = json.dumps(cfg, indent=2, default=str)
    html += f'<div class="json-block">{pretty}</div>'
    return html
