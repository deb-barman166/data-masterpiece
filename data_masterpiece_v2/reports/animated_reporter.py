"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ANIMATED HTML REPORT GENERATOR - Make Beautiful Reports!        ║
║                                                                            ║
║  Creates stunning, animated HTML reports with:                            ║
║  ✨ Dark theme with neon accents                                           ║
║  📊 Interactive charts and visualizations                                  ║
║  🎬 Smooth animations and transitions                                       ║
║  📱 Responsive design                                                       ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
    • Particle animation background
    • Animated stat counters
    • Interactive expandable sections
    • Color-coded data quality badges
    • Professional model ranking cards
    • Correlation visualizations
    • Data distribution charts

Usage:
    >>> from data_masterpiece_v2.reports import AnimatedReportGenerator
    >>> generator = AnimatedReportGenerator()
    >>> path = generator.generate(results)
    >>> print(f"Report saved to: {path}")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import base64

from data_masterpiece_v2.utils.logger import get_logger
from data_masterpiece_v2.utils.helpers import format_bytes, format_duration

logger = get_logger("AnimatedReporter")


class AnimatedReportGenerator:
    """
    ═══════════════════════════════════════════════════════════════════════════
    ANIMATED HTML REPORT GENERATOR
    ═══════════════════════════════════════════════════════════════════════════

    Creates beautiful, animated HTML reports with a dark theme!

    Parameters
    ----------
    output_path : str
        Where to save the HTML report.
    title : str
        Report title.

    Examples
    --------
    Basic usage:

        >>> generator = AnimatedReportGenerator()
        >>> path = generator.generate(results)
        >>> print(f"Report: {path}")

    Custom title:

        >>> generator = AnimatedReportGenerator(
        ...     output_path="my_report.html",
        ...     title="My Analysis Report"
        ... )
        >>> generator.generate(results)

    ═══════════════════════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        output_path: str = "output/report.html",
        title: str = "Data Masterpiece Analysis Report"
    ):
        """Initialize the report generator."""
        self.output_path = output_path
        self.title = title

    def generate(self, results: Dict[str, Any]) -> str:
        """
        Generate the animated HTML report.

        Parameters
        ----------
        results : Dict
            Pipeline results containing all analysis data.

        Returns
        -------
        str
            Path to the generated report.
        """
        logger.info(f"Generating animated HTML report...")

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML content
        html = self._generate_html(results)

        # Save report
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Report saved: {self.output_path}")

        return self.output_path

    def _generate_html(self, results: Dict[str, Any]) -> str:
        """Generate the complete HTML content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract data
        df_processed = results.get('df_processed', pd.DataFrame())
        target = results.get('target', 'unknown')
        elapsed = results.get('elapsed_formatted', 'N/A')
        n_rows = len(df_processed)
        n_cols = len(df_processed.columns)

        # Get intelligence results
        profile = results.get('profile', {})
        features = results.get('features', {})
        relationships = results.get('relationships', {})
        recommendations = results.get('recommendations', {})
        ml_results = results.get('best_model', {})

        # Build HTML sections
        sections = []
        sections.append(self._section_hero(target, n_rows, n_cols, timestamp, elapsed))
        sections.append(self._section_executive_summary(results))
        sections.append(self._section_data_overview(df_processed, target))
        sections.append(self._section_correlations(relationships))
        sections.append(self._section_feature_importance(features))
        sections.append(self._section_model_recommendations(recommendations))

        if ml_results:
            sections.append(self._section_ml_results(ml_results))

        # Combine all
        html = self._HTML_TEMPLATE.format(
            title=self.title,
            timestamp=timestamp,
            sections='\n'.join(sections)
        )

        return html

    def _section_hero(
        self,
        target: str,
        n_rows: int,
        n_cols: int,
        timestamp: str,
        elapsed: str
    ) -> str:
        """Generate the hero section with animated stats."""
        return f'''
        <div class="hero">
            <canvas id="particles"></canvas>
            <div class="hero-content">
                <h1 class="logo">DATA MASTERPIECE</h1>
                <div class="version">VERSION 2.0 - LEGEND LEVEL</div>
                <div class="subtitle">From Raw Data to ML Masterpiece</div>
                <div class="glow-line"></div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value" data-count="{n_rows}">{n_rows:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-value">{n_cols}</div>
                <div class="stat-label">Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-value" style="font-size:1.5em">{elapsed}</div>
                <div class="stat-label">Processing Time</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-icon">🎯</div>
                <div class="stat-value">{target}</div>
                <div class="stat-label">Target Variable</div>
            </div>
        </div>
        '''

    def _section_executive_summary(self, results: Dict) -> str:
        """Generate executive summary section."""
        stages = results.get('stages_completed', [])
        problem_type = results.get('problem_type', 'N/A')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">📋 Executive Summary</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Pipeline Stages</h3>
                        <div class="pipeline-stages">
                            <span class="stage {'completed' if 'preprocessing' in stages else ''}">Preprocessing</span>
                            <span class="stage-arrow">→</span>
                            <span class="stage {'completed' if 'intelligence' in stages else ''}">Analysis</span>
                            <span class="stage-arrow">→</span>
                            <span class="stage {'completed' if 'ml_building' in stages else ''}">ML Models</span>
                            <span class="stage-arrow">→</span>
                            <span class="stage {'completed' if 'report' in stages else ''}">Report</span>
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Problem Type</h3>
                        <div class="problem-badge">{problem_type.upper()}</div>
                    </div>
                </div>
            </div>
        </div>
        '''

    def _section_data_overview(self, df: pd.DataFrame, target: str) -> str:
        """Generate data overview section."""
        if df.empty:
            return ''

        # Get column info
        col_info = []
        for col in df.columns[:20]:
            dtype = str(df[col].dtype)
            null_pct = df[col].isna().sum() / len(df) * 100
            unique = df[col].nunique()

            col_info.append({
                'name': col,
                'dtype': dtype,
                'null_pct': null_pct,
                'unique': unique,
                'is_target': col == target
            })

        rows = []
        for info in col_info:
            badge = '<span class="badge badge-gold">TARGET</span>' if info['is_target'] else ''
            quality_class = 'quality-high' if info['null_pct'] < 5 else 'quality-med' if info['null_pct'] < 20 else 'quality-low'

            rows.append(f'''
                <tr>
                    <td><strong>{info['name']}</strong> {badge}</td>
                    <td><span class="badge badge-blue">{info['dtype']}</span></td>
                    <td>{info['unique']:,}</td>
                    <td><span class="{quality_class}">{info['null_pct']:.1f}%</span></td>
                </tr>
            ''')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">📊 Data Overview</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Type</th>
                                <th>Unique</th>
                                <th>Null %</th>
                            </tr>
                        </thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
            </div>
        </div>
        '''

    def _section_correlations(self, relationships: Dict) -> str:
        """Generate correlation analysis section."""
        target_corr = relationships.get('target_correlations', {})
        strong_pairs = relationships.get('strong_pairs', [])

        if not target_corr:
            return ''

        # Top correlations
        top_corr = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        corr_rows = []
        for feat, corr in top_corr:
            bar_width = abs(corr) * 100
            bar_class = 'corr-high' if abs(corr) > 0.5 else 'corr-med' if abs(corr) > 0.3 else 'corr-low'

            corr_rows.append(f'''
                <div class="corr-row">
                    <div class="corr-label">{feat}</div>
                    <div class="corr-bar">
                        <div class="corr-bar-fill {bar_class}" style="width:{bar_width}%"></div>
                    </div>
                    <div class="corr-value {bar_class}">{corr:.3f}</div>
                </div>
            ''')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">🔗 Correlation Analysis</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <div class="correlation-chart">
                    <h3>Feature Correlations with Target</h3>
                    <div class="corr-list">{"".join(corr_rows)}</div>
                </div>
            </div>
        </div>
        '''

    def _section_feature_importance(self, features: Dict) -> str:
        """Generate feature importance section."""
        if not features or 'scores' not in features:
            return ''

        scores = features.get('scores', {})
        selected = features.get('selected_features', [])

        top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]

        rows = []
        for i, (feat, score) in enumerate(top_features):
            is_selected = feat in selected
            badge = '<span class="badge badge-green">SELECTED</span>' if is_selected else ''

            bar_width = (score / top_features[0][1] * 100) if top_features[0][1] > 0 else 0

            rows.append(f'''
                <tr>
                    <td>{i+1}</td>
                    <td><strong>{feat}</strong> {badge}</td>
                    <td>
                        <div class="mini-bar">
                            <div class="mini-bar-fill" style="width:{bar_width}%"></div>
                        </div>
                    </td>
                    <td><strong>{score:.4f}</strong></td>
                </tr>
            ''')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">⭐ Feature Importance</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <p class="section-desc">Top features ranked by importance to the target variable.</p>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr><th>#</th><th>Feature</th><th>Importance</th><th>Score</th></tr>
                        </thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
            </div>
        </div>
        '''

    def _section_model_recommendations(self, recommendations: Dict) -> str:
        """Generate model recommendations section."""
        models = recommendations.get('models', [])

        if not models:
            return ''

        model_cards = []
        for i, model in enumerate(models[:5]):
            priority_class = f'priority-{i+1}'
            pros = ''.join([f'<li>✅ {p}</li>' for p in model.get('pros', [])[:2]])
            cons = ''.join([f'<li>⚠️ {c}</li>' for c in model.get('cons', [])[:1]])

            model_cards.append(f'''
                <div class="model-card {priority_class}">
                    <div class="model-header">
                        <div class="model-rank">#{i+1}</div>
                        <div class="model-name">{model.get('name', 'Unknown')}</div>
                    </div>
                    <div class="model-body">
                        <div class="model-score">
                            <svg class="score-ring" viewBox="0 0 36 36">
                                <circle class="score-ring-bg" cx="18" cy="18" r="15.9"/>
                                <circle class="score-ring-fill" cx="18" cy="18" r="15.9"
                                    stroke-dasharray="{model.get('score', 0) * 100}, 100"/>
                            </svg>
                        </div>
                        <div class="model-details">
                            <div class="detail-row">
                                <span class="detail-label">Category:</span> {model.get('category', 'N/A')}
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Complexity:</span> {model.get('complexity', 'N/A')}
                            </div>
                            <ul class="pros-cons">
                                {pros}{cons}
                            </ul>
                        </div>
                    </div>
                </div>
            ''')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">🤖 Model Recommendations</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <div class="best-model-banner">
                    <span class="crown">👑</span>
                    <span>Recommended Model: <strong>{models[0].get('name', 'N/A')}</strong></span>
                </div>
                <div class="models-grid">{"".join(model_cards)}</div>
            </div>
        </div>
        '''

    def _section_ml_results(self, ml_results: Dict) -> str:
        """Generate ML results section."""
        name = ml_results.get('name', 'N/A')
        score = ml_results.get('score', 0)
        metrics = ml_results.get('metrics', {})

        metric_rows = []
        for metric, value in metrics.items():
            metric_rows.append(f'''
                <div class="metric-item">
                    <span class="metric-name">{metric.upper()}</span>
                    <span class="metric-value">{value:.4f}</span>
                </div>
            ''')

        return f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-title">🏆 Trained Model Results</span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="section-content">
                <div class="best-model-banner">
                    <span class="crown">🏆</span>
                    <span>Best Model: <strong>{name}</strong></span>
                </div>
                <div class="ml-metrics">
                    <div class="ml-score">
                        <div class="big-score">{score:.4f}</div>
                        <div class="score-label">Test Score</div>
                    </div>
                    <div class="ml-metrics-grid">{"".join(metric_rows)}</div>
                </div>
            </div>
        </div>
        '''

    # ═══════════════════════════════════════════════════════════════════════════
    # HTML TEMPLATE - The complete styling and JavaScript
    # ═══════════════════════════════════════════════════════════════════════════

    _HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* ═══════════════════════════════════════════════════════════════════
           DATA MASTERPIECE V2 - LEGENDARY DARK THEME
           ═══════════════════════════════════════════════════════════════════ */

        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #0f0f1a;
            --bg-card: #12121f;
            --bg-hover: #1a1a2e;
            --border: #2a2a3e;
            --text-primary: #e0e0ff;
            --text-secondary: #a0a0c0;
            --text-muted: #606080;
            --cyan: #00f5ff;
            --pink: #ff00a0;
            --green: #39ff14;
            --orange: #ff6b35;
            --purple: #b967ff;
            --gold: #ffd700;
            --red: #ff3366;
            --blue: #00bfff;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        /* Hero Section */
        .hero {{
            position: relative;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #1a0a2e 100%);
            overflow: hidden;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 50% 50%, rgba(0,245,255,0.05) 0%, transparent 50%);
            animation: pulse 8s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.2); opacity: 0.8; }}
        }}

        .hero-content {{
            position: relative;
            z-index: 2;
            text-align: center;
        }}

        .logo {{
            font-size: 2.5em;
            font-weight: 900;
            letter-spacing: 0.15em;
            background: linear-gradient(90deg, var(--cyan), var(--purple), var(--pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s ease-in-out infinite;
        }}

        @keyframes shimmer {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.3); }}
        }}

        .version {{
            font-size: 1em;
            color: var(--cyan);
            margin-top: 8px;
            letter-spacing: 0.3em;
        }}

        .subtitle {{
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-top: 10px;
        }}

        .glow-line {{
            width: 200px;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), transparent);
            margin: 15px auto 0;
            animation: glowExpand 4s ease-in-out infinite;
        }}

        @keyframes glowExpand {{
            0%, 100% {{ width: 200px; opacity: 0.6; }}
            50% {{ width: 300px; opacity: 1; }}
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            max-width: 1200px;
            margin: -40px auto 30px;
            padding: 0 20px;
            position: relative;
            z-index: 10;
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 15px;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-4px);
            border-color: var(--cyan);
            box-shadow: 0 8px 32px rgba(0,245,255,0.1);
        }}

        .stat-card.highlight {{
            border-color: var(--gold);
            background: linear-gradient(135deg, rgba(255,215,0,0.1), transparent);
        }}

        .stat-icon {{ font-size: 1.5em; margin-bottom: 8px; }}
        .stat-value {{
            font-size: 2em;
            font-weight: 800;
            color: var(--cyan);
        }}
        .stat-label {{
            font-size: 0.75em;
            color: var(--text-secondary);
            margin-top: 4px;
            text-transform: uppercase;
        }}

        /* Sections */
        .section {{
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }}

        .section-header {{
            background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 24px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }}

        .section-header:hover {{
            border-color: var(--cyan);
        }}

        .section-title {{
            font-size: 1.1em;
            font-weight: 600;
        }}

        .toggle-icon {{
            color: var(--cyan);
            transition: transform 0.3s ease;
        }}

        .section-header.collapsed .toggle-icon {{ transform: rotate(-90deg); }}

        .section-content {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 24px;
            animation: slideDown 0.3s ease;
        }}

        .section-header.collapsed + .section-content {{ display: none; }}

        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .section-desc {{
            color: var(--text-secondary);
            margin-bottom: 16px;
            font-size: 0.9em;
        }}

        /* Tables */
        .table-container {{ overflow-x: auto; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }}

        .data-table th {{
            background: var(--bg-hover);
            color: var(--cyan);
            padding: 12px 14px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75em;
            border-bottom: 2px solid var(--border);
        }}

        .data-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }}

        .data-table tbody tr:hover {{ background: rgba(0,245,255,0.04); }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
        }}

        .badge-blue {{ background: rgba(0,191,255,0.15); color: var(--blue); }}
        .badge-green {{ background: rgba(57,255,20,0.15); color: var(--green); }}
        .badge-gold {{ background: rgba(255,215,0,0.2); color: var(--gold); }}

        .quality-high {{ color: var(--green); }}
        .quality-med {{ color: var(--gold); }}
        .quality-low {{ color: var(--red); }}

        /* Summary */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
        }}

        .summary-card h3 {{
            color: var(--cyan);
            margin-bottom: 15px;
            font-size: 0.9em;
            text-transform: uppercase;
        }}

        .pipeline-stages {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .stage {{
            background: var(--bg-hover);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            color: var(--text-muted);
        }}

        .stage.completed {{
            background: rgba(57,255,20,0.15);
            color: var(--green);
        }}

        .stage-arrow {{ color: var(--cyan); }}

        .problem-badge {{
            display: inline-block;
            padding: 8px 20px;
            background: linear-gradient(135deg, var(--purple), var(--cyan));
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.1em;
        }}

        /* Correlation */
        .corr-list {{ margin-top: 15px; }}
        .corr-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }}

        .corr-label {{
            width: 150px;
            font-size: 0.85em;
            color: var(--text-secondary);
        }}

        .corr-bar {{
            flex: 1;
            height: 8px;
            background: var(--bg-hover);
            border-radius: 4px;
            overflow: hidden;
        }}

        .corr-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
        }}

        .corr-high {{ color: var(--green); background: linear-gradient(90deg, var(--green), var(--cyan)); }}
        .corr-med {{ color: var(--gold); background: var(--gold); }}
        .corr-low {{ color: var(--text-muted); background: var(--text-muted); }}

        .corr-value {{
            width: 60px;
            text-align: right;
            font-weight: 600;
            font-size: 0.85em;
        }}

        /* Mini Bar */
        .mini-bar {{
            width: 100px;
            height: 6px;
            background: var(--bg-hover);
            border-radius: 3px;
            overflow: hidden;
        }}

        .mini-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            border-radius: 3px;
        }}

        /* Model Cards */
        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .model-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .model-card:hover {{
            border-color: var(--purple);
            transform: translateY(-4px);
        }}

        .model-card.priority-1 {{ border-left: 3px solid var(--gold); }}
        .model-card.priority-2 {{ border-left: 3px solid var(--cyan); }}
        .model-card.priority-3 {{ border-left: 3px solid var(--purple); }}

        .model-header {{
            padding: 14px 18px;
            background: var(--bg-hover);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .model-rank {{
            background: var(--purple);
            color: white;
            width: 28px; height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: 700;
        }}

        .model-name {{ font-weight: 700; flex: 1; }}

        .model-body {{
            padding: 18px;
            display: flex;
            gap: 18px;
        }}

        .model-score {{ text-align: center; }}
        .score-ring {{
            width: 60px; height: 60px;
        }}
        .score-ring svg {{ transform: rotate(-90deg); }}
        .score-ring-bg {{
            fill: none;
            stroke: var(--bg-hover);
            stroke-width: 3;
        }}
        .score-ring-fill {{
            fill: none;
            stroke: var(--cyan);
            stroke-width: 3;
            stroke-linecap: round;
            transition: stroke-dasharray 1.5s ease;
        }}

        .model-details {{ flex: 1; }}
        .detail-row {{
            font-size: 0.85em;
            margin-bottom: 4px;
            color: var(--text-secondary);
        }}
        .detail-label {{ color: var(--text-muted); }}

        .pros-cons {{
            list-style: none;
            margin-top: 10px;
            font-size: 0.8em;
        }}

        .pros-cons li {{ margin-bottom: 4px; }}

        /* Best Model Banner */
        .best-model-banner {{
            background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,107,53,0.1));
            border: 1px solid var(--gold);
            border-radius: 10px;
            padding: 14px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .crown {{ font-size: 1.3em; }}

        /* ML Metrics */
        .ml-metrics {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 30px;
            align-items: center;
        }}

        .ml-score {{ text-align: center; }}
        .big-score {{
            font-size: 3em;
            font-weight: 800;
            color: var(--cyan);
        }}
        .score-label {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}

        .ml-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }}

        .metric-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}

        .metric-name {{
            display: block;
            font-size: 0.75em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.2em;
            font-weight: 700;
            color: var(--cyan);
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
            font-size: 0.85em;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .ml-metrics {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    {sections}

    <div class="footer">
        <p>Generated by Data Masterpiece V2 | {timestamp}</p>
        <p>✨ Making Data Science Easy for Everyone ✨</p>
    </div>

    <script>
        // Toggle section visibility
        function toggleSection(header) {{
            header.classList.toggle('collapsed');
        }}

        // Particle animation
        const canvas = document.getElementById('particles');
        if (canvas) {{
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = 300;

            const particles = [];
            for (let i = 0; i < 50; i++) {{
                particles.push({{
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    size: Math.random() * 2
                }});
            }}

            function animate() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                particles.forEach(p => {{
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                    ctx.fillStyle = 'rgba(0, 245, 255, 0.3)';
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                }});
                requestAnimationFrame(animate);
            }}
            animate();
        }}

        // Animate counters
        document.querySelectorAll('[data-count]').forEach(el => {{
            const target = parseInt(el.dataset.count);
            let current = 0;
            const increment = target / 50;
            const timer = setInterval(() => {{
                current += increment;
                if (current >= target) {{
                    el.textContent = target.toLocaleString();
                    clearInterval(timer);
                }} else {{
                    el.textContent = Math.floor(current).toLocaleString();
                }}
            }}, 30);
        }});
    </script>
</body>
</html>'''
