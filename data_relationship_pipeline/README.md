<div align="center">

# 🔬 Data Relationship Pipeline

### *Discover Hidden Patterns in Your Data — Automatically*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.9%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)]()

---

**A professional Python CLI pipeline that automatically finds, tests, and visualizes statistical relationships between columns in any dataset — and delivers a beautiful interactive HTML report.**

[🚀 Quick Start](#-quick-start) · [📖 Documentation](#-how-it-works) · [📊 Features](#-features) · [🏗️ Architecture](#%EF%B8%8F-architecture) · [💡 Examples](#-usage-examples)

---

</div>

## 📸 What It Does

```
User:     python main.py --data sales.csv
Pipeline: ──────────────────────────────────────────────────────────────────────
          [1] 📂 Loads data  (CSV · Excel · JSON · TSV · URL)
          [2] 🔍 Detects column types  (numeric · categorical · datetime)
          [3] 📊 Computes descriptive statistics  (mean · std · skewness · …)
          [4] 🔗 Tests every column pair  (Pearson · Spearman · ANOVA · Chi²)
          [5] 🧠 Calculates Mutual Information  (captures non-linear patterns)
          [6] ⚠️  Detects outliers  (IQR method, per column)
          [7] 📈 Generates 8 interactive Plotly charts
          [8] 📝 Builds a dark-themed, self-contained report.html
Output:   report.html  ✅
```

---

## ✨ Features

### 🔬 Statistical Analysis Engine
| Pair Type | Statistical Test | Effect Size Metric |
|---|---|---|
| Numeric ↔ Numeric | Pearson + Spearman Correlation | r (correlation coefficient) |
| Numeric ↔ Categorical | One-way ANOVA | η² (eta-squared) |
| Categorical ↔ Categorical | Chi-Square Test | Cramér's V |
| All Pairs | Mutual Information | MI Score (non-linear) |
| Numeric Columns | IQR Outlier Detection | Outlier % per column |

### 📊 8 Auto-Generated Interactive Charts
| # | Chart | Purpose |
|---|---|---|
| 1 | **Missing Value Bar Chart** | See which columns have incomplete data |
| 2 | **Distribution Histograms** | Shape of every numeric column |
| 3 | **Pearson Correlation Heatmap** | All numeric relationships at a glance |
| 4 | **Scatter Matrix (Pairplot)** | Visual patterns between numeric pairs |
| 5 | **Categorical Bar Charts** | Value frequency for each category |
| 6 | **Top Correlation Scatter Plots** | Strongest pairs with trendlines |
| 7 | **Box Plots (Outlier Detection)** | Spread and outliers per column |
| 8 | **Mutual Information Heatmap** | Non-linear dependency between all columns |

### 🖥️ Smart CLI Interface
- ✅ Full **interactive mode** — just run `python main.py` and answer prompts
- ✅ Full **flag mode** — specify everything in one command
- ✅ **Beautiful terminal UI** powered by Rich (progress bars, colored panels)
- ✅ **Interactive column picker** — see column names, types, sample values before selecting
- ✅ **Row filter** — analyze specific row indices only

### 📂 Universal Data Loader
- ✅ Auto-detects file format from extension (CSV, Excel, JSON, TSV, Parquet)
- ✅ **Downloads from any URL** (HTTP/HTTPS) — paste a GitHub raw link and go
- ✅ Auto-detects CSV separator (`,` `;` `|` `\t`)
- ✅ Auto-detects encoding (UTF-8, Latin-1, CP1252, UTF-16)
- ✅ Non-destructive cleaning (strips whitespace, converts "nan" strings to NaN, infers numeric types)

---

## 🏗️ Architecture

```
data_relationship_pipeline/
│
├── main.py          ← CLI entry point · argument parsing · interactive prompts · pipeline orchestrator
├── loader.py        ← Universal data loader · format detection · URL download · smart cleaning
├── analyzer.py      ← Statistical engine · pairwise tests · chart generation · outlier detection
├── reporter.py      ← HTML report builder · dark UI · chart embedding · JavaScript filtering
├── requirements.txt ← All Python dependencies
└── report.html      ← Generated output (created at runtime)
```

### Design Principles (First Principles Architecture)

> **"Every component has exactly one job. No component knows what the others look like internally."**

| File | Single Responsibility | Knows About |
|---|---|---|
| `main.py` | Orchestration & UX | loader, analyzer, reporter (interfaces only) |
| `loader.py` | Data ingestion | pandas, requests, pathlib |
| `analyzer.py` | Statistical computation + charts | scipy, plotly, sklearn |
| `reporter.py` | HTML rendering | string formatting, results dict |

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the pipeline

**Interactive mode (recommended for first use):**
```bash
python main.py
```
The pipeline will ask you for:
1. Your data path or URL
2. Which columns to analyze
3. Whether to filter specific rows
4. Output filename

**One-liner with flags:**
```bash
python main.py --data your_data.csv
```

**That's it.** Open `report.html` in any browser.

---

## 💡 Usage Examples

### Example 1 — Local CSV file
```bash
python main.py --data sales_data.csv
```

### Example 2 — Select specific columns
```bash
python main.py --data sales_data.csv --columns Age,Salary,Department,Performance
```

### Example 3 — Download from URL (GitHub raw, Kaggle, etc.)
```bash
python main.py --data https://raw.githubusercontent.com/plotly/datasets/master/iris.csv
```

### Example 4 — Full control
```bash
python main.py \
  --data employees.xlsx \
  --columns Age,Salary,Department,YearsExperience \
  --output employee_analysis.html \
  --sample 1000
```

### Example 5 — Analyze specific rows
```bash
python main.py --data data.csv --rows 0,10,50,100,200
```

### Example 6 — Skip interactive prompts (for scripts/automation)
```bash
python main.py --data data.csv --no-interactive --output report.html
```

---

## 🧪 CLI Flags Reference

| Flag | Short | Description | Default |
|---|---|---|---|
| `--data` | `-d` | Path to file or URL | *(interactive prompt)* |
| `--columns` | `-c` | Comma-separated column names | *(all columns)* |
| `--rows` | `-r` | Comma-separated row indices | *(all rows)* |
| `--output` | `-o` | Output HTML filename | `report.html` |
| `--sample` | `-s` | Max rows to sample | *(all rows)* |
| `--no-interactive` | — | Skip all prompts | `False` |

---

## 📊 Supported File Formats

| Format | Extensions | Notes |
|---|---|---|
| CSV | `.csv` | Auto-detects separator and encoding |
| Excel | `.xlsx`, `.xls` | Reads first sheet by default |
| JSON | `.json` | Tries multiple orientations automatically |
| TSV | `.tsv` | Tab-separated values |
| Parquet | `.parquet` | High-performance columnar format |
| **URL** | `http://`, `https://` | Downloads and auto-detects format |

---

## 📈 Report Sections

The generated `report.html` contains **10 fully interactive sections**:

| # | Section | What You Learn |
|---|---|---|
| 1 | 📋 Dataset Overview | Shape, data types, memory, health score |
| 2 | 📈 Descriptive Statistics | Mean, median, std, skewness, quartiles per column |
| 3 | ❓ Missing Value Analysis | Which columns are incomplete and by how much |
| 4 | 📊 Distributions | Shape of each numeric column's distribution |
| 5 | 🌡️ Correlation Heatmap | All numeric correlations in one view |
| 6 | 🔗 Pairwise Relationships | Every pair tested — filterable by type/significance |
| 7 | 🔵 Scatter Plots | Top correlated pairs with OLS trendlines |
| 8 | 🏷️ Categorical Charts | Frequency distribution of categorical columns |
| 9 | ⚠️ Outlier Detection | IQR-based outlier counts and fences |
| 10 | 🧠 Mutual Information | Non-linear dependencies between all columns |

### Interactive Features in the Report
- 🔍 **Filter relationships** by type (Numeric↔Numeric, Numeric↔Categorical, Categorical↔Categorical)
- ⚡ **Show only significant** relationships (p < 0.05)
- 📌 **Sticky side navigation** — jump to any section instantly
- 🖱️ **Zoom, pan, hover** on all Plotly charts
- 💾 **Download charts** as PNG from the chart toolbar

---

## 🔬 Statistical Test Interpretation Guide

### Correlation Strength (r / Cramér's V)
| Range | Strength | Meaning |
|---|---|---|
| 0.80 – 1.00 | Very Strong | Near-linear relationship |
| 0.60 – 0.79 | Strong | Clear pattern in data |
| 0.40 – 0.59 | Moderate | Noticeable but noisy pattern |
| 0.20 – 0.39 | Weak | Slight tendency |
| 0.00 – 0.19 | Very Weak | No meaningful relationship |

### ANOVA Effect Size (η² / eta-squared)
| η² | Interpretation |
|---|---|
| ≥ 0.14 | Large effect — group differences are practically significant |
| 0.06 – 0.13 | Medium effect |
| 0.01 – 0.05 | Small effect |
| < 0.01 | Negligible effect |

### p-value (Statistical Significance)
| p-value | Interpretation |
|---|---|
| < 0.001 | Extremely significant |
| 0.001 – 0.01 | Highly significant |
| 0.01 – 0.05 | Significant (standard threshold) |
| > 0.05 | Not statistically significant |

---

## 🧰 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5.0 | Data loading and manipulation |
| `numpy` | ≥ 1.21.0 | Numerical operations |
| `scipy` | ≥ 1.9.0 | Statistical tests (Pearson, Spearman, ANOVA, Chi²) |
| `plotly` | ≥ 5.10.0 | Interactive chart generation |
| `scikit-learn` | ≥ 1.1.0 | Mutual information calculation |
| `requests` | ≥ 2.28.0 | URL-based data download |
| `openpyxl` | ≥ 3.0.0 | Excel (.xlsx) file reading |
| `xlrd` | ≥ 2.0.0 | Excel (.xls) legacy file reading |
| `rich` | ≥ 12.0.0 | Beautiful terminal UI (progress bars, panels) |

---

## ⚠️ Known Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| Very large files | Files >500MB may be slow | Use `--sample 10000` to sample |
| High-cardinality categoricals | 100+ unique values may clutter bar charts | Pipeline limits bar charts to top 15 values |
| Multi-sheet Excel | Only reads the first sheet | Pre-process with pandas to select sheet |
| Non-UTF-8 URLs | Some data portals block automated downloads | Download manually, then pass local path |
| DateTime columns | Time-series analysis not yet included | Planned for v2.0 |

---

## 🗺️ Roadmap

### v1.0 (Current)
- [x] CSV, Excel, JSON, TSV, Parquet loading
- [x] URL-based data download
- [x] Pearson, Spearman, ANOVA, Chi-Square, Cramér's V
- [x] Mutual Information (linear + non-linear)
- [x] IQR outlier detection
- [x] 8 interactive Plotly charts
- [x] Dark-themed self-contained HTML report
- [x] Rich CLI with interactive prompts

### v2.0 (Planned)
- [ ] Time-series relationship detection
- [ ] Regression analysis (linear, polynomial)
- [ ] Clustering-based relationship discovery
- [ ] PDF export of the report
- [ ] Config file support (`pipeline.yaml`)
- [ ] Dataset comparison mode (before vs. after)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with First Principles Architecture · Python · Plotly · SciPy · Rich**

*"Don't just look at data. Understand it."*

</div>
