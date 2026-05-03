# 📖 Data Relationship Pipeline — Complete Usage Guide

> **Environments covered:** Google Colab · VS Code · PyCharm · Terminal (Windows / macOS / Linux)

---

## 📋 Table of Contents

1. [Before You Start — Prerequisites](#-before-you-start--prerequisites)
2. [Environment A — Google Colab](#-environment-a--google-colab-recommended-for-beginners)
3. [Environment B — VS Code (Local)](#-environment-b--vs-code-local)
4. [Environment C — PyCharm (Local)](#-environment-c--pycharm-local)
5. [Environment D — Terminal / Command Line](#-environment-d--terminal--command-line-only)
6. [How to Provide Your Data](#-how-to-provide-your-data)
7. [CLI Flags — Full Reference](#-cli-flags--full-reference)
8. [Reading Your Report](#-reading-your-report)
9. [Troubleshooting](#-troubleshooting)
10. [Quick Cheat Sheet](#-quick-cheat-sheet)

---

## ✅ Before You Start — Prerequisites

### What You Need
| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.8 or higher | `python --version` |
| pip | 21.0+ | `pip --version` |
| Internet | For URL datasets + Colab | — |
| Browser | Any modern browser | For viewing `report.html` |

### The 4 Pipeline Files
Make sure you have all four `.py` files in the **same folder**:

```
your_project_folder/
├── main.py           ← Run this file
├── loader.py         ← Data loader (don't edit)
├── analyzer.py       ← Statistical engine (don't edit)
├── reporter.py       ← HTML builder (don't edit)
└── requirements.txt  ← Dependencies list
```

> ⚠️ **Important:** All four `.py` files must be in the same directory. The pipeline will not run if they are separated.

---

## 🌐 Environment A — Google Colab (Recommended for Beginners)

Google Colab gives you a free cloud Python environment — **no installation needed on your computer.**

### Step 1 — Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and click **"New notebook"**.

---

### Step 2 — Upload the Pipeline Files

In the left sidebar, click the **📁 folder icon** to open the Files panel.

Then click the **⬆️ Upload** button and upload all four files:
- `main.py`
- `loader.py`
- `analyzer.py`
- `reporter.py`

> 💡 **Tip:** You can select all 4 files at once in the upload dialog.

After uploading, your file panel should look like:

```
/content/
├── main.py
├── loader.py
├── analyzer.py
└── reporter.py
```

---

### Step 3 — Install Dependencies

In the first cell, run:

```python
!pip install pandas numpy scipy plotly scikit-learn requests openpyxl xlrd rich -q
```

Wait for all packages to finish installing (usually 30–60 seconds).

---

### Step 4A — Upload Your Data File

**Option A: Upload a CSV/Excel/JSON file from your computer**

```python
from google.colab import files
uploaded = files.upload()   # A dialog box will appear — choose your file
```

After uploading, your file (e.g. `sales.csv`) will be available at `/content/sales.csv`.

---

### Step 4B — Use a URL Dataset

**Option B: Use a public dataset URL directly** (no upload needed)

```python
# Iris dataset example
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv"

# Titanic dataset
# DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
```

---

### Step 5 — Run the Pipeline

**For a local uploaded file:**

```python
import subprocess
result = subprocess.run(
    ["python", "main.py",
     "--data", "sales.csv",          # ← replace with your filename
     "--output", "report.html",
     "--no-interactive"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
```

**For a URL:**

```python
import subprocess
result = subprocess.run(
    ["python", "main.py",
     "--data", "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
     "--output", "report.html",
     "--no-interactive"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
```

**To analyze specific columns only:**

```python
import subprocess
result = subprocess.run(
    ["python", "main.py",
     "--data", "sales.csv",
     "--columns", "Age,Salary,Department,Performance",   # ← your column names
     "--output", "report.html",
     "--no-interactive"],
    capture_output=True, text=True
)
print(result.stdout)
```

---

### Step 6 — Download the Report

After the pipeline finishes, download `report.html` to your computer:

```python
from google.colab import files
files.download("report.html")
```

> 💡 Open the downloaded `report.html` in **Chrome, Firefox, or Edge** for the best experience.

---

### Step 7 — (Optional) Preview Inside Colab

You can preview the report directly inside Colab:

```python
from IPython.display import IFrame
IFrame(src="report.html", width="100%", height=800)
```

> ⚠️ Note: Some interactive chart features work better when the file is opened in a standalone browser tab.

---

### 🔄 Complete Colab Notebook (Copy-Paste Ready)

Copy this entire block into a new Colab notebook for a one-click run:

```python
# ════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ════════════════════════════════════════════════
!pip install pandas numpy scipy plotly scikit-learn requests openpyxl xlrd rich -q
print("✅ Dependencies installed")

# ════════════════════════════════════════════════
# CELL 2 — Upload pipeline files
# ════════════════════════════════════════════════
from google.colab import files
print("Upload main.py, loader.py, analyzer.py, reporter.py")
uploaded = files.upload()
print(f"✅ Uploaded: {list(uploaded.keys())}")

# ════════════════════════════════════════════════
# CELL 3 — Upload your data (skip if using URL)
# ════════════════════════════════════════════════
print("Upload your data file (CSV/Excel/JSON):")
data_files = files.upload()
data_filename = list(data_files.keys())[0]
print(f"✅ Data file: {data_filename}")

# ════════════════════════════════════════════════
# CELL 4 — Run the pipeline
# ════════════════════════════════════════════════
import subprocess

result = subprocess.run(
    ["python", "main.py",
     "--data", data_filename,
     "--output", "report.html",
     "--no-interactive"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# ════════════════════════════════════════════════
# CELL 5 — Download the report
# ════════════════════════════════════════════════
files.download("report.html")
print("✅ Report downloaded!")
```

---

## 💻 Environment B — VS Code (Local)

### Step 1 — Install Python

Download from [python.org/downloads](https://www.python.org/downloads/).

During installation on **Windows**, check ✅ **"Add Python to PATH"**.

Verify:
```bash
python --version
# Should print: Python 3.8.x or higher
```

---

### Step 2 — Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com).

Install the **Python extension** by Microsoft:
- Press `Ctrl+Shift+X` (Extensions panel)
- Search `Python`
- Click **Install** on the Microsoft Python extension

---

### Step 3 — Open Your Project Folder

1. Open VS Code
2. Go to **File → Open Folder**
3. Select the folder containing your 4 pipeline files

Your VS Code Explorer should show:
```
EXPLORER
└── your-folder/
    ├── main.py
    ├── loader.py
    ├── analyzer.py
    ├── reporter.py
    └── requirements.txt
```

---

### Step 4 — Open the Integrated Terminal

Press `` Ctrl+` `` (backtick) or go to **Terminal → New Terminal**.

The terminal opens at the bottom of VS Code, already pointing to your folder.

---

### Step 5 — (Recommended) Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal prompt — this means it's active.

> 💡 **Why virtual environments?** They keep this project's packages separate from other Python projects on your computer. Best practice every time.

---

### Step 6 — Install Dependencies

```bash
pip install -r requirements.txt
```

Wait for all packages to install. You should see `Successfully installed ...` at the end.

---

### Step 7 — Run the Pipeline

**Interactive mode (VS Code terminal):**
```bash
python main.py
```

The pipeline will ask you:
```
→ Data path or URL: sales.csv
→ Columns to analyze [Enter for ALL]: Age,Salary,Department
→ Filter specific rows? (No = use all rows): N
```

**With flags (non-interactive):**
```bash
python main.py --data sales.csv --output report.html
```

**With specific columns:**
```bash
python main.py --data sales.csv --columns "Age,Salary,Department" --output report.html
```

---

### Step 8 — Open the Report

After the pipeline finishes, `report.html` appears in your project folder.

**Open it directly from VS Code:**
- Right-click `report.html` in the Explorer
- Click **"Reveal in File Explorer"** (Windows) or **"Reveal in Finder"** (macOS)
- Double-click `report.html` — it opens in your browser

**Or from terminal:**
```bash
# Windows
start report.html

# macOS
open report.html

# Linux
xdg-open report.html
```

---

### VS Code Tips

| Shortcut | Action |
|---|---|
| `` Ctrl+` `` | Open / close terminal |
| `Ctrl+Shift+P` | Command palette |
| `Ctrl+Shift+E` | Toggle file explorer |
| `F5` | Run current Python file with debugger |
| `Ctrl+/` | Toggle comment on selected lines |

---

## 🐍 Environment C — PyCharm (Local)

### Step 1 — Install PyCharm

Download **PyCharm Community Edition** (free) from [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/).

---

### Step 2 — Create a New Project

1. Open PyCharm → Click **"New Project"**
2. Set the **Location** to your project folder
3. Under **Python Interpreter**, select **"New environment using Virtualenv"**
4. Click **Create**

PyCharm automatically creates a virtual environment for you.

---

### Step 3 — Add Pipeline Files

Copy `main.py`, `loader.py`, `analyzer.py`, `reporter.py`, and `requirements.txt` into your project folder.

They will appear in the **Project panel** on the left side.

---

### Step 4 — Install Dependencies

**Option A — Using PyCharm's GUI:**
1. Go to **File → Settings** (Windows) or **PyCharm → Preferences** (macOS)
2. Navigate to **Project → Python Interpreter**
3. Click the **+** button
4. Search for each package (`pandas`, `scipy`, `plotly`, etc.) and click **Install Package**

**Option B — Using PyCharm's Terminal (faster):**
1. Go to **View → Tool Windows → Terminal**
2. Run:
```bash
pip install -r requirements.txt
```

---

### Step 5 — Configure the Run

1. Click the **▶ Run** dropdown at the top → **"Edit Configurations..."**
2. Click the **+** button → **Python**
3. Set:
   - **Script path:** `main.py`
   - **Parameters:** `--data your_file.csv --output report.html --no-interactive`
   - **Working directory:** your project folder
4. Click **OK**

---

### Step 6 — Run the Pipeline

Click the **▶ green Run button** (top right) or press `Shift+F10`.

Watch the output in PyCharm's **Run panel** at the bottom.

---

### Step 7 — Open the Report

After the run completes, right-click `report.html` in the Project panel → **"Open in → Browser → Chrome"**.

---

## ⌨️ Environment D — Terminal / Command Line Only

For users comfortable with the command line — no IDE needed.

### Windows (Command Prompt or PowerShell)

```cmd
:: 1. Navigate to your project folder
cd C:\Users\YourName\projects\data_pipeline

:: 2. Create virtual environment
python -m venv venv

:: 3. Activate it
venv\Scripts\activate

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. Run the pipeline
python main.py --data your_data.csv --output report.html

:: 6. Open the report
start report.html
```

### macOS / Linux (Bash or Zsh)

```bash
# 1. Navigate to your project folder
cd ~/projects/data_pipeline

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the pipeline
python main.py --data your_data.csv --output report.html

# 6. Open the report
open report.html          # macOS
xdg-open report.html      # Linux
```

---

## 📂 How to Provide Your Data

### Option 1 — Local File Path

```bash
# Relative path (file in same folder as main.py)
python main.py --data sales.csv
python main.py --data data/employees.xlsx
python main.py --data reports/q3_data.json

# Absolute path (full path from root)
python main.py --data /home/user/datasets/titanic.csv
python main.py --data C:\Users\Name\Downloads\data.csv
```

### Option 2 — URL (Public Dataset)

```bash
# GitHub raw file
python main.py --data https://raw.githubusercontent.com/plotly/datasets/master/iris.csv

# Any public CSV URL
python main.py --data https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv

# Kaggle (download first, then use local path)
# Note: Kaggle URLs require authentication, download manually
```

### Option 3 — Well-Known Public Datasets for Testing

| Dataset | URL |
|---|---|
| Iris | `https://raw.githubusercontent.com/plotly/datasets/master/iris.csv` |
| Titanic | `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv` |
| Tips | `https://raw.githubusercontent.com/plotly/datasets/master/tips.csv` |
| Gapminder | `https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv` |
| Diamonds | `https://raw.githubusercontent.com/plotly/datasets/master/diamonds.csv` |

---

## 🔧 CLI Flags — Full Reference

```
python main.py [OPTIONS]

OPTIONS:
  -d, --data TEXT        Path to data file (CSV/Excel/JSON/TSV/Parquet) or
                         HTTP/HTTPS URL.
                         Examples:
                           --data sales.csv
                           --data data/file.xlsx
                           --data https://example.com/data.csv

  -c, --columns TEXT     Comma-separated list of column names to analyze.
                         If omitted, ALL columns are analyzed.
                         Examples:
                           --columns Age,Salary,Department
                           --columns "Product Name,Price,Category,Rating"

  -r, --rows TEXT        Comma-separated row indices (0-based) to include.
                         If omitted, ALL rows are used.
                         Examples:
                           --rows 0,5,10,50,100
                           --rows 1,2,3

  -o, --output TEXT      Filename for the output HTML report.
                         Default: report.html
                         Examples:
                           --output my_analysis.html
                           --output results/sales_report.html

  -s, --sample INT       Maximum number of rows to randomly sample.
                         Useful for very large datasets (>100,000 rows).
                         Examples:
                           --sample 5000
                           --sample 10000

  --no-interactive       Skip all interactive prompts.
                         Use this when running from scripts or automation.

  -h, --help             Show help message and exit.
```

---

## 📊 Reading Your Report

Once `report.html` opens in your browser, here's how to navigate it:

### Side Navigation Bar (Left Edge)
Click any icon to jump to that section instantly:
```
📋 → Dataset Overview
📈 → Descriptive Statistics
❓ → Missing Value Analysis
📊 → Column Distributions
🌡️ → Correlation Heatmap
🔗 → Pairwise Relationships ← Most important section
🔵 → Scatter Plots
🏷️ → Categorical Charts
⚠️ → Outlier Detection
🧠 → Mutual Information
```

### Understanding the Pairwise Relationships Section

This is the core of the report. Each card shows one column pair:

```
📈  Age  ↔  Salary                    [Very Strong] [✓ Significant] [Pearson]
─────────────────────────────────────────────────────────────────────────────
r = 0.847    p = 0.000001    Pearson r = 0.847    Spearman ρ = 0.831    n = 500
─────────────────────────────────────────────────────────────────────────────
Very strong positive correlation (r=0.847, p=0.000001).
Statistically significant.
```

**What each field means:**
| Field | Meaning |
|---|---|
| `r` | Correlation coefficient (-1 to +1). Closer to ±1 = stronger relationship |
| `p` | p-value. Below 0.05 = statistically significant |
| `Pearson r` | Linear correlation (best for normally distributed data) |
| `Spearman ρ` | Rank-based correlation (best for skewed data) |
| `n` | Number of data points used (after removing missing values) |
| `F` | F-statistic from ANOVA (numeric vs categorical pairs) |
| `η²` | Eta-squared — practical effect size for ANOVA |
| `χ²` | Chi-square statistic (categorical vs categorical pairs) |
| `Cramér's V` | Effect size for Chi-square (0 = no association, 1 = perfect) |

### Filter Buttons
Use the filter bar to focus on what matters:
- **All** — shows every tested pair
- **📈 Numeric↔Numeric** — only correlation-type pairs
- **📊 Numeric↔Categorical** — only ANOVA-type pairs
- **🔲 Categorical↔Categorical** — only Chi-square pairs
- **⚡ Significant Only** — only pairs with p < 0.05

### Chart Interactions
Every Plotly chart is interactive:
| Action | How |
|---|---|
| Zoom in | Click and drag on the chart |
| Zoom out | Double-click on the chart |
| Pan | Hold Shift + drag |
| Hover for values | Move your mouse over any data point |
| Hide a series | Click its name in the legend |
| Download as PNG | Click the camera icon in the chart toolbar |
| Reset view | Click the house icon in the chart toolbar |

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'plotly'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt

# Or install the specific missing package
pip install plotly
```

### ❌ `FileNotFoundError: File not found: data.csv`
```bash
# Check your current directory
pwd           # macOS/Linux
cd            # Windows

# List files in current directory
ls            # macOS/Linux
dir           # Windows

# Make sure main.py and your data file are in the SAME folder
# Or use the full absolute path:
python main.py --data /full/path/to/your/data.csv
```

### ❌ `ConnectionError: Cannot connect to URL`
- Check your internet connection
- Make sure the URL is a **direct file link** (not a webpage)
- GitHub links should use `raw.githubusercontent.com`, not `github.com`

```bash
# Wrong ❌
python main.py --data https://github.com/user/repo/blob/main/data.csv

# Correct ✅
python main.py --data https://raw.githubusercontent.com/user/repo/main/data.csv
```

### ❌ `ValueError: Unsupported file format: '.txt'`
The pipeline supports: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.parquet`

Convert your file first:
```python
import pandas as pd
df = pd.read_csv("data.txt", sep="\t")   # if tab-separated
df.to_csv("data.csv", index=False)
```

### ❌ Column names not found after `--columns`
Column names are **case-sensitive** and must match exactly.

```bash
# Check exact column names first:
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"

# Then use the exact names:
python main.py --data data.csv --columns "Exact Column Name,Another Column"
```

### ❌ Report opens but charts are blank
- Make sure you're opening `report.html` in a **modern browser** (Chrome, Firefox, Edge)
- Do not open via `file://` in some browsers — instead, drag-and-drop the file into Chrome
- In Colab, use `files.download("report.html")` then open locally

### ❌ `UnicodeDecodeError` when loading CSV
The pipeline tries multiple encodings automatically. If it still fails:
```bash
# Convert to UTF-8 first:
python -c "
import pandas as pd
df = pd.read_csv('data.csv', encoding='latin-1')
df.to_csv('data_utf8.csv', index=False, encoding='utf-8')
"
python main.py --data data_utf8.csv
```

### ❌ Pipeline is very slow on a large file
```bash
# Use --sample to randomly sample N rows
python main.py --data huge_file.csv --sample 10000 --output report.html
```

### ❌ `python` command not found (Windows)
```cmd
:: Try python3 instead
python3 main.py --data data.csv

:: Or add Python to PATH (reinstall Python and check "Add to PATH")
```

---

## ⚡ Quick Cheat Sheet

```bash
# ── SETUP (run once) ────────────────────────────────────────────
pip install -r requirements.txt

# ── BASIC RUNS ──────────────────────────────────────────────────

# Analyze all columns in a CSV
python main.py --data data.csv

# Analyze a URL dataset
python main.py --data https://raw.githubusercontent.com/plotly/datasets/master/iris.csv

# Analyze specific columns
python main.py --data data.csv --columns Age,Salary,Department

# Custom output filename
python main.py --data data.csv --output my_report.html

# Sample large dataset (for speed)
python main.py --data big_data.csv --sample 5000

# Skip all prompts (for scripts)
python main.py --data data.csv --no-interactive

# ── FULL COMMAND EXAMPLE ─────────────────────────────────────────
python main.py \
  --data employees.xlsx \
  --columns "Age,Salary,Department,Years Experience,Performance" \
  --output employee_analysis.html \
  --sample 10000 \
  --no-interactive

# ── OPEN REPORT ──────────────────────────────────────────────────
# Windows
start report.html

# macOS
open report.html

# Linux
xdg-open report.html

# ── CHECK COLUMN NAMES ───────────────────────────────────────────
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"
```

---

## 🌐 Environment Comparison

| Feature | Google Colab | VS Code | PyCharm | Terminal Only |
|---|---|---|---|---|
| Setup difficulty | ⭐ Easiest | ⭐⭐ Easy | ⭐⭐ Easy | ⭐⭐⭐ Moderate |
| Installation required | ❌ None | ✅ Python + VS Code | ✅ Python + PyCharm | ✅ Python only |
| Debugger | Basic | ✅ Full debugger | ✅ Full debugger | ❌ None |
| Works offline | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Best for | Quick analysis, sharing | Daily development | Full project work | Automation/scripts |
| GPU support | ✅ Free GPU | ❌ | ❌ | ❌ |
| File size limit | ~100MB uploads | No limit | No limit | No limit |

---

*Data Relationship Pipeline v1.0 · Documentation*
