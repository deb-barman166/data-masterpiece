# ⚡ Data Masterpiece v3

> **Legend-Level Python Data Science Pipeline**
> Auto + Manual Mode | ML-Ready Output | Animated HTML Report | AutoML

---

## 🚀 What is Data Masterpiece v3?

Data Masterpiece v3 is a **professional-grade, all-in-one data science pipeline** written in pure Python.

You give it raw, messy data. It gives you back:
- ✅ **Clean, ML-ready data** (numpy arrays + CSV)
- ✅ **13+ professional dark-theme charts**
- ✅ **Animated Legend-level HTML report**
- ✅ **Automatic ML model training** (optional)
- ✅ **Deep statistical analysis** of every column

**So easy a 12-year-old can use it. So powerful a data scientist will love it.**

---

## 📦 Installation

```bash
# 1. Clone or download the project
cd data_masterpiece_v3

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install for PyTorch AutoML
pip install torch

# 4. Install as a package (optional)
pip install -e .
```

---

## ⚡ Quick Start (3 Lines!)

```python
from data_masterpiece_v3 import MasterPipeline

# That's it — one call runs the ENTIRE pipeline!
result = MasterPipeline().run("my_data.csv", target="price")
```

**Outputs generated automatically:**
```
output/
├── processed.csv          ← clean, ML-ready CSV
├── report.html            ← animated Legend HTML report
├── plots/
│   ├── 00_missing_values.png
│   ├── 01_correlation_heatmap.png
│   ├── 02_target_distribution.png
│   ├── 03_feature_distributions.png
│   ├── 04_boxplot_grid.png
│   ├── 05_scatter_vs_target.png
│   ├── 06_violin_plots.png
│   ├── 07_skewness.png
│   ├── 08_feature_importance.png
│   ├── 09_pairplot.png
│   ├── 10_categorical_bars.png
│   ├── 11_pie_charts.png
│   └── 12_class_balance.png
└── ml_ready/
    ├── X_train.npy        ← ready for model.fit()!
    ├── X_val.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_val.npy
    ├── y_test.npy
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── scaler.pkl
    ├── feature_names.txt
    ├── metadata.json
    └── pytorch_dataset.py
```

---

## 🎛️ Two Modes

### AUTO Mode (default)
The pipeline decides everything automatically. Zero configuration needed!

```python
from data_masterpiece_v3 import MasterPipeline
result = MasterPipeline().run("data.csv", target="survived")
```

### MANUAL Mode (full control)
You control every step via a JSON config file or Python dict.

```python
from data_masterpiece_v3 import MasterPipeline, Config

cfg = Config.from_json("my_config.json")
result = MasterPipeline(cfg).run("data.csv", target="survived")
```

**Generate a starter config to edit:**
```python
Config().save_json("starter_config.json")
# Now open starter_config.json and edit anything you want!
```

---

## 🤖 AutoML

Enable automatic model training with one config option:

```python
cfg = Config(
    run_automl      = True,
    automl_backends = ["sklearn"],        # or ["sklearn", "pytorch"]
    automl_max_models = 8,
    pytorch_epochs  = 50,                 # only if pytorch is in backends
)
result = MasterPipeline(cfg).run("data.csv", target="price")
```

**Models trained automatically:**

| Type | Models |
|------|--------|
| Classification | LogisticRegression, RandomForest, GradientBoosting, ExtraTrees, SVC, KNN, DecisionTree, GaussianNB, AdaBoost |
| Regression | LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, ExtraTrees, SVR, KNN, DecisionTree |
| Deep Learning | PyTorch MLP (configurable architecture) |

---

## 🔧 Manual Config JSON Reference

```json
{
  "mode": "manual",
  "active_agents": ["cleaning", "type_conversion", "missing", "encoding", "feature", "validation"],

  "global": {
    "drop_duplicates": true,
    "null_drop_threshold": 0.6,
    "normalize": false,
    "scale_method": "minmax"
  },

  "cleaning": {
    "drop_columns": ["id", "row_id"]
  },

  "missing": {
    "age":    "median",
    "salary": "mean",
    "city":   "unknown",
    "date":   "ffill"
  },

  "encoding": {
    "gender":   "binary",
    "city":     "onehot",
    "tags":     "multihot",
    "user_id":  "frequency"
  },

  "type_conversion": {
    "price":    "float",
    "count":    "int"
  },

  "features": {
    "derived": [
      {"type": "ratio",    "col_a": "revenue", "col_b": "cost",  "name": "profit_margin"},
      {"type": "log1p",    "col":   "salary"},
      {"type": "square",   "col":   "age"},
      {"type": "agg_mean", "cols":  ["s1","s2","s3"], "name": "avg_score"}
    ]
  },

  "relationship_columns": [
    ["age", "income", "score"]
  ],

  "run_automl": false,
  "automl_backends": ["sklearn", "pytorch"],
  "output_path": "output/processed.csv"
}
```

---

## 📊 Pipeline Stages

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1 | Loader | Reads CSV, Excel, JSON, Parquet, URL, DataFrame |
| 2 | CleaningAgent | Removes duplicates, high-null cols, zero-variance cols |
| 3 | TypeAgent | Converts dates, booleans, numeric strings automatically |
| 4 | MissingAgent | Fills NaN with mean/median/mode/ffill/constant/etc |
| 5 | EncodingAgent | Converts text→numbers (label/onehot/binary/frequency/multihot) |
| 6 | FeatureAgent | Creates new features (ratio/diff/log/square/polynomial/etc) |
| 7 | ValidationAgent | Final quality check, optional scaling |
| 8 | StatsEngine | Deep statistical analysis of every column |
| 9 | OutlierEngine | IQR/Z-score outlier detection + clip/remove |
| 10 | FeatureSelector | Drops low-variance & high-correlation features |
| 11 | DataSplitter | Train/val/test split → saves numpy arrays |
| 12 | ChartEngine | 13+ professional dark-theme charts |
| 13 | AutoMLBuilder | Trains sklearn models + PyTorch MLP (optional) |
| 14 | ReportBuilder | Animated Legend-level HTML report |

---

## 🎨 HTML Report Features

- **Animated particle background** with neon glow
- **Animated counters** (numbers count up on load)
- **Animated progress bars** for model scores
- **Interactive tabs**: Overview, Columns, Charts, Correlation, Statistics, Pipeline, Data Split, AutoML, Config
- **AutoML Leaderboard** with rank badges (🥇🥈🥉)
- **13+ embedded charts** (no external dependencies needed)
- **Full column-by-column statistics**
- **Deep dark neon theme** — cyan, pink, purple, gold

---

## 📁 Project Structure

```
data_masterpiece_v3/
├── data_masterpiece_v3/
│   ├── __init__.py          ← import MasterPipeline here
│   ├── master.py            ← MasterPipeline (main entry point)
│   ├── config.py            ← Config class (all settings)
│   ├── agents/
│   │   ├── cleaning_agent.py
│   │   ├── missing_agent.py
│   │   ├── encoding_agent.py
│   │   ├── type_agent.py
│   │   ├── feature_agent.py
│   │   └── validation_agent.py
│   ├── intelligence/
│   │   ├── stats.py         ← Deep statistical analysis
│   │   ├── outlier.py       ← Outlier detection
│   │   ├── selector.py      ← Feature selection
│   │   └── splitter.py      ← Train/val/test split
│   ├── automl/
│   │   ├── builder.py       ← AutoML orchestrator
│   │   ├── sklearn_models.py← All sklearn models
│   │   └── pytorch_models.py← PyTorch MLP
│   ├── visualization/
│   │   └── charts.py        ← 13+ chart types
│   ├── reporting/
│   │   └── report_builder.py← Animated HTML report
│   └── utils/
│       ├── logger.py        ← Neon terminal logger
│       └── loader.py        ← Smart data loader
├── examples/
│   ├── auto_mode_example.py
│   ├── manual_mode_example.py
│   └── sample_config.json   ← Edit this for manual mode!
├── README.md
├── DOCUMENTATION.md
├── requirements.txt
└── setup.py
```

---

## 💡 Tips for Best Results

1. **Always specify `target`** — the column you want to predict
2. **In AUTO mode**, the pipeline handles everything — just check the report!
3. **In MANUAL mode**, start by generating a config: `Config().save_json("cfg.json")`
4. **For AutoML**, install PyTorch: `pip install torch`
5. **For relationship analysis**, add column groups to `relationship_columns`

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

Made with ❤️ and 10,000+ hours of Python experience.
**Data Masterpiece v3 — Legend Level** ⚡
