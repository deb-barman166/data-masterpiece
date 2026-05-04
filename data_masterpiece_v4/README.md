# Data Masterpiece v4 — God Level
## Pure Power • Pure Precision • Pure Magic

---

## What is Data Masterpiece v4?

**God-Level** Python Data Science Pipeline that transforms raw data into ML-ready output with maximum intelligence and minimal configuration.

### Three Modes of Power:

1. **GOD AUTO Mode** — Maximum intelligence, zero configuration
2. **SMART AUTO Mode** — Intelligent defaults
3. **MANUAL Mode** — Full control via JSON config

---

## Installation

```bash
pip install data_masterpiece_v4
```

Or for development:

```bash
git clone https://github.com/datamasterpiece/data_masterpiece_v4
cd data_masterpiece_v4
pip install -e .
```

---

## Quick Start

### Python API

```python
from data_masterpiece_v4 import MasterPipeline

# God Auto Mode (maximum intelligence)
result = MasterPipeline(mode="god_auto").run("data.csv", target="price")

# Smart Auto Mode
result = MasterPipeline(mode="smart_auto").run("data.csv", target="price")

# Manual Mode
from data_masterpiece_v4 import Config
cfg = Config.from_json("my_config.json")
result = MasterPipeline(cfg).run("data.csv", target="price")
```

### CLI Interface

```bash
# God Auto Mode (suggests target automatically)
dm4 auto data.csv --target price

# Auto Mode with AutoML
dm4 run data.csv --target price --automl

# Inspect dataset first
dm4 inspect data.csv

# Generate starter config
dm4 config generate --output my_config.json
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `dm4 auto <file>` | Run God-Level Auto Mode |
| `dm4 run <file> -t <target>` | Run pipeline with AUTO mode |
| `dm4 inspect <file>` | Inspect and analyze dataset |
| `dm4 manual <file> <config>` | Run with custom config |
| `dm4 preprocess <file>` | Run preprocessing only |
| `dm4 config generate` | Generate starter config |
| `dm4 config validate <file>` | Validate config file |
| `dm4 info` | Show version info |

---

## Output Files

```
output/
├── processed.csv          # Clean, ML-ready CSV
├── report.html           # Animated God-Level HTML report
├── plots/
│   ├── 00_missing_values.png
│   ├── 01_correlation_heatmap.png
│   ├── 02_target_distribution.png
│   └── ... (14+ charts)
└── ml_ready/
    ├── X_train.npy
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
    └── metadata.json
```

---

## God-Level Features

### Auto Target Detection
Automatically detects the target column based on data characteristics.

### Intelligent Imputation
- Numeric with low skew → mean
- Numeric with high skew → median
- Categorical → mode
- Time series → forward-fill

### Context-Aware Encoding
- Binary (2 values)
- One-hot (low cardinality ≤10)
- Label (medium cardinality ≤50)
- Frequency (high cardinality >50)

### God-Level AutoML
- 10+ sklearn models
- Cross-validation scoring
- Model ranking
- Ensemble of top 3 models

### Caching System
- Cache intermediate results
- Fast pipeline reruns
- Configurable TTL

---

## Configuration Example

```json
{
  "mode": "god_auto",
  "active_agents": ["cleaning", "type_conversion", "missing", "encoding", "feature", "validation"],

  "global": {
    "drop_duplicates": true,
    "null_drop_threshold": 0.6,
    "normalize": false,
    "scale_method": "minmax"
  },

  "missing": {
    "age": "median",
    "salary": "mean",
    "city": "unknown"
  },

  "encoding": {
    "gender": "binary",
    "city": "onehot",
    "category": "label"
  },

  "features": {
    "derived": [
      {"type": "ratio", "col_a": "revenue", "col_b": "cost", "name": "profit_margin"},
      {"type": "log1p", "col": "salary"}
    ]
  },

  "run_automl": true,
  "automl_max_models": 10,
  "automl_ensemble": true
}
```

---

## Pipeline Stages

1. **LOAD** → Multi-format data loader
2. **PROFILE** → God-Level data profiling
3. **CLEAN** → Remove duplicates, nulls, garbage
4. **TYPE** → Smart type conversion
5. **MISSING** → Intelligent imputation
6. **ENCODE** → Categorical → numeric
7. **FEATURES** → Derived feature engineering
8. **VALIDATE** → Quality check + scaling
9. **STATS** → Deep statistical analysis
10. **OUTLIERS** → Detect and handle
11. **SELECT** → Feature selection
12. **SPLIT** → Train/val/test split
13. **CHARTS** → 14+ professional charts
14. **AUTOML** → Train models + ensemble
15. **REPORT** → Animated HTML report

---

## License

MIT License — Free to use, modify, and distribute.

---

Made with ❤️
**Data Masterpiece v4 — God Level** ⚡
