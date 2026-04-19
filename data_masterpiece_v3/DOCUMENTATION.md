# 📚 Data Masterpiece v3 — Full Documentation

---

## Table of Contents

1. [MasterPipeline](#masterpipeline)
2. [Config Reference](#config-reference)
3. [Agent Reference](#agent-reference)
4. [Intelligence Engine](#intelligence-engine)
5. [AutoML Reference](#automl-reference)
6. [Chart Engine](#chart-engine)
7. [Report Builder](#report-builder)
8. [Feature Engineering Guide](#feature-engineering-guide)
9. [Encoding Guide](#encoding-guide)
10. [Missing Value Strategies](#missing-value-strategies)
11. [Output Files Reference](#output-files-reference)
12. [FAQ](#faq)

---

## MasterPipeline

The single entry point for the entire pipeline.

### Constructor

```python
MasterPipeline(config: Config = None)
```

If `config` is None, uses AUTO mode with all defaults.

### Methods

#### `run(source, target, ask_automl=True) → dict`

Runs the full end-to-end pipeline.

**Parameters:**
- `source` — file path (CSV/Excel/JSON/Parquet), URL, or DataFrame
- `target` — column name to predict (ML target column)
- `ask_automl` — if True, prompts user before training models

**Returns dict with:**
```python
{
  "df_raw":             pd.DataFrame,   # original untouched data
  "df_clean":           pd.DataFrame,   # after preprocessing
  "df_processed":       pd.DataFrame,   # final ML-ready data
  "stats":              dict,           # full statistical analysis
  "preprocess_summary": dict,           # what each agent did
  "split":              SplitResult,    # train/val/test split object
  "split_info":         dict,           # split row counts
  "automl_results":     dict,           # model leaderboard (if run)
  "charts":             list[str],      # paths to generated charts
  "report_path":        str,            # path to HTML report
  "csv_path":           str,            # path to processed CSV
  "elapsed_s":          float,          # total time in seconds
}
```

#### `preprocess_only(source, target=None) → (DataFrame, dict)`

Runs only preprocessing stages 1–7. No charts, no report, no AutoML.

```python
df_clean, summary = MasterPipeline().preprocess_only("data.csv")
```

#### `generate_starter_config(path="starter_config.json") → str`

Saves a full config JSON with all default values — edit it for manual mode.

```python
MasterPipeline().generate_starter_config("my_config.json")
```

---

## Config Reference

```python
from data_masterpiece_v3 import Config
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"auto"` | `"auto"` or `"manual"` |
| `active_agents` | list | all | Which agents to run |
| `drop_duplicates` | bool | True | Remove exact duplicate rows |
| `null_drop_threshold` | float | 0.60 | Drop col if >60% null |
| `variance_threshold` | float | 1e-10 | Drop near-zero-variance cols |
| `low_card_threshold` | int | 10 | ≤N unique → treat as categorical |
| `normalize` | bool | False | Scale features after preprocessing |
| `scale_method` | str | `"minmax"` | `"minmax"`, `"standard"`, `"robust"` |
| `log_transform_skewed` | bool | False | Auto log1p on skewed columns |
| `test_size` | float | 0.20 | Fraction for test set |
| `val_size` | float | 0.10 | Fraction for validation set |
| `stratify` | bool | True | Stratified split (classification) |
| `run_automl` | bool | False | Run AutoML (requires user confirmation) |
| `automl_backends` | list | `["sklearn"]` | `"sklearn"` and/or `"pytorch"` |
| `automl_max_models` | int | 8 | Max sklearn models to try |
| `pytorch_epochs` | int | 50 | Neural network training epochs |
| `pytorch_hidden_sizes` | list | `[128,64,32]` | MLP hidden layer sizes |
| `relationship_columns` | list | `[]` | Column groups for relationship charts |
| `outlier_method` | str | `"auto"` | `"iqr"`, `"zscore"`, `"auto"`, `"none"` |
| `outlier_strategy` | str | `"clip"` | `"clip"` or `"remove"` |

### Loading from JSON

```python
cfg = Config.from_json("my_config.json")
```

### Saving to JSON

```python
cfg = Config()
cfg.save_json("my_config.json")
# Now edit my_config.json in any text editor!
```

### Building from dict

```python
cfg = Config.from_dict({
    "mode": "manual",
    "normalize": True,
    "test_size": 0.2
})
```

---

## Agent Reference

All agents follow the same pattern:
```python
agent = SomeAgent(cfg_dict)
df_clean = agent.run(df_raw)
summary = agent.summary  # dict of what was done
```

### CleaningAgent

Removes noise, garbage, and duplicate data.

**What it does:**
- Strips whitespace from column names → converts to lowercase_underscore
- Removes exact duplicate rows
- Drops columns with >N% missing (controlled by `null_drop_threshold`)
- Drops near-zero-variance columns
- Normalizes text columns (lowercase + strip)

**Config keys:** `cleaning.drop_columns`, `global.drop_duplicates`, `global.null_drop_threshold`

### TypeAgent

Auto-detects and converts data types.

**AUTO detection order:**
1. Boolean-like (`true/false/yes/no/0/1`) → int (0/1)
2. Numeric strings → float
3. Date-like strings → year/month/day columns

**MANUAL:** Specify exact type per column in `type_conversion`.

**Supported types:** `int`, `float`, `bool`, `datetime`, `str`

### MissingAgent

Fills in missing values.

**Strategies:**
| Strategy | Description |
|----------|-------------|
| `mean` | Fill with column mean |
| `median` | Fill with column median |
| `mode` | Fill with most common value |
| `ffill` | Forward-fill (good for time series) |
| `bfill` | Backward-fill |
| `zero` | Fill with 0 |
| `unknown` | Fill with string "unknown" |
| `drop` | Drop rows with this column's NaN |
| `constant:X` | Fill with the value X |

**AUTO logic:**
- Numeric, low skew → `mean`
- Numeric, high skew → `median`
- Categorical → `mode`

### EncodingAgent

Converts categorical columns to numbers.

**Encodings:**
| Type | When to use |
|------|-------------|
| `binary` | 2 categories (male/female, yes/no) |
| `onehot` | ≤10 categories (city names, colors) |
| `label` | 10–50 categories |
| `frequency` | 50+ categories (user IDs, product codes) |
| `ordinal` | Ordered categories (low/medium/high) |
| `multihot` | Comma-separated tags ("python,ml,data") |

**AUTO logic:**
- 2 unique → binary
- ≤`low_card_threshold` unique → onehot
- ≤`med_card_threshold` unique → label
- Otherwise → frequency

### FeatureAgent

Creates new features from existing ones.

See [Feature Engineering Guide](#feature-engineering-guide) for all types.

### ValidationAgent

Final quality check:
- Drops remaining non-numeric columns (warns user)
- Replaces infinity values with median
- Optionally scales features

---

## Intelligence Engine

### StatsEngine

Computes per-column statistics:
- `mean`, `std`, `min`, `max`, `q25`, `median`, `q75`
- `skewness`, `kurtosis`, `IQR`
- Null counts and percentages
- Top values for categorical columns
- Shapiro-Wilk normality test
- Pearson correlation matrix
- Target correlation ranking

### OutlierEngine

Detects and handles outliers.

**Methods:**
- `iqr` — Interquartile Range rule (good for skewed data)
- `zscore` — Standard deviation rule (good for normal data)
- `auto` — Picks IQR for skewed cols, zscore for normal cols

**Strategies:**
- `clip` — Replace outliers with boundary values (keeps row count)
- `remove` — Drop outlier rows (reduces row count)

### FeatureSelector

Removes low-quality features:
1. **Variance threshold** — drops near-zero-variance features
2. **Correlation threshold** — drops one of each highly-correlated pair

### DataSplitter

Creates train/val/test splits:
- Stratified split for classification (preserves class ratios)
- Saves numpy arrays, CSV files, scaler, and metadata
- Auto-generates PyTorch Dataset class

---

## AutoML Reference

### Sklearn Models

**Classification:**
- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier
- ExtraTreesClassifier
- SVC (Support Vector Classifier)
- KNeighborsClassifier
- DecisionTreeClassifier
- GaussianNB
- AdaBoostClassifier

**Regression:**
- LinearRegression
- Ridge, Lasso, ElasticNet
- RandomForestRegressor
- GradientBoostingRegressor
- ExtraTreesRegressor
- SVR (Support Vector Regression)
- KNeighborsRegressor
- DecisionTreeRegressor
- AdaBoostRegressor

**Metrics per model:**
- `test_accuracy` / `test_r2`
- `test_f1_macro` / `test_mae` / `test_rmse`
- `cv_mean`, `cv_std` (5-fold cross-validation)
- `train_accuracy` / `train_r2` (overfitting check)
- `overfit_gap`, `overfit_risk`
- `feature_importance` (top 20 features)
- `training_time`
- `confusion_matrix` (classification)

### PyTorch MLP

Architecture: `Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output`

**Config:**
```python
Config(
    pytorch_epochs       = 100,
    pytorch_hidden_sizes = [256, 128, 64, 32],
    pytorch_lr           = 1e-3,
    pytorch_batch_size   = 64,
)
```

Optimizer: Adam with CosineAnnealingLR scheduler  
Regularization: Dropout (0.3) + L2 weight decay + gradient clipping

---

## Chart Engine

### Generated Charts

| File | Chart | Description |
|------|-------|-------------|
| `00_missing_values.png` | Horizontal bar | % missing per column |
| `01_correlation_heatmap.png` | Heatmap | Pearson correlation matrix |
| `02_target_distribution.png` | Histogram + KDE + boxplot | Target column analysis |
| `03_feature_distributions.png` | Grid of histograms | Distribution of all features |
| `04_boxplot_grid.png` | Box plots | Outlier visualization |
| `05_scatter_vs_target.png` | Scatter grid | Each feature vs target |
| `06_violin_plots.png` | Violin plots | Distribution shape |
| `07_skewness.png` | Horizontal bar | Skewness per feature |
| `08_feature_importance.png` | Horizontal bar | Feature importance scores |
| `09_pairplot.png` | Pair matrix | Top 6 features pairwise |
| `10_categorical_bars.png` | Bar charts | Category value counts |
| `11_pie_charts.png` | Pie charts | Category proportions |
| `12_class_balance.png` | Bar + pie | Target class balance |
| `13_relationship_*.png` | Scatter matrix | Multi-column relationships |

---

## Feature Engineering Guide

All feature types available in `features.derived`:

```json
{"type": "ratio",       "col_a": "A", "col_b": "B",    "name": "A_div_B"}
{"type": "diff",        "col_a": "A", "col_b": "B",    "name": "A_minus_B"}
{"type": "product",     "col_a": "A", "col_b": "B",    "name": "A_times_B"}
{"type": "interaction", "col_a": "A", "col_b": "B",    "name": "A_x_B"}
{"type": "agg_mean",    "cols": ["A","B","C"],          "name": "avg_ABC"}
{"type": "agg_sum",     "cols": ["A","B","C"],          "name": "sum_ABC"}
{"type": "agg_max",     "cols": ["A","B"],              "name": "max_AB"}
{"type": "agg_min",     "cols": ["A","B"],              "name": "min_AB"}
{"type": "log1p",       "col":  "A"}
{"type": "square",      "col":  "A"}
{"type": "sqrt",        "col":  "A"}
{"type": "zscore",      "col":  "A"}
{"type": "bin",         "col":  "A", "n_bins": 5}
{"type": "rank",        "col":  "A"}
{"type": "polynomial",  "col":  "A"}
```

---

## Output Files Reference

```
output/
├── processed.csv              ← Final ML-ready CSV (all features numeric)
├── report.html                ← Animated HTML report (open in browser)
├── plots/
│   └── *.png                  ← All charts (dark theme)
├── ml_ready/
│   ├── X_train.npy            ← Training features matrix
│   ├── X_val.npy              ← Validation features matrix
│   ├── X_test.npy             ← Test features matrix
│   ├── y_train.npy            ← Training labels
│   ├── y_val.npy              ← Validation labels
│   ├── y_test.npy             ← Test labels
│   ├── train.csv              ← Human-readable training set
│   ├── val.csv                ← Human-readable validation set
│   ├── test.csv               ← Human-readable test set
│   ├── scaler.pkl             ← Fitted MinMaxScaler (for new data)
│   ├── feature_names.txt      ← Ordered list of feature column names
│   ├── metadata.json          ← Split info, feature count, target name
│   └── pytorch_dataset.py     ← Ready-to-use PyTorch Dataset class
└── models/ (if AutoML enabled)
    ├── sklearn_results.json   ← All sklearn model results
    ├── pytorch_results.json   ← PyTorch MLP results
    ├── pytorch_model.pt       ← Saved PyTorch model weights
    └── automl_results.json    ← Combined ranked leaderboard
```

### Using ML-Ready Output

```python
import numpy as np
import pickle

# Load your data
X_train = np.load("output/ml_ready/X_train.npy")
y_train = np.load("output/ml_ready/y_train.npy")
X_test  = np.load("output/ml_ready/X_test.npy")
y_test  = np.load("output/ml_ready/y_test.npy")

# Load scaler (to transform new data the same way)
with open("output/ml_ready/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names
with open("output/ml_ready/feature_names.txt") as f:
    feature_names = f.read().splitlines()

# Train your own model!
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

```python
# PyTorch usage
from output.ml_ready.pytorch_dataset import MasterDataset
from torch.utils.data import DataLoader

train_ds = MasterDataset("output/ml_ready/X_train.npy", "output/ml_ready/y_train.npy")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

---

## FAQ

**Q: The pipeline is slow on large data. How do I speed it up?**  
A: Reduce `max_viz_cols` in Config (fewer charts). Also set `skip_selection=False` to use feature selection before charts.

**Q: How do I add a new encoding type?**  
A: Edit `agents/encoding_agent.py` and add a new `elif enc_type == "mytype"` block.

**Q: Can I use the pipeline without a target column?**  
A: Yes! Just don't pass `target`. Preprocessing, stats, and charts still run. Only the split and AutoML are skipped.

**Q: How do I transform NEW data (production) the same way?**  
A: Load `output/ml_ready/scaler.pkl` and call `scaler.transform(new_data)`. Use `feature_names.txt` to ensure column order.

**Q: PyTorch AutoML is not running.**  
A: Install PyTorch: `pip install torch`. Then set `automl_backends=["sklearn","pytorch"]` in your config.

**Q: How do I analyze relationships between multiple columns?**  
A: Use `relationship_columns` in config:
```python
Config(relationship_columns=[["age","income","score"], ["price","qty"]])
```

**Q: The report is huge. How do I reduce its size?**  
A: Set `chart_dpi=100` (lower image quality) or generate fewer charts by limiting `max_viz_cols`.

---

*Data Masterpiece v3 — Built with ❤️ in Python* ⚡
