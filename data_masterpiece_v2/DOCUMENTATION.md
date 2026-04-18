# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║                    DATA MASTERPIECE V2 - DOCUMENTATION                      ║
# ║                                                                            ║
# ║              Complete Guide for Users of All Levels!                        ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration System](#configuration-system)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

---

## Introduction

### What is Data Masterpiece V2?

Data Masterpiece V2 is a comprehensive data analysis and machine learning pipeline that automates the entire process from raw data to ML-ready output.

### Who is it for?

- **Beginners**: No coding experience needed!
- **Students**: Learn data science easily
- **Professionals**: Speed up your workflow
- **Researchers**: Focus on insights, not code

### Key Features

| Feature | Description |
|---------|-------------|
| 🤖 Auto ML | Automatically builds and compares ML models |
| 📊 Data Analysis | Statistical profiling, correlations, feature importance |
| 📄 Reports | Beautiful animated HTML reports |
| ⚡ Fast | Optimized for speed |
| 🔧 Customizable | Full control when you need it |

---

## Installation

### System Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux
- 4GB RAM minimum (8GB recommended)

### Install Dependencies

```bash
# Option 1: All dependencies
pip install -r requirements.txt

# Option 2: Minimal (for basic usage)
pip install numpy pandas scikit-learn matplotlib requests openpyxl
```

### Verify Installation

```python
import data_masterpiece_v2
print(data_masterpiece_v2.__version__)
```

---

## Quick Start

### The Simplest Way (3 Lines!)

```python
# Import the pipeline
from data_masterpiece_v2 import MasterPipeline

# Create and run
pipeline = MasterPipeline()
result = pipeline.run("your_data.csv", target="what_to_predict")

# That's it!
print(result.keys())  # See what's available
```

### What You Get

After running, you'll have:
- ✅ Clean, processed data
- ✅ Statistical analysis
- ✅ Correlation insights
- ✅ Model recommendations
- ✅ Beautiful HTML report
- ✅ (Optional) Trained ML models

---

## Configuration System

### Auto Mode (For Beginners)

Let Data Masterpiece handle everything automatically:

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline(mode="auto")
result = pipeline.run("data.csv", target="survived")
```

### Manual Mode (For Control)

Customize every aspect:

```python
from data_masterpiece_v2 import Config

config = Config(mode="manual")

# Preprocessing
config.preprocessing.drop_duplicates = True
config.preprocessing.missing_strategy = "median"
config.preprocessing.encoding_strategy = "onehot"

# ML Builder
config.ml_builder.enable_auto_ml = True
config.ml_builder.max_models = 10

# Create pipeline
pipeline = MasterPipeline(config=config)
result = pipeline.run("data.csv", target="survived")
```

### JSON Configuration

Save and load configurations:

```python
# Save config to JSON
config = Config.create_manual_config()
config.to_json("my_config.json")

# Load from JSON
config = Config.from_json("my_config.json")
```

### JSON Config File Example

```json
{
    "mode": "manual",
    "preprocessing": {
        "drop_duplicates": true,
        "missing_strategy": "median",
        "encoding_strategy": "onehot",
        "scale_method": "standard"
    },
    "intelligence": {
        "outlier_method": "iqr",
        "outlier_strategy": "clip",
        "feature_selection_method": "correlation"
    },
    "ml_builder": {
        "enable_auto_ml": true,
        "task_type": "auto",
        "max_models": 5
    },
    "output": {
        "save_csv": true,
        "save_report": true,
        "save_models": true,
        "output_dir": "output"
    }
}
```

---

## API Reference

### MasterPipeline

Main class for running the complete pipeline.

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline(config=None, mode="auto")
```

#### Methods

##### `run(source, target, **kwargs)`

Run the complete pipeline.

**Parameters:**
- `source` (str, Path, or DataFrame): Your data
- `target` (str): Target column to predict
- `save_csv` (bool): Save processed data? Default: True
- `save_report` (bool): Generate HTML report? Default: True
- `build_models` (bool): Build ML models? Default: from config
- `show_plots` (bool): Display plots? Default: False

**Returns:** Dict with results

**Example:**
```python
result = pipeline.run(
    source="titanic.csv",
    target="Survived",
    save_report=True,
    build_models=True
)
```

##### `preprocess_only(source, target)`

Run only preprocessing.

**Example:**
```python
df_clean = pipeline.preprocess_only("data.csv", target="y")
```

##### `analyze_only(df, target)`

Run only intelligence analysis.

**Example:**
```python
analysis = pipeline.analyze_only(df_clean, target="y")
```

### Config

Configuration management class.

```python
from data_masterpiece_v2 import Config

# Auto config
config = Config(mode="auto")

# Manual config
config = Config(mode="manual")

# From file
config = Config.from_json("config.json")
```

#### Sub-Configs

##### PreprocessingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| drop_duplicates | bool | True | Remove duplicate rows |
| missing_strategy | str | "auto" | How to handle missing: "auto", "mean", "median", "mode", "drop" |
| encoding_strategy | str | "auto" | How to encode: "auto", "onehot", "label", "target" |
| scale_method | str | "auto" | How to scale: "auto", "standard", "minmax", "robust" |
| normalize | bool | False | Normalize features? |

##### IntelligenceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| outlier_method | str | "auto" | Detection: "auto", "iqr", "zscore" |
| outlier_strategy | str | "clip" | Handle: "clip", "drop", "flag", "none" |
| feature_selection_method | str | "auto" | Selection: "auto", "variance", "correlation" |
| test_size | float | 0.2 | Test set proportion |
| stratify | bool | True | Stratified split for classification? |

##### MLBuilderConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enable_auto_ml | bool | True | Build models automatically? |
| task_type | str | "auto" | Problem: "auto", "classification", "regression" |
| max_models | int | 10 | Max models to try |
| cv_folds | int | 5 | Cross-validation folds |
| search_strategy | str | "random" | Hyperparameter search: "random", "grid" |

---

## Advanced Usage

### Custom Preprocessing Pipeline

```python
from data_masterpiece_v2 import Config, MasterPipeline

config = Config(mode="manual")

# Very specific preprocessing
config.preprocessing.drop_duplicates = True
config.preprocessing.null_drop_threshold = 0.5  # Drop cols with >50% nulls
config.preprocessing.missing_strategy = "median"
config.preprocessing.create_statistical_features = True
config.preprocessing.create_interaction_features = True

pipeline = MasterPipeline(config=config)
result = pipeline.run("data.csv", target="target")
```

### Using Pre-split Data

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline()

# Use your own split data
result = pipeline.run(
    df,
    target="survived",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
```

### Multiple Data Sources

```python
from data_masterpiece_v2.preprocessing import DataLoader

loader = DataLoader()

# Load and concatenate multiple files
df = loader.load_multiple(["data1.csv", "data2.csv", "data3.csv"])

# Or load from URL
df = loader.load_from_url("https://example.com/data.csv")

pipeline = MasterPipeline()
result = pipeline.run(df, target="target")
```

### Custom Analysis Steps

```python
# Skip some steps
result = pipeline.run(
    df,
    target="survived",
    skip_outlier=True,        # Skip outlier detection
    skip_selection=False,     # Run feature selection
    skip_recommendations=False  # Get model recommendations
)
```

### Model Prediction

```python
# After training
result = pipeline.run(df, target="survived", build_models=True)

# Make predictions
predictions = pipeline.predict(new_data)
```

---

## Output Files

### Generated Files

```
output/
├── processed_data.csv      # Clean, ML-ready data
├── report.html            # Beautiful HTML report
├── models/                # Saved ML models
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── scaler.joblib
└── logs/                   # Processing logs
```

### Report Sections

The generated HTML report includes:

1. **Hero Section** - Animated title and overview stats
2. **Executive Summary** - Pipeline stages and problem type
3. **Data Overview** - Column statistics and quality
4. **Correlation Analysis** - Feature correlations
5. **Feature Importance** - Ranked features
6. **Model Recommendations** - Suggested algorithms
7. **ML Results** - Trained model performance

---

## Troubleshooting

### Common Issues

#### "Module not found" error

```bash
pip install -r requirements.txt
```

#### "File not found" error

Check your file path:
```python
import os
print(os.path.exists("your_file.csv"))
```

#### Memory issues with large data

```python
# Sample your data
loader = DataLoader(sample_size=10000)
df = loader.load("large_file.csv")
```

#### "Target column not found" error

```python
# Check column names
print(df.columns.tolist())

# Use exact name
result = pipeline.run(df, target="exact_column_name")
```

### Getting Help

1. Check this documentation
2. Look at the examples
3. Open a GitHub issue

---

## Examples

### Example 1: Titanic Dataset

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline()
pipeline.config.ml_builder.enable_auto_ml = True

result = pipeline.run(
    source="titanic.csv",
    target="Survived",
    build_models=True
)

print(f"Best Model: {result['best_model']['name']}")
print(f"Accuracy: {result['best_model']['score']:.2%}")
```

### Example 2: House Prices

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline(mode="manual")
pipeline.config.preprocessing.missing_strategy = "median"
pipeline.config.preprocessing.scale_method = "standard"
pipeline.config.ml_builder.enable_auto_ml = True

result = pipeline.run(
    source="house_prices.csv",
    target="SalePrice",
    build_models=True
)

print(f"Best Model: {result['best_model']['name']}")
print(f"R² Score: {result['best_model']['metrics']['r2']:.4f}")
```

### Example 3: Custom Analysis Only

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline()
pipeline.config.ml_builder.enable_auto_ml = False
pipeline.config.output.save_csv = False

result = pipeline.run(
    source="data.csv",
    target="target"
)

# Access analysis results
profile = result['profile']
correlations = result['relationships']
recommendations = result['recommendations']
```

### Example 4: Using JSON Config

```python
from data_masterpiece_v2 import Config, MasterPipeline

# Create and save config
config = Config.create_manual_config()
config.preprocessing.drop_duplicates = True
config.preprocessing.missing_strategy = "mean"
config.ml_builder.enable_auto_ml = True
config.to_json("my_config.json")

# Later, load and use
config = Config.from_json("my_config.json")
pipeline = MasterPipeline(config=config)
result = pipeline.run("data.csv", target="target")
```

---

## API Quick Reference

| Task | Code |
|------|------|
| Create pipeline | `pipeline = MasterPipeline()` |
| Run everything | `pipeline.run(data, target="y")` |
| Run with ML | `pipeline.run(data, target="y", build_models=True)` |
| Just preprocess | `pipeline.preprocess_only(data, target="y")` |
| Just analyze | `pipeline.analyze_only(df, target="y")` |
| Make predictions | `pipeline.predict(new_data)` |
| Get best model | `result['best_model']` |
| Get report path | `result['report_path']` |
| Get clean data | `result['df_processed']` |

---

## Glossary

| Term | Definition |
|------|-----------|
| Target | The column you want to predict |
| Feature | An input column for the model |
| Preprocessing | Cleaning and transforming raw data |
| Training | Teaching a model from data |
| Prediction | Using a model on new data |
| Cross-validation | Testing model on different data splits |

---

## Support

For questions or issues:
- GitHub Issues: [Link](https://github.com/deb-barman166/data-masterpiece/issues)
- Documentation: [DOCUMENTATION.md](DOCUMENTATION.md)

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated: 2024**
**Version: 2.0.0**

*Making Data Science Accessible to Everyone! ✨*
