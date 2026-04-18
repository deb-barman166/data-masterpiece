# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║           ███████╗ ██████╗ ██████╗ ███████╗███████╗████████╗             ║
# ║           ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝             ║
# ║           ███████╗██║   ██║██████╔╝█████╗  ███████╗   ██║                ║
# ║           ╚════██║██║   ██║██╔══██╗██╔══╝  ╚════██║   ██║                ║
# ║           ███████║╚██████╔╝██████╔╝███████╗███████║   ██║                ║
# ║           ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝                ║
# ║                                                                            ║
# ║                    ★ VERSION 2 - LEGEND LEVEL ★                             ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Data Masterpiece V2
### From Raw Data to ML-Ready Masterpiece - Made Easy! ✨

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/deb-barman166/data-masterpiece?style=social)](https://github.com/deb-barman166/data-masterpiece)

---

## 🎯 What is Data Masterpiece V2?

**Data Masterpiece V2** is the ultimate data analysis and machine learning pipeline that takes you from raw data to production-ready ML models with just a few lines of code!

Whether you're a **beginner**, a **student**, or a **professional data scientist**, Data Masterpiece V2 makes complex data science tasks simple and accessible.

---

## ✨ Features

### 🤖 Auto ML/DL Model Building
- Automatically builds multiple ML models
- Compares and ranks models
- Selects the best model for your data
- Supports both Classification and Regression

### 📊 Intelligent Data Analysis
- Automatic data profiling
- Outlier detection
- Feature selection
- Correlation analysis
- Model recommendations

### 📄 Beautiful Animated Reports
- Dark theme with neon accents
- Interactive expandable sections
- Animated stat counters
- Professional visualizations

### 🔧 Both Auto & Manual Modes
- **Auto Mode**: Let the magic happen automatically!
- **Manual Mode**: Full control over every setting

### ⚡ Easy to Use
Designed for everyone - from 12-year-old beginners to expert data scientists!

---

## 🚀 Quick Start

### The Easiest Way (3 Lines!)

```python
from data_masterpiece_v2 import MasterPipeline

# Create pipeline and run!
pipeline = MasterPipeline()
result = pipeline.run("your_data.csv", target="your_target")

# That's it! You get:
# ✅ Clean data
# ✅ Analysis report
# ✅ ML models (optional)
```

### With Auto ML (Build Models Automatically!)

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline()
pipeline.config.ml_builder.enable_auto_ml = True

result = pipeline.run(
    source="titanic.csv",
    target="Survived",  # What you want to predict
    build_models=True    # Build ML models!
)

# See your best model
print(result['best_model'])
```

### Using Config Files (Manual Mode)

```python
from data_masterpiece_v2 import Config, MasterPipeline
import json

# Create a config file
config = Config.create_manual_config()
config.to_json("my_config.json")

# Edit the JSON file with your settings
with open("my_config.json", "r") as f:
    custom_settings = json.load(f)

# Load and use
config = Config.from_json("my_config.json")
pipeline = MasterPipeline(config=config)
result = pipeline.run("data.csv", target="price")
```

---

## 📁 Project Structure

```
data_masterpiece_v2/
├── __init__.py              # Package initialization
├── master.py                # Main pipeline (MasterPipeline class)
├── config.py                # Configuration system
├── requirements.txt         # Python dependencies
├── README.md               # This file!
├── DOCUMENTATION.md        # Full documentation
│
├── preprocessing/           # Data preprocessing
│   ├── controller.py        # Main preprocessing controller
│   └── core/
│       └── loader.py       # Data loading utilities
│
├── intelligence/            # Data analysis & insights
│   ├── controller.py       # Main intelligence controller
│   ├── profiler.py         # Statistical profiling
│   ├── outlier.py          # Outlier detection
│   ├── feature_selection.py # Feature selection
│   ├── relationship.py     # Correlation analysis
│   ├── recommender.py      # Model recommendations
│   └── splitter.py         # Data splitting
│
├── ml_builder/             # Auto ML module
│   └── auto_builder.py     # Auto ML model builder
│
├── reports/                # Report generation
│   └── animated_reporter.py # Animated HTML reports
│
└── utils/                  # Utility functions
    ├── logger.py           # Logging system
    └── helpers.py          # Helper functions
```

---

## 🔧 Installation

### Option 1: Install Everything (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Minimal Install

```bash
pip install numpy pandas scikit-learn matplotlib requests openpyxl
```

### Option 3: From GitHub

```bash
git clone https://github.com/deb-barman166/data-masterpiece.git
cd data-masterpiece
pip install -e .
```

---

## 📖 Documentation

For full documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

### Quick Reference

| Task | Code |
|------|------|
| Load data | `pipeline.run("data.csv", target="y")` |
| Skip ML | `pipeline.run("data.csv", target="y", build_models=False)` |
| Custom config | `pipeline = MasterPipeline(config=my_config)` |
| Auto mode | `pipeline = MasterPipeline(mode="auto")` |
| Manual mode | `pipeline = MasterPipeline(mode="manual")` |
| Just preprocess | `df_clean = pipeline.preprocess_only(data, target="y")` |

---

## 💡 Usage Examples

### Titanic Dataset

```python
from data_masterpiece_v2 import MasterPipeline

pipeline = MasterPipeline()
result = pipeline.run(
    source="titanic.csv",
    target="Survived"
)

print(f"Best Model: {result['best_model']['name']}")
print(f"Accuracy: {result['best_model']['score']:.2%}")
```

### Custom Preprocessing

```python
from data_masterpiece_v2 import Config

config = Config.create_manual_config()
config.preprocessing.drop_duplicates = True
config.preprocessing.missing_strategy = "median"
config.preprocessing.encoding_strategy = "onehot"

pipeline = MasterPipeline(config=config)
result = pipeline.run("data.csv", target="survived")
```

### Generate Only Report

```python
pipeline = MasterPipeline()
pipeline.config.output.save_csv = False
pipeline.config.ml_builder.enable_auto_ml = False
result = pipeline.run("data.csv", target="survived")
```

---

## 🎨 Output Files

When you run the pipeline, you get:

```
output/
├── processed_data.csv     # Clean, ML-ready data
├── report.html           # Beautiful animated HTML report
├── models/                # Saved ML models (if enabled)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── ...
└── logs/                  # Processing logs
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/deb-barman166/data-masterpiece/issues)
- **Documentation**: [Full docs](DOCUMENTATION.md)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Created with ❤️ for the data science community
- Inspired by the need to make ML accessible to everyone
- Special thanks to all contributors!

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐!

```bash
git clone https://github.com/deb-barman166/data-masterpiece.git
```

---

## 🏆 What's New in V2?

- 🔥 **Auto ML Builder** - Build models automatically!
- 🎬 **Animated Reports** - Beautiful, interactive HTML reports
- ⚡ **Faster Processing** - Optimized algorithms
- 🎯 **Better Recommendations** - Smarter model suggestions
- 📊 **Enhanced Visualizations** - Professional charts
- 🔧 **JSON Configuration** - Customize everything easily
- 👶 **Beginner Friendly** - Even easier to use!

---

**Made with ❤️ by Data Masterpiece Team**

*Making Data Science Easy for Everyone!*
