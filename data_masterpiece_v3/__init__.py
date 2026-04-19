"""
Data Masterpiece v3
═══════════════════
Legend-Level Python Data Science Pipeline.

A 12-year-old can use it. A professional will love it.

Quick Start
-----------
    from data_masterpiece_v3 import MasterPipeline

    # Done! That's it — one line to run the whole pipeline:
    result = MasterPipeline().run("my_data.csv", target="price")

    # Your outputs:
    # ✅ output/processed.csv       → clean ML-ready CSV
    # ✅ output/ml_ready/           → numpy arrays for training
    # ✅ output/plots/              → 13+ beautiful dark-theme charts
    # ✅ output/report.html         → animated Legend HTML report
    # ✅ output/models/             → trained model results (if AutoML)

Manual Mode (you control everything)
--------------------------------------
    from data_masterpiece_v3 import MasterPipeline, Config

    cfg = Config.from_json("my_config.json")
    result = MasterPipeline(cfg).run("my_data.csv", target="price")

    # Generate a starter config to edit:
    Config().save_json("starter_config.json")
"""

from .master import MasterPipeline
from .config import Config

__version__ = "3.0.0"
__author__  = "Data Masterpiece Team"

__all__ = ["MasterPipeline", "Config", "__version__"]
