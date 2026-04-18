"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATA MASTERPIECE  ★  VERSION 2 ★                        ║
║                                                                            ║
║          The Ultimate Data Analysis & ML Pipeline for Everyone!             ║
║                                                                            ║
║   Easy • Powerful • Professional • Legend Level                           ║
║                                                                            ║
║   Created with ❤️ for Data Scientists, Students, and Enthusiasts!           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Data Masterpiece V2 - Your Journey from Raw Data to ML-Ready Masterpiece!

Features:
  ✨ Auto Mode - Let the magic happen automatically!
  ✨ Manual Mode - You control every detail
  ✨ Auto ML/DL - Builds models for you automatically
  ✨ Animated Reports - Beautiful, interactive dark theme reports
  ✨ Professional Visualizations - Charts that tell stories
  ✨ JSON Config - Customize everything in Manual mode

Quick Start:
  >>> from data_masterpiece_v2 import MasterPipeline
  >>> pipeline = MasterPipeline()
  >>> result = pipeline.run("your_data.csv", target="your_target")

Version: 2.0.0
Author: Data Masterpiece Team
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Data Masterpiece Team"
__email__ = "support@datamasterpiece.io"

from data_masterpiece_v2.master import MasterPipeline
from data_masterpiece_v2.config import Config, load_config_from_json

# Friendly exports for easy access
__all__ = [
    "MasterPipeline",
    "Config",
    "load_config_from_json",
    "__version__",
]

# Print welcome banner
def _welcome():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   ███████╗███████╗███████╗ █████╗ ██████╗ ███╗   ███╗                  ║
    ║   ██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗████╗ ████║                  ║
    ║   ███████╗█████╗  █████╗  ███████║██████╔╝██╔████╔██║                  ║
    ║   ╚════██║██╔══╝  ██╔══╝  ██╔══██║██╔══██╗██║╚██╔╝██║                  ║
    ║   ███████║███████╗███████╗██║  ██║██║  ██║██║ ╚═╝ ██║                  ║
    ║   ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝                  ║
    ║                                                                           ║
    ║                    ★  VERSION 2 - LEGEND LEVEL  ★                         ║
    ║                                                                           ║
    ║         From Raw Data to ML-Ready Masterpiece - Made Easy!              ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

# Uncomment to show welcome banner on import
# _welcome()
