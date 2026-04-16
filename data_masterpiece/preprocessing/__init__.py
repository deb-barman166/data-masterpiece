"""
data_masterpiece.preprocessing  --  Data preprocessing pipeline.

A 6-agent sequential pipeline that transforms raw datasets into
fully numeric, ML-ready DataFrames:

    Cleaning -> TypeConversion -> MissingValues -> Encoding -> FeatureTransform -> Validation
"""

from data_masterpiece.preprocessing.controller import PipelineController
from data_masterpiece.preprocessing.core.loader import load_data
from data_masterpiece.preprocessing.core.auto_config import generate_auto_config

__all__ = [
    "PipelineController",
    "load_data",
    "generate_auto_config",
]
