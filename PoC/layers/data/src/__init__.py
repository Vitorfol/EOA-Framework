"""
Data Layer - Hydraulic Systems Dataset Processing

This package provides utilities for loading, processing, and transforming
the Condition Monitoring of Hydraulic Systems dataset.

Public API:
    - get_original_dataframe: Load processed DataFrames (normalized or aggregated)
    - get_filtered_dataframe: Filter DataFrame by features and targets
    - get_features_metadata: Load feature metadata for agent consumption
    - prepare_for_xgboost: Prepare normalized wide/long time series for XGBoost training

Example:
    >>> from layers.data.src import get_original_dataframe, get_features_metadata
    >>> df = get_original_dataframe(variant='aggregated')
    >>> metadata = get_features_metadata()
"""

# Import public API from utils module
from .utils import (
    get_original_dataframe,
    get_filtered_dataframe,
    get_features_metadata,
    prepare_for_xgboost,
)

__all__ = [
    "get_original_dataframe",
    "get_filtered_dataframe",
    "get_features_metadata",
    "prepare_for_xgboost",
]
