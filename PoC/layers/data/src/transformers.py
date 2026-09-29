"""
Data Transformers for Hydraulic Systems Dataset

This module provides functions for transforming DataFrames into formats
suitable for machine learning models (XGBoost, ARIMA).

Functions:
    filter_features: Select subset of features and targets
    timeseries_to_supervised: Convert time series to supervised learning format (XGBoost)
"""

from typing import List, Optional, Tuple

import pandas as pd

from .loaders import get_sensor_info, get_target_names


def filter_features(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    target_list: Optional[List[str]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Filter DataFrame to include only specified features and targets.
    
    Args:
        df: Input DataFrame with features and targets
        feature_list: List of feature column names to include.
            If None, includes all non-target columns.
        target_list: List of target column names to include.
            If None, includes all target columns.
        validate: If True, raises error for missing columns
        
    Returns:
        Filtered DataFrame with selected columns
        
    Raises:
        ValueError: If validate=True and requested columns don't exist
        
    Example:
        >>> df = get_original_dataframe(variant='aggregated')
        >>> # Keep only temperature sensors and cooler condition
        >>> features = ['TS1_mean', 'TS2_mean', 'TS3_mean', 'TS4_mean']
        >>> targets = ['cooler_condition']
        >>> df_filtered = filter_features(df, features, targets)
        >>> df_filtered.shape
        (2205, 5)  # 4 features + 1 target
    """
    all_target_names = get_target_names()
    
    # Determine columns to keep
    columns_to_keep = []
    
    # Handle features
    if feature_list is None:
        # Include all non-target columns
        feature_list = [col for col in df.columns if col not in all_target_names]
    
    # Handle targets
    if target_list is None:
        # Include all available target columns
        target_list = [col for col in all_target_names if col in df.columns]
    
    # Combine feature and target lists
    columns_to_keep = feature_list + target_list
    
    # Validate column existence
    if validate:
        missing_cols = set(columns_to_keep) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Requested columns not found in DataFrame: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )
    else:
        # Only keep columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # Return filtered DataFrame
    return df[columns_to_keep].copy()


def timeseries_to_supervised(
    df: pd.DataFrame,
    sensor_info: Optional[dict] = None,
    n_lags: int = 5,
    target_columns: Optional[List[str]] = None,
    dropnan: bool = True,
    input_format: str = 'wide'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert normalized time series to supervised learning format for XGBoost.
    
    This function creates lag features from the normalized time series,
    transforming temporal data into a tabular format suitable for
    gradient boosting models.
    
    Args:
        df: Normalized DataFrame in WIDE or LONG format.
        sensor_info: Sensor metadata. If None, uses get_sensor_info()
        n_lags: Number of time lags to create (e.g., t-1, t-2, ..., t-n_lags)
        target_columns: List of target columns. If None, uses all targets.
        dropnan: If True, drops rows with NaN values from lag creation
        input_format: Format of normalized input DataFrame:
            - 'wide': columns like PS1_t0000, PS1_t0001, ...
            - 'long': columns like time_sec, PS1, PS2, ...
        
    Returns:
        Tuple of (X, y) where:
        - X: DataFrame with lag features (shape: [samples, n_sensors * n_lags])
        - y: DataFrame with target variables
        
    Example:
        >>> df_norm = get_original_dataframe(variant='normalized', format='wide')
        >>> X, y = timeseries_to_supervised(df_norm, n_lags=5, input_format='wide')
        >>> X.shape
        (2205, 850)  # 17 sensors * 5 lags * 10 selected timesteps
        >>> y.shape
        (2205, 5)  # 5 target variables

        >>> df_long = get_original_dataframe(variant='normalized', format='long')
        >>> X, y = timeseries_to_supervised(df_long, n_lags=5, input_format='long')
        
    Note:
        In WIDE format, we sample evenly spaced timesteps to reduce dimensionality
        while preserving temporal patterns. In LONG format, lag features are
        created directly from the available non-target columns, which works well
        with filtered DataFrames that contain only a subset of sensors.
    """
    if sensor_info is None:
        sensor_info = get_sensor_info()
    
    if target_columns is None:
        target_columns = get_target_names()
    
    # Validate targets exist
    available_targets = [t for t in target_columns if t in df.columns]
    if not available_targets:
        raise ValueError("No target columns found in DataFrame")

    if input_format not in {'wide', 'long'}:
        raise ValueError("input_format must be 'wide' or 'long'")

    lag_feature_map = {}

    if input_format == 'wide':
        feature_cols = [col for col in df.columns if col not in target_columns]

        # For 60 timesteps, sample 10 evenly spaced points.
        sample_rate = 6

        for sensor_name in sensor_info.keys():
            sensor_cols = sorted(
                col for col in feature_cols if col.startswith(f"{sensor_name}_t")
            )

            if not sensor_cols:
                continue

            sampled_cols = sensor_cols[::sample_rate]

            for lag in range(1, n_lags + 1):
                for col in sampled_cols:
                    timestep = int(col.split('_t')[1])
                    lag_feature_map[f"{sensor_name}_t{timestep:04d}_lag{lag}"] = df[col].shift(lag)
    else:
        feature_cols = [
            col for col in df.columns
            if col not in target_columns and col != 'time_sec'
        ]

        for lag in range(1, n_lags + 1):
            for col in feature_cols:
                lag_feature_map[f"{col}_lag{lag}"] = df[col].shift(lag)

    if not lag_feature_map:
        raise ValueError(
            "No feature columns available to build XGBoost lag features. "
            "Check the input_format and the filtered DataFrame columns."
        )

    X = pd.DataFrame(lag_feature_map, index=df.index)
    
    # Extract targets
    y = df[available_targets].copy()
    
    # Drop NaN rows if requested
    if dropnan:
        # Find rows with any NaN
        valid_mask = ~X.isnull().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
    
    return X, y



