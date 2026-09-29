"""
Public API for Data Layer

This module provides the main interface for accessing processed data
and utilities across all layers (training, agent, view).

Public Functions:
    get_original_dataframe: Load processed DataFrames
    get_filtered_dataframe: Filter DataFrame by features and targets
    get_features_metadata: Load feature metadata for agent consumption
    prepare_for_xgboost: Prepare time series for XGBoost
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .loaders import (
    load_full_dataset,
    get_sensor_info,
    get_target_names
)
from .preprocessors import (
    normalize_to_base_frequency,
    create_aggregated_features,
    validate_dataframe
)
from .transformers import (
    filter_features,
    timeseries_to_supervised,
)
from .metadata import (
    load_metadata,
    generate_features_metadata,
    get_target_metadata
)


def get_processed_data_path() -> Path:
    """
    Get path to processed data directory.
    
    Returns:
        Path object pointing to layers/data/processed/
    """
    return Path(__file__).parent.parent / 'processed'


def get_original_dataframe(
    variant: str = 'normalized',
    format: str = 'long',
    force_reload: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load processed DataFrame (normalized or aggregated).
    
    This is the primary function for accessing data from other layers.
    It loads cached CSV files if available, otherwise processes from scratch.
    
    Args:
        variant: Type of DataFrame to load:
            - 'normalized': Time series normalized to 1 Hz
            - 'aggregated': Statistical features (136 features + 5 targets)
            - 'raw': Raw sensor data (43,680 features + 5 targets) - NOT CACHED
        format: For 'normalized' variant only:
            - 'long': Each row = one time point (RECOMMENDED)
              Shape: (132,300 rows, ~23 cols)
              Columns: time_sec, PS1, PS2, ..., targets
              time_sec is CONTINUOUS from 0 to 132,299
            - 'wide': Each row = one cycle
              Shape: (2,205 rows, ~1,025 cols)
              Columns: PS1_t0000, PS1_t0001, ..., targets
        force_reload: If True, reprocesses data even if cached file exists
        verbose: If True, prints loading information
        
    Returns:
        DataFrame with requested variant and format
        
    Raises:
        ValueError: If variant is not recognized
        FileNotFoundError: If cached file doesn't exist and raw data unavailable
        
    Example:
        >>> # For time series analysis (RECOMMENDED)
        >>> df = get_original_dataframe(variant='normalized', format='long')
        >>> df.shape
        (132300, 23)  # 132,300 continuous timesteps
        >>> df.head()
        time_sec   PS1    PS2  ...  cooler_condition
        0        152.3  148.2  ...  100
        1        151.8  149.1  ...  100
        2        152.1  148.5  ...  100
        
        >>> # For traditional ML (XGBoost, Random Forest)
        >>> df = get_original_dataframe(variant='aggregated')
        >>> df.shape
        (2205, 141)  # Statistical features
        
    Note:
        First call may take longer as it processes raw data.
        Subsequent calls load from cached CSV files for speed.
        
        LONG format is recommended for:
        - Time series visualization
        - Temporal analysis  
        - LSTM/RNN models (after reshaping)
        - Easy pandas plotting
        - time_sec is now CONTINUOUS (no cycle breaks)
        
        WIDE format is mainly for compatibility but less intuitive.
    """
    valid_variants = ['normalized', 'aggregated', 'raw']
    if variant not in valid_variants:
        raise ValueError(
            f"Invalid variant '{variant}'. Choose from {valid_variants}"
        )
    
    processed_path = get_processed_data_path()
    
    # Handle raw data (never cached due to size)
    if variant == 'raw':
        if verbose:
            print("Loading raw dataset (not cached)...")
        return load_full_dataset(verbose=verbose)
    
    # Determine cache file path
    # For normalized variant, include format suffix in filename
    if variant == 'normalized':
        cache_file = processed_path / f'hydraulic_{variant}_{format}.csv'
    else:
        cache_file = processed_path / f'hydraulic_{variant}.csv'
    
    # Check if cached file exists and should be used
    if cache_file.exists() and not force_reload:
        if verbose:
            print(f"Loading cached {variant} DataFrame from {cache_file}...")
        
        df = pd.read_csv(cache_file)
        
        if verbose:
            print(f"✓ Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")
        
        return df
    
    # Need to process from scratch
    if verbose:
        print(f"Cached file not found. Processing {variant} DataFrame from raw data...")
    
    # Load raw data
    df_raw = load_full_dataset(verbose=verbose)
    
    # Process based on variant
    if variant == 'normalized':
        df = normalize_to_base_frequency(df_raw, format=format, verbose=verbose)
    elif variant == 'aggregated':
        df = create_aggregated_features(df_raw, verbose=verbose)
    
    # Cache the result
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    
    if verbose:
        print(f"✓ Cached {variant} DataFrame to {cache_file}")
    
    return df


def get_filtered_dataframe(
    df: Optional[pd.DataFrame] = None,
    feature_list: Optional[List[str]] = None,
    target_list: Optional[List[str]] = None,
    variant: str = 'aggregated',
    validate: bool = True
) -> pd.DataFrame:
    """
    Get DataFrame with only specified features and targets.
    
    Args:
        df: Input DataFrame. If None, loads using get_original_dataframe()
        feature_list: List of feature column names. If None, includes all features.
        target_list: List of target column names. If None, includes all targets.
        variant: If df is None, which variant to load ('normalized' or 'aggregated')
        validate: If True, raises error for missing columns
        
    Returns:
        Filtered DataFrame
        
    Example:
        >>> # Get only temperature features and cooler condition
        >>> features = ['TS1_mean', 'TS2_mean', 'TS3_mean', 'TS4_mean']
        >>> targets = ['cooler_condition']
        >>> df = get_filtered_dataframe(
        ...     feature_list=features,
        ...     target_list=targets
        ... )
        >>> df.shape
        (2205, 5)
    """
    if df is None:
        df = get_original_dataframe(variant=variant)
    
    return filter_features(df, feature_list, target_list, validate)


def get_features_metadata(
    metadata_path: Optional[Path] = None,
    force_regenerate: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Load feature metadata for agent consumption.
    
    This function loads metadata from JSON and automatically includes the
    domain file content, providing complete context in a single call.
    
    Args:
        metadata_path: Path to metadata JSON. If None, uses default location.
        force_regenerate: If True, regenerates metadata even if file exists
        verbose: If True, prints loading information
        
    Returns:
        Dictionary with feature metadata including:
        - domain_file: Path to domain knowledge text file
        - domain_content: Content of domain file (loaded automatically)
        - dataset_info: General dataset information
        - features: List of feature metadata
        - targets: List of target metadata
        
    Example:
        >>> metadata = get_features_metadata()
        >>> metadata['dataset_info']['total_samples']
        2205
        >>> len(metadata['features'])
        1020  # For normalized variant
        >>> 
        >>> # Domain content is loaded automatically
        >>> print(metadata['domain_content'][:100])
        'Hydraulic System Components and Operation...'
        >>> 
        >>> metadata['features'][0]
        {
            'name': 'CE_t0000',
            'sensor_group': 'CE',
            'original_frequency_hz': 1,
            'normalized': True,
            ...
        }
    """
    if metadata_path is None:
        processed_path = get_processed_data_path()
        metadata_path = processed_path / 'features_metadata.json'
    
    # Check if we need to regenerate
    if metadata_path.exists() and not force_regenerate:
        return load_metadata(metadata_path, verbose=verbose)
    
    # Generate metadata from normalized DataFrame
    if verbose:
        print("Generating feature metadata...")
    
    df = get_original_dataframe(variant='normalized', verbose=verbose)
    metadata = generate_features_metadata(df, normalized=True)
    
    # Save for future use
    from .metadata import save_metadata
    save_metadata(metadata, metadata_path, verbose=verbose)
    
    return metadata


def prepare_for_xgboost(
    df: Optional[pd.DataFrame] = None,
    n_lags: int = 5,
    target_columns: Optional[List[str]] = None,
    dropnan: bool = True,
    input_format: str = 'wide'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare normalized time series data for XGBoost training.
    
    Converts time series to supervised learning format with lag features.
    
    Args:
        df: Normalized DataFrame. If None, loads automatically using input_format.
            Can also be a filtered DataFrame with only the desired feature columns.
        n_lags: Number of time lags to create
        target_columns: Target columns to include. If None, uses all targets.
        dropnan: If True, drops rows with NaN values from lag creation
        input_format: Format of normalized input DataFrame:
            - 'wide': cycle-based columns like PS1_t0000, PS1_t0001, ...
            - 'long': continuous rows with columns like time_sec, PS1, PS2, ...
        
    Returns:
        Tuple of (X, y) DataFrames ready for XGBoost
        
    Example:
        >>> X, y = prepare_for_xgboost(n_lags=5)
        >>> X.shape
        (2205, 850)  # Lag features
        >>> y.shape
        (2205, 5)    # Targets

        >>> df_long = get_original_dataframe(variant='normalized', format='long')
        >>> X, y = prepare_for_xgboost(df_long, n_lags=5, input_format='long')

        >>> df_filtered = get_filtered_dataframe(
        ...     df=df_long,
        ...     feature_list=['PS1', 'PS2'],
        ...     target_list=['cooler_condition']
        ... )
        >>> X, y = prepare_for_xgboost(
        ...     df_filtered,
        ...     n_lags=3,
        ...     target_columns=['cooler_condition'],
        ...     input_format='long'
        ... )
        
        >>> # Use with XGBoost
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    """
    if input_format not in {'wide', 'long'}:
        raise ValueError("input_format must be 'wide' or 'long'")

    if df is None:
        df = get_original_dataframe(variant='normalized', format=input_format)
    
    sensor_info = get_sensor_info()
    
    return timeseries_to_supervised(
        df,
        sensor_info=sensor_info,
        n_lags=n_lags,
        target_columns=target_columns,
        dropnan=dropnan,
        input_format=input_format
    )


# Convenience exports
__all__ = [
    'get_original_dataframe',
    'get_filtered_dataframe',
    'get_features_metadata',
    'prepare_for_xgboost',
]
