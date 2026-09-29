"""
Data Preprocessors for Hydraulic Systems Dataset

This module provides functions for normalizing and aggregating time series data
from sensors with different sampling frequencies.

Functions:
    normalize_to_base_frequency: Downsample all sensors to 1 Hz using median
    create_aggregated_features: Extract statistical features from time series
    validate_dataframe: Validate DataFrame structure and completeness
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loaders import get_sensor_info, get_target_names


def normalize_to_base_frequency(
    df: pd.DataFrame,
    sensor_info: Optional[Dict] = None,
    base_freq: int = 1,
    aggregation: str = 'median',
    format: str = 'long',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Normalize all sensors to the base frequency (1 Hz) using aggregation.
    
    This function downsamples high-frequency sensors (10 Hz, 100 Hz) to the
    base frequency (1 Hz) by aggregating values within each second using
    the specified method (default: median).
    
    Returns data in LONG FORMAT by default, where each row represents one
    time point with CONTINUOUS time_sec, and each sensor has ONE column.
    
    Why median? It's more robust to outliers than mean, which is important
    for high-frequency pressure sensors that may have spikes.
    
    Args:
        df: DataFrame with raw sensor data (43,680+ columns)
        sensor_info: Sensor metadata dict. If None, uses get_sensor_info()
        base_freq: Target frequency in Hz (default: 1)
        aggregation: Aggregation method ('median', 'mean', 'min', 'max')
        format: Output format - 'long' (default) or 'wide'
            - 'long': Each row = one time point, columns = time_sec, sensor1, sensor2, ...
              time_sec is CONTINUOUS from 0 to 132,299 (treats all data as one sequence)
              Shape: (132,300 rows, ~23 cols)
            - 'wide': Each row = one cycle, columns = sensor_t0000, sensor_t0001, ...
              Shape: (2205, 17 sensors * 60 + targets) = (2205 rows, ~1,025 cols)
        verbose: If True, prints progress information
        
    Returns:
        DataFrame with normalized sensors in specified format
        
    Example (LONG FORMAT - DEFAULT):
        >>> df_raw = load_full_dataset()
        >>> df_normalized = normalize_to_base_frequency(df_raw, format='long', verbose=True)
        >>> df_normalized.shape
        (132300, 23)  # 132,300 timesteps, 17 sensors + time_sec + 5 targets
        >>> df_normalized.head(3)
        time_sec   PS1    PS2  ...  TS4   cooler_condition
        0        152.3  148.2  ...  35.2  100
        1        151.8  149.1  ...  35.3  100
        2        152.1  148.5  ...  35.4  100
        
    Example (WIDE FORMAT):
        >>> df_wide = normalize_to_base_frequency(df_raw, format='wide', verbose=True)
        >>> df_wide.shape
        (2205, 1025)  # 2205 cycles, 17 sensors × 60 timesteps + 5 targets
        
    Note:
        - 1 Hz sensors (60 samples): kept as-is
        - 10 Hz sensors (600 samples): aggregate every 10 samples
        - 100 Hz sensors (6000 samples): aggregate every 100 samples
        - LONG format treats all measurements as continuous sequence (no cycle breaks)
        - Targets repeat in blocks of 60 (each original cycle's targets)
    """
    if sensor_info is None:
        sensor_info = get_sensor_info()
    
    # Validate format parameter
    valid_formats = ['long', 'wide']
    if format not in valid_formats:
        raise ValueError(
            f"Invalid format '{format}'. Choose from {valid_formats}"
        )
    
    if verbose:
        print(f"Normalizing sensors to {base_freq} Hz using {aggregation}...")
        print(f"Output format: {format.upper()}")
        print("=" * 80)
    
    # Determine aggregation function
    agg_functions = {
        'median': np.median,
        'mean': np.mean,
        'min': np.min,
        'max': np.max
    }
    
    if aggregation not in agg_functions:
        raise ValueError(
            f"Invalid aggregation: {aggregation}. "
            f"Choose from {list(agg_functions.keys())}"
        )
    
    agg_func = agg_functions[aggregation]
    
    normalized_data = {}
    target_names = get_target_names()
    
    for sensor_name, info in sensor_info.items():
        # Get columns for this sensor
        sensor_cols = [col for col in df.columns if col.startswith(f"{sensor_name}_t")]
        
        if not sensor_cols:
            if verbose:
                print(f"⚠️  {sensor_name:5s}: Not found in DataFrame, skipping")
            continue
        
        sensor_freq = info['freq']
        original_samples = info['samples']
        
        # Calculate target samples at base frequency
        cycle_duration_sec = 60  # Each cycle is 60 seconds
        target_samples = cycle_duration_sec * base_freq
        
        if sensor_freq == base_freq:
            # Already at base frequency, keep as-is
            for col in sensor_cols:
                normalized_data[col] = df[col]
            
            if verbose:
                print(f"✓ {sensor_name:5s} ({sensor_freq:3d} Hz): "
                      f"kept as-is ({target_samples} samples)")
        else:
            # Need to downsample
            samples_per_aggregate = sensor_freq // base_freq
            
            if sensor_freq % base_freq != 0:
                raise ValueError(
                    f"Sensor {sensor_name} frequency ({sensor_freq} Hz) is not "
                    f"a multiple of base frequency ({base_freq} Hz)"
                )
            
            # Aggregate in groups
            sensor_data = df[sensor_cols].values
            
            # Reshape to (n_cycles, target_samples, samples_per_aggregate)
            n_cycles = sensor_data.shape[0]
            reshaped = sensor_data.reshape(
                n_cycles,
                target_samples,
                samples_per_aggregate
            )
            
            # Apply aggregation along the last axis
            aggregated = agg_func(reshaped, axis=2)
            
            # Create new column names
            new_cols = [f"{sensor_name}_t{i:04d}" for i in range(target_samples)]
            
            # Add to normalized data
            for i, col in enumerate(new_cols):
                normalized_data[col] = aggregated[:, i]
            
            if verbose:
                print(f"✓ {sensor_name:5s} ({sensor_freq:3d} Hz → {base_freq} Hz): "
                      f"{original_samples} → {target_samples} samples "
                      f"(aggregate every {samples_per_aggregate})")
    
    # Create normalized DataFrame
    df_normalized = pd.DataFrame(normalized_data)
    
    # Add targets if present in original DataFrame
    for target in target_names:
        if target in df.columns:
            df_normalized[target] = df[target].values
    
    # Convert to desired format
    if format == 'wide':
        # Wide format: each row = one cycle, columns = sensor_t0000, sensor_t0001, ...
        result = df_normalized
        
        if verbose:
            print("=" * 80)
            print(f"✅ Normalization complete (WIDE format)!")
            print(f"   Original shape: {df.shape}")
            print(f"   Normalized shape: {result.shape}")
            print(f"   Feature columns: {result.shape[1] - len([t for t in target_names if t in result.columns])}")
            print(f"   Memory: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return result
    
    else:  # format == 'long'
        # Long format: each row = one time point
        # Columns: time_sec, PS1, PS2, ..., TS4, VS1, CE, CP, SE, targets
        # time_sec is CONTINUOUS from 0 to (total_cycles * 60 - 1)
        
        if verbose:
            print("\n🔄 Converting to LONG format (continuous time_sec, sensors)...")
        
        # Calculate number of timesteps
        cycle_duration_sec = 60
        target_samples = cycle_duration_sec * base_freq
        
        # Reorganize data to long format with continuous time
        long_data = []
        continuous_time = 0  # Start at 0 and increment continuously
        
        for row_idx in range(len(df_normalized)):
            # Get target values for this row (constant across original 60-second measurement)
            row_targets = {
                target: df_normalized.loc[row_idx, target]
                for target in target_names 
                if target in df_normalized.columns
            }
            
            # For each timestep in this 60-second measurement
            for timestep in range(target_samples):
                long_row = {
                    'time_sec': continuous_time  # Continuous time from 0 to 132,299
                }
                
                # Add sensor values at this timestep
                for sensor_name in sensor_info.keys():
                    col_name = f"{sensor_name}_t{timestep:04d}"
                    if col_name in df_normalized.columns:
                        long_row[sensor_name] = df_normalized.loc[row_idx, col_name]
                
                # Add targets (same for all 60 timesteps from this measurement)
                long_row.update(row_targets)
                
                long_data.append(long_row)
                continuous_time += 1  # Increment continuous time
        
        result = pd.DataFrame(long_data)
        
        # Reorder columns: time_sec, sensors (sorted), targets
        sensor_cols = sorted([s for s in sensor_info.keys() if s in result.columns])
        target_cols = [t for t in target_names if t in result.columns]
        column_order = ['time_sec'] + sensor_cols + target_cols
        result = result[column_order]
        
        if verbose:
            print("=" * 80)
            print(f"✅ Normalization complete (LONG format)!")
            print(f"   Original shape: {df.shape}")
            print(f"   Normalized shape: {result.shape}")
            print(f"   Total timesteps: {result.shape[0]:,} (continuous from 0 to {result.shape[0]-1})")
            print(f"   Columns: time_sec + {len(sensor_cols)} sensors + {len(target_cols)} targets")
            print(f"   Sensor columns: {sensor_cols}")
            print(f"   Memory: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"\n   Structure:")
            print(f"   - Each row = one time point (1 second)")
            print(f"   - Each sensor = ONE column (not {target_samples})")
            print(f"   - time_sec is CONTINUOUS (no cycle breaks)")
            print(f"   - Total: {result.shape[0]:,} observations × {result.shape[1]} variables")
        
        return result


def create_aggregated_features(
    df: pd.DataFrame,
    sensor_info: Optional[Dict] = None,
    statistics: Optional[List[str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Extract statistical features from time series data.
    
    Reduces dimensionality by computing statistics over each sensor's
    time series, transforming from thousands of columns to ~136 features
    (8 statistics × 17 sensors).
    
    Args:
        df: DataFrame with sensor time series (normalized or raw)
        sensor_info: Sensor metadata dict. If None, uses get_sensor_info()
        statistics: List of statistics to compute. If None, uses all 8.
            Options: 'mean', 'std', 'min', 'max', 'median', 'range', 'q25', 'q75'
        verbose: If True, prints progress information
        
    Returns:
        DataFrame with aggregated features (136 columns) + targets (5 columns)
        Each sensor contributes 8 statistical features.
        
    Example:
        >>> df_raw = load_full_dataset()
        >>> df_agg = create_aggregated_features(df_raw, verbose=True)
        >>> df_agg.shape
        (2205, 141)  # 17 sensors × 8 stats + 5 targets
        >>> df_agg.columns[:5]
        ['CE_mean', 'CE_std', 'CE_min', 'CE_max', 'CE_median']
    """
    if sensor_info is None:
        sensor_info = get_sensor_info()
    
    # Default statistics
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max', 'median', 'range', 'q25', 'q75']
    
    # Validate statistics
    valid_stats = {'mean', 'std', 'min', 'max', 'median', 'range', 'q25', 'q75'}
    invalid_stats = set(statistics) - valid_stats
    if invalid_stats:
        raise ValueError(f"Invalid statistics: {invalid_stats}. Choose from {valid_stats}")
    
    if verbose:
        print(f"Creating aggregated features ({len(statistics)} statistics per sensor)...")
        print("=" * 80)
    
    aggregated_features = {}
    target_names = get_target_names()
    
    for sensor_name in sensor_info.keys():
        # Get columns for this sensor
        sensor_cols = [col for col in df.columns if col.startswith(f"{sensor_name}_t")]
        
        if not sensor_cols:
            if verbose:
                print(f"⚠️  {sensor_name:5s}: Not found in DataFrame, skipping")
            continue
        
        sensor_df = df[sensor_cols]
        
        # Compute each statistic
        if 'mean' in statistics:
            aggregated_features[f'{sensor_name}_mean'] = sensor_df.mean(axis=1)
        
        if 'std' in statistics:
            aggregated_features[f'{sensor_name}_std'] = sensor_df.std(axis=1)
        
        if 'min' in statistics:
            aggregated_features[f'{sensor_name}_min'] = sensor_df.min(axis=1)
        
        if 'max' in statistics:
            aggregated_features[f'{sensor_name}_max'] = sensor_df.max(axis=1)
        
        if 'median' in statistics:
            aggregated_features[f'{sensor_name}_median'] = sensor_df.median(axis=1)
        
        if 'range' in statistics:
            aggregated_features[f'{sensor_name}_range'] = (
                sensor_df.max(axis=1) - sensor_df.min(axis=1)
            )
        
        if 'q25' in statistics:
            aggregated_features[f'{sensor_name}_q25'] = sensor_df.quantile(0.25, axis=1)
        
        if 'q75' in statistics:
            aggregated_features[f'{sensor_name}_q75'] = sensor_df.quantile(0.75, axis=1)
        
        if verbose:
            print(f"✓ {sensor_name:5s}: {len(statistics)} features created")
    
    # Create aggregated DataFrame
    df_aggregated = pd.DataFrame(aggregated_features)
    
    # Add targets if present in original DataFrame
    for target in target_names:
        if target in df.columns:
            df_aggregated[target] = df[target].values
    
    if verbose:
        print("=" * 80)
        print(f"✅ Aggregation complete!")
        print(f"   Original shape: {df.shape}")
        print(f"   Aggregated shape: {df_aggregated.shape}")
        print(f"   Feature columns: {df_aggregated.shape[1] - len([t for t in target_names if t in df_aggregated.columns])}")
        print(f"   Reduction: {df.shape[1]} → {df_aggregated.shape[1]} columns")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB → "
              f"{df_aggregated.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_aggregated


def validate_dataframe(
    df: pd.DataFrame,
    expected_rows: Optional[int] = None,
    check_missing: bool = True,
    check_targets: bool = True,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Validate DataFrame structure and data quality.
    
    Args:
        df: DataFrame to validate
        expected_rows: Expected number of rows (default: 2205)
        check_missing: If True, checks for missing values
        check_targets: If True, validates target columns exist
        verbose: If True, prints validation details
        
    Returns:
        Dictionary with validation results:
        - valid: bool indicating if all checks passed
        - issues: list of issues found
        - stats: dictionary with statistics
        
    Example:
        >>> df = load_full_dataset()
        >>> result = validate_dataframe(df, verbose=True)
        >>> result['valid']
        True
    """
    if expected_rows is None:
        expected_rows = 2205  # Standard dataset size
    
    issues = []
    stats = {}
    
    if verbose:
        print("Validating DataFrame...")
        print("=" * 80)
    
    # Check dimensions
    stats['rows'] = df.shape[0]
    stats['columns'] = df.shape[1]
    
    if df.shape[0] != expected_rows:
        issues.append(
            f"Row count mismatch: expected {expected_rows}, got {df.shape[0]}"
        )
    
    if verbose:
        print(f"✓ Dimensions: {df.shape}")
    
    # Check for missing values
    if check_missing:
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        stats['missing_values'] = int(total_missing)
        
        if total_missing > 0:
            issues.append(f"Found {total_missing} missing values")
            stats['missing_columns'] = missing_counts[missing_counts > 0].to_dict()
        
        if verbose:
            if total_missing == 0:
                print("✓ No missing values")
            else:
                print(f"⚠️  Missing values: {total_missing}")
    
    # Check for target columns
    if check_targets:
        target_names = get_target_names()
        present_targets = [t for t in target_names if t in df.columns]
        missing_targets = [t for t in target_names if t not in df.columns]
        
        stats['present_targets'] = present_targets
        stats['missing_targets'] = missing_targets
        
        if missing_targets and len(missing_targets) == len(target_names):
            # This is okay if it's a features-only DataFrame
            if verbose:
                print("⚠️  No target columns found (features-only DataFrame)")
        elif missing_targets:
            issues.append(f"Missing target columns: {missing_targets}")
        
        if verbose and present_targets:
            print(f"✓ Target columns: {len(present_targets)}/{len(target_names)}")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats['numeric_columns'] = len(numeric_cols)
    stats['non_numeric_columns'] = df.shape[1] - len(numeric_cols)
    
    if stats['non_numeric_columns'] > 0:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        issues.append(f"Non-numeric columns found: {non_numeric}")
    
    if verbose:
        print(f"✓ Numeric columns: {len(numeric_cols)}/{df.shape[1]}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    stats['memory_mb'] = round(memory_mb, 2)
    
    if verbose:
        print(f"✓ Memory usage: {memory_mb:.2f} MB")
    
    # Final validation status
    valid = len(issues) == 0
    stats['valid'] = valid
    
    if verbose:
        print("=" * 80)
        if valid:
            print("✅ Validation passed!")
        else:
            print(f"⚠️  Validation found {len(issues)} issue(s):")
            for issue in issues:
                print(f"   - {issue}")
    
    return {
        'valid': valid,
        'issues': issues,
        'stats': stats
    }
