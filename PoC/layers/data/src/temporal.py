"""
Temporal Utilities for Time Series Analysis (WIDE FORMAT ONLY)

This module provides utilities for working with the WIDE format of the
normalized hydraulic systems dataset. These functions work with DataFrames
where each row represents one cycle with timestamped columns.

⚠️  IMPORTANT: These functions expect WIDE format, not LONG format.
    Use get_original_dataframe(variant='normalized', format='wide')

Functions:
    get_temporal_index: Extract timestamp indices from column names
    extract_sensor_timeseries: Extract time series for a specific sensor from WIDE format
    plot_sensor_timeseries: Visualize sensor time series from WIDE format
    get_temporal_statistics: Compute statistics across cycles

Note:
    For LONG format (continuous time_sec), use pandas operations directly:
    df[df['time_sec'] < 300]  # First 5 minutes
    df.plot(x='time_sec', y='PS1')  # Plot sensor
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd

from .loaders import get_sensor_info, get_target_names


def get_temporal_index(
    df: pd.DataFrame,
    sensor_name: Optional[str] = None,
    return_seconds: bool = True
) -> Union[np.ndarray, range]:
    """
    Get the temporal index from WIDE format DataFrame.
    
    ⚠️  Expects WIDE format where columns are like 'PS1_t0000', 'PS1_t0001', ... 'PS1_t0059'
    representing measurements at t=0s, t=1s, ... t=59s within each 60-second cycle.
    
    Args:
        df: Normalized DataFrame in WIDE format with timestamped columns
        sensor_name: Specific sensor to extract timestamps from (e.g., 'PS1').
            If None, uses the first sensor found.
        return_seconds: If True, returns array [0, 1, 2, ..., 59].
            If False, returns range object.
            
    Returns:
        Array or range representing time in seconds (0 to 59)
        
    Example:
        >>> df_wide = get_original_dataframe(variant='normalized', format='wide')
        >>> time_index = get_temporal_index(df_wide)
        >>> time_index
        array([ 0,  1,  2, ..., 57, 58, 59])
        
        >>> # Use for plotting individual cycles
        >>> import matplotlib.pyplot as plt
        >>> cycle_0_data = extract_sensor_timeseries(df_wide, 'PS1', cycle_idx=0)
        >>> plt.plot(time_index, cycle_0_data)
        >>> plt.xlabel('Time (seconds)')
        >>> plt.ylabel('Pressure')
    """
    # Find a sensor to extract temporal structure
    if sensor_name is None:
        # Use first sensor found
        sensor_info = get_sensor_info()
        sensor_name = list(sensor_info.keys())[0]
    
    # Get columns for this sensor
    sensor_cols = [col for col in df.columns if col.startswith(f"{sensor_name}_t")]
    
    if not sensor_cols:
        raise ValueError(f"Sensor '{sensor_name}' not found in DataFrame")
    
    # Extract timestamps from column names
    # Column format: 'SENSOR_t0000', 'SENSOR_t0001', etc.
    timestamps = []
    for col in sorted(sensor_cols):
        try:
            # Extract the numeric part after '_t'
            timestamp_str = col.split('_t')[1]
            timestamp = int(timestamp_str)
            timestamps.append(timestamp)
        except (IndexError, ValueError):
            continue
    
    if return_seconds:
        return np.array(timestamps)
    else:
        return range(len(timestamps))


def extract_sensor_timeseries(
    df: pd.DataFrame,
    sensor_name: str,
    cycle_idx: Optional[Union[int, List[int]]] = None,
    return_timestamps: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Extract time series data for a specific sensor from WIDE format.
    
    ⚠️  Expects WIDE format where each row = one cycle with timestamped columns.
    
    Args:
        df: Normalized DataFrame in WIDE format
        sensor_name: Sensor name (e.g., 'PS1', 'TS1')
        cycle_idx: ROW index/indices to extract (0-2204). If None, extracts all cycles.
            Can be single integer or list of integers.
        return_timestamps: If True, also returns timestamp array
        
    Returns:
        If cycle_idx is single int and return_timestamps=False:
            1D array of sensor values over time (60 timesteps)
        If cycle_idx is single int and return_timestamps=True:
            Tuple of (timestamps, values)
        If cycle_idx is list or None:
            2D array of shape (n_cycles, 60)
            
    Example:
        >>> df_wide = get_original_dataframe(variant='normalized', format='wide')
        >>> 
        >>> # Single cycle (row 0)
        >>> ps1_data = extract_sensor_timeseries(df_wide, 'PS1', cycle_idx=0)
        >>> ps1_data.shape
        (60,)  # 60 seconds
        >>> 
        >>> # With timestamps
        >>> time, ps1_data = extract_sensor_timeseries(
        ...     df_wide, 'PS1', cycle_idx=0, return_timestamps=True
        ... )
        >>> 
        >>> # Multiple cycles (rows 0, 1, 2)
        >>> ps1_multiple = extract_sensor_timeseries(df_wide, 'PS1', cycle_idx=[0, 1, 2])
        >>> ps1_multiple.shape
        (3, 60)  # 3 cycles × 60 seconds
        >>> 
        >>> # All cycles
        >>> ps1_all = extract_sensor_timeseries(df_wide, 'PS1')
        >>> ps1_all.shape
        (2205, 60)  # All cycles × 60 seconds
    """
    # Get columns for this sensor
    sensor_cols = [col for col in df.columns if col.startswith(f"{sensor_name}_t")]
    
    if not sensor_cols:
        raise ValueError(f"Sensor '{sensor_name}' not found in DataFrame")
    
    # Sort columns by timestamp
    sensor_cols = sorted(sensor_cols)
    
    # Extract data
    if cycle_idx is None:
        # All cycles
        data = df[sensor_cols].values
    elif isinstance(cycle_idx, (list, tuple)):
        # Multiple cycles
        data = df.loc[cycle_idx, sensor_cols].values
    else:
        # Single cycle
        data = df.loc[cycle_idx, sensor_cols].values
    
    if return_timestamps:
        timestamps = get_temporal_index(df, sensor_name)
        return timestamps, data
    else:
        return data


def reshape_to_long_format(
    df: pd.DataFrame,
    sensor_names: Optional[List[str]] = None,
    include_targets: bool = True,
    time_column_name: str = 'time_sec',
    cycle_column_name: str = 'cycle_id'
) -> pd.DataFrame:
    """
    ⚠️  DEPRECATED: Use get_original_dataframe(format='long') instead.
    
    This function is kept for backward compatibility but is REDUNDANT.
    The official LONG format is generated by preprocessors.py with:
    - Continuous time_sec (0 to 132,299)
    - NO cycle_id column
    
    This function creates a DIFFERENT long format with cycle_id,
    which is inconsistent with the official format.
    
    Recommended migration:
        # OLD (deprecated):
        df_long = reshape_to_long_format(df_wide)
        
        # NEW (official):
        df_long = get_original_dataframe(variant='normalized', format='long')
        
        # To get a specific time window:
        df_window = df_long[(df_long['time_sec'] >= 0) & (df_long['time_sec'] < 60)]
    """
    import warnings
    warnings.warn(
        "reshape_to_long_format() is deprecated. "
        "Use get_original_dataframe(variant='normalized', format='long') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    sensor_info = get_sensor_info()
    target_names = get_target_names()
    
    # Determine which sensors to include
    if sensor_names is None:
        sensor_names = list(sensor_info.keys())
    
    # Validate sensor names
    invalid_sensors = set(sensor_names) - set(sensor_info.keys())
    if invalid_sensors:
        raise ValueError(f"Invalid sensor names: {invalid_sensors}")
    
    # Get number of timesteps (should be 60 for normalized data)
    first_sensor = sensor_names[0]
    sensor_cols = [col for col in df.columns if col.startswith(f"{first_sensor}_t")]
    n_timesteps = len(sensor_cols)
    
    # Create long format data
    long_data = []
    
    for cycle_idx in range(len(df)):
        # Extract targets for this cycle (constant across time)
        if include_targets:
            cycle_targets = {target: df.loc[cycle_idx, target] 
                           for target in target_names if target in df.columns}
        else:
            cycle_targets = {}
        
        # For each timestep
        for timestep in range(n_timesteps):
            row = {
                cycle_column_name: cycle_idx,
                time_column_name: timestep
            }
            
            # Add sensor values at this timestep
            for sensor_name in sensor_names:
                col_name = f"{sensor_name}_t{timestep:04d}"
                if col_name in df.columns:
                    row[sensor_name] = df.loc[cycle_idx, col_name]
            
            # Add targets (same for all timesteps in a cycle)
            row.update(cycle_targets)
            
            long_data.append(row)
    
    df_long = pd.DataFrame(long_data)
    
    return df_long


def plot_sensor_timeseries(
    df: pd.DataFrame,
    sensor_name: str,
    cycle_indices: Optional[List[int]] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    show_legend: bool = True
):
    """
    Plot time series for one or more cycles from WIDE format.
    
    ⚠️  Expects WIDE format where each row = one cycle.
    
    Args:
        df: Normalized DataFrame in WIDE format
        sensor_name: Sensor to plot (e.g., 'PS1', 'TS1')
        cycle_indices: List of ROW indices (0-2204) to plot. If None, plots first 3 cycles.
        figsize: Figure size (width, height)
        title: Plot title. If None, auto-generates title.
        show_legend: Whether to show legend
        
    Returns:
        matplotlib Figure and Axes objects
        
    Example:
        >>> df_wide = get_original_dataframe(variant='normalized', format='wide')
        >>> fig, ax = plot_sensor_timeseries(
        ...     df_wide, 
        ...     sensor_name='PS1',
        ...     cycle_indices=[0, 100, 200]
        ... )
        >>> plt.show()
        
    Note:
        For LONG format plotting, use pandas directly:
        >>> df_long = get_original_dataframe(variant='normalized', format='long')
        >>> df_long[df_long['time_sec'] < 300].plot(x='time_sec', y='PS1')
    """
    import matplotlib.pyplot as plt
    
    sensor_info = get_sensor_info()
    
    if sensor_name not in sensor_info:
        raise ValueError(f"Sensor '{sensor_name}' not found")
    
    # Default to first 3 cycles
    if cycle_indices is None:
        cycle_indices = [0, 1, 2]
    
    # Get temporal index
    time_index = get_temporal_index(df, sensor_name)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cycle
    for cycle_idx in cycle_indices:
        time, data = extract_sensor_timeseries(
            df, sensor_name, cycle_idx, return_timestamps=True
        )
        ax.plot(time, data, label=f'Cycle {cycle_idx}', alpha=0.7, linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    sensor_desc = sensor_info[sensor_name]['desc']
    if title is None:
        title = f'{sensor_name} - {sensor_desc}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show_legend:
        ax.legend(loc='best')
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def get_temporal_statistics(
    df: pd.DataFrame,
    sensor_name: str,
    statistic: str = 'mean'
) -> np.ndarray:
    """
    Compute statistics across all cycles for each timestep (WIDE format only).
    
    ⚠️  Expects WIDE format where each row = one cycle.
    This shows how sensor values typically evolve over the 60-second cycle.
    
    Args:
        df: Normalized DataFrame in WIDE format
        sensor_name: Sensor to analyze
        statistic: Statistic to compute ('mean', 'std', 'median', 'min', 'max')
        
    Returns:
        Array of length 60 with the statistic at each timestep
        
    Example:
        >>> df_wide = get_original_dataframe(variant='normalized', format='wide')
        >>> ps1_mean_profile = get_temporal_statistics(df_wide, 'PS1', 'mean')
        >>> ps1_std_profile = get_temporal_statistics(df_wide, 'PS1', 'std')
        >>> 
        >>> # Plot typical temporal pattern
        >>> import matplotlib.pyplot as plt
        >>> time = get_temporal_index(df_norm)
        >>> plt.plot(time, ps1_mean_profile, label='Mean')
        >>> plt.fill_between(
        ...     time,
        ...     ps1_mean_profile - ps1_std_profile,
        ...     ps1_mean_profile + ps1_std_profile,
        ...     alpha=0.3, label='±1 std'
        ... )
        >>> plt.xlabel('Time (seconds)')
        >>> plt.ylabel('PS1 Pressure')
        >>> plt.legend()
        >>> plt.show()
    """
    # Extract all cycles
    data = extract_sensor_timeseries(df, sensor_name)  # Shape: (n_cycles, 60)
    
    # Compute statistic across cycles (axis=0)
    stat_functions = {
        'mean': np.mean,
        'std': np.std,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }
    
    if statistic not in stat_functions:
        raise ValueError(
            f"Invalid statistic '{statistic}'. Choose from {list(stat_functions.keys())}"
        )
    
    result = stat_functions[statistic](data, axis=0)
    
    return result
