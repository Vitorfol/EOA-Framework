"""
Data Loaders for Hydraulic Systems Dataset

This module provides functions for loading raw sensor data and targets
from the Condition Monitoring of Hydraulic Systems dataset.

Functions:
    get_sensor_info: Returns metadata for all 17 sensors
    load_raw_sensors: Loads all sensor time series data
    load_targets: Loads target variables (profile.txt)
    load_full_dataset: Loads complete dataset (sensors + targets)
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def get_sensor_info() -> Dict[str, Dict]:
    """
    Get metadata information for all sensors in the hydraulic system.
    
    Returns:
        Dict mapping sensor names to their characteristics:
        - file: filename in the dataset
        - freq: sampling frequency in Hz
        - samples: number of samples per 60-second cycle
        - desc: human-readable description
        
    Example:
        >>> sensor_info = get_sensor_info()
        >>> sensor_info['PS1']
        {'file': 'PS1.txt', 'freq': 100, 'samples': 6000, 
         'desc': 'Pressure sensor 1'}
    """
    return {
        # Low frequency sensors (1 Hz = 60 samples/cycle)
        'CE': {
            'file': 'CE.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Cooling efficiency (virtual sensor)'
        },
        'CP': {
            'file': 'CP.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Cooling power (virtual sensor)'
        },
        'SE': {
            'file': 'SE.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Efficiency factor (virtual sensor)'
        },
        'TS1': {
            'file': 'TS1.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Temperature sensor 1'
        },
        'TS2': {
            'file': 'TS2.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Temperature sensor 2'
        },
        'TS3': {
            'file': 'TS3.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Temperature sensor 3'
        },
        'TS4': {
            'file': 'TS4.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Temperature sensor 4'
        },
        'VS1': {
            'file': 'VS1.txt',
            'freq': 1,
            'samples': 60,
            'desc': 'Vibration sensor'
        },
        # Medium frequency sensors (10 Hz = 600 samples/cycle)
        'FS1': {
            'file': 'FS1.txt',
            'freq': 10,
            'samples': 600,
            'desc': 'Volume flow sensor (pump)'
        },
        'FS2': {
            'file': 'FS2.txt',
            'freq': 10,
            'samples': 600,
            'desc': 'Volume flow sensor (cooler)'
        },
        # High frequency sensors (100 Hz = 6000 samples/cycle)
        'PS1': {
            'file': 'PS1.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 1'
        },
        'PS2': {
            'file': 'PS2.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 2'
        },
        'PS3': {
            'file': 'PS3.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 3'
        },
        'PS4': {
            'file': 'PS4.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 4'
        },
        'PS5': {
            'file': 'PS5.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 5'
        },
        'PS6': {
            'file': 'PS6.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Pressure sensor 6'
        },
        'EPS1': {
            'file': 'EPS1.txt',
            'freq': 100,
            'samples': 6000,
            'desc': 'Motor power'
        },
    }


def _get_dataset_path() -> Path:
    """
    Get the path to the hydraulic dataset from environment or default.
    
    Returns:
        Path object pointing to the dataset directory
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
    """
    # Load environment variables
    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / '.env')
    
    # Get path from environment or use default
    hydraulic_path_str = os.getenv(
        'HYDRAULIC_DATASET_PATH',
        'datasets/Condition monitoring of hydraulic systems'
    )
    hydraulic_path = Path(hydraulic_path_str)
    
    # Make absolute if relative
    if not hydraulic_path.is_absolute():
        hydraulic_path = (project_root / hydraulic_path_str).resolve()
    
    # Validate existence
    if not hydraulic_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {hydraulic_path}. "
            f"Please set HYDRAULIC_DATASET_PATH in .env or place dataset at default location."
        )
    
    return hydraulic_path


def load_targets() -> pd.DataFrame:
    """
    Load target variables from profile.txt.
    
    Returns:
        DataFrame with 2,205 rows and 5 target columns:
        - cooler_condition: Cooler condition state (3, 20, 100)
        - valve_condition: Valve condition state (73, 80, 90, 100)
        - pump_leakage: Pump leakage level (0, 1, 2)
        - accumulator: Accumulator pressure state (90, 100, 115, 130)
        - stable_flag: System stability flag (0, 1)
        
    Example:
        >>> targets = load_targets()
        >>> targets.shape
        (2205, 5)
        >>> targets.columns.tolist()
        ['cooler_condition', 'valve_condition', 'pump_leakage', 
         'accumulator', 'stable_flag']
    """
    dataset_path = _get_dataset_path()
    profile_file = dataset_path / 'profile.txt'
    
    if not profile_file.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_file}")
    
    profile_cols = [
        'cooler_condition',
        'valve_condition',
        'pump_leakage',
        'accumulator',
        'stable_flag'
    ]
    
    targets = pd.read_csv(profile_file, sep='\t', names=profile_cols)
    
    return targets


def load_raw_sensors(
    sensors: Optional[list] = None,
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load raw sensor time series data.
    
    Args:
        sensors: List of sensor names to load. If None, loads all sensors.
        verbose: If True, prints loading progress.
        
    Returns:
        Dictionary mapping sensor names to DataFrames with columns named
        as '{sensor}_t{index:04d}' (e.g., 'PS1_t0000', 'PS1_t0001', ...)
        
    Raises:
        FileNotFoundError: If sensor file doesn't exist
        
    Example:
        >>> sensor_data = load_raw_sensors(['PS1', 'TS1'])
        >>> sensor_data['PS1'].shape
        (2205, 6000)
        >>> sensor_data['TS1'].shape
        (2205, 60)
    """
    dataset_path = _get_dataset_path()
    sensor_info = get_sensor_info()
    
    # Use all sensors if none specified
    if sensors is None:
        sensors = list(sensor_info.keys())
    
    # Validate sensor names
    invalid_sensors = set(sensors) - set(sensor_info.keys())
    if invalid_sensors:
        raise ValueError(f"Invalid sensor names: {invalid_sensors}")
    
    sensor_data = {}
    
    for sensor_name in sensors:
        info = sensor_info[sensor_name]
        filepath = dataset_path / info['file']
        
        if not filepath.exists():
            raise FileNotFoundError(f"Sensor file not found: {filepath}")
        
        try:
            # Load data using numpy for efficiency
            data = np.loadtxt(filepath, delimiter='\t')
            
            # Create column names
            col_names = [f"{sensor_name}_t{i:04d}" for i in range(data.shape[1])]
            
            # Create DataFrame
            sensor_df = pd.DataFrame(data, columns=col_names)
            sensor_data[sensor_name] = sensor_df
            
            if verbose:
                print(f"✓ Loaded {sensor_name:5s} ({info['freq']:3d} Hz): "
                      f"shape {data.shape} - {info['desc']}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading {sensor_name}: {str(e)}")
    
    return sensor_data


def load_full_dataset(
    sensors: Optional[list] = None,
    include_targets: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load complete dataset with all sensors and optionally targets.
    
    Args:
        sensors: List of sensor names to load. If None, loads all 17 sensors.
        include_targets: If True, appends target columns to the DataFrame.
        verbose: If True, prints loading progress.
        
    Returns:
        DataFrame with shape (2205, N) where N depends on sensors loaded:
        - All sensors: 43,680 feature columns
        - With targets: 43,680 features + 5 targets = 43,685 columns
        
    Example:
        >>> df = load_full_dataset(verbose=True)
        >>> df.shape
        (2205, 43685)
        >>> df.memory_usage(deep=True).sum() / 1024**2
        ~130.0  # MB
    """
    if verbose:
        print("Loading Hydraulic Systems Dataset...")
        print("=" * 80)
    
    # Load sensor data
    sensor_data = load_raw_sensors(sensors=sensors, verbose=verbose)
    
    # Concatenate all sensors horizontally
    sensor_dfs = [sensor_data[s] for s in sorted(sensor_data.keys())]
    df = pd.concat(sensor_dfs, axis=1)
    
    # Add targets if requested
    if include_targets:
        targets = load_targets()
        df = pd.concat([df, targets], axis=1)
        
        if verbose:
            print("\n✓ Targets loaded: 5 columns")
    
    if verbose:
        print("=" * 80)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def get_target_names() -> list:
    """
    Get list of target column names.
    
    Returns:
        List of 5 target column names
        
    Example:
        >>> get_target_names()
        ['cooler_condition', 'valve_condition', 'pump_leakage', 
         'accumulator', 'stable_flag']
    """
    return [
        'cooler_condition',
        'valve_condition',
        'pump_leakage',
        'accumulator',
        'stable_flag'
    ]
