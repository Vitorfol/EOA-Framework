"""
Metadata Generation for Hydraulic Systems Dataset

This module provides functions for generating and managing metadata about
features, sensors, and targets for consumption by the agent layer.

Functions:
    generate_features_metadata: Create comprehensive feature metadata
    save_metadata: Save metadata to JSON file
    load_metadata: Load metadata from JSON file
    get_target_metadata: Get metadata for target variables
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .loaders import get_sensor_info, get_target_names

def generate_features_metadata(
    df: pd.DataFrame,
    sensor_info: Optional[Dict] = None,
    normalized: bool = True,
    domain_file_path: Optional[str] = None,
    include_statistics: bool = True
) -> Dict:
    """
    Generate comprehensive metadata for all features in the dataset.
    
    This metadata is designed for consumption by the agent layer to understand
    the dataset structure, feature characteristics, and domain context.
    
    Args:
        df: DataFrame with features (normalized or aggregated)
        sensor_info: Sensor metadata. If None, uses get_sensor_info()
        normalized: Whether features are normalized to 1 Hz
        domain_file_path: Path to domain knowledge text file
        include_statistics: If True, includes basic statistics for each feature
        
    Returns:
        Dictionary with complete metadata structure including:
        - domain_file: Path to domain knowledge
        - dataset_info: General dataset information
        - features: List of feature metadata dicts
        - targets: List of target metadata dicts
        
    Example:
        >>> df = get_original_dataframe(variant='normalized')
        >>> metadata = generate_features_metadata(df, normalized=True)
        >>> len(metadata['features'])
        1020  # 17 sensors × 60 timesteps
        >>> metadata['features'][0]
        {
            'name': 'CE_t0000',
            'sensor_group': 'CE',
            'original_frequency_hz': 1,
            'normalized': True,
            'type': 'feature',
            'description': 'Cooling efficiency (virtual sensor) at timestep 0'
        }
    """
    if sensor_info is None:
        sensor_info = get_sensor_info()
    
    target_names = get_target_names()
    
    # Initialize metadata structure
    metadata = {
        'domain_file': domain_file_path or 'domain/hydraulic_system_domain.txt',
        'dataset_info': {
            'total_samples': len(df),
            'total_features': len([col for col in df.columns if col not in target_names]),
            'total_targets': len([col for col in df.columns if col in target_names]),
            'normalized_to_1hz': normalized,
            'description': 'Condition Monitoring of Hydraulic Systems Dataset'
        },
        'features': [],
        'targets': []
    }
    
    # Generate feature metadata
    for col in df.columns:
        if col in target_names:
            continue  # Handle targets separately
        
        # Determine sensor group and characteristics
        sensor_group = None
        original_freq = None
        sensor_desc = None
        
        # Check if it's a time series column or aggregated feature
        for sensor_name, info in sensor_info.items():
            if col.startswith(f"{sensor_name}_"):
                sensor_group = sensor_name
                original_freq = info['freq']
                sensor_desc = info['desc']
                break
        
        # Determine feature type and description
        if '_t' in col:
            # Time series feature (e.g., 'PS1_t0000')
            try:
                timestep = int(col.split('_t')[1])
                feature_desc = f"{sensor_desc} at timestep {timestep}"
            except (IndexError, ValueError):
                feature_desc = sensor_desc
        else:
            # Aggregated feature (e.g., 'PS1_mean')
            stat_type = col.split('_')[-1] if '_' in col else 'unknown'
            feature_desc = f"{sensor_desc} - {stat_type}"
        
        # Create feature metadata entry
        feature_meta = {
            'name': col,
            'sensor_group': sensor_group,
            'original_frequency_hz': original_freq,
            'normalized': normalized,
            'type': 'feature',
            'description': feature_desc
        }
        
        # Add statistics if requested
        if include_statistics and sensor_group:
            try:
                feature_meta['statistics'] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            except Exception:
                pass  # Skip if statistics can't be computed
        
        metadata['features'].append(feature_meta)
    
    # Generate target metadata
    target_metadata = get_target_metadata()
    for target_name in target_names:
        if target_name not in df.columns:
            continue
        
        target_info = target_metadata.get(target_name, {})
        
        # Get class distribution
        class_counts = df[target_name].value_counts().sort_index()
        
        target_meta = {
            'name': target_name,
            'type': 'target',
            'description': target_info.get('description', target_name),
            'classes': class_counts.index.tolist(),
            'class_distribution': class_counts.to_dict(),
            'num_classes': len(class_counts),
            'problem_type': target_info.get('problem_type', 'classification')
        }
        
        metadata['targets'].append(target_meta)
    
    return metadata


def get_target_metadata() -> Dict[str, Dict]:
    """
    Get detailed metadata for target variables.
    
    Returns:
        Dictionary mapping target names to their characteristics:
        - description: Human-readable description
        - problem_type: 'classification' or 'regression'
        - classes: Expected class values
        - recommendation: Modeling recommendations
        
    Example:
        >>> target_meta = get_target_metadata()
        >>> target_meta['cooler_condition']
        {
            'description': 'Cooler condition state',
            'problem_type': 'classification',
            'classes': [3, 20, 100],
            'recommendation': 'Use class weights due to moderate imbalance'
        }
    """
    return {
        'cooler_condition': {
            'description': 'Cooler condition state (percentage of optimal)',
            'problem_type': 'classification',
            'classes': [3, 20, 100],
            'recommendation': 'Use class weights due to moderate imbalance',
            'interpretation': {
                3: 'Close to total failure',
                20: 'Reduced efficiency',
                100: 'Full efficiency / Optimal'
            }
        },
        'valve_condition': {
            'description': 'Valve condition state (percentage of optimal)',
            'problem_type': 'classification',
            'classes': [73, 80, 90, 100],
            'recommendation': 'Use class weights due to moderate imbalance',
            'interpretation': {
                73: 'Severe lag',
                80: 'Moderate lag',
                90: 'Slight lag',
                100: 'Optimal switching behavior'
            }
        },
        'pump_leakage': {
            'description': 'Pump leakage severity level',
            'problem_type': 'classification',
            'classes': [0, 1, 2],
            'recommendation': 'Well-balanced classes, no special treatment needed',
            'interpretation': {
                0: 'No leakage',
                1: 'Weak leakage',
                2: 'Severe leakage'
            }
        },
        'accumulator': {
            'description': 'Accumulator pressure state (bar)',
            'problem_type': 'classification',
            'classes': [90, 100, 115, 130],
            'recommendation': 'Use SMOTE or class weights due to high imbalance',
            'interpretation': {
                90: 'Reduced pressure',
                100: 'Slightly reduced',
                115: 'Optimal pressure',
                130: 'Near maximum permissible'
            }
        },
        'stable_flag': {
            'description': 'System stability indicator',
            'problem_type': 'classification',
            'classes': [0, 1],
            'recommendation': 'Well-balanced binary classification',
            'interpretation': {
                0: 'Unstable operation / Conditions were not stable',
                1: 'Stable operation / Conditions reached stable regime'
            }
        }
    }


def save_metadata(
    metadata: Dict,
    output_path: Path,
    indent: int = 2,
    verbose: bool = False
) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary to save
        output_path: Path where JSON file will be saved
        indent: JSON indentation for readability
        verbose: If True, prints save confirmation
        
    Example:
        >>> metadata = generate_features_metadata(df)
        >>> save_metadata(metadata, Path('processed/features_metadata.json'))
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=indent, ensure_ascii=False)
    
    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"✓ Metadata saved to {output_path}")
        print(f"  - Total features: {len(metadata.get('features', []))}")
        print(f"  - Total targets: {len(metadata.get('targets', []))}")
        print(f"  - File size: {file_size:.2f} KB")


def load_metadata(
    metadata_path: Path,
    verbose: bool = False
) -> Dict:
    """
    Load metadata from JSON file and include domain file content.
    
    This function reads the metadata JSON and automatically loads the content
    of the domain file (if specified) into the 'domain_content' field.
    
    Args:
        metadata_path: Path to JSON metadata file
        verbose: If True, prints load confirmation
        
    Returns:
        Metadata dictionary with:
        - domain_file: Path to domain knowledge file (from JSON)
        - domain_content: Content of domain file (loaded automatically)
        - dataset_info: General dataset information
        - features: List of feature metadata
        - targets: List of target metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        
    Example:
        >>> metadata = load_metadata(Path('processed/features_metadata.json'))
        >>> metadata['dataset_info']['total_samples']
        2205
        >>> metadata['domain_content']  # Content loaded automatically
        'Hydraulic System Components...'
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load domain file content if path is specified
    if 'domain_file' in metadata and metadata['domain_file']:
        domain_file_str = metadata['domain_file']
        
        # Try to resolve the path (works for both absolute and relative)
        domain_path = Path(domain_file_str)
        
        # If it's not absolute, try relative to project root
        if not domain_path.is_absolute():
            # metadata_path is: .../PoC/layers/data/processed/features_metadata.json
            # Go up 4 levels to get to PoC (project root)
            project_root = metadata_path.parent.parent.parent.parent
            domain_path = project_root / domain_file_str
        
        if domain_path.exists():
            try:
                with open(domain_path, 'r', encoding='utf-8') as f:
                    metadata['domain_content'] = f.read()
                    
                if verbose:
                    content_size = len(metadata['domain_content']) / 1024
                    print(f"  - Domain content loaded: {content_size:.2f} KB")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Could not load domain file: {e}")
                metadata['domain_content'] = None
        else:
            if verbose:
                print(f"  ⚠️  Domain file not found: {domain_path}")
            metadata['domain_content'] = None
    
    if verbose:
        print(f"✓ Metadata loaded from {metadata_path}")
        print(f"  - Total features: {len(metadata.get('features', []))}")
        print(f"  - Total targets: {len(metadata.get('targets', []))}")
    
    return metadata


def create_feature_summary(
    metadata: Dict,
    output_format: str = 'markdown'
) -> str:
    """
    Create human-readable summary of features from metadata.
    
    Args:
        metadata: Metadata dictionary
        output_format: 'markdown' or 'text'
        
    Returns:
        Formatted string with feature summary
        
    Example:
        >>> metadata = load_metadata(Path('processed/features_metadata.json'))
        >>> summary = create_feature_summary(metadata)
        >>> print(summary)
    """
    if output_format not in ['markdown', 'text']:
        raise ValueError("output_format must be 'markdown' or 'text'")
    
    lines = []
    
    # Header
    if output_format == 'markdown':
        lines.append("# Dataset Feature Summary\n")
        lines.append(f"**Total Samples**: {metadata['dataset_info']['total_samples']}")
        lines.append(f"**Total Features**: {metadata['dataset_info']['total_features']}")
        lines.append(f"**Total Targets**: {metadata['dataset_info']['total_targets']}\n")
        lines.append("## Features by Sensor\n")
    else:
        lines.append("Dataset Feature Summary")
        lines.append("=" * 80)
        lines.append(f"Total Samples: {metadata['dataset_info']['total_samples']}")
        lines.append(f"Total Features: {metadata['dataset_info']['total_features']}")
        lines.append(f"Total Targets: {metadata['dataset_info']['total_targets']}\n")
        lines.append("Features by Sensor:")
        lines.append("-" * 80)
    
    # Group features by sensor
    features_by_sensor = {}
    for feature in metadata['features']:
        sensor = feature.get('sensor_group', 'unknown')
        if sensor not in features_by_sensor:
            features_by_sensor[sensor] = []
        features_by_sensor[sensor].append(feature)
    
    # List features by sensor
    for sensor, features in sorted(features_by_sensor.items()):
        if output_format == 'markdown':
            lines.append(f"### {sensor}")
            lines.append(f"- **Count**: {len(features)}")
            lines.append(f"- **Frequency**: {features[0].get('original_frequency_hz', 'N/A')} Hz")
            lines.append(f"- **Description**: {features[0].get('description', 'N/A').split(' at timestep')[0]}\n")
        else:
            lines.append(f"\n{sensor}:")
            lines.append(f"  Count: {len(features)}")
            lines.append(f"  Frequency: {features[0].get('original_frequency_hz', 'N/A')} Hz")
            lines.append(f"  Description: {features[0].get('description', 'N/A').split(' at timestep')[0]}")
    
    # Targets section
    if output_format == 'markdown':
        lines.append("\n## Target Variables\n")
    else:
        lines.append("\n" + "-" * 80)
        lines.append("Target Variables:")
        lines.append("-" * 80)
    
    for target in metadata['targets']:
        if output_format == 'markdown':
            lines.append(f"### {target['name']}")
            lines.append(f"- **Description**: {target.get('description', 'N/A')}")
            lines.append(f"- **Classes**: {target.get('classes', [])}")
            lines.append(f"- **Problem Type**: {target.get('problem_type', 'classification')}\n")
        else:
            lines.append(f"\n{target['name']}:")
            lines.append(f"  Description: {target.get('description', 'N/A')}")
            lines.append(f"  Classes: {target.get('classes', [])}")
            lines.append(f"  Problem Type: {target.get('problem_type', 'classification')}")
    
    return '\n'.join(lines)
