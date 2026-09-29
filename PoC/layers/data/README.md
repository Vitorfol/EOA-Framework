# Data Layer - Hydraulic Systems Dataset

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Integration Flows](#integration-flows)
   - [Training Layer](#training-layer-flow)
   - [Agent Layer](#agent-layer-flow)
   - [View Layer](#view-layer-flow)
5. [Public API](#public-api)
6. [Module Reference](#module-reference)
7. [Usage Examples](#usage-examples)
8. [Data Variants](#data-variants)
9. [Getting Started](#getting-started)

---

## Overview

The **Data Layer** is responsible for loading, processing, and transforming the Condition Monitoring of Hydraulic Systems dataset. It provides a clean, modular interface for all other layers (training, agent, view) to access data in different formats optimized for their specific needs.

### Key Features

✅ **Multi-frequency normalization**: Downsamples sensors from 100 Hz, 10 Hz, and 1 Hz to a common 1 Hz baseline  
✅ **Statistical aggregation**: Extracts 8 statistical features per sensor (136 total features)  
✅ **Metadata generation**: Provides comprehensive feature metadata for agent consumption  
✅ **Multiple data formats**: Supports normalized time series, aggregated features, and raw data  
✅ **Efficient caching**: Processes data once and caches results for fast loading  
✅ **Domain knowledge**: Includes hydraulic system domain description for context

### Dataset Characteristics

- **Samples**: 2,205 cycles (60 seconds each)
- **Sensors**: 17 sensors with varying frequencies (1 Hz, 10 Hz, 100 Hz)
- **Targets**: 5 condition variables (cooler, valve, pump, accumulator, stable_flag)
- **Raw dimensions**: 43,680 feature columns + 5 target columns
- **Processed dimensions**: 1,020 (normalized) or 136 (aggregated) + 5 targets

---

## Architecture

### Directory Structure

```
layers/data/
├── README.md                          # This file
├── src/                               # Core modules
│   ├── __init__.py                    # Public API exports
│   ├── loaders.py                     # Data loading utilities
│   ├── preprocessors.py               # Normalization and aggregation
│   ├── transformers.py                # Format conversions
│   ├── metadata.py                    # Metadata generation
│   └── utils.py                       # High-level public API
├── processed/                         # Cached processed data
│   ├── hydraulic_normalized.csv       # Time series (1 Hz, 1,020 features)
│   ├── hydraulic_aggregated.csv       # Statistical features (136 features)
│   ├── features_metadata.json         # Feature metadata for agent
│   └── feature_summary.md             # Human-readable summary
├── domain/                            # Domain knowledge
│   └── hydraulic_system_domain.txt    # Hydraulic physics and context
├── scripts/                           # Processing scripts
│   └── process_data.py                # One-time data processing
└── tests/                             # Test API functions
    └── test_api.py                    # Python module using pytest 
└── notebooks/                         # Exploratory analysis
    └── data_explore.ipynb             # Initial data exploration
```

### Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| **loaders.py** | Load raw sensor data and targets | `load_full_dataset()`, `load_targets()`, `get_sensor_info()` |
| **preprocessors.py** | Normalize and aggregate data | `normalize_to_base_frequency()`, `create_aggregated_features()` |
| **transformers.py** | Transform data formats | `timeseries_to_supervised()`, `filter_features()` |
| **metadata.py** | Generate feature metadata | `generate_features_metadata()`, `get_target_metadata()` |
| **utils.py** | Public API interface | `get_original_dataframe()`, `get_filtered_dataframe()`, `get_features_metadata()`, `prepare_for_xgboost()` |

---

## Data Processing Pipeline


### Processing Script

Run once to generate all processed files:

```bash
python layers/data/scripts/process_data.py --verbose
```

This creates:
1. `hydraulic_normalized.csv` - Normalized time series (1 Hz)
2. `hydraulic_aggregated.csv` - Statistical features
3. `features_metadata.json` - Feature metadata
4. `feature_summary.md` - Human-readable summary

---

## Integration Flows

### Training Layer Flow

**Purpose**: Train machine learning models to predict component conditions

**Data Requirements**:
- Aggregated features for traditional ML (XGBoost, Random Forest)
- Normalized time series in WIDE or LONG format for lag-based XGBoost
- Optional filtered DataFrames with only the sensors/targets needed

**Integration Code**:

```python
# File: layers/training/train.py
from layers.data.src import (
    get_original_dataframe,
    prepare_for_xgboost
)

# Option 1: Traditional ML (XGBoost, Random Forest)
df = get_original_dataframe(variant='aggregated', verbose=True)
# Split with sklearn in training layer

# Option 2: Time series with XGBoost from WIDE format
df_wide = get_original_dataframe(variant='normalized', format='wide')
X, y = prepare_for_xgboost(df_wide, n_lags=5, input_format='wide')

# Option 3: Time series with XGBoost from LONG format
df_long = get_original_dataframe(variant='normalized', format='long')
X, y = prepare_for_xgboost(df_long, n_lags=5, input_format='long')

# Option 4: Filter before preparing lag features
df_filtered = get_filtered_dataframe(
    df=df_long,
    feature_list=['PS1', 'PS2', 'TS1'],
    target_list=['cooler_condition']
)
X, y = prepare_for_xgboost(
    df_filtered,
    n_lags=3,
    target_columns=['cooler_condition'],
    input_format='long'
)
```

**What Training Layer Receives**:
- `X_train`, `X_test`: Feature DataFrames
- `y_train`, `y_test`: Target DataFrames (can be multi-target)
- Clean data, no missing values, properly scaled indices

**Training Layer Workflow**:
```
Data Layer → get_original_dataframe() → Training preprocesses
         ↓
    Train models → Save to results/models/
         ↓
    Compute SHAP → Save to results/shap_values/
         ↓
    Predictions → Save to results/predictions/
```

---

### Agent Layer Flow

**Purpose**: Provide LLM-based analysis and interpretation of model results

**Data Requirements**:
- Comprehensive feature metadata (names, types, frequencies)
- Domain knowledge context (hydraulic system physics)
- Target variable descriptions and interpretations
- Model predictions and SHAP values (from training layer)

**Integration Code**:

```python
# File: layers/agent/agent.py
from layers.data.src import get_features_metadata, get_target_metadata
import json

# Load feature metadata
metadata = get_features_metadata(verbose=True)

# Access domain knowledge natively embedded in metadata
domain_context = metadata.get('domain_content', '')

# Get feature information
features = metadata['features']
# Example feature:
# {
#     'name': 'PS1_t0000',
#     'sensor_group': 'PS1',
#     'original_frequency_hz': 100,
#     'normalized': True,
#     'type': 'feature',
#     'description': 'Pressure sensor 1 at timestep 0'
# }

# Get target information
targets = metadata['targets']
target_details = get_target_metadata()
# Example:
# {
#     'cooler_condition': {
#         'description': 'Cooler condition state',
#         'classes': [3, 20, 100],
#         'interpretation': {3: 'Close to total failure', ...}
#     }
# }

# Build context for LLM prompt
context = f"""
Dataset: {metadata['dataset_info']['description']}
Features: {metadata['dataset_info']['total_features']}
Samples: {metadata['dataset_info']['total_samples']}

Domain Context:
{domain_context}

Feature Groups:
{json.dumps([f for f in features[:5]], indent=2)}  # Sample

Target Variables:
{json.dumps(target_details, indent=2)}
"""

# Use in Gemini/LLM prompt...
```

**What Agent Layer Receives**:
- **JSON metadata**: All features with descriptions, frequencies, statistics
- **Domain knowledge**: Text file with hydraulic system physics and failure modes
- **Target metadata**: Class interpretations, balancing recommendations
- **Feature groupings**: Sensors organized by type (pressure, temperature, etc.)

**Agent Layer Workflow**:
```
Data Layer → get_features_metadata() → Build LLM context
         ↓
Domain knowledge → Enhance understanding
         ↓
Training results → Load predictions/SHAP
         ↓
LLM Analysis → Generate insights
         ↓
Judge evaluation → Validate quality
```

---

### View Layer Flow

**Purpose**: Visualize data, model results, and provide interactive exploration

**Data Requirements**:
- Original DataFrames (normalized or aggregated)
- Filtered subsets (specific features/targets)
- Feature metadata for labels and descriptions
- Flexible filtering and selection

**Integration Code**:

```python
# File: layers/view/renderer.py
from layers.data.src import (
    get_original_dataframe,
    get_filtered_dataframe,
    get_features_metadata,
    get_sensor_info
)
import matplotlib.pyplot as plt

# Load data for visualization
df_agg = get_original_dataframe(variant='aggregated')
df_norm = get_original_dataframe(variant='normalized')

# Example 1: Visualize temperature sensors over time
temp_features = ['TS1_t0000', 'TS2_t0000', 'TS3_t0000', 'TS4_t0000']
df_temps = get_filtered_dataframe(
    df_norm,
    feature_list=[f'TS{i}_t{j:04d}' for i in range(1,5) for j in range(60)],
    target_list=['cooler_condition']
)

# Plot time series (first 60 seconds)
for i in range(1, 5):
    cols = [f'TS{i}_t{j:04d}' for j in range(60)]
    plt.plot(df_temps.loc[0, cols].values, label=f'TS{i}')
plt.legend()
plt.title('Temperature Sensors - First 60 Seconds')
plt.show()

# Example 2: Feature correlation heatmap
import seaborn as sns
corr_matrix = df_agg.corr()
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Example 3: Target distribution
metadata = get_features_metadata()
for target in metadata['targets']:
    name = target['name']
    classes = target['classes']
    distribution = target['class_distribution']
    
    plt.bar(distribution.keys(), distribution.values())
    plt.title(f"{name} Distribution")
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

# Example 4: Interactive filtering by sensor type
sensor_info = get_sensor_info()
pressure_sensors = [s for s, info in sensor_info.items() if 'Pressure' in info['desc']]
# Filter DataFrame to only pressure features...
```

**What View Layer Receives**:
- **Full DataFrames**: For comprehensive visualizations
- **Filtered DataFrames**: For specific feature/target combinations
- **Metadata**: For proper labeling and descriptions
- **Sensor information**: For grouping by frequency, type, etc.

**View Layer Workflow**:
```
Data Layer → get_original_dataframe() → Choose variant
         ↓
Filter features → get_filtered_dataframe()
         ↓
Get metadata → Proper labels
         ↓
Render visualizations → Charts, tables, plots
         ↓
User interaction → Dynamic filtering
```

---

## Public API

### Core Functions

#### `get_original_dataframe(variant='normalized', force_reload=False, verbose=False)`

Load processed DataFrame (normalized or aggregated).

**Parameters**:
- `variant` (str): 'normalized', 'aggregated', or 'raw'
- `force_reload` (bool): Force reprocessing even if cached
- `verbose` (bool): Print loading information

**Returns**: `pd.DataFrame`

**Use Cases**:
- Training: Load aggregated for XGBoost/RF or normalized for LSTM
- View: Load for visualization
- Agent: Load to understand data structure

---

#### `get_filtered_dataframe(df=None, feature_list=None, target_list=None, variant='aggregated', validate=True)`

Filter DataFrame to specific features and targets.

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame (loads if None)
- `feature_list` (List[str]): Feature columns to keep
- `target_list` (List[str]): Target columns to keep
- `variant` (str): Which variant to load if df is None
- `validate` (bool): Raise error if columns missing

**Returns**: `pd.DataFrame`

**Use Cases**:
- Training: Select specific targets for single-output models
- View: Focus on sensor subsets for visualization
- Experimentation: Test with reduced feature sets

---

#### `get_features_metadata(metadata_path=None, force_regenerate=False, verbose=False)`

Load comprehensive feature metadata.

**Parameters**:
- `metadata_path` (Path): Custom metadata file location
- `force_regenerate` (bool): Regenerate even if exists
- `verbose` (bool): Print loading information

**Returns**: `Dict` with structure:
```python
{
    'domain_file': 'domain/hydraulic_system_domain.txt',
    'dataset_info': {
        'total_samples': 2205,
        'total_features': 1020,
        'total_targets': 5,
        'normalized_to_1hz': True
    },
    'features': [
        {
            'name': 'PS1_t0000',
            'sensor_group': 'PS1',
            'original_frequency_hz': 100,
            'normalized': True,
            'type': 'feature',
            'description': 'Pressure sensor 1 at timestep 0',
            'statistics': {'mean': 153.2, 'std': 12.4, ...}
        },
        ...
    ],
    'targets': [
        {
            'name': 'cooler_condition',
            'type': 'target',
            'description': 'Cooler condition state',
            'classes': [3, 20, 100],
            'class_distribution': {3: 500, 20: 800, 100: 905},
            'num_classes': 3,
            'problem_type': 'classification'
        },
        ...
    ]
}
```

**Use Cases**:
- Agent: Build LLM context with feature descriptions
- Training: Understand feature characteristics
- Documentation: Auto-generate feature documentation

---

#### `prepare_for_xgboost(df=None, n_lags=5, target_columns=None, dropnan=True, input_format='wide')`

Prepare normalized time series for XGBoost training.

**Parameters**:
- `df` (pd.DataFrame): Normalized DataFrame (loads if None)
- `n_lags` (int): Number of time lags to create
- `target_columns` (List[str]): Targets to include
- `dropnan` (bool): Drop NaN rows from lag creation
- `input_format` (str): Explicitly match the topology of the provided df ('wide' or 'long'). Default is 'wide'.

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame]` - (X, y)

**Use Cases**:
- Training: Convert time series to tabular format for gradient boosting

---



## Module Reference

### src/loaders.py

**Functions**:
- `get_sensor_info()`: Returns metadata for all 17 sensors
- `load_targets()`: Load target variables from profile.txt
- `load_raw_sensors(sensors=None)`: Load raw sensor time series
- `load_full_dataset(sensors=None, include_targets=True)`: Load complete dataset
- `get_target_names()`: Get list of target column names

### src/preprocessors.py

**Functions**:
- `normalize_to_base_frequency(df, base_freq=1, aggregation='median')`: Downsample to 1 Hz
- `create_aggregated_features(df, statistics=None)`: Extract statistical features
- `validate_dataframe(df, expected_rows=2205)`: Validate data quality

### src/transformers.py

**Functions**:
- `filter_features(df, feature_list, target_list)`: Filter to specific columns
- `timeseries_to_supervised(df, n_lags=5)`: Create lag features for XGBoost
- `reshape_for_lstm(df)`: Create 3D arrays for LSTM
- `split_features_targets(df)`: Split into X and y
- `create_train_test_split(df, stratify_column=None)`: Stratified split

### src/metadata.py

**Functions**:
- `generate_features_metadata(df, normalized=True)`: Create metadata JSON
- `get_target_metadata()`: Get target variable details
- `save_metadata(metadata, output_path)`: Save metadata to JSON
- `load_metadata(metadata_path)`: Load metadata from JSON
- `create_feature_summary(metadata)`: Create human-readable summary

### src/utils.py

High-level public API that wraps lower-level modules.

### src/temporal.py

**Functions for working with temporal dimension**:
- `get_temporal_index(df, sensor_name=None)`: Extract timestamp array [0, 1, ..., 59]
- `extract_sensor_timeseries(df, sensor_name)`: Get time series for specific sensor
- `reshape_to_long_format(df, sensor_names)`: Convert to long format with explicit time column
- `plot_sensor_timeseries(df, sensor_name, cycle_indices)`: Visualize sensor time series
- `get_temporal_statistics(df, sensor_name, statistic)`: Compute stats across cycles for each timestep

**See**: [Temporal Structure](#temporal-structure-time-series) section for detailed examples.

---

## Data Variants

### 1. Raw Data

**Dimensions**: 2,205 rows × 43,685 columns  
**Features**: 43,680 (17 sensors × varying timesteps)  
**Targets**: 5  
**Memory**: ~130 MB  
**Use Case**: Full resolution time series analysis (rarely needed)

**How to Load**:
```python
df_raw = get_original_dataframe(variant='raw')
```

### 2. Normalized Data (1 Hz)

**Dimensions**: 2,205 rows × 1,025 columns  
**Features**: 1,020 (17 sensors × 60 timesteps)  
**Targets**: 5  
**Memory**: ~50 MB  
**Processing**: Median aggregation to downsample all sensors to 1 Hz  
**Use Case**: Time series models (LSTM, GRU), temporal analysis

**Why 1 Hz**:
- Lowest sensor frequency in dataset (temperature, vibration)
- Reasonable temporal resolution for hydraulic dynamics
- Median aggregation preserves signal while reducing noise

**How to Load**:
```python
df_norm = get_original_dataframe(variant='normalized')
```

### 3. Aggregated Data (Statistical Features)

**Dimensions**: 2,205 rows × 141 columns  
**Features**: 136 (17 sensors × 8 statistics)  
**Targets**: 5  
**Memory**: ~0.2 MB  
**Statistics**: mean, std, min, max, median, range, q25, q75  
**Use Case**: Traditional ML (XGBoost, Random Forest, SVM)

**Statistics Computed**:
- `mean`: Average value across cycle
- `std`: Standard deviation (variability)
- `min`: Minimum value
- `max`: Maximum value
- `median`: Median value (robust to outliers)
- `range`: max - min (spread)
- `q25`: 25th percentile
- `q75`: 75th percentile

**How to Load**:
```python
df_agg = get_original_dataframe(variant='aggregated')
```

---

## Temporal Structure (Time Series)

### How Timestamps Are Embedded

**Important**: The normalized dataset **preserves the temporal dimension** through column naming!

#### Column Naming Convention

Each timestamped column follows the pattern: `{SENSOR}_t{TIMESTEP:04d}`

Example for Pressure Sensor 1:
- `PS1_t0000` → Measurement at **t = 0 seconds**
- `PS1_t0001` → Measurement at **t = 1 second**
- `PS1_t0002` → Measurement at **t = 2 seconds**
- ...
- `PS1_t0059` → Measurement at **t = 59 seconds**

Each sensor has **60 timesteps** (one per second) within each 60-second cycle.

#### Dataset Structure

**Wide Format** (default):
```
Cycle | PS1_t0000 | PS1_t0001 | ... | PS1_t0059 | TS1_t0000 | ... | cooler_condition
------|-----------|-----------|-----|-----------|-----------|-----|------------------
0     | 152.3     | 151.8     | ... | 153.1     | 35.2      | ... | 100
1     | 148.9     | 149.2     | ... | 149.5     | 36.1      | ... | 100
```

- **Each ROW** = one complete 60-second cycle
- **Each COLUMN** = one time point for one sensor
- **1,020 feature columns** = 17 sensors × 60 timesteps

### Summary

✅ **Timestamps ARE preserved** in the normalized dataset  
✅ **Column names encode time**: `SENSOR_t0000` to `SENSOR_t0059`  
✅ **Each column = one second** in the 60-second cycle  
✅ **Utilities available** to extract and visualize temporal patterns  
✅ **Long format conversion** for easy pandas/matplotlib plotting  


---

## Getting Started

### 1. Process Raw Data (First Time Only)

```bash
cd layers/data
python scripts/process_data.py --verbose
```

This generates:
- `processed/hydraulic_normalized.csv`
- `processed/hydraulic_aggregated.csv`
- `processed/features_metadata.json`
- `processed/feature_summary.md`

### 2. Use in Training Layer

```python
# Import from data layer
from layers.data.src import get_original_dataframe, get_train_test_split

# Load aggregated data
df = get_original_dataframe(variant='aggregated')

# Get stratified split
X_train, X_test, y_train, y_test = get_train_test_split(
    df,
    stratify_column='cooler_condition',
    test_size=0.2
)

# Train model...
```

### 3. Use in Agent Layer

```python
from layers.data.src import get_features_metadata
from pathlib import Path

# Load metadata
metadata = get_features_metadata()

# Load domain knowledge
domain_file = Path('layers/data') / metadata['domain_file']
with open(domain_file, 'r') as f:
    domain_context = f.read()

# Build LLM prompt...
```

### 4. Use in View Layer

```python
from layers.data.src import (
    get_original_dataframe,
    get_filtered_dataframe,
    get_sensor_info
)

# Load data
df = get_original_dataframe(variant='aggregated')

# Filter to temperature sensors
temp_features = [f'TS{i}_mean' for i in range(1, 5)]
df_temp = get_filtered_dataframe(
    df,
    feature_list=temp_features,
    target_list=['cooler_condition']
)

# Visualize...
```
---

### Lista de Atributos e Targets

| Nome do Grupo | Tipo | Medição | Como Montar a Time Series |
| :--- | :--- | :--- | :--- |
| **PS1 a PS6** (Pressão) | Atributo (Feature) | Tempo Variado (100 Hz) | Cada um é uma série de 6000 pontos por ciclo. |
| **EPS1** (Potência) | Atributo (Feature) | Tempo Variado (100 Hz) | Uma série de 6000 pontos por ciclo. |
| **FS1, FS2** (Fluxo) | Atributo (Feature) | Tempo Variado (10 Hz) | Cada um é uma série de 600 pontos por ciclo. |
| **TS1 a TS4** (Temperatura) | Atributo (Feature) | Tempo Variado (1 Hz) | Cada um é uma série de 60 pontos por ciclo. |
| **VS1** (Vibração) | Atributo (Feature) | Tempo Variado (1 Hz) | Uma série de 60 pontos por ciclo. |
| **CE, CP, SE** (Virtuais) | Atributo (Feature) | Tempo Variado (1 Hz) | Cada um é uma série de 60 pontos por ciclo. |
| **Cooler, Valve, Pump, Accumulator** | **Target** | **Um valor por ciclo** | Para cada ciclo de 60s, há um valor de target que descreve o estado do componente *naquele* ciclo. A condição varia *entre* ciclos, mas é considerada constante *dentro* de um ciclo. |



---

## Notes

### Performance Considerations

- **First load**: Processes raw data (~1-2 minutes)
- **Subsequent loads**: Fast CSV loading (~1 second)
- **Force reload**: Use `force_reload=True` only when raw data changes

### Data Quality

- ✅ No missing values in dataset
- ✅ All sensors functional
- ✅ Validated structure and types
- ⚠️ Class imbalance in some targets (use stratification)

### Best Practices

1. **Always use processed data**: Don't load raw unless necessary
2. **Choose right variant**: Aggregated for ML, normalized for DL
3. **Stratify splits**: Maintain class balance in train/test
4. **Cache results**: Process once, load many times
5. **Document usage**: Add examples to notebooks/

---

## Support

For questions or issues:
- Check `notebooks/data_explore.ipynb` for examples
- Review `domain/hydraulic_system_domain.txt` for domain knowledge
- Examine `processed/feature_summary.md` for feature overview
