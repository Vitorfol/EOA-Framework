# Data Layer Quick Start Guide

## Overview

The data layer provides **three data formats** for different use cases:

| Format | Shape | Best For | 
|--------|-------|----------|
| **Long** | 132,300 × 23 | Visualization, temporal analysis |
| **Wide** | 2,205 × 1,025 | LSTM, sequence models | 
| **Aggregated** | 2,205 × 141 | Traditional ML (XGBoost, RF) |

## ✅ Data Processing Complete

All data files have been generated and are ready to use:

```bash
layers/data/processed/
├── hydraulic_normalized_long.csv    # ONE column per sensor + time_sec
├── hydraulic_normalized_wide.csv    # sensor_t0000, sensor_t0001, ...
├── hydraulic_aggregated.csv         # 8 statistics per sensor
├── features_metadata.json           # For agent layer
└── feature_summary.md               # Human-readable summary
```

## 🚀 Quick Usage

### Option 1: Direct CSV Loading (Simple)

```python
import pandas as pd

# For visualization and exploration
df_long = pd.read_csv('layers/data/processed/hydraulic_normalized_long.csv')

# For traditional ML
df_agg = pd.read_csv('layers/data/processed/hydraulic_aggregated.csv')

# For LSTM/sequence models  
df_wide = pd.read_csv('layers/data/processed/hydraulic_normalized_wide.csv')
```

### Option 2: Using the Data API (Recommended)

```python
from layers.data.src import get_original_dataframe

# Get long format (default)
df = get_original_dataframe(variant='normalized', format='long')
# → 132,300 rows × 23 columns
# → Columns: time_sec, CE, CP, ..., VS1, targets
# → time_sec is CONTINUOUS from 0 to 132,299

# Get wide format
df = get_original_dataframe(variant='normalized', format='wide')
# → 2,205 rows × 1,025 columns  
# → Columns: CE_t0000, CE_t0001, ..., PS1_t0000, ...

# Get aggregated features
df = get_original_dataframe(variant='aggregated')
# → 2,205 rows × 141 columns
# → Columns: CE_mean, CE_std, CE_min, ..., targets
```

## 📊 Long Format Structure (PRIMARY)

This is the recommended format for most use cases:

```
   time_sec     CE     CP   EPS1   ...   TS1    TS2  cooler_condition
0         0  47.20  2.184  2552.9  ...  35.57  40.96                 3
1         1  47.27  2.184  2941.8  ...  35.49  40.95                 3
2         2  47.25  2.184  2910.9  ... 35.47  40.97                 3
...     ...    ...    ...     ...  ...    ...    ...               ...
132299 132299  ...    ...     ...  ...    ...    ...                 ...
```

**Key Features:**
- Each row = one time point (1 second)
- **time_sec is CONTINUOUS** from 0 to 132,299 (treats all data as one sequence)
- ONE column per sensor (not 60 columns!)
- 132,300 total timesteps (2,205 original cycles × 60 seconds)
- Targets repeat in blocks of 60 (each original cycle's condition)

## 🎯 Integration Examples

### Training Layer (Traditional ML)

```python
# train.py
from layers.data.src import get_original_dataframe
from sklearn.ensemble import RandomForestClassifier

# Load aggregated features
df = get_original_dataframe(variant='aggregated')

# Separate features and targets
X = df.drop(['cooler_condition', 'valve_condition', 'pump_leakage', 
             'accumulator', 'stable_flag'], axis=1)
y = df['cooler_condition']

# Train model
model = RandomForestClassifier()
model.fit(X, y)
```

### Training Layer (LSTM)

```python
# train.py  
from layers.data.src import get_original_dataframe
import numpy as np

# Load wide format
df = get_original_dataframe(variant='normalized', format='wide')

# Extract targets
targets = df[['cooler_condition', 'valve_condition', 'pump_leakage', 
              'accumulator', 'stable_flag']]

# Get sensor columns (remove targets)
sensor_cols = [c for c in df.columns if c not in targets.columns]
X = df[sensor_cols].values

# Reshape for LSTM: (samples, timesteps, features)
n_samples = len(df)
n_timesteps = 60  # seconds
n_features = len(sensor_cols) // n_timesteps

X_lstm = X.reshape(n_samples, n_timesteps, n_features)
# Shape: (2205, 60, 17)
```

### Agent Layer

```python
# agent.py
from layers.data.src import get_original_dataframe, get_features_metadata
from pathlib import Path

# Load data for analysis
df = get_original_dataframe(variant='normalized', format='long')

# Load metadata for LLM context
metadata = get_features_metadata()

# Load domain knowledge
domain_path = Path('layers/data/domain/hydraulic_system_domain.txt')
with open(domain_path) as f:
    domain_knowledge = f.read()

# Now you can provide to LLM:
# 1. Actual data (df) for specific queries
# 2. Metadata (JSON) for feature understanding
# 3. Domain knowledge (text) for contextual reasoning
```

### View Layer (Visualization)

```python
# renderer.py
from layers.data.src import get_original_dataframe
import matplotlib.pyplot as plt

# Load long format
df = get_original_dataframe(variant='normalized', format='long')

# Plot a time window (first 300 seconds)
time_window = df[df['time_sec'] < 300]

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(time_window['time_sec'], time_window['PS1'], label='Pressure PS1')
plt.xlabel('Time (seconds)')
plt.ylabel('Pressure (bar)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_window['time_sec'], time_window['TS1'], label='Temperature TS1', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 📝 Common Operations

### Get Time Range

```python
# Get first minute (seconds 0-59, original cycle 0)
first_minute = df[df['time_sec'] < 60]
# → 60 rows
```

### Get Specific Time Window

```python
# Get time range 1000-1059 (roughly original cycle 16-17)
window = df[(df['time_sec'] >= 1000) & (df['time_sec'] < 1060)]
# → 60 rows
```

### Plot Time Series

```python
import matplotlib.pyplot as plt

# Plot entire pressure sensor PS1
plt.figure(figsize=(15, 4))
plt.plot(df['time_sec'], df['PS1'])
plt.xlabel('Time (seconds)')
plt.ylabel('Pressure (bar)')
plt.title('PS1 - Continuous Time Series')
plt.show()
```

## 🔧 Running the Processing Pipeline

If you need to regenerate the data:

```bash
# Activate environment
source venv/bin/activate

# Run processing script
python layers/data/scripts/process_data.py --verbose

# Force regeneration (overwrite existing files)
python layers/data/scripts/process_data.py --verbose --force
```

## 📚 Available Scripts

1. **process_data.py** - Generate all data files (one-time)
2. **test_api.py** - Pytest to make sure the functions implemented by the API are correct

```bash
# Run processment
python layers/data/scripts/process_data.py
```

## 🎓 Next Steps

### Training Layer
- [ ] Update `train.py` to use `get_original_dataframe`
- [ ] Test XGBoost with long or wide format
- [ ] Test ARIMA with long format

### Agent Layer  
- [ ] Use `get_features_metadata` into LLM context, this function is gonna load the json and read the domain context
- [ ] Use long format for temporal reasoning queries

### View Layer
- [ ] Create visualization dashboard using long format
- [ ] Plot sensor trends over time
- [ ] Compare cycles with different conditions

## 📖 Full Documentation

See [README.md](README.md) for complete architecture documentation.

