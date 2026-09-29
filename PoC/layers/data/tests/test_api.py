"""
Tests for the Data Layer public API.

Tests the 4 public functions:
    - get_original_dataframe
    - get_filtered_dataframe
    - get_features_metadata
    - prepare_for_xgboost
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
import pandas as pd
from layers.data.src import (
    get_original_dataframe,
    get_filtered_dataframe,
    get_features_metadata,
    prepare_for_xgboost,
)

TARGET_NAMES = [
    'cooler_condition', 'valve_condition', 'pump_leakage',
    'accumulator', 'stable_flag'
]


# ─── get_original_dataframe ──────────────────────────────────────────────────

class TestGetOriginalDataframe:

    def test_aggregated_shape(self):
        df = get_original_dataframe(variant='aggregated')
        assert df.shape[0] == 2205, f"Expected 2205 rows, got {df.shape[0]}"
        assert df.shape[1] > 100, f"Expected >100 cols, got {df.shape[1]}"

    def test_aggregated_has_targets(self):
        df = get_original_dataframe(variant='aggregated')
        for target in TARGET_NAMES:
            assert target in df.columns, f"Missing target: {target}"

    def test_normalized_long_shape(self):
        df = get_original_dataframe(variant='normalized', format='long')
        assert df.shape[0] == 132300, f"Expected 132300 rows, got {df.shape[0]}"
        assert 'time_sec' in df.columns

    def test_normalized_long_continuous_time(self):
        df = get_original_dataframe(variant='normalized', format='long')
        assert df['time_sec'].min() == 0
        assert df['time_sec'].max() == 132299
        assert df['time_sec'].is_monotonic_increasing

    def test_normalized_long_has_targets(self):
        df = get_original_dataframe(variant='normalized', format='long')
        for target in TARGET_NAMES:
            assert target in df.columns, f"Missing target: {target}"

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            get_original_dataframe(variant='invalid')


# ─── get_filtered_dataframe ───────────────────────────────────────────────────

class TestGetFilteredDataframe:

    def test_filter_features_and_target(self):
        df = get_filtered_dataframe(
            feature_list=['TS1_mean', 'TS2_mean'],
            target_list=['cooler_condition']
        )
        assert list(df.columns) == ['TS1_mean', 'TS2_mean', 'cooler_condition']
        assert df.shape[0] == 2205

    def test_filter_only_targets(self):
        df = get_filtered_dataframe(target_list=['pump_leakage'])
        assert 'pump_leakage' in df.columns
        assert df.shape[0] == 2205

    def test_filter_invalid_column_raises(self):
        with pytest.raises(ValueError):
            get_filtered_dataframe(feature_list=['nonexistent_column'])

    def test_filter_returns_subset(self):
        df_full = get_original_dataframe(variant='aggregated')
        df_filtered = get_filtered_dataframe(
            feature_list=['PS1_mean', 'PS2_mean'],
            target_list=['cooler_condition']
        )
        assert df_filtered.shape[1] < df_full.shape[1]
        assert df_filtered.shape[0] == df_full.shape[0]


# ─── get_features_metadata ────────────────────────────────────────────────────

class TestGetFeaturesMetadata:

    def test_returns_dict(self):
        metadata = get_features_metadata()
        assert isinstance(metadata, dict)

    def test_has_required_keys(self):
        metadata = get_features_metadata()
        required = ['domain_file', 'domain_content', 'dataset_info', 'features', 'targets']
        for key in required:
            assert key in metadata, f"Missing key: {key}"

    def test_domain_content_loaded(self):
        metadata = get_features_metadata()
        assert metadata['domain_content'] is not None
        assert isinstance(metadata['domain_content'], str)
        assert len(metadata['domain_content']) > 1000, "Domain content too short"

    def test_dataset_info(self):
        metadata = get_features_metadata()
        info = metadata['dataset_info']
        assert info['total_samples'] == 132300
        assert info['total_features'] == 18
        assert info['total_targets'] == 5

    def test_features_list(self):
        metadata = get_features_metadata()
        assert len(metadata['features']) == 18
        feature_names = [f['name'] for f in metadata['features']]
        assert 'time_sec' in feature_names

    def test_targets_list(self):
        metadata = get_features_metadata()
        assert len(metadata['targets']) == 5
        target_names = [t['name'] for t in metadata['targets']]
        for target in TARGET_NAMES:
            assert target in target_names, f"Missing target: {target}"

    def test_target_has_classes(self):
        metadata = get_features_metadata()
        cooler = next(t for t in metadata['targets'] if t['name'] == 'cooler_condition')
        assert cooler['classes'] == [3, 20, 100]
        assert cooler['num_classes'] == 3
        assert cooler['problem_type'] == 'classification'


# ─── prepare_for_xgboost ─────────────────────────────────────────────────────

class TestPrepareForXgboost:

    def test_returns_tuple(self):
        X, y = prepare_for_xgboost(n_lags=3)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

    def test_X_has_lag_features(self):
        X, y = prepare_for_xgboost(n_lags=3)
        lag_cols = [col for col in X.columns if '_lag' in col]
        assert len(lag_cols) > 0

    def test_y_has_all_targets(self):
        X, y = prepare_for_xgboost(n_lags=3)
        for target in TARGET_NAMES:
            assert target in y.columns, f"Missing target in y: {target}"

    def test_no_nan_by_default(self):
        X, y = prepare_for_xgboost(n_lags=3, dropnan=True)
        assert not X.isnull().any().any(), "X contains NaN values"
        assert not y.isnull().any().any(), "y contains NaN values"

    def test_shapes_match(self):
        X, y = prepare_for_xgboost(n_lags=3)
        assert X.shape[0] == y.shape[0], "X and y have different number of rows"

    def test_filter_to_single_target(self):
        X, y = prepare_for_xgboost(n_lags=3, target_columns=['cooler_condition'])
        assert list(y.columns) == ['cooler_condition']

    def test_accepts_long_format(self):
        df_long = get_original_dataframe(variant='normalized', format='long')
        X, y = prepare_for_xgboost(df=df_long, n_lags=3, input_format='long')
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert 'PS1_lag1' in X.columns
        assert X.shape[0] == y.shape[0]

    def test_loads_long_format_when_requested(self):
        X, y = prepare_for_xgboost(n_lags=3, input_format='long')
        assert 'PS1_lag1' in X.columns
        assert X.shape[0] == 132300 - 3
        assert y.shape[0] == 132300 - 3

    def test_accepts_filtered_long_dataframe(self):
        df_long = get_original_dataframe(variant='normalized', format='long')
        df_filtered = get_filtered_dataframe(
            df=df_long,
            feature_list=['PS1', 'PS2'],
            target_list=['cooler_condition']
        )
        X, y = prepare_for_xgboost(
            df=df_filtered,
            n_lags=2,
            target_columns=['cooler_condition'],
            input_format='long'
        )
        assert list(y.columns) == ['cooler_condition']
        assert sorted(X.columns.tolist()) == ['PS1_lag1', 'PS1_lag2', 'PS2_lag1', 'PS2_lag2']

    def test_invalid_input_format_raises(self):
        with pytest.raises(ValueError):
            prepare_for_xgboost(n_lags=3, input_format='invalid')
