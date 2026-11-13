"""Reproducible pipeline for Cooler Condition classification (Random Forest) and XAI (SHAP).

Reads the raw ZeMA hydraulic test rig sensor matrices from `infrastructure/raw/`,
computes per-cycle mean features across the 14 physical sensors (excluding virtual
efficiency sensors CE, CP, SE to avoid target leakage), trains a RandomForestClassifier
to predict `Cooler condition / %` (Column 1 of `profile.txt`: 3, 20, 100), and computes
exact out-of-sample SHAP values via `shap.TreeExplainer`.

Outputs:
- `cooler_incidents.csv`: 9 representative out-of-sample test instances (4 at 3%,
  3 at 20%, and 2 at 100%) with cycle sensor means, model predictions, and SHAP values.
- `cooler_baseline_stats.json`: Empirical reference distribution (mean, std, min,
  p05, p50, p95, max) computed strictly from healthy training cycles (Cooler == 100%).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

INFRA_DIR = Path(__file__).resolve().parent
RAW_DIR = INFRA_DIR / "raw"
OUTPUT_CSV = INFRA_DIR / "cooler_incidents.csv"
OUTPUT_BASELINE_JSON = INFRA_DIR / "cooler_baseline_stats.json"

RANDOM_SEED = 42

# The 14 strictly physical sensors defined in description.md / documentation.md
# (Virtual sensors CE, CP, SE are excluded to prevent circular target leakage).
PHYSICAL_SENSORS: list[tuple[str, str, str, str, str]] = [
    ("PS1", "Pressure", "bar", "100 Hz", "PS1_mean_bar"),
    ("PS2", "Pressure", "bar", "100 Hz", "PS2_mean_bar"),
    ("PS3", "Pressure", "bar", "100 Hz", "PS3_mean_bar"),
    ("PS4", "Pressure", "bar", "100 Hz", "PS4_mean_bar"),
    ("PS5", "Pressure", "bar", "100 Hz", "PS5_mean_bar"),
    ("PS6", "Pressure", "bar", "100 Hz", "PS6_mean_bar"),
    ("EPS1", "Motor power", "W", "100 Hz", "EPS1_mean_W"),
    ("FS1", "Volume flow", "l/min", "10 Hz", "FS1_mean_l_min"),
    ("FS2", "Volume flow", "l/min", "10 Hz", "FS2_mean_l_min"),
    ("TS1", "Temperature", "°C", "1 Hz", "TS1_mean_C"),
    ("TS2", "Temperature", "°C", "1 Hz", "TS2_mean_C"),
    ("TS3", "Temperature", "°C", "1 Hz", "TS3_mean_C"),
    ("TS4", "Temperature", "°C", "1 Hz", "TS4_mean_C"),
    ("VS1", "Vibration", "mm/s", "1 Hz", "VS1_mean_mm_s"),
]

COOLER_LABELS: dict[int, str] = {
    3: "3% (close to total failure)",
    20: "20% (reduced efficiency)",
    100: "100% (full efficiency)",
}


def load_cycle_means(raw_dir: Path) -> np.ndarray:
    """Compute the arithmetic mean per 60-second cycle for each physical sensor."""
    n_cycles = 2205
    n_features = len(PHYSICAL_SENSORS)
    feature_matrix = np.zeros((n_cycles, n_features), dtype=np.float64)

    for feat_idx, (sensor_name, _, _, _, _) in enumerate(PHYSICAL_SENSORS):
        sensor_file = raw_dir / f"{sensor_name}.txt"
        if not sensor_file.is_file():
            raise FileNotFoundError(f"Missing raw sensor file: {sensor_file}")

        with sensor_file.open("r", encoding="latin-1") as f:
            for row_idx, line in enumerate(f):
                values = np.fromstring(line, dtype=np.float64, sep="\t")
                feature_matrix[row_idx, feat_idx] = float(np.mean(values))

    return feature_matrix


def load_profile_targets(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Cooler condition (col 0) and Stable flag (col 4) from profile.txt."""
    profile_file = raw_dir / "profile.txt"
    profile_data = np.loadtxt(profile_file, dtype=np.int64, delimiter="\t")
    cooler_target = profile_data[:, 0]
    stable_flag = profile_data[:, 4]
    return cooler_target, stable_flag


def build_baseline_stats(
    X_train: np.ndarray,
    y_train: np.ndarray,
    accuracy: float,
    f1_macro: float,
) -> dict[str, Any]:
    """Compute reference sensor statistics on healthy training cycles (Cooler == 100%)."""
    healthy_mask = y_train == 100
    X_healthy = X_train[healthy_mask]

    sensors_stats: dict[str, Any] = {}
    for feat_idx, (tag, quantity, unit, rate, col_name) in enumerate(PHYSICAL_SENSORS):
        col = X_healthy[:, feat_idx]
        sensors_stats[tag] = {
            "physical_quantity": quantity,
            "unit": unit,
            "sampling_rate": rate,
            "csv_column": col_name,
            "healthy_100pct_mean": round(float(np.mean(col)), 4),
            "healthy_100pct_std": round(float(np.std(col)), 4),
            "healthy_100pct_min": round(float(np.min(col)), 4),
            "healthy_100pct_p05": round(float(np.percentile(col, 5)), 4),
            "healthy_100pct_p50": round(float(np.percentile(col, 50)), 4),
            "healthy_100pct_p95": round(float(np.percentile(col, 95)), 4),
            "healthy_100pct_max": round(float(np.max(col)), 4),
        }

    return {
        "dataset": "ZeMA Hydraulic Test Rig (Condition Monitoring of Hydraulic Systems)",
        "target_variable": "Cooler condition / %",
        "target_classes": COOLER_LABELS,
        "feature_extraction": "Cycle arithmetic mean (60s window) across 14 physical sensors",
        "excluded_virtual_sensors": ["CE", "CP", "SE"],
        "model": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": RANDOM_SEED,
            "train_cycles": int(X_train.shape[0]),
            "healthy_baseline_cycles": int(np.sum(healthy_mask)),
            "test_accuracy": round(accuracy, 6),
            "test_f1_macro": round(f1_macro, 6),
        },
        "sensors": sensors_stats,
    }


def extract_class_shap_vector(
    raw_shap: Any,
    sample_pos: int,
    class_idx: int,
) -> np.ndarray:
    """Handle both list-of-arrays and 3D-array formats across SHAP versions."""
    if isinstance(raw_shap, list):
        return np.asarray(raw_shap[class_idx][sample_pos], dtype=np.float64)
    arr = np.asarray(raw_shap, dtype=np.float64)
    if arr.ndim == 3:
        return arr[sample_pos, :, class_idx]
    return arr[sample_pos]


def main() -> None:
    print("1. Loading raw sensor matrices and computing 60s cycle means...")
    X = load_cycle_means(RAW_DIR)
    y_cooler, stable_flags = load_profile_targets(RAW_DIR)
    cycle_ids = np.arange(1, X.shape[0] + 1, dtype=np.int64)

    print("2. Splitting dataset (80% train / 20% test, stratified, seed=42)...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        cycles_train,
        cycles_test,
        stable_train,
        stable_test,
    ) = train_test_split(
        X,
        y_cooler,
        cycle_ids,
        stable_flags,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_cooler,
    )

    print("3. Training RandomForestClassifier(n_estimators=100, random_state=42)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    print(f"   Test Accuracy: {acc:.4%} | Macro F1-Score: {f1:.4%}")
    print(classification_report(y_test, y_pred, digits=4))

    print("4. Computing exact out-of-sample SHAP values via shap.TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    raw_shap_values = explainer.shap_values(X_test)
    expected_values = np.asarray(explainer.expected_value, dtype=np.float64).ravel()
    classes_list = [int(c) for c in clf.classes_]

    # Select 9 curated out-of-sample test instances ordered by cycle_id within each tier:
    # - 4 instances of 3% (close to total failure)
    # - 3 instances of 20% (reduced efficiency)
    # - 2 instances of 100% (full efficiency, healthy baseline reference)
    selection_plan = [(3, 4), (20, 3), (100, 2)]
    selected_positions: list[int] = []

    for target_code, count in selection_plan:
        matching_pos = [
            pos
            for pos in range(len(y_test))
            if int(y_pred[pos]) == target_code and int(y_test[pos]) == target_code
        ]
        # Sort by cycle_id for deterministic, evenly spaced selection across the test set
        matching_pos.sort(key=lambda pos: int(cycles_test[pos]))
        indices = np.linspace(0, len(matching_pos) - 1, num=count, dtype=int)
        for idx in indices:
            selected_positions.append(matching_pos[int(idx)])

    print(f"5. Exporting {len(selected_positions)} curated test incidents to {OUTPUT_CSV.name}...")
    sensor_tags = [s[0] for s in PHYSICAL_SENSORS]
    sensor_cols = [s[4] for s in PHYSICAL_SENSORS]

    fieldnames = [
        "incident_id",
        "cycle_id",
        "actual_condition_code",
        "actual_condition_label",
        "predicted_condition_code",
        "predicted_condition_label",
        "confidence",
        "stable_flag",
        *sensor_cols,
        "base_value",
        "shap_attributions_json",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inc_id, pos in enumerate(selected_positions, start=1):
            pred_code = int(y_pred[pos])
            actual_code = int(y_test[pos])
            class_idx = classes_list.index(pred_code)
            confidence = float(y_proba[pos, class_idx])
            base_val = float(expected_values[class_idx])

            shap_vec = extract_class_shap_vector(raw_shap_values, pos, class_idx)
            # Order SHAP dictionary by descending absolute attribution magnitude
            shap_pairs = sorted(
                zip(sensor_tags, shap_vec, strict=True),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            shap_dict = {tag: round(float(val), 4) for tag, val in shap_pairs}

            row_dict: dict[str, Any] = {
                "incident_id": inc_id,
                "cycle_id": int(cycles_test[pos]),
                "actual_condition_code": actual_code,
                "actual_condition_label": COOLER_LABELS[actual_code],
                "predicted_condition_code": pred_code,
                "predicted_condition_label": COOLER_LABELS[pred_code],
                "confidence": round(confidence, 4),
                "stable_flag": int(stable_test[pos]),
                "base_value": round(base_val, 4),
                "shap_attributions_json": json.dumps(shap_dict),
            }
            for feat_idx, col_name in enumerate(sensor_cols):
                row_dict[col_name] = round(float(X_test[pos, feat_idx]), 4)

            writer.writerow(row_dict)

    print(f"6. Exporting baseline statistics to {OUTPUT_BASELINE_JSON.name}...")
    baseline_payload = build_baseline_stats(X_train, y_train, acc, f1)
    with OUTPUT_BASELINE_JSON.open("w", encoding="utf-8") as jf:
        json.dump(baseline_payload, jf, indent=2, ensure_ascii=False)

    print("Done! All artifacts generated successfully.")


if __name__ == "__main__":
    main()
