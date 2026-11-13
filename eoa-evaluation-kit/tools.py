"""Domain tools and data loaders for the EOA Hydraulic System Evaluation Kit.

Provides zero-dependency helpers (built strictly with Python's standard library)
to inspect the ZeMA Hydraulic Test Rig dataset, load faithful domain documentation,
and equip EOA agents with deterministic or autonomous tools.

Available functions:
- `list_incidents`: List the 9 curated out-of-sample test incidents.
- `get_incident`: Retrieve `prediction` and `xai` artifacts for an incident.
- `get_sensor_readings`: Retrieve the 14 physical sensor cycle averages.
- `get_baseline_comparison`: Compare incident readings against healthy (100%) operation.
- `load_domain_context`: Load ZeMA documentation (`docs/*.md`) and sensor metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

KIT_DIR = Path(__file__).resolve().parent
INFRA_DIR = KIT_DIR / "infrastructure"
DOCS_DIR = INFRA_DIR / "docs"
INCIDENTS_CSV_PATH = INFRA_DIR / "cooler_incidents.csv"
BASELINE_JSON_PATH = INFRA_DIR / "cooler_baseline_stats.json"


def _load_incidents_rows() -> list[dict[str, str]]:
    """Load all rows from `infrastructure/cooler_incidents.csv`."""
    if not INCIDENTS_CSV_PATH.exists():
        msg = f"Incidents file not found at {INCIDENTS_CSV_PATH}"
        raise FileNotFoundError(msg)
    with INCIDENTS_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_baseline_stats() -> dict[str, Any]:
    """Load baseline statistics from `infrastructure/cooler_baseline_stats.json`."""
    if not BASELINE_JSON_PATH.exists():
        msg = f"Baseline stats file not found at {BASELINE_JSON_PATH}"
        raise FileNotFoundError(msg)
    return json.loads(BASELINE_JSON_PATH.read_text(encoding="utf-8"))


def _find_incident_row(
    incident_id: int | str | None = None,
    cycle_id: int | str | None = None,
) -> dict[str, str]:
    """Resolve an incident row by explicit `cycle_id` or `incident_id`."""
    rows = _load_incidents_rows()

    # 1. Explicit cycle_id lookup
    if cycle_id is not None:
        try:
            target_cycle = int(str(cycle_id).strip())
        except ValueError as exc:
            msg = f"Invalid cycle_id '{cycle_id}'. Expected an integer."
            raise ValueError(msg) from exc
        for row in rows:
            if int(row["cycle_id"]) == target_cycle:
                return row
        valid_cycles = [int(r["cycle_id"]) for r in rows]
        msg = (
            f"Cycle '{cycle_id}' not found in curated test incidents. "
            f"Valid cycle_id values: {valid_cycles}."
        )
        raise KeyError(msg)

    # 2. Lookup by incident_id (1..9), with fallback to cycle_id if > 9
    if incident_id is not None:
        try:
            target_id = int(str(incident_id).strip())
        except ValueError as exc:
            msg = f"Invalid incident_id '{incident_id}'. Expected an integer (e.g., 1..9)."
            raise ValueError(msg) from exc

        # Match incident_id (1..9)
        for row in rows:
            if int(row["incident_id"]) == target_id:
                return row

        # Fallback: match cycle_id if someone passed a cycle number as incident_id (e.g. 241, 492)
        for row in rows:
            if int(row["cycle_id"]) == target_id:
                return row

        valid_ids = [int(r["incident_id"]) for r in rows]
        valid_cycles = [int(r["cycle_id"]) for r in rows]
        msg = (
            f"Incident '{incident_id}' not found. "
            f"Valid incident_id values: {valid_ids} (or cycle_id: {valid_cycles})."
        )
        raise KeyError(msg)

    msg = "Either incident_id (1..9) or cycle_id must be provided."
    raise ValueError(msg)


def list_incidents(condition_code: int | None = None) -> list[dict[str, Any]]:
    """List the curated out-of-sample hydraulic cooler incidents.

    Args:
        condition_code: Optional filter by cooler condition code (`3`, `20`, or `100`).
            If None, returns all 9 curated test incidents.

    Returns:
        List of dictionaries summarizing each incident (`incident_id`, `cycle_id`,
        `actual_condition_label`, `predicted_condition_label`, `confidence`,
        `stable_flag`, and `top_3_shap_features`).

    Example:
        >>> from tools import list_incidents
        >>> failures = list_incidents(condition_code=3)
        >>> len(failures)
        4
    """
    rows = _load_incidents_rows()
    summaries: list[dict[str, Any]] = []

    for row in rows:
        pred_code = int(row["predicted_condition_code"])
        if condition_code is not None and pred_code != int(condition_code):
            continue

        shap_dict: dict[str, float] = json.loads(row["shap_attributions_json"])
        top_3 = sorted(
            shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:3]

        summaries.append(
            {
                "incident_id": int(row["incident_id"]),
                "cycle_id": int(row["cycle_id"]),
                "actual_condition_code": int(row["actual_condition_code"]),
                "actual_condition_label": row["actual_condition_label"],
                "predicted_condition_code": pred_code,
                "predicted_condition_label": row["predicted_condition_label"],
                "confidence": float(row["confidence"]),
                "stable_flag": int(row["stable_flag"]),
                "top_3_shap_features": dict(top_3),
            }
        )

    return summaries


def get_incident(
    incident_id: int | None = None,
    *,
    cycle_id: int | None = None,
) -> dict[str, Any]:
    """Fetch the ML prediction and TreeSHAP attributions for a given incident.

    Compatible with both Direct/Push Mode (passing `result["prediction"]` and
    `result["xai"]` to `layer.explain(...)`) and Autonomous/Pull Mode (registering
    `get_incident` on `ArtifactsAgent` and calling `layer.explain(incident_id=...)`).

    Args:
        incident_id: Curated incident identifier (`1` to `9`), or cycle number if > 9.
        cycle_id: Optional keyword argument to look up strictly by ZeMA `cycle_id`.

    Returns:
        Dictionary with `"prediction"` (matching `PredictionArtifact`) and
        `"xai"` (matching `XAIArtifact`).

    Example:
        >>> from tools import get_incident
        >>> inc = get_incident(2)
        >>> inc["prediction"]["predicted_value"]
        '3% (close to total failure)'
    """
    row = _find_incident_row(incident_id=incident_id, cycle_id=cycle_id)
    shap_dict: dict[str, float] = json.loads(row["shap_attributions_json"])
    stable_flag = int(row["stable_flag"])

    return {
        "prediction": {
            "target_variable": "Cooler condition / %",
            "predicted_value": row["predicted_condition_label"],
            "confidence_or_probability": float(row["confidence"]),
            "status_label": row["predicted_condition_label"],
            "model_name": "RandomForestClassifier (14 physical sensors)",
            "metadata": {
                "incident_id": int(row["incident_id"]),
                "cycle_id": int(row["cycle_id"]),
                "predicted_condition_code": int(row["predicted_condition_code"]),
                "actual_condition_code": int(row["actual_condition_code"]),
                "actual_condition_label": row["actual_condition_label"],
                "stable_flag": stable_flag,
                "stable_flag_meaning": (
                    "0: conditions were stable"
                    if stable_flag == 0
                    else "1: static conditions might not have been reached yet"
                ),
            },
        },
        "xai": {
            "method": "TreeSHAP (shap.TreeExplainer)",
            "feature_attributions": shap_dict,
            "baseline_values": {"base_value": float(row["base_value"])},
            "raw_artifacts": {
                "incident_id": int(row["incident_id"]),
                "cycle_id": int(row["cycle_id"]),
            },
        },
    }


def get_sensor_readings(
    incident_id: int | None = None,
    *,
    cycle_id: int | None = None,
) -> dict[str, Any]:
    """Fetch the 60-second cycle mean readings for all 14 physical sensors.

    Can be called directly in Python or registered as a tool on
    `ContextManagerAgent`, `SpecializedAgent`, or `ExplainerAgent`.

    Args:
        incident_id: Curated incident identifier (`1` to `9`), or cycle number if > 9.
        cycle_id: Optional keyword argument to look up strictly by ZeMA `cycle_id`.

    Returns:
        Dictionary with sensor readings, units, sampling rates, `stable_flag`,
        and a `documents` digest for automatic `DomainContext` ingestion.

    Example:
        >>> from tools import get_sensor_readings
        >>> telemetry = get_sensor_readings(2)
        >>> telemetry["sensors"]["TS1"]["value"]
        54.2448
    """
    row = _find_incident_row(incident_id=incident_id, cycle_id=cycle_id)
    baseline = _load_baseline_stats()
    sensors_meta: dict[str, Any] = baseline["sensors"]

    inc_id = int(row["incident_id"])
    cyc_id = int(row["cycle_id"])
    stable_flag = int(row["stable_flag"])
    stable_desc = (
        "0 (conditions were stable)"
        if stable_flag == 0
        else "1 (static conditions might not have been reached yet)"
    )

    readings: dict[str, dict[str, Any]] = {}
    doc_lines: list[str] = [
        f"Sensor Telemetry for Incident {inc_id} (Cycle {cyc_id}, "
        f"Stable Flag = {stable_desc}):"
    ]

    for sensor_id, meta in sensors_meta.items():
        col_name = meta["csv_column"]
        val = float(row[col_name])
        unit = meta["unit"]
        qty = meta["physical_quantity"]
        rate = meta["sampling_rate"]
        readings[sensor_id] = {
            "value": val,
            "unit": unit,
            "physical_quantity": qty,
            "sampling_rate": rate,
        }
        doc_lines.append(f"- {sensor_id} ({qty}): {val:.4f} {unit} (sampled at {rate})")

    return {
        "incident_id": inc_id,
        "cycle_id": cyc_id,
        "stable_flag": stable_flag,
        "stable_flag_description": stable_desc,
        "sensors": readings,
        "documents": {
            f"incident_{inc_id}_sensor_readings": "\n".join(doc_lines),
        },
    }


def get_baseline_comparison(
    incident_id: int | None = None,
    *,
    cycle_id: int | None = None,
) -> dict[str, Any]:
    """Compare an incident's 14 physical sensor readings against healthy (100%) baseline.

    Uses the training-set reference statistics (`Cooler condition = 100%`, N=593 cycles)
    stored in `infrastructure/cooler_baseline_stats.json` to compute exact deviations
    and z-scores for each physical sensor.

    Args:
        incident_id: Curated incident identifier (`1` to `9`), or cycle number if > 9.
        cycle_id: Optional keyword argument to look up strictly by ZeMA `cycle_id`.

    Returns:
        Dictionary containing per-sensor baseline comparison metrics and a
        `documents` entry formatted for `ContextManagerAgent` and `ValidatorAgent`.

    Example:
        >>> from tools import get_baseline_comparison
        >>> comp = get_baseline_comparison(2)
        >>> comp["comparisons"]["TS1"]["z_score"] > 20
        True
    """
    row = _find_incident_row(incident_id=incident_id, cycle_id=cycle_id)
    baseline = _load_baseline_stats()
    sensors_meta: dict[str, Any] = baseline["sensors"]

    inc_id = int(row["incident_id"])
    cyc_id = int(row["cycle_id"])
    comparisons: dict[str, dict[str, Any]] = {}

    doc_lines: list[str] = [
        f"Healthy Baseline Comparison for Incident {inc_id} (Cycle {cyc_id}) "
        f"vs. 100% Full Efficiency Reference (N={baseline['model']['healthy_baseline_cycles']} training cycles):"
    ]

    for sensor_id, meta in sensors_meta.items():
        col_name = meta["csv_column"]
        val = float(row[col_name])
        mean_100 = float(meta["healthy_100pct_mean"])
        std_100 = float(meta["healthy_100pct_std"])
        p05 = float(meta["healthy_100pct_p05"])
        p95 = float(meta["healthy_100pct_p95"])
        unit = meta["unit"]

        delta = round(val - mean_100, 4)
        z_score = round((val - mean_100) / std_100, 2) if std_100 > 0 else 0.0
        outside_p05_p95 = val < p05 or val > p95

        comparisons[sensor_id] = {
            "current_value": val,
            "unit": unit,
            "healthy_100pct_mean": mean_100,
            "healthy_100pct_std": std_100,
            "healthy_100pct_p05": p05,
            "healthy_100pct_p95": p95,
            "delta_from_healthy_mean": delta,
            "z_score": z_score,
            "outside_healthy_p05_p95": outside_p05_p95,
        }
        status_tag = "OUTSIDE [p05, p95]" if outside_p05_p95 else "within [p05, p95]"
        doc_lines.append(
            f"- {sensor_id} ({meta['physical_quantity']}): current={val:.4f} {unit} | "
            f"healthy_mean={mean_100:.4f} {unit} (std={std_100:.4f}, "
            f"p05..p95=[{p05:.4f}, {p95:.4f}] {unit}) | "
            f"delta={delta:+.4f} {unit} (z={z_score:+.2f}, {status_tag})"
        )

    return {
        "incident_id": inc_id,
        "cycle_id": cyc_id,
        "comparisons": comparisons,
        "documents": {
            f"incident_{inc_id}_baseline_comparison": "\n".join(doc_lines),
        },
    }


def load_domain_context(include_baseline_thresholds: bool = True) -> dict[str, Any]:
    """Load the official ZeMA Hydraulic System documentation and sensor dictionary.

    Reads `infrastructure/docs/description.md`, `infrastructure/docs/documentation.md`,
    and `infrastructure/cooler_baseline_stats.json` into a dictionary ready to be
    passed to `layer.explain(context=...)` or `DomainContext.from_input(...)`.

    Args:
        include_baseline_thresholds: If True, populates `thresholds` with the
            healthy (100% cooler efficiency) reference ranges alongside the
            target condition codes.

    Returns:
        Dictionary structured for EOA's `DomainContext` contract (`variables`,
        `thresholds`, `documents`, `schemas`).
    """
    baseline = _load_baseline_stats()
    sensors_meta: dict[str, Any] = baseline["sensors"]

    variables: dict[str, str] = {}
    for sensor_id, meta in sensors_meta.items():
        variables[sensor_id] = (
            f"{meta['physical_quantity']} sensor (unit: {meta['unit']}, "
            f"sampling rate: {meta['sampling_rate']})"
        )

    thresholds: dict[str, Any] = {
        "Cooler condition / %": (
            "3: close to total failure | 20: reduced efficiency | 100: full efficiency"
        ),
        "Stable flag": (
            "0: conditions were stable | "
            "1: static conditions might not have been reached yet"
        ),
    }

    if include_baseline_thresholds:
        for sensor_id, meta in sensors_meta.items():
            unit = meta["unit"]
            thresholds[f"{sensor_id}_healthy_100pct_reference"] = (
                f"mean={meta['healthy_100pct_mean']} {unit}, "
                f"std={meta['healthy_100pct_std']} {unit}, "
                f"p05..p95=[{meta['healthy_100pct_p05']}, {meta['healthy_100pct_p95']}] {unit}"
            )

    documents: dict[str, str] = {}
    desc_path = DOCS_DIR / "description.md"
    doc_path = DOCS_DIR / "documentation.md"
    if desc_path.exists():
        documents["zema_description"] = desc_path.read_text(encoding="utf-8")
    if doc_path.exists():
        documents["zema_documentation"] = doc_path.read_text(encoding="utf-8")

    schemas: dict[str, Any] = {
        "zema_hydraulic_test_rig": {
            "total_cycles": 2205,
            "cycle_duration_seconds": 60,
            "physical_sensors_used": list(sensors_meta.keys()),
            "excluded_virtual_sensors": baseline.get(
                "excluded_virtual_sensors", ["CE", "CP", "SE"]
            ),
            "target_variable": baseline["target_variable"],
        }
    }

    return {
        "variables": variables,
        "thresholds": thresholds,
        "documents": documents,
        "schemas": schemas,
    }
