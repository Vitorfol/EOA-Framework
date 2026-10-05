"""Comprehensive unit tests for EOA data contracts (Epic 02).

Verifies strict validation, boundary constraints, serialization/deserialization,
and polymorphic ingestion (dict vs. typed instance) for all contracts:
- PredictionArtifact (Level 1)
- XAIArtifact (Level 2)
- DomainContext (Domain Contract & Metadata)
- AgentStepTrace (Auditability)
- ExplanationResponse (Level 3 Output)
"""

import pytest
from pydantic import ValidationError

import eoa
from eoa.contracts import (
    AgentStepTrace,
    DomainContext,
    ExplanationResponse,
    PredictionArtifact,
    XAIArtifact,
)

# ==============================================================================
# 1. PredictionArtifact Tests
# ==============================================================================


def test_prediction_artifact_minimal() -> None:
    """Test creating PredictionArtifact with only mandatory fields."""
    pred = PredictionArtifact(
        target_variable="Cooler Condition",
        predicted_value="Efficiency Reduced",
    )
    assert pred.target_variable == "Cooler Condition"
    assert pred.predicted_value == "Efficiency Reduced"
    assert pred.model_name is None
    assert pred.confidence_or_probability is None
    assert pred.status_label is None
    assert pred.metadata == {}


def test_prediction_artifact_full() -> None:
    """Test creating PredictionArtifact with full domain metadata."""
    pred = PredictionArtifact(
        target_variable="Cooler Condition",
        predicted_value="Efficiency Reduced",
        model_name="xgboost_hydraulic_v1",
        confidence_or_probability=0.82,
        status_label="Efficiency Reduced (20% tier)",
        metadata={"cycle_id": 4091, "plant": "Hydroelectric Plant 01"},
    )
    assert pred.model_name == "xgboost_hydraulic_v1"
    assert pred.confidence_or_probability == 0.82
    assert pred.status_label == "Efficiency Reduced (20% tier)"
    assert pred.metadata["cycle_id"] == 4091


def test_prediction_artifact_allows_arbitrary_extra_fields() -> None:
    """Verify that arbitrary extra fields are preserved due to extra='allow'."""
    pred = PredictionArtifact.model_validate(
        {
            "target_variable": "Cooler Condition",
            "predicted_value": 0,
            "custom_metric": 42.5,
            "batch_tag": "BATCH_2026_Q1",
        }
    )
    assert pred.custom_metric == 42.5
    assert pred.batch_tag == "BATCH_2026_Q1"


def test_prediction_artifact_probability_boundaries() -> None:
    """Ensure confidence_or_probability rejects numbers outside [0.0, 1.0]."""
    # Valid boundaries
    p_zero = PredictionArtifact(
        target_variable="T", predicted_value=1, confidence_or_probability=0.0
    )
    p_one = PredictionArtifact(
        target_variable="T", predicted_value=1, confidence_or_probability=1.0
    )
    assert p_zero.confidence_or_probability == 0.0
    assert p_one.confidence_or_probability == 1.0

    # Invalid: > 1.0
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        PredictionArtifact(
            target_variable="T", predicted_value=1, confidence_or_probability=1.01
        )

    # Invalid: < 0.0
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        PredictionArtifact(
            target_variable="T", predicted_value=1, confidence_or_probability=-0.05
        )


def test_prediction_artifact_missing_mandatory_fields() -> None:
    """Verify validation error when required fields are missing."""
    with pytest.raises(ValidationError):
        PredictionArtifact.model_validate(
            {"target_variable": "Cooler"}
        )  # missing predicted_value


def test_prediction_artifact_from_input_polymorphic() -> None:
    """Verify from_input handles both dict and existing instance."""
    raw_dict = {
        "target_variable": "Pump",
        "predicted_value": 0,
        "confidence_or_probability": 0.95,
    }
    obj_from_dict = PredictionArtifact.from_input(raw_dict)
    assert isinstance(obj_from_dict, PredictionArtifact)
    assert obj_from_dict.confidence_or_probability == 0.95

    # Passing existing instance should return the same object
    same_obj = PredictionArtifact.from_input(obj_from_dict)
    assert same_obj is obj_from_dict


# ==============================================================================
# 2. XAIArtifact Tests
# ==============================================================================


def test_xai_artifact_defaults() -> None:
    """Test default values when initializing empty XAIArtifact."""
    xai = XAIArtifact()
    assert xai.method == "feature_importance"
    assert xai.feature_attributions == {}
    assert xai.baseline_values is None
    assert xai.counterfactuals is None
    assert xai.raw_artifacts == {}
    assert xai.top_features() == []


def test_xai_artifact_top_features_ranking() -> None:
    """Test ranking features by absolute weight and raw value."""
    xai = XAIArtifact(
        method="shap",
        feature_attributions={
            "PS1": -0.45,
            "EPS1": 0.35,
            "TS2": 0.20,
            "VS1": -0.10,
            "SE": 0.02,
        },
    )

    # Ranking by absolute magnitude: |PS1| = 0.45, |EPS1| = 0.35, |TS2| = 0.20
    top_3_abs = xai.top_features(n=3, absolute=True)
    assert top_3_abs == [("PS1", -0.45), ("EPS1", 0.35), ("TS2", 0.20)]

    # Ranking by raw numerical value (descending): EPS1 (0.35), TS2 (0.20)
    top_2_raw = xai.top_features(n=2, absolute=False)
    assert top_2_raw == [("EPS1", 0.35), ("TS2", 0.20)]


def test_xai_artifact_from_input_polymorphic() -> None:
    """Verify from_input handles None, dict, and existing instance."""
    assert XAIArtifact.from_input(None).feature_attributions == {}

    xai_dict = {"method": "lime", "feature_attributions": {"sensor_A": 0.8}}
    obj = XAIArtifact.from_input(xai_dict)
    assert obj.method == "lime"
    assert obj.feature_attributions["sensor_A"] == 0.8

    same_obj = XAIArtifact.from_input(obj)
    assert same_obj is obj


# ==============================================================================
# 3. DomainContext Tests
# ==============================================================================


def test_domain_context_generic_attributes() -> None:
    """Verify that DomainContext supports generic variables and thresholds."""
    ctx = DomainContext(
        variables={"FICO": "Credit score metric", "PS1": "Pump pressure"},
        thresholds={"min_score": 600, "PS1_max": 200.0},
        documents={"policy": "Risk assessment guidelines"},
        hierarchy={"subsystem": "Hydraulics"},
        schemas={"sensor_series": {"sampling_rate": "100Hz"}},
    )
    assert ctx.variables["FICO"] == "Credit score metric"
    assert ctx.variables["PS1"] == "Pump pressure"
    assert ctx.thresholds["min_score"] == 600
    assert ctx.documents["policy"] == "Risk assessment guidelines"
    assert ctx.hierarchy == {"subsystem": "Hydraulics"}
    assert ctx.schemas["sensor_series"] == {"sampling_rate": "100Hz"}

    ctx.add_variable("DTI", "Debt to income ratio")
    assert ctx.variables["DTI"] == "Debt to income ratio"


def test_domain_context_iterative_enrichment() -> None:
    """Verify that DomainContext can be enriched incrementally."""
    ctx = DomainContext()
    assert ctx.variables == {}
    assert ctx.thresholds == {}
    assert ctx.documents == {}
    assert ctx.schemas == {}

    ctx.add_variable("PS1", "Main pump axial pressure")
    ctx.add_variable("EPS1", "Motor electrical power consumption")
    assert len(ctx.variables) == 2
    assert ctx.variables["PS1"] == "Main pump axial pressure"

    ctx.add_document("cooling_manual", "Thermal dissipation guidelines")
    assert "cooling_manual" in ctx.documents

    ctx.add_threshold("PS1_max", 180.0)
    assert ctx.thresholds["PS1_max"] == 180.0

    ctx.add_schema("telemetry", {"frequency": "100Hz", "type": "float"})
    assert ctx.schemas["telemetry"]["frequency"] == "100Hz"


def test_domain_context_merge() -> None:
    """Verify merging a static global context with incident-specific metadata."""
    global_ctx = DomainContext(
        variables={"PS1": "Pressure 1", "TS1": "Temperature 1"},
        thresholds={"PS1_max": 200.0},
        documents={"base_doc": "Base manual"},
        schemas={"base_table": {"columns": ["time", "val"]}},
    )

    incident_ctx = {
        "variables": {
            "TS1": "Temperature 1 (Recalibrated)",
            "VS1": "Vibration 1",
        },
        "documents": {"alert_log": "Incident report from shift 2"},
        "schemas": {"incident_table": {"columns": ["error_code"]}},
    }

    merged = global_ctx.merge(incident_ctx)

    # Variables: TS1 was overwritten, PS1 kept, VS1 added
    assert merged.variables["PS1"] == "Pressure 1"
    assert merged.variables["TS1"] == "Temperature 1 (Recalibrated)"
    assert merged.variables["VS1"] == "Vibration 1"

    # Thresholds preserved from global
    assert merged.thresholds["PS1_max"] == 200.0

    # Documents merged
    assert "base_doc" in merged.documents
    assert "alert_log" in merged.documents

    # Schemas merged
    assert "base_table" in merged.schemas
    assert "incident_table" in merged.schemas

    # Original global_ctx was not mutated
    assert "VS1" not in global_ctx.variables
    assert "incident_table" not in global_ctx.schemas


def test_domain_context_from_input_polymorphic() -> None:
    """Verify from_input handles None, dict, and instance."""
    assert DomainContext.from_input(None).variables == {}

    ctx = DomainContext.from_input({"variables": {"S1": "Flow"}})
    assert ctx.variables["S1"] == "Flow"


# ==============================================================================
# 4. AgentStepTrace & ExplanationResponse Tests
# ==============================================================================


def test_agent_step_trace_valid() -> None:
    """Test valid instantiation and duration constraints for AgentStepTrace."""
    trace = AgentStepTrace(
        agent_name="ArtifactsAgent",
        execution_time_ms=15.4,
        summary="Extracted top SHAP features",
        status="success",
        details={"features_found": 3},
    )
    assert trace.agent_name == "ArtifactsAgent"
    assert trace.execution_time_ms == 15.4
    assert trace.details["features_found"] == 3

    # Negative execution time should fail
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AgentStepTrace(agent_name="EA", execution_time_ms=-1.0, summary="Error")


def test_explanation_response_validation_score_constraints() -> None:
    """Test validation_score rounding and configurable score boundaries."""
    res = ExplanationResponse(
        narrative="Risk is driven by axial vibration (PS1).",
        validation_score=9.5678,
        validation_passed=True,
    )
    # Score must be rounded to two decimal places
    assert res.validation_score == 9.57
    assert res.attempts_used == 1
    assert not res.fallback_used
    assert res.execution_trace == []

    # Score > 10.0 must raise ValidationError on default scale
    with pytest.raises(ValidationError, match="between 0.0 and 10.0"):
        ExplanationResponse(
            narrative="Test", validation_score=10.5, validation_passed=True
        )

    # Score < 0.0 must raise ValidationError on default scale
    with pytest.raises(ValidationError, match="between 0.0 and 10.0"):
        ExplanationResponse(
            narrative="Test", validation_score=-0.1, validation_passed=False
        )

    # Custom scale (e.g. percentage 0-100) can be configured via ClassVar
    ExplanationResponse.max_score = 100.0
    try:
        res_percent = ExplanationResponse(
            narrative="Custom percentage validator",
            validation_score=95.556,
            validation_passed=True,
        )
        assert res_percent.validation_score == 95.56
    finally:
        ExplanationResponse.max_score = 10.0


def test_explanation_response_without_validation() -> None:
    """Ensure response can be created without validation (fast mode / skipped)."""
    res = ExplanationResponse(narrative="Direct unvalidated explanation.")
    assert res.narrative == "Direct unvalidated explanation."
    assert res.validation_score is None
    assert res.validation_passed is None
    assert res.validation_critique is None


def test_explanation_response_json_serialization_roundtrip() -> None:
    """Verify full JSON serialization and deserialization (Level 3 trace)."""
    step1 = AgentStepTrace(
        agent_name="CMA",
        execution_time_ms=10.2,
        summary="Resolved sensor dictionary",
    )
    step2 = AgentStepTrace(
        agent_name="VA",
        execution_time_ms=180.5,
        summary="Technical fidelity scored 9.8/10",
        details={"critique": "No hallucination detected"},
    )

    response = ExplanationResponse(
        narrative="Cooling unit efficiency reduced due to thermal loss.",
        validation_score=9.8,
        validation_passed=True,
        validation_critique=None,
        attempts_used=2,
        fallback_used=False,
        execution_trace=[step1, step2],
    )

    # Convert to JSON string
    json_str = response.model_dump_json()
    assert isinstance(json_str, str)
    assert "Cooling unit efficiency reduced" in json_str
    assert "9.8" in json_str

    # Parse back from JSON
    restored = ExplanationResponse.model_validate_json(json_str)
    assert restored.narrative == response.narrative
    assert restored.validation_score == 9.8
    assert len(restored.execution_trace) == 2
    assert restored.execution_trace[0].agent_name == "CMA"
    assert (
        restored.execution_trace[1].details["critique"] == "No hallucination detected"
    )


def test_contracts_resilient_to_none_dicts() -> None:
    """Ensure None values for dictionary fields are coerced to empty dicts."""
    # PredictionArtifact metadata=None
    pred = PredictionArtifact(
        target_variable="Pump", predicted_value=1, metadata=None
    )
    assert pred.metadata == {}

    # XAIArtifact feature_attributions=None, raw_artifacts=None
    xai = XAIArtifact(feature_attributions=None, raw_artifacts=None)
    assert xai.feature_attributions == {}
    assert xai.raw_artifacts == {}

    # DomainContext with explicit None dictionaries
    ctx = DomainContext(
        variables=None,
        thresholds=None,
        documents=None,
        hierarchy=None,
        schemas=None,
    )
    assert ctx.variables == {}
    assert ctx.thresholds == {}
    assert ctx.documents == {}
    assert ctx.hierarchy == {}
    assert ctx.schemas == {}

    # AgentStepTrace with details=None
    trace = AgentStepTrace(
        agent_name="CMA", execution_time_ms=5.0, summary="Done", details=None
    )
    assert trace.details == {}

    # ExplanationResponse with execution_trace=None
    res = ExplanationResponse(narrative="Test", execution_trace=None)
    assert res.execution_trace == []


# ==============================================================================
# 5. Root Package Import Verification
# ==============================================================================


def test_package_exports_contracts() -> None:
    """Ensure all core contracts are directly exported at root eoa package."""
    assert hasattr(eoa, "PredictionArtifact")
    assert hasattr(eoa, "XAIArtifact")
    assert hasattr(eoa, "DomainContext")
    assert hasattr(eoa, "AgentStepTrace")
    assert hasattr(eoa, "ExplanationResponse")
    assert hasattr(eoa, "LLM")
    assert not hasattr(eoa, "LLMProvider")
