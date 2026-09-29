"""Renderer: formatacao de outputs para terminal (console)."""

import json
from typing import Dict, Any, List


def render_dataframe_summary(df, head_rows: int = 5) -> str:
    lines = []
    lines.append("\n=== DataFrame ===")
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Colunas: {len(df.columns)}")
    lines.append("\nAmostra (head):")
    lines.append(df.head(head_rows).to_string(index=False))
    return "\n".join(lines)


def _format_prediction_value(prediction: Dict[str, Any]) -> str:
    pred = prediction.get("prediction", {}) if isinstance(prediction, dict) else {}
    if "predicted_class" in pred:
        return f"class={pred.get('predicted_class')}"
    if "predicted_value" in pred:
        return f"value={pred.get('predicted_value')}"
    return "valor_indisponivel"


def render_prediction(prediction: Dict[str, Any], detail_level: str = "standard") -> str:
    model = prediction.get("model_info", {}) if isinstance(prediction, dict) else {}
    pred_value = _format_prediction_value(prediction)
    lines = []
    lines.append("\n=== Predicao ===")
    lines.append(f"Id: {prediction.get('id', '-')}")
    lines.append(f"Model: {model.get('model_name', '-')}")
    lines.append(f"Tipo: {model.get('model_type', '-')}")
    lines.append(f"Target: {model.get('target_variable', '-')}")
    lines.append(f"Task: {model.get('task_type', '-')}")
    lines.append(f"Valor: {pred_value}")

    if detail_level == "full":
        lines.append("\nDetalhes:")
        lines.append(json_dumps_safe(prediction))

    return "\n".join(lines)


def render_explanation(result) -> str:
    lines = []
    lines.append("\n=== Explicacao ===")
    if hasattr(result, "payload"):
        payload = getattr(result, "payload", {})
        lines.append(f"Prediction id: {payload.get('id', '-')}")
        model = payload.get("model_info", {})
        lines.append(f"Model: {model.get('model_name', '-')}")
        lines.append(f"Target: {model.get('target_variable', '-')}")
    if hasattr(result, "judge"):
        judge = getattr(result, "judge", {}) or {}
        score = judge.get("score", "-")
        lines.append(f"Judge score: {score}")
    if hasattr(result, "explanation"):
        lines.append("\nTexto:")
        lines.append(getattr(result, "explanation", ""))
    if hasattr(result, "answer"):
        lines.append("\nResposta:")
        lines.append(getattr(result, "answer", ""))
    return "\n".join(lines)


def json_dumps_safe(obj: Dict[str, Any]) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
