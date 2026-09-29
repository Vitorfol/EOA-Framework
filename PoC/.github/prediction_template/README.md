# Prediction JSON template (canonical)

This folder contains the canonical prediction JSON template used by the PoC CLI and LLM prompts.

Required fields:

- `id` (int): unique identifier for the prediction.
- `model_info` (object): must include `model_name`, `model_type`, `target_variable`, `task_type`.
- `prediction` (object): REQUIRED. Must contain a concise `summary` object describing the model output (e.g., `class` and `class_probabilities` for classification, or `value` for regression).
- `features` (object): a flat mapping of feature_name -> numeric_value for the most important inputs.

Recommended optional fields:

- `metadata` (object): `timestamp`, `source`, `dataset_reference`, and `units` to help the LLM interpret values.
- `generic` (object): free-form guidance for the LLM/judge such as `llm_hint`, `interpretation_hints`, or `human_label_map`.
- `raw_payload` (object): original model input/payload for debugging.

Why `prediction` is required

The explainer and judge expect a short summary of the model output in a predictable location. Requiring `prediction.summary` ensures the LLM receives the minimal interpretable model output (label/value and optional confidences) while leaving the rest of the JSON flexible.

Usage:

1. Copy `prediction_template.json` into your `AGENT_PAYLOAD_DIR` (configured in environment) and rename it; ensure `id` is unique.
2. The CLI loads all `.json` files from `AGENT_PAYLOAD_DIR` and uses `features`, `prediction.summary`, `model_info` and `generic` when building LLM input.
