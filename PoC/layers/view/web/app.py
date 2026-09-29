from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
os.sys.path.insert(0, str(ROOT))

from layers.agent import ExplanationAgent
from layers.data.src import get_original_dataframe


def get_predictions_dir() -> Path:
    value = os.getenv("AGENT_PAYLOAD_DIR")
    if not value:
        raise ValueError("AGENT_PAYLOAD_DIR nao configurado no ambiente.")
    return Path(value)


def load_predictions() -> List[Dict[str, Any]]:
    base = get_predictions_dir()
    preds: List[Dict[str, Any]] = []
    if not base.exists():
        return preds
    for p in sorted(base.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                preds.append(json.load(f))
        except Exception:
            continue
    return sorted(preds, key=lambda x: x.get("id", 0))


def sanitize_pred(pred: Dict[str, Any]) -> Dict[str, Any]:
    pred = dict(pred)
    pred.setdefault("model_info", {})
    pred.setdefault("prediction", {})
    pred.setdefault("features", {})
    pred.setdefault("metadata", {})
    pred.setdefault("generic", {})
    for k in ("model_info", "prediction", "features", "metadata", "generic"):
        v = pred.get(k)
        if v is None:
            pred[k] = {}
        elif not isinstance(v, dict):
            try:
                pred[k] = dict(v)
            except Exception:
                pred[k] = {"value": v}
    return pred


def compact_pred_display(pred: Dict[str, Any]) -> str:
    prediction_obj = pred.get("prediction", {})
    if isinstance(prediction_obj, dict):
        summary = prediction_obj.get("summary") if "summary" in prediction_obj else prediction_obj
        if isinstance(summary, dict):
            if "class" in summary:
                label = summary.get("class_label")
                return f"class={summary.get('class')}" + (f" ({label})" if label else "")
            if "value" in summary:
                return f"value={summary.get('value')}"
    return "-"


app = Flask(__name__, template_folder="templates", static_folder="static")

# Single agent instance to mirror CLI behavior
try:
    GLOBAL_AGENT: Optional[ExplanationAgent] = ExplanationAgent()
    AGENT_INIT_ERROR: Optional[str] = None
except Exception as e:
    GLOBAL_AGENT = None
    AGENT_INIT_ERROR = str(e)


@app.route("/")
def index():
    preds = load_predictions()
    sanitized = []
    for p in preds:
        sp = sanitize_pred(p)
        sp["predicted_display"] = compact_pred_display(sp)
        sanitized.append(sp)
    return render_template("index.html", predictions=sanitized)


@app.route("/prediction/<int:pid>")
def prediction_view(pid: int):
    preds = load_predictions()
    pred = next((p for p in preds if p.get("id") == pid), None)
    if not pred:
        return redirect(url_for("index"))
    pred = sanitize_pred(pred)
    return render_template("prediction.html", pred=pred)


# In-memory conversation store: pid -> list[{role, text}]
conversations: Dict[int, List[Dict[str, str]]] = {}


@app.route("/api/conversation/<int:pid>", methods=["GET"])
def api_get_conversation(pid: int):
    conv = conversations.get(pid, [])
    return jsonify({"conversation": conv})


@app.route("/api/conversation/<int:pid>", methods=["POST"])
def api_post_conversation(pid: int):
    data = request.json or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    conv = conversations.setdefault(pid, [])
    conv.append({"role": "user", "text": question})
    if GLOBAL_AGENT is None:
        return jsonify({"error": f"Agent not available: {AGENT_INIT_ERROR}"}), 500
    try:
        answer = GLOBAL_AGENT.ask_question(question, prediction_id=pid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    assistant_text = None
    try:
        assistant_text = getattr(answer, "answer", None) or str(answer)
    except Exception:
        assistant_text = str(answer)
    if not (len(conv) and conv[-1].get("role") == "assistant" and conv[-1].get("text") == assistant_text):
        conv.append({"role": "assistant", "text": assistant_text})
    return jsonify({"answer": assistant_text, "conversation": conv})


@app.route("/dataframes")
def dataframes_view():
    sources = [
        ("normalized_long", "Normalizado (long, series no tempo)"),
        ("normalized_wide", "Normalizado (wide, por ciclo)"),
        ("aggregated", "Agregado (features estatisticas)"),
    ]
    previews = []
    for key, label in sources:
        try:
            if key == "normalized_long":
                df = get_original_dataframe(variant="normalized", format="long")
            elif key == "normalized_wide":
                df = get_original_dataframe(variant="normalized", format="wide")
            else:
                df = get_original_dataframe(variant="aggregated")
            preview = {
                "key": key,
                "label": label,
                "shape": df.shape,
                "head": df.head(5).to_dict(orient="records"),
            }
        except Exception as exc:
            preview = {"key": key, "label": label, "error": str(exc)}
        previews.append(preview)
    return render_template("dataframes.html", previews=previews)


@app.route("/api/explain/<int:pid>", methods=["POST"])
def api_explain(pid: int):
    if GLOBAL_AGENT is None:
        return jsonify({"error": f"Agent not available: {AGENT_INIT_ERROR}"}), 500
    try:
        result = GLOBAL_AGENT.explain_prediction(pid)
        try:
            GLOBAL_AGENT.explained_prediction_ids.add(pid)
        except Exception:
            pass
        conv = conversations.setdefault(pid, [])
        conv_msg = {"role": "assistant", "text": result.explanation}
        if not (len(conv) and conv[-1].get("role") == "assistant" and conv[-1].get("text") == conv_msg["text"]):
            conv.append(conv_msg)
        out = {"explanation": result.explanation, "judge": result.judge, "payload": result.payload}
        return jsonify({"result": out})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ask/<int:pid>", methods=["POST"])
def api_ask(pid: int):
    data = request.json or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    if GLOBAL_AGENT is None:
        return jsonify({"error": f"Agent not available: {AGENT_INIT_ERROR}"}), 500
    try:
        qr = GLOBAL_AGENT.ask_question(question, prediction_id=pid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    answer_text = getattr(qr, "answer", None) or str(qr)
    # append to conversation
    conv = conversations.setdefault(pid, [])
    if not (len(conv) and conv[-1].get("role") == "assistant" and conv[-1].get("text") == answer_text):
        conv.append({"role": "assistant", "text": answer_text})
    out = {"answer": answer_text, "judge": getattr(qr, "judge", {}), "blocked": getattr(qr, "blocked", False)}
    return jsonify({"result": out})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
