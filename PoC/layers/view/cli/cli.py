#!/usr/bin/env python3
"""CLI da camada view.

Interface de terminal para explorar dataframes, previsoes mockadas
e interagir com o agente de explicacao.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers.data.src import get_original_dataframe
from layers.agent import ExplanationAgent

from renderer import render_prediction, render_explanation, render_dataframe_summary


load_dotenv()


def _find_project_root(max_levels: int = 6) -> Path:
    p = Path(__file__).resolve()
    for i in range(max_levels):
        cand = p.parents[i]
        # heuristics: project root likely contains requirements.txt or readme.md
        if (cand / "requirements.txt").exists() or (cand / "readme.md").exists() or (cand / "readme.MD").exists():
            return cand
    # fallback to parent[4] (two levels up from layers)
    try:
        return p.parents[4]
    except Exception:
        return p.parents[len(p.parents) - 1]


def _resolve_env_paths() -> None:
    root = _find_project_root()
    # canonicalize common env paths so running from subfolders still works
    for key in ("AGENT_PAYLOAD_DIR", "AGENT_SYSTEM_PROMPT_PATH", "AGENT_JUDGE_PROMPT_PATH"):
        val = os.getenv(key)
        if not val:
            continue
        pv = Path(val)
        if not pv.is_absolute():
            resolved = (root / pv).resolve()
            os.environ[key] = str(resolved)


_resolve_env_paths()


def _prompt(text: str) -> str:
    return input(text).strip()


def _prompt_int(text: str, allow_empty: bool = False) -> Optional[int]:
    raw = _prompt(text)
    if allow_empty and raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_predictions_dir() -> Path:
    value = os.getenv("AGENT_PAYLOAD_DIR")
    if not value:
        raise ValueError("AGENT_PAYLOAD_DIR nao configurado no ambiente.")
    return Path(value)


def load_predictions() -> List[Dict[str, Any]]:
    base = _get_predictions_dir()
    if not base.exists():
        raise FileNotFoundError(f"Diretorio de predicoes nao encontrado: {base}")

    predictions: List[Dict[str, Any]] = []
    for path in sorted(base.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                predictions.append(data)
            except json.JSONDecodeError:
                continue

    return sorted(predictions, key=lambda x: x.get("id", 0))


def find_prediction(predictions: List[Dict[str, Any]], pred_id: int) -> Optional[Dict[str, Any]]:
    for pred in predictions:
        if pred.get("id") == pred_id:
            return pred
    return None


def list_predictions(predictions: List[Dict[str, Any]]) -> None:
    if not predictions:
        print("Nenhuma previsao encontrada.")
        return

    print("\nPredicoes disponiveis:")
    for pred in predictions:
        pid = pred.get("id")
        model = pred.get("model_info", {})
        model_name = model.get("model_name", "-")
        target = model.get("target_variable", "-")
        task = model.get("task_type", "-")
        # derive a compact human-friendly predicted value
        predicted_display = "-"
        prediction_obj = pred.get("prediction", {})
        if isinstance(prediction_obj, dict):
            summary = prediction_obj.get("summary") if "summary" in prediction_obj else prediction_obj
            if isinstance(summary, dict):
                if "class" in summary:
                    cls = summary.get("class")
                    label = summary.get("class_label") or summary.get("classLabel")
                    predicted_display = f"class={cls}" + (f" ({label})" if label else "")
                elif "value" in summary:
                    predicted_display = f"value={summary.get('value')}"
        print(f"  - id={pid} | model={model_name} | target={target} | task={task} | predicted={predicted_display}")


def _get_dataframe_sources() -> List[Tuple[str, str]]:
    return [
        ("normalized_long", "Normalizado (long, series no tempo)"),
        ("normalized_wide", "Normalizado (wide, por ciclo)"),
        ("aggregated", "Agregado (features estatisticas)"),
    ]


def _load_dataframe_by_key(key: str):
    if key == "normalized_long":
        return get_original_dataframe(variant="normalized", format="long")
    if key == "normalized_wide":
        return get_original_dataframe(variant="normalized", format="wide")
    if key == "aggregated":
        return get_original_dataframe(variant="aggregated")
    raise ValueError("Chave de dataframe invalida.")


def view_dataframes(cache: Dict[str, Any]) -> None:
    sources = _get_dataframe_sources()
    print("\nDataframes disponiveis:")
    for idx, (_, label) in enumerate(sources, start=1):
        print(f"  [{idx}] {label}")

    choice = _prompt_int("Escolha um dataframe: ")
    if choice is None or choice < 1 or choice > len(sources):
        print("Opcao invalida.")
        return

    key = sources[choice - 1][0]
    if key not in cache:
        cache[key] = _load_dataframe_by_key(key)
    df = cache[key]
    print(render_dataframe_summary(df))


def view_predictions(predictions: List[Dict[str, Any]]) -> None:
    list_predictions(predictions)
    pred_id = _prompt_int("Digite o id para ver detalhes (ou enter para voltar): ", allow_empty=True)
    if pred_id is None:
        return
    pred = find_prediction(predictions, pred_id)
    if not pred:
        print("Id nao encontrado.")
        return
    print(render_prediction(pred, detail_level="full"))


def llm_interaction(agent: ExplanationAgent, predictions: List[Dict[str, Any]]) -> None:
    last_explained_id: Optional[int] = None
    while True:
        print("\nInteracao com LLM:")
        print("  [1] Explain prediction")
        print("  [2] Ask question")
        print("  [0] Voltar")

        choice = _prompt("Escolha: ")
        if choice == "0":
            return
        if choice == "1":
            pred_id = _prompt_int("Id da predicao: ")
            if pred_id is None:
                print("Id invalido.")
                continue
            pred = find_prediction(predictions, pred_id)
            if not pred:
                print("Id nao encontrado.")
                continue
            try:
                result = agent.explain_prediction(pred_id)
            except Exception as exc:
                print(f"Falha ao gerar explicacao: {exc}")
                continue
            last_explained_id = pred_id
            print(render_explanation(result))
            continue
        if choice == "2":
            question = _prompt("Pergunta: ")
            if question == "":
                print("Pergunta vazia.")
                continue
            pred_id_input = _prompt("Id da predicao (enter para geral): ")
            pred_id = None
            if pred_id_input != "":
                try:
                    pred_id = int(pred_id_input)
                except ValueError:
                    print("Id invalido, usando pergunta geral.")
                    pred_id = None
            elif last_explained_id is not None:
                # se o usuario nao passou id, mantem pergunta geral
                pred_id = None
            try:
                result = agent.ask_question(question, prediction_id=pred_id)
            except Exception as exc:
                print(f"Falha ao enviar pergunta: {exc}")
                continue
            print(render_explanation(result))
            continue
        print("Opcao invalida.")


def demo_flow(agent: ExplanationAgent, predictions: List[Dict[str, Any]]) -> None:
    if not predictions:
        print("Nao ha predicoes para demonstrar.")
        return
    pred = predictions[0]
    pred_id = pred.get("id")
    if not isinstance(pred_id, int):
        print("Predicao invalida para demo.")
        return

    print("\n=== Demo ===")
    print(render_prediction(pred, detail_level="standard"))
    print("\nGerando explicacao...")
    try:
        result = agent.explain_prediction(pred_id)
    except Exception as exc:
        print(f"Falha ao gerar explicacao: {exc}")
        return
    print(render_explanation(result))

    print("\nEnviando pergunta de follow-up...")
    try:
        answer = agent.ask_question(
            "Explique os principais fatores desta previsao.",
            prediction_id=pred_id,
        )
    except Exception as exc:
        print(f"Falha ao enviar pergunta: {exc}")
        return
    print(render_explanation(answer))


def main() -> None:
    df_cache: Dict[str, Any] = {}
    try:
        predictions = load_predictions()
    except Exception as exc:
        print(f"Falha ao carregar predicoes: {exc}")
        predictions = []

    try:
        agent = ExplanationAgent()
    except Exception as exc:
        agent = None
        print(f"Falha ao inicializar ExplanationAgent: {exc}")

    while True:
        print("\n=== View CLI ===")
        print("  [1] Ver dataframe")
        print("  [2] Ver predicao")
        print("  [3] Interagir com LLM")
        print("  [4] Rodar demo")
        print("  [0] Sair")
        choice = _prompt("Escolha: ")

        if choice == "0":
            print("Saindo...")
            return
        if choice == "1":
            view_dataframes(df_cache)
            continue
        if choice == "2":
            view_predictions(predictions)
            continue
        if choice == "3":
            if agent is None:
                print("Agent indisponivel. Verifique variaveis de ambiente.")
                continue
            llm_interaction(agent, predictions)
            continue
        if choice == "4":
            if agent is None:
                print("Agent indisponivel. Verifique variaveis de ambiente.")
                continue
            demo_flow(agent, predictions)
            continue
        print("Opcao invalida.")


if __name__ == "__main__":
    main()
