#!/usr/bin/env python3
"""Agent Principal: mini API de explicações (XAI + LLM).

Esta camada expõe uma API simples para a view consumir explicações
sobre predições, usando:
- mocks em JSON (enquanto a camada training não está plugada)
- GeminiAdapter para geração de texto
- ExplanationJudge para validação em loop até aprovação
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from layers.data.src import get_features_metadata

from .gemini_adapter import GeminiAdapter, GeminiSession
from .judge import ExplanationJudge


SYSTEM_PROMPT_ENV = "AGENT_SYSTEM_PROMPT_PATH"
PAYLOAD_DIR_ENV = "AGENT_PAYLOAD_DIR"
JUDGE_MIN_SCORE_ENV = "AGENT_JUDGE_MIN_SCORE"
MAX_ATTEMPTS_ENV = "AGENT_MAX_ATTEMPTS"


def _load_system_prompt() -> str:
    prompt_path = os.getenv(SYSTEM_PROMPT_ENV)
    if not prompt_path:
        raise ValueError(f"{SYSTEM_PROMPT_ENV} não configurado no ambiente.")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_domain_text() -> str:
    """Carrega texto de domínio a partir do metadata da camada de dados."""

    metadata = get_features_metadata()
    return metadata.get("domain_content", "")



def _load_payload_by_id(prediction_id: int) -> Dict[str, Any]:
    """Carrega um payload (mock ou real) a partir do diretório configurado.

    Args:
        prediction_id: Identificador numérico da predição.
    """

    payload_dir = os.getenv(PAYLOAD_DIR_ENV)
    if not payload_dir:
        raise ValueError(f"{PAYLOAD_DIR_ENV} não configurado no ambiente.")

    base_path = Path(payload_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Diretório de payloads não encontrado: {base_path}")

    for path in base_path.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("id") == prediction_id:
            return data

    raise ValueError(f"Prediction id {prediction_id} não encontrado em {base_path}.")


@dataclass
class ExplanationResult:
    """Resultado final de uma explicação aprovada."""

    explanation: str
    judge: Dict[str, Any]
    payload: Dict[str, Any]


@dataclass
class QuestionResult:
    """Resposta do agent para perguntas do usuario."""

    answer: str
    judge: Dict[str, Any]
    blocked: bool
    blocked_reason: Optional[str]
    prediction_id: Optional[int]


class ExplanationAgent:
    """Mini API de explicações para consumo pela camada view.

    Por enquanto, consome mocks de predição em JSON. Futuramente, será
    plugado diretamente na camada training.
    """

    def __init__(self, llm_adapter: GeminiAdapter | None = None) -> None:
        self.llm = llm_adapter or GeminiAdapter()
        self.judge = ExplanationJudge(self.llm)
        self.system_prompt = _load_system_prompt()
        min_score_env = os.getenv(JUDGE_MIN_SCORE_ENV)
        try:
            self.min_score = float(min_score_env)
        except (TypeError, ValueError):
            raise ValueError("AGENT_JUDGE_MIN_SCORE não configurado ou inválido no ambiente.")

        max_attempts_env = os.getenv(MAX_ATTEMPTS_ENV)
        try:
            self.max_attempts = int(max_attempts_env)
        except (TypeError, ValueError):
            raise ValueError("AGENT_MAX_ATTEMPTS não configurado ou inválido no ambiente.")

        if self.max_attempts < 1:
            raise ValueError("AGENT_MAX_ATTEMPTS deve ser >= 1.")

        self.session_id = uuid4().hex
        self.explained_prediction_ids: set[int] = set()
        self.agent_session: GeminiSession = self._start_agent_session()

    def _start_agent_session(self) -> GeminiSession:
        domain_text = _load_domain_text()
        system_instruction = f"{self.system_prompt}\n\nDOMAIN:\n{domain_text}"
        return self.llm.start_session(
            system_instruction=system_instruction,
            initial_input="Iniciar explicador.",
        )

    def _build_llm_input(self, payload: Dict[str, Any]) -> str:
        """Monta o JSON de entrada para o LLM com domínio + prompt."""

        enriched = {
            "prediction_payload": payload,
        }

        return json.dumps(enriched, ensure_ascii=False)



    def explain_from_payload(self, payload: Dict[str, Any]) -> ExplanationResult:
        """Gera e valida explicação a partir de um payload completo.

        Args:
            payload: Dicionário com predição, XAI, modelo e domínio.
            max_attempts: Número máximo de tentativas até aprovação.
        """

        last_judge: Dict[str, Any] | None = None
        explanation_text: str | None = None
        best_score = float("-inf")
        best_judge: Dict[str, Any] | None = None
        best_explanation: str | None = None

        for _ in range(self.max_attempts):
            llm_input = self._build_llm_input(payload)
            explanation_text = self.agent_session.send(llm_input)

            judge_payload = {
                **payload,
                "explanation": explanation_text,
            }
            last_judge = self.judge.evaluate_explanation(judge_payload)

            try:
                score = float(last_judge.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            if score > best_score:
                best_score = score
                best_judge = last_judge
                best_explanation = explanation_text

            if score >= self.min_score:
                break

        return ExplanationResult(
            explanation=best_explanation or explanation_text or "",
            judge=best_judge or last_judge or {},
            payload=payload,
        )

    def explain_prediction(self, prediction_id: int) -> ExplanationResult:
        """Gera explicação usando um payload de predição por id."""

        payload = _load_payload_by_id(prediction_id)
        result = self.explain_from_payload(payload)
        if isinstance(payload.get("id"), int):
            self.explained_prediction_ids.add(payload["id"])
        return result

    def ask_question(self, question: str, prediction_id: Optional[int] = None) -> QuestionResult:
        """Permite ao usuario tirar duvidas apos explicacoes.

        Args:
            question: texto da pergunta do usuario.
            prediction_id: id da previsao a qual a pergunta se refere.
                Se None, trata-se de uma pergunta geral.
                Se um id válido, requer que a previsão já tenha sido explicada.
        """

        # Bloqueio: se referencia uma predição ainda não explicada, nega resposta.
        if isinstance(prediction_id, int) and prediction_id not in self.explained_prediction_ids:
            return QuestionResult(
                answer="Preciso explicar essa previsao antes de responder perguntas sobre ela.",
                judge={},
                blocked=True,
                blocked_reason="prediction_not_explained",
                prediction_id=prediction_id,
            )

        best_score = float("-inf")
        best_judge: Dict[str, Any] | None = None
        best_answer: str | None = None

        for _ in range(self.max_attempts):
            answer = self.agent_session.send(question)
            judge_payload = {
                "question": question,
                "answer": answer,
                "prediction_id": prediction_id,
            }
            judge_result = self.judge.evaluate_explanation(judge_payload)

            try:
                score = float(judge_result.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            if score > best_score:
                best_score = score
                best_judge = judge_result
                best_answer = answer

            if score >= self.min_score:
                break

        return QuestionResult(
            answer=best_answer or "",
            judge=best_judge or {},
            blocked=False,
            blocked_reason=None,
            prediction_id=prediction_id,
        )


# package-level exports are managed by layers/agent/__init__.py

