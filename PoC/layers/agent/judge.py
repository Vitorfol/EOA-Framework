"""LLM as a Judge - Avaliador de Qualidade de Explicações.

Esta classe encapsula o fluxo de avaliação de explicações usando um
LLM (Gemini) com um prompt específico de judge.

Ela recebe o contexto completo da explicação (domínio, modelo,
predição, XAI, instância e texto da explicação) e retorna um dicionário
estruturado com score geral, feedback e notas por critério.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .gemini_adapter import GeminiAdapter, GeminiSession


JUDGE_PROMPT_ENV = "AGENT_JUDGE_PROMPT_PATH"


def _load_judge_prompt() -> str:
    """Carrega o prompt de judge do arquivo .txt."""

    prompt_path = os.getenv(JUDGE_PROMPT_ENV)
    if not prompt_path:
        raise ValueError(f"{JUDGE_PROMPT_ENV} não configurado no ambiente.")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class ExplanationJudge:
    """Avalia qualidade de explicações usando LLM as a Judge."""

    def __init__(self, llm_adapter: GeminiAdapter | None = None) -> None:
        """Inicializa o judge.

        Args:
            llm_adapter: Instância de GeminiAdapter. Se None, será criada
                uma instância padrão usando variáveis de ambiente.
        """

        self.llm = llm_adapter or GeminiAdapter()
        self.prompt_template = _load_judge_prompt()
        self.session = self.llm.start_session(
            system_instruction=self.prompt_template,
            initial_input="Iniciar avaliacao.",
        )

    def evaluate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia qualidade de uma explicação.

        Args:
            payload: Dicionário com o contexto completo da explicação,
                contendo pelo menos a chave "explanation".

        Returns:
            Dicionário com score geral, feedback e critérios.
        """

        raw_response = self.session.send(json.dumps(payload, ensure_ascii=False))

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback defensivo: empacotar resposta bruta
            result = {
                "score": 0.0,
                "feedback": "Falha ao interpretar JSON do judge. Resposta bruta retornada.",
                "criteria": {
                    "clarity": 0.0,
                    "technical_accuracy": 0.0,
                    "completeness": 0.0,
                    "coherence": 0.0,
                },
                "raw_response": raw_response,
            }

        return result


# package-level exports are managed by layers/agent/__init__.py

