"""Package `layers.agent` — reexportar API pública da camada agent.

Este arquivo centraliza as exportações para facilitar imports como:

    from layers.agent import ExplanationAgent

Em cada módulo mantemos `__all__` localmente; o package-level
`__init__` define a API pública consumida por outras camadas.
"""

from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env quando o package é importado.
load_dotenv()

# Validação básica das variáveis essenciais da camada agent.
_required = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "AGENT_SYSTEM_PROMPT_PATH",
    "AGENT_JUDGE_PROMPT_PATH",
    "AGENT_PAYLOAD_DIR",
    "AGENT_JUDGE_MIN_SCORE",
    "AGENT_MAX_ATTEMPTS",
]
_missing = [k for k in _required if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"Faltando variáveis de ambiente para layers.agent: {_missing}")

from .agent import ExplanationAgent, ExplanationResult, QuestionResult
from .judge import ExplanationJudge
from .gemini_adapter import GeminiAdapter

__all__ = [
    "ExplanationAgent",
    "ExplanationResult",
    "QuestionResult",
    "ExplanationJudge",
    "GeminiAdapter",
]
