"""Google Gemini adapter for natural-language explanations.

Esta classe encapsula o cliente do Google Gemini e lê as
configurações (API key e modelo) de variáveis de ambiente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google import genai


class GeminiAdapter:
    """Adapter para Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Inicializa adapter.

        Args:
            api_key: Chave da API Gemini. Se None, usa GEMINI_API_KEY do .env.
            model:  Nome do modelo Gemini. Se None, usa GEMINI_MODEL do .env.
        """

        api_key_env = os.getenv("GEMINI_API_KEY")
        self.api_key = api_key or api_key_env
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não configurada no ambiente nem passada ao adapter.")

        self.model = model or os.getenv("GEMINI_MODEL")
        if not self.model:
            raise ValueError("GEMINI_MODEL não configurado no ambiente nem passado ao adapter.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_explanation(self, context: str | Dict[str, Any]) -> str:
        """Gera explicação em linguagem natural.

        Args:
            context: Texto ou dicionário com dados de contexto para explicação.

        Returns:
            Explicação formatada em texto.
        """

        prompt = context if isinstance(context, str) else str(context)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        return text if text else str(response)

    def create_interaction(
        self,
        input_text: str,
        system_instruction: Optional[str] = None,
        previous_interaction_id: Optional[str] = None,
        store: bool = True,
    ):
        return self.client.interactions.create(
            model=self.model,
            input=input_text,
            system_instruction=system_instruction,
            previous_interaction_id=previous_interaction_id,
            store=store,
        )

    def _extract_text(self, interaction) -> str:
        outputs = getattr(interaction, "outputs", None) or []
        for output in reversed(outputs):
            text = getattr(output, "text", None)
            if text:
                return text
        return str(interaction)

    def start_session(self, system_instruction: str, initial_input: str) -> "GeminiSession":
        interaction = self.create_interaction(
            input_text=initial_input,
            system_instruction=system_instruction,
            previous_interaction_id=None,
            store=True,
        )
        return GeminiSession(
            adapter=self,
            system_instruction=system_instruction,
            previous_interaction_id=interaction.id,
        )

    def run_stateless(self, input_text: str, system_instruction: Optional[str] = None) -> str:
        interaction = self.create_interaction(
            input_text=input_text,
            system_instruction=system_instruction,
            previous_interaction_id=None,
            store=False,
        )
        return self._extract_text(interaction)


@dataclass
class GeminiSession:
    adapter: GeminiAdapter
    system_instruction: str
    previous_interaction_id: Optional[str] = None

    def send(self, input_text: str) -> str:
        interaction = self.adapter.create_interaction(
            input_text=input_text,
            system_instruction=self.system_instruction,
            previous_interaction_id=self.previous_interaction_id,
            store=True,
        )
        self.previous_interaction_id = interaction.id
        return self.adapter._extract_text(interaction)


# package-level exports are managed by layers/agent/__init__.py
