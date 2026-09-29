# Camada 3: Agent (Explicabilidade)

## Objetivo
Gerar explicações em linguagem natural sobre predições de ML usando técnicas de XAI e LLM.

## Responsabilidades
- **Consumir** SHAP values gerados pela layer training
- Integrar com LLM (Gemini) para gerar narrativas em linguagem natural
- Traduzir outputs técnicos (SHAP) em explicações claras
- **Avaliar qualidade** das explicações via LLM as a Judge
- Manter agente em modo Read-Only (sem atuação)

## Componentes
- `gemini_adapter.py` - Integração com Gemini API (geração de narrativas)
- `agent.py` - Orchestrator principal (coordena geração de explicações)
- `judge.py` - **LLM as a Judge** (avalia qualidade das explicações)
- `prompts/` - Prompts do sistema
  - `system_prompt.md` - Prompt para geração de explicações
  - `judge_prompt.md` - Prompt para avaliação de qualidade

## Fluxo
1. **Carrega** SHAP values de `training/results/`
2. **Envia** para Gemini com `system_prompt.md` → gera explicação
3. **Avalia** explicação com Judge (LLM as a Judge) → score + feedback
4. **Retorna** explicação (+ opcionalmente score de qualidade)

## Uso rápido

A camada `layers.agent` expõe as principais classes:

- `ExplanationAgent` — orquestra geração de explicações e Q&A.
- `ExplanationJudge` — avalia qualidade de explicações usando LLM.
- `GeminiAdapter` — adapter para Google Gemini (padrão). Para testes, passe um adapter falso ao construtor de `ExplanationAgent`.

Exemplo mínimo (PoC):

```python
from layers.agent.agent import ExplanationAgent
from layers.agent.gemini_adapter import GeminiAdapter

# cria agent com adapter real (requer variáveis de ambiente)
agent = ExplanationAgent()

# gerar explicação para predição id=1 (usa AGENT_PAYLOAD_DIR)
result = agent.explain_prediction(1)
print(result.explanation)

# fazer pergunta de follow-up (após explicar a predição)
q = agent.ask_question("Por que esse atributo influenciou a predição?", prediction_id=1)
print(q.answer)
```

## Variáveis de ambiente necessárias

Configure um arquivo `.env` na raiz do projeto com as seguintes chaves (exemplo):

- `GEMINI_API_KEY` — chave da API Gemini (somente para execuções reais).
- `GEMINI_MODEL` — nome do modelo Gemini.
- `AGENT_SYSTEM_PROMPT_PATH` — caminho para `layers/agent/prompts/system_prompt.txt`.
- `AGENT_JUDGE_PROMPT_PATH` — caminho para `layers/agent/prompts/judge_prompt.txt`.
- `AGENT_PAYLOAD_DIR` — diretório contendo mocks de predições (JSON) — em PoC usamos `layers/agent/mock`.
- `AGENT_JUDGE_MIN_SCORE` — score mínimo (float) exigido pelo judge para aceitar uma explicação/resposta.
- `AGENT_MAX_ATTEMPTS` — número máximo de tentativas ao gerar/avaliar (int).

## Testes

A camada inclui uma suíte pytest em `layers/agent/tests` que não depende da SDK real.
Para executar os testes:

```bash
pip install -r requirements.txt
pytest -q layers/agent/tests
```

Os testes usam um adapter falso que emula sessões do LLM, permitindo validação determinística das
funções `explain_prediction`, `ask_question` e `ExplanationJudge.evaluate_explanation`.

Se você estiver usando um ambiente virtual (venv) na raiz do projeto, ative-o antes de instalar
dependências e rodar os testes. Exemplos:

```bash
# venv criado em .venv
source .venv/bin/activate
# ou, se em `venv`
source venv/bin/activate

# então:
pip install -r requirements.txt
pytest -q layers/agent/tests
```

Se não houver venv local, você pode executar com o Python do sistema:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q layers/agent/tests
```

## Observações

- O package-level `layers.agent` valida presença de variáveis de ambiente ao ser importado —
  os testes definem valores falsos antes da importação para evitar falhas.
- Para PoC, a resolução de qual `prediction_id` uma pergunta refere-se é responsabilidade da camada
  de view; `ask_question` aceita um `prediction_id` explícito e bloqueia queries sobre predições
  que ainda não tenham sido explicadas.

