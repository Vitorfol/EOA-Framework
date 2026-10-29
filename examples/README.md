# EOA Framework — Examples, Notebooks & Public API Reference

Este diretório contém os notebooks interativos de demonstração e o **Guia Completo de Referência da API Pública** do framework **EOA (Explanation-Oriented Agents)** v0.1.0.

---

## 📂 1. Notebooks Interativos (`examples/`)

Separamos os notebooks em dois fluxos claros para facilitar tanto os testes rápidos de desenvolvimento (sem gastar tokens) quanto a avaliação científica real:

| Arquivo | Propósito | Requisitos |
| :--- | :--- | :--- |
| **[`01_walkthrough_mock.ipynb`](./01_walkthrough_mock.ipynb)** | **Fluxo Offline / Determinístico**: Percorre todos os cenários do EOA (Plug & Play, customização de agentes, Multi-Turn SQLite, modo autônomo com Tools e Fail-Safe "Unplugged") usando dados simulados e `MockLLM`. | Roda instantaneamente sem chave de API e com **zero custo de tokens**. |
| **[`02_walkthrough_live_gemini.ipynb`](./02_walkthrough_live_gemini.ipynb)** | **Fluxo Real com Google Gemini**: Executa a explicação multi-agente real, auditoria LLM-as-a-Judge (`ValidatorAgent`), inspeção de latência (`execution_trace`) e conversa multi-turn no SQLite. | Requer `GEMINI_API_KEY` (lida do `.env` na primeira célula ou via ambiente do SO). |

> **Dica de Instalação**: Sempre execute comandos `pip install -e .` ou `pytest` a partir da **raiz do repositório** (`/home/vitor/Desktop/Dev/IA/academia`), onde está localizado o arquivo `pyproject.toml`.

---

## 🛠️ 2. Referência Completa da API Pública (`import eoa`)

Abaixo estão **todas** as classes, métodos e propriedades públicas disponíveis para o desenvolvedor, organizadas por camada arquitetural.

### A. Fachada Principal e Inicialização (`eoa.layer`)

| Método / Função | Assinatura e Papel Arquitetural |
| :--- | :--- |
| **`ExplanationLayer.default_layer(...)`** | `default_layer(llm=None, *, provider=None, api_key=None, model=None, session_store=None, threshold=7.0, max_attempts=3, **llm_kwargs) -> ExplanationLayer`<br>Inicializa e retorna uma `ExplanationLayer` já populada com os 4 agentes canônicos (`context_manager`, `artifacts`, `explainer`, `validator`) em 1 linha. Lê `GEMINI_API_KEY` das variáveis de ambiente e suporta os 3 modos de LLM (Gemini, BYO Client ou Callable). |
| **`eoa.ExplanationLayer(...)`** | `ExplanationLayer(llm=None, *, provider=None, api_key=None, model=None, session_store=None, max_attempts=3, auto_create_agents=False, **llm_kwargs)`<br>Construtor principal da camada de explicabilidade (*Builder Pattern*). Permite montar topologias customizadas de agentes do zero, chamar `.create_default_agents(threshold=...)` para customização prévia, ou deixar que `.explain()` crie os defaults sob demanda. |
| **`layer.explain(...)`** | `explain(user_query="...", *, prediction=None, xai=None, context=None, query_params=None, incident_id=None, session_id=None) -> ExplanationResponse`<br>**O método único de execução síncrona**. Suporta Modo Direto/Push (`prediction`, `xai`, `context`), Modo Autônomo/Pull (`incident_id`, `query_params` ou extração autônoma de parâmetros via LLM reasoning), Modo Zero-ML (apenas `user_query` + `context`) e Multi-Turn (`session_id` com cache automático de artefatos entre turnos). |
| **`await layer.aexplain(...)`** | `async aexplain(...) -> ExplanationResponse`<br>Variante **100% assíncrona** não-bloqueante de `explain(...)` (ideal para rotas `async def` no FastAPI). |
| **`layer.create_default_agents(...)`** | `create_default_agents(threshold: float = 7.0) -> ExplanationLayer`<br>Instancia e registra os 4 agentes canônicos padrão (`context_manager`, `artifacts`, `explainer`, `validator`). Retorna `self` para permitir encadeamento. |
| **`layer.register_agent(...)`** | `register_agent(agent: BaseAgent, prompt_source: str | Path | None = None) -> None`<br>Registra um `SpecializedAgent` (ex.: especialista em cavitação de turbinas) ou substitui qualquer agente canônico existente. |
| **`layer.remove_agent(...)`** | `remove_agent(name: str) -> None`<br>Desativa e remove qualquer agente da camada pelo seu identificador (ex.: `layer.remove_agent("context_manager")`). |
| **`layer.get_agent(...)`** | `get_agent(name: str) -> BaseAgent`<br>Recupera a instância ativa de um agente para registrar ferramentas (`aa.register_tool(fn)`) ou plugar busca vetorial/RAG (`cma.set_retriever(fn)`). |
| **`layer.override_agent_prompt(...)`** | `override_agent_prompt(name: str, prompt: str | Path | None) -> None`<br>Sobrescreve cirurgicamente o prompt de um agente ativo via texto direto (`str`) ou arquivo Markdown local (`Path("meu_prompt.md")`) sem perder as tools registradas. Passar `None` restaura o prompt padrão embutido. |
| **`layer.reset_agent_prompt(...)`** | `reset_agent_prompt(name: str) -> None`<br>Restaura o template de prompt de um agente para o seu padrão original embutido (`Built-in`). |
| **`layer.get_history(...)`** | `get_history(session_id: str) -> list[ChatMessage]`<br>Recupera a lista de mensagens (`ChatMessage`, modelo Pydantic `BaseModel`) gravadas na sessão em ordem cronológica. |
| **`layer.clear_session(...)`** | `clear_session(session_id: str) -> None`<br>Limpa o histórico de mensagens e os artefatos em cache de uma sessão específica. |
| **`layer.inspect(...)`** | `inspect(print_output: bool = True) -> str | None`<br>Imprime uma tabela formatada no terminal/notebook mostrando todos os agentes ativos, seus papéis, quantidade de tools registradas e a origem do prompt (`Built-in`, caminho `.md` ou `Inline (str)`). Se `print_output=False`, retorna a string formatada sem imprimir. |

---

### B. Agentes Modulares e Ferramentas (`eoa.agents` & `eoa.tools`)

Quando o desenvolvedor recupera um agente via `layer.get_agent(...)` ou instancia um agente diretamente, ele conta com os seguintes métodos:

| Classe / Método | Assinatura e Papel Arquitetural |
| :--- | :--- |
| **`ContextManagerAgent`** | `ContextManagerAgent(prompt=None, role="Context Manager", retriever=None, tools=None)`<br>Agente canônico (`"context_manager"`) que resolve esquemas, dicionários de variáveis e manuais técnicos. |
| **`cma.set_retriever(...)`** | `set_retriever(retriever: ContextRetriever | Callable[[str], str]) -> None`<br>Pluga um mecanismo customizado de busca ou RAG vetorial (ex.: FAISS, Chroma, LlamaIndex) para manuais extensos de centenas de páginas. |
| **`ArtifactsAgent`** | `ArtifactsAgent(prompt=None, role="Artifacts Analyst", tools=None)`<br>Agente canônico (`"artifacts"`) que interpreta predições de ML e atribuições XAI (SHAP, LIME). Possui otimização **Zero-ML** (pula chamada ao LLM se não houver predição). |
| **`ExplainerAgent`** | `ExplainerAgent(prompt=None, role="Explainer", tools=None)`<br>Agente canônico (`"explainer"`) que sintetiza a narrativa final combinando AA, CMA, especialistas, histórico de chat e críticas de refinamento. |
| **`ValidatorAgent`** | `ValidatorAgent(threshold: float = 7.0, prompt=None, role="Quality Gate Auditor", tools=None)`<br>Agente canônico (`"validator"`) que atua como *Quality Gate* (LLM-as-a-Judge), atribuindo nota de `0.0` a `10.0` e gerando críticas estruturadas para auto-correção. |
| **`SpecializedAgent`** | `SpecializedAgent(name: str, prompt: str | Path, role: str | None = None, tools: list | None = None)`<br>Agente customizável para subsistemas ou domínios específicos. Injeta seus pareceres automaticamente em `state.specialist_insights[name]`. |
| **`agent.register_tool(...)`** | `register_tool(tool_or_fn: Tool | Callable[..., Any]) -> Tool`<br>Disponível em **qualquer agente**. Registra uma função Python (síncrona ou assíncrona) como ferramenta. Opera de forma híbrida: (1) *Fast-Path determinístico* quando `query_params` / `incident_id` já fornece os argumentos exigidos; (2) *Agentic Reasoning* quando o agente avalia a `user_query`, decide se a tool é necessária e extrai os parâmetros tipados (`parameters_schema`) autonomamente do texto. |
| **`agent.set_prompt(...)`** | `set_prompt(source: str | Path) -> None`<br>Atualiza o template de prompt da instância do agente. |
| **`Tool(fn, ...)`** | `Tool(fn: Callable, name: str | None = None, description: str | None = None)`<br>Wrapper de introspecção de ferramentas com suporte a `.execute(**kwargs)` e `await .aexecute(**kwargs)`. |

---

### C. Contratos de Dados e Resposta (`eoa.contracts`)

Todos os contratos são modelos **Pydantic v2** com serialização nativa (`.model_dump()`, `.model_dump_json()`):

| Contrato / Propriedade | Campos e Utilidades |
| :--- | :--- |
| **`ExplanationResponse`** | Retornado por `layer.explain(...)`:<br>• `narrative: str` — O texto final da explicação entregue ao usuário.<br>• `validation_score: float | None` — Nota de fidelidade (0.0 a 10.0) atribuída pelo `ValidatorAgent`.<br>• `validation_passed: bool | None` — `True` se `validation_score >= threshold`.<br>• `validation_critique: str | None` — Crítica técnica do validador (se houver).<br>• `attempts_used: int` — Quantidade de iterações gastas no loop EA ↔ VA.<br>• `fallback_used: bool` — `True` se o modo Fail-Safe ("Unplugged") foi acionado.<br>• `session_id: str | None` — Identificador da sessão de conversa.<br>• `execution_trace: list[AgentStepTrace]` — Rastro passo a passo com tempo em `ms` de cada agente.<br>• `total_execution_time_ms: float` *(property)* — Soma da latência de todos os passos registrados no trace. |
| **`PredictionArtifact`** | Contrato de Nível 1 (Predição de ML): `target_variable`, `predicted_value`, `confidence_or_probability`, `status_label`, `model_name`, `metadata`. Aceita sinônimos automáticamente via `PredictionArtifact.from_input(dict)`. |
| **`XAIArtifact`** | Contrato de Nível 2 (Explicabilidade): `method` (ex.: `"SHAP"`), `feature_attributions: dict[str, float]`, `baseline_values`, `counterfactuals`. Método útil: `xai.top_features(n=5)` retorna as top $N$ variáveis ordenadas por impacto absoluto (`|score|`). |
| **`DomainContext`** | Contrato de Contexto Operacional: `variables`, `thresholds`, `documents`, `hierarchy`, `schemas`. Métodos úteis: `.is_empty()`, `.merge(other)`, `.add_variable()`, `.add_document()`. |

---

### D. Memória, Sessões e Auditoria SQLite (`eoa.memory`)

| Classe / Método | Assinatura e Papel Arquitetural |
| :--- | :--- |
| **`SQLiteSessionStore`** | `SQLiteSessionStore(db_path: str | Path = "eoa.db")` — Armazenamento relacional persistente usando `sqlite3` padrão do Python (zero dependências externas, modo WAL concorrente). |
| **`InMemorySessionStore`** | `InMemorySessionStore()` — Armazenamento efêmero em RAM (padrão quando `session_store` não é especificado). |
| **`store.get_history(...)`** | `get_history(session_id: str) -> list[ChatMessage]` — Retorna todas as mensagens (`role`, `content`, `timestamp`, `metadata`) em ordem cronológica. |
| **`store.get_validation_logs(...)`** | `get_validation_logs(session_id: str) -> list[dict]` *(SQLiteSessionStore)* — Retorna o histórico de auditorias do `ValidatorAgent` (`score`, `passed`, `critique`, `timestamp`, `metadata`) daquela sessão. |
| **`store.export_dict(...)`** | `export_dict(session_id: str | None = None) -> dict` *(SQLiteSessionStore)* — Exporta `sessions`, `messages` e `validation_logs` como dicionários Python prontos para auditoria externa, pandas DataFrames ou relatórios do artigo. |
| **`store.clear(...)` / `delete_session(...)`** | Limpa as mensagens de uma sessão (`clear`) ou remove a sessão inteira do banco (`delete_session`). |
| **`store.list_sessions()`** | Retorna a lista de todos os `session_id` registrados no banco. |

---

### E. Conexão com LLMs e Gerenciamento de Prompts (`eoa.llm` & `eoa.prompts`)

| Componente | Assinatura e Papel Arquitetural |
| :--- | :--- |
| **`eoa.create_llm(...)`** | Factory unificada que resolve os 3 modos de conexão: Modo 1 (`provider="gemini"` ou auto-load do `.env`), Modo 2 (`llm=genai_client`), Modo 3 (`llm=minha_funcao_python`). |
| **`eoa.GeminiProvider`** | Adaptador oficial `google-genai` com suporte síncrono (`generate`), assíncrono (`agenerate`), auto-load de `.env` e retry com backoff exponencial para erros transientes (`429`/`503`). |
| **`eoa.CallableLLMAdapter`** | Envolve qualquer função Python `def fn(prompt: str) -> str` ou `async def fn(prompt: str) -> str` no protocolo `LLM`. |
| **`eoa.PromptManager`** | Utilitário estático e stateful para resolver (`PromptManager.resolve(agent, source)`) e interpolar (`PromptManager.render(template, **vars)`) prompts Markdown PTCF preservando chaves JSON intactas. |
