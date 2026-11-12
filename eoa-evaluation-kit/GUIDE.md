# EOA - Guia de Referência da API e Ferramentas

Este guia reúne, por tópico, as funções e os parâmetros que você pode chamar no framework EOA e no módulo auxiliar [`tools.py`](./tools.py), sempre com a tabela de referência seguida de exemplos práticos.

---

## 1. Inicialização da `ExplanationLayer`

| Função / Classe | Parâmetros principais | Quando chamar |
| :--- | :--- | :--- |
| `ExplanationLayer.default_layer` | `(llm=None, *, provider=None, api_key=None, model=None, session_store=None, threshold=7.0, max_attempts=3, temperature=None, **llm_kwargs)` | Forma mais rápida de começar: já cria a camada com os 4 agentes padrão ativos (`context_manager`, `artifacts`, `explainer`, `validator`). |
| `ExplanationLayer` | `(llm=None, *, provider=None, api_key=None, model=None, session_store=None, max_attempts=3, auto_create_agents=False, **llm_kwargs)` | Quando você quer montar a camada manualmente e escolher quais agentes registrar antes de executar. |
| `layer.create_default_agents` | `(threshold=7.0, temperature=None)` | Usado após `ExplanationLayer(...)` se quiser ativar os 4 agentes padrão depois. |
| `SQLiteSessionStore` | `(db_path="eoa.db", timeout=10.0)` | Para persistir histórico de conversa, artefatos e notas do validador em SQLite nativo (passe no parâmetro `session_store`). |

### Exemplos de uso

```python
from google import genai
from eoa import ExplanationLayer, ExplainerAgent, SQLiteSessionStore, ValidatorAgent

# 1. Padrão detectando GEMINI_API_KEY do .env
layer = ExplanationLayer.default_layer()

# 2. Configurando provedor, modelo, nota de corte e persistência em SQLite
store = SQLiteSessionStore("hydraulic_audit.db")
layer_sqlite = ExplanationLayer.default_layer(
    provider="gemini",
    model="gemini-3.6-flash",
    threshold=7.5,
    max_attempts=3,
    temperature=0.0,
    session_store=store,
)

# 3. Trazendo um cliente genai.Client já configurado
client = genai.Client(api_key="sua-chave")
layer_client = ExplanationLayer.default_layer(llm=client, model="gemini-3.6-flash")

# 4. Conectando uma função Python customizada (modelo local ou simulado)
def meu_llm(prompt: str) -> str:
    return "Resposta customizada."

layer_custom = ExplanationLayer.default_layer(llm=meu_llm)

# 5. Construtor direto com criação automática ou manual de agentes
layer_direto = ExplanationLayer(provider="gemini", model="gemini-3.6-flash", auto_create_agents=True)
layer_manual = ExplanationLayer(provider="gemini", model="gemini-3.6-flash")
layer_manual.register_agent(ExplainerAgent())
layer_manual.register_agent(ValidatorAgent(threshold=8.0))
```

---

## 2. Gerando e Lendo Explicações (`explain` e `ExplanationResponse`)

| Método | Parâmetros principais | Quando chamar |
| :--- | :--- | :--- |
| `layer.explain` | `(user_query=..., *, prediction=None, xai=None, context=None, query_params=None, incident_id=None, session_id=None, metadata=None)` | Ponto principal de execução síncrona. Atende Zero-ML (só `context`), Push (`prediction` + `xai`), Pull (`incident_id` + tools) e Multi-Turno (`session_id`). |
| `await layer.aexplain` | Mesmos parâmetros de `layer.explain` | Quando sua aplicação for assíncrona (`asyncio`). |

Campos principais retornados em **`ExplanationResponse`**:

| Atributo | Tipo | Significado |
| :--- | :--- | :--- |
| `narrative` | `str` | Explicação final produzida para o usuário. |
| `validation_score` | `float \| None` | Nota dada pelo `ValidatorAgent` (`0.0` a `10.0`). |
| `validation_passed` | `bool \| None` | `True` se atingiu o `threshold` configurado. |
| `validation_critique` | `str \| None` | Feedback do auditor quando pede refinamento. |
| `attempts_used` | `int` | Quantas rodadas de geração/validação ocorreram (`>= 1`). |
| `fallback_used` | `bool` | `True` se o LLM falhou e entrou o modo *fail-safe* determinístico. |
| `session_id` | `str \| None` | ID da sessão utilizada. |
| `execution_trace` | `list[AgentStepTrace]` | Lista com `agent_name`, `status`, `execution_time_ms` e `summary` de cada etapa. |

### Exemplos de uso

```python
from tools import get_incident, load_domain_context

ctx = load_domain_context()
inc = get_incident(2)

# 1. Modo Zero-ML (explora apenas documentação e sensores, sem modelo preditivo)
r_zero = layer.explain(
    context=ctx,
    user_query="Quais são os 14 sensores físicos do sistema e o que significa cada código de condição?",
)

# 2. Modo Push (envia predição e SHAP diretamente)
r_push = layer.explain(
    prediction=inc["prediction"],
    xai=inc["xai"],
    context=ctx,
    user_query="Por que este ciclo foi classificado como 3% (próximo à falha total)?",
)

# 3. Modo Pull (os agentes buscam os dados sozinhos via tools registradas)
r_pull = layer.explain(
    incident_id=2,
    context=load_domain_context(include_baseline_thresholds=False),
    user_query="Diagnostique o incidente 2 comparando temperaturas e pressões com a operação normal.",
)

# 4. Modo Multi-Turno (reaproveita histórico e artefatos da sessão)
r_t1 = layer.explain(incident_id=2, session_id="op_01", context=ctx, user_query="Resuma a falha do incidente 2.")
r_t2 = layer.explain(session_id="op_01", user_query="Quanto o sensor TS1 subiu acima da média saudável?")

# 5. Lendo o rastro de execução
print(r_push.narrative)
print(f"Nota: {r_push.validation_score}/10 | Tentativas: {r_push.attempts_used}")
for step in r_push.execution_trace:
    print(f"[{step.agent_name}] {step.status} ({step.execution_time_ms:.1f} ms): {step.summary}")
```

---

## 3. Gerenciando Agentes, Prompts e Ferramentas

| Função / Classe | Parâmetros principais | Quando chamar |
| :--- | :--- | :--- |
| `layer.inspect` | `(print_output=True)` | Para conferir rapidamente quais agentes estão ativos, quantas tools têm e qual prompt usam. |
| `layer.get_agent` | `(name)` | Para pegar um agente (`"context_manager"`, `"artifacts"`, `"explainer"`, `"validator"` ou customizado) e configurar tools/retriever. |
| `agent.register_tool` | `(tool_or_fn)` | Para anexar qualquer função Python tipada (com docstring) como ferramenta do agente. |
| `layer.override_agent_prompt` | `(name, prompt)` | Para trocar o prompt de um agente usando uma `str` ou um `Path("arquivo.md")`. |
| `layer.reset_agent_prompt` | `(name)` | Para restaurar o prompt padrão embutido do agente. |
| `SpecializedAgent` | `(name, prompt, role=None, tools=None, temperature=None)` | Para criar um novo agente especialista com foco em um domínio ou regra específica. |
| `layer.register_agent` | `(agent, prompt_source=None)` | Para adicionar um `SpecializedAgent` à camada ou substituir um agente existente. |
| `layer.remove_agent` | `(name)` | Para desligar um agente do pipeline (ex.: remover `"validator"` em testes rápidos). |
| `cma.set_retriever` | `(retriever)` | Para trocar o buscador padrão do `ContextManagerAgent` por uma função `fn(query)` ou objeto RAG próprio. |

### Exemplos de uso

```python
from eoa import SpecializedAgent
from tools import get_baseline_comparison, get_incident, get_sensor_readings

# 1. Registrando tools para o Modo Pull e inspecionando a camada
layer.get_agent("artifacts").register_tool(get_incident)
layer.get_agent("context_manager").register_tool(get_sensor_readings)
layer.get_agent("context_manager").register_tool(get_baseline_comparison)
layer.inspect()

# 2. Sobrescrevendo o prompt do Explainer e restaurando depois
layer.override_agent_prompt(
    "explainer",
    """Responda em 3 tópicos curtos usando apenas os dados verificados abaixo:
1. Estado previsto e confiança
2. Top 3 sensores no TreeSHAP
3. Desvios em relação à referência de 100%

Evidências: {artifacts}
Contexto: {domain_context}
Pergunta: {user_query}
""",
)
# layer.reset_agent_prompt("explainer")

# 3. Adicionando um agente especialista de domínio (executa em paralelo na Fase 1)
especialista = SpecializedAgent(
    name="analista_estabilidade",
    role="Especialista em Regime Térmico",
    prompt=(
        "Verifique nos dados coletados ({tool_outputs}) se o ciclo está estável "
        "(stable_flag=0) ou em transição (stable_flag=1) e explique o impacto disso em 2 frases."
    ),
    tools=[get_sensor_readings],
)
layer.register_agent(especialista)
```

---

## 4. Sessões, Histórico e Auditoria (`SQLiteSessionStore`)

| Método | Parâmetros principais | Quando chamar |
| :--- | :--- | :--- |
| `layer.get_history` | `(session_id)` | Para recuperar todas as mensagens (`ChatMessage`) trocadas na sessão em ordem cronológica. |
| `layer.clear_session` | `(session_id)` | Para apagar o histórico e o cache de artefatos de uma sessão antes de recomeçar. |
| `store.get_validation_logs` | `(session_id)` | Para consultar notas, aprovações e críticas gravadas pelo `ValidatorAgent` no SQLite. |
| `store.list_sessions` | `()` | Para listar todos os `session_id` já registrados no banco SQLite. |
| `store.export_dict` | `(session_id=None)` | Para exportar sessões, mensagens e logs de validação como dicionário Python/JSON. |

### Exemplos de uso

```python
# Ver diálogo gravado na sessão
for msg in layer_sqlite.get_history("op_01"):
    print(f"[{msg.role}] {msg.content}")

# Ver notas do ValidatorAgent gravadas no banco SQLite
for log in store.get_validation_logs("op_01"):
    print(f"Nota: {log['score']}/10 | Aprovado: {log['passed']} | Meta: {log['metadata']}")

# Exportar auditoria completa da sessão e limpar ao final
auditoria = store.export_dict(session_id="op_01")
layer_sqlite.clear_session("op_01")
```

---

## 5. Contratos de Dados (`PredictionArtifact`, `XAIArtifact`, `DomainContext`)

Você pode enviar dicionários Python simples (como os retornados por `tools.py`) ou instanciar as classes Pydantic diretamente:

| Contrato | Principais campos | Quando usar / observar |
| :--- | :--- | :--- |
| `PredictionArtifact` | `target_variable`, `predicted_value`, `confidence_or_probability`, `status_label`, `model_name`, `metadata` | Representa a predição do modelo (Nível 1). `target_variable` e `predicted_value` são obrigatórios. |
| `XAIArtifact` | `method`, `feature_attributions`, `baseline_values`, `counterfactuals`, `raw_artifacts` | Representa pesos de explicabilidade (Nível 2). Traz `.top_features(n=5, absolute=True)` pronto. |
| `DomainContext` | `variables`, `thresholds`, `documents`, `hierarchy`, `schemas` | Agrupa dicionário de sensores, faixas de referência e textos técnicos. Possui `.merge(...)`. |

### Exemplos de uso

```python
from eoa import DomainContext, PredictionArtifact, XAIArtifact

pred = PredictionArtifact(
    target_variable="Cooler condition / %",
    predicted_value="3% (close to total failure)",
    confidence_or_probability=1.0,
)
xai = XAIArtifact(
    method="TreeSHAP",
    feature_attributions={"PS6": 0.1670, "PS5": 0.1557, "TS4": 0.1340},
)
ctx = DomainContext(
    variables={"TS4": "Temperature sensor (°C, 1 Hz)"},
    thresholds={"Cooler condition / %": "3: close to total failure | 20: reduced | 100: full"},
)
```

---

## 6. Funções Prontas do Kit ([`tools.py`](./tools.py))

Todas usam apenas a biblioteca padrão do Python (`csv`, `json`, `pathlib`) para acessar os dados já processados na pasta `infrastructure/`:

| Função | Parâmetros | O que retorna / quando chamar |
| :--- | :--- | :--- |
| `list_incidents` | `(condition_code=None)` | Lista os 9 incidentes curados (ou filtra por `3`, `20` ou `100`) com confiança, `stable_flag` e Top 3 no TreeSHAP. |
| `get_incident` | `(incident_id=None, *, cycle_id=None)` | Retorna `{"prediction": ..., "xai": ...}` para o incidente (`1` a `9`) ou ciclo. Ideal para Modo Push ou tool do `ArtifactsAgent`. |
| `get_sensor_readings` | `(incident_id=None, *, cycle_id=None)` | Retorna as médias de 60s dos 14 sensores físicos com unidades e `stable_flag`, mais texto estruturado em `documents`. |
| `get_baseline_comparison` | `(incident_id=None, *, cycle_id=None)` | Compara os 14 sensores do incidente contra a referência saudável de 100% (`N=593` ciclos), calculando `delta`, `z_score` e faixa `p05..p95`. |
| `load_domain_context` | `(include_baseline_thresholds=True)` | Carrega [`description.md`](./infrastructure/docs/description.md), [`documentation.md`](./infrastructure/docs/documentation.md), dicionário dos 14 sensores e limiares saudáveis. |

### Exemplos de uso

```python
from tools import (
    get_baseline_comparison,
    get_incident,
    get_sensor_readings,
    list_incidents,
    load_domain_context,
)

# Listar apenas incidentes críticos (3%)
criticos = list_incidents(condition_code=3)

# Buscar artefatos de ML/XAI, leituras e desvio da linha de base do incidente 2
inc2 = get_incident(2)
leituras = get_sensor_readings(2)
desvios = get_baseline_comparison(2)
print(desvios["comparisons"]["TS1"])

# Carregar documentação e dicionário de sensores para o context=
contexto_completo = load_domain_context()
```

---

## 7. Para Curiosos: Estrutura Interna do `src/eoa/` (Leitura Opcional)

> **Nota:** Esta seção é apenas uma visão rápida de como a biblioteca instalada no `.whl` está organizada por dentro. **Não é leitura obrigatória** e você não precisa consultar nenhum módulo interno para fazer os testes no notebook.

```text
src/eoa/
├── __init__.py                   # Exportações públicas da API (v0.1.0)
├── layer.py                      # Fachada ExplanationLayer, cache de sessão e gerenciamento de agentes
├── contracts.py                  # Schemas Pydantic v2 (PredictionArtifact, XAIArtifact, DomainContext, ExplanationResponse)
├── engine/
│   └── orchestrator.py           # Fluxo multi-agente: coleta paralela, loop Explainer/Validator e fallback fail-safe
├── agents/
│   ├── base.py                   # BaseAgent, AgentState e despacho automático de ferramentas
│   ├── context.py                # ContextManagerAgent e StructuredContextRetriever (pluggable para RAG)
│   ├── artifacts.py              # ArtifactsAgent (Push, Pull e economia de chamadas no modo Zero-ML)
│   ├── specialized.py            # SpecializedAgent para especialistas plugáveis de domínio
│   ├── explainer.py              # ExplainerAgent (síntese da explicação com evidências e histórico)
│   └── validator.py              # ValidatorAgent (auditoria automática LLM-as-a-Judge com nota 0-10)
├── prompts/
│   ├── manager.py                # PromptManager (resolução de defaults, overrides inline/.md e interpolação segura)
│   └── *_default.md              # Prompts padrão dos agentes e template de seleção de ferramentas
├── tools/
│   └── base.py                   # Wrapper Tool (extração de assinatura, schema de parâmetros e coerção de tipos)
├── llm/
│   ├── base.py                   # Protocolo base LLM (generate / agenerate)
│   ├── factory.py                # create_llm() unificando provider, cliente externo e função Python
│   ├── gemini.py                 # GeminiProvider (google-genai com retry exponencial para 429/503)
│   └── callable.py               # CallableLLMAdapter para adaptar funções sync/async como LLM
└── memory/
    ├── base.py                   # Protocolo SessionStore e ChatMessage
    ├── in_memory.py              # Armazenamento padrão em memória
    └── sqlite.py                 # SQLiteSessionStore persistente (tabelas sessions, messages e validation_logs)
```

*Novamente, essa visão interna é totalmente opcional. Tudo o que você precisa para explorar e avaliar o EOA está nas seções 1 a 6 acima e no [`exploration.ipynb`](./exploration.ipynb).*
