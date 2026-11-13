# EOA - Guia de Referência da API e Ferramentas

Este documento sitematiza as classes, métodos e parâmetros disponibilizados pelo framework EOA e pelo módulo auxiliar [`tools.py`](./tools.py). Cada seção apresenta uma tabela de especificação técnica acompanhada de exemplos de implementação.

---

## 1. Inicialização da `ExplanationLayer`

| Classe / Método | Parâmetros Principais | Finalidade e Contexto de Aplicação |
| :--- | :--- | :--- |
| `ExplanationLayer.default_layer` | `(llm=None, *, provider=None, api_key=None, model=None, session_store=None, threshold=7.0, max_attempts=3, temperature=None, **llm_kwargs)` | Instanciação simplificada: inicializa a camada de explicabilidade com os quatro agentes padrão pré-configurados (`context_manager`, `artifacts`, `explainer`, `validator`). |
| `ExplanationLayer` | `(llm=None, *, provider=None, api_key=None, model=None, session_store=None, max_attempts=3, auto_create_agents=False, **llm_kwargs)` | Instanciação customizada: permite compor a arquitetura manualmente e selecionar quais agentes serão registrados antes da execução. |
| `layer.create_default_agents` | `(threshold=7.0, temperature=None)` | Usado  após a instanciação direta de `ExplanationLayer(...)` para registrar posteriormente os quatro agentes padrão. |
| `SQLiteSessionStore` | `(db_path="eoa.db", timeout=10.0)` | Habilita a persistência nativa em SQLite do histórico conversacional, dos artefatos recuperados e dos registros de auditoria do validador (fornecido via parâmetro `session_store`). |

### Exemplos de Implementação

```python
from google import genai
from eoa import ExplanationLayer, ExplainerAgent, SQLiteSessionStore, ValidatorAgent

# 1. Inicialização padrão com detecção automática de GEMINI_API_KEY via .env
layer = ExplanationLayer.default_layer()

# 2. Configuração explícita de provedor, modelo, limiar de aprovação e persistência SQLite
store = SQLiteSessionStore("hydraulic_audit.db")
layer_sqlite = ExplanationLayer.default_layer(
    provider="gemini",
    model="gemini-3.6-flash",
    threshold=7.5,
    max_attempts=3,
    temperature=0.0,
    session_store=store,
)

# 3. Injeção de um cliente genai.Client previamente instanciado
client = genai.Client(api_key="sua-chave")
layer_client = ExplanationLayer.default_layer(llm=client, model="gemini-3.6-flash")

# 4. coplamento de uma função Python customizada (modelo local ou simulado)
def meu_llm(prompt: str) -> str:
    return "Resposta customizada."

layer_custom = ExplanationLayer.default_layer(llm=meu_llm)

# 5. Construtor direto com registro automático ou manual de agentes
layer_direto = ExplanationLayer(provider="gemini", model="gemini-3.6-flash", auto_create_agents=True)
layer_manual = ExplanationLayer(provider="gemini", model="gemini-3.6-flash")
layer_manual.register_agent(ExplainerAgent())
layer_manual.register_agent(ValidatorAgent(threshold=8.0))
```

---

## 2. Gerando e Lendo Explicações (`explain` e `ExplanationResponse`)

| Método | Parâmetros principais |  Finalidade e Contexto de Aplicação |
| :--- | :--- | :--- |
| `layer.explain` | `(user_query=..., *, prediction=None, xai=None, context=None, query_params=None, incident_id=None, session_id=None, metadata=None)` | Interface principal de execução síncrona. Suporta os modos Zero-ML (só `context`), Push (`prediction` + `xai`), Pull (`incident_id` + tools) e Multi-Turno (`session_id`). |
| `await layer.aexplain` | Idênticos aos de `layer.explain` |  Interface de execução assíncrona destinada a aplicações baseadas em `asyncio`. |

Principais atributos retornados na estrutura **`ExplanationResponse`**:

| Atributo | Tipo | Significado |
| :--- | :--- | :--- |
| `narrative` | `str` | Narrativa explicativa final destinada ao usuário. |
| `validation_score` | `float \| None` | Pontuação atribuída pelo `ValidatorAgent` (escala de `0.0` a `10.0`). |
| `validation_passed` | `bool \| None` | Indica se a explicação atingiu ou superou o limiar (`threshold`) configurado. |
| `validation_critique` | `str \| None` | Parecer técnico emitido pelo agente auditor quando um ciclo de refinamento é solicitado. |
| `attempts_used` | `int` | Número de iterações de geração e validação executadas (`>= 1`). |
| `fallback_used` | `bool` | Indica se houve falha na comunicação com o LLM e acionamento do modo *fail-safe* determinístico. |
| `session_id` | `str \| None` | Identificador único da sessão conversacional utilizada. |
| `execution_trace` | `list[AgentStepTrace]` | Registro estruturado contendo `agent_name`, `status`, `execution_time_ms` e `summary` de cada etapa. |

### Exemplos de Implementação

```python
from tools import get_incident, load_domain_context

ctx = load_domain_context()
inc = get_incident(2)

# 1. Modo Zero-ML (consulta exclusivamente documentação e sensores, sem modelo preditivo)
r_zero = layer.explain(
    context=ctx,
    user_query="Quais são os 14 sensores físicos do sistema e o que significa cada código de condição?",
)

# 2. Modo Push (fornecimento direto da predição e dos valores SHAP)
r_push = layer.explain(
    prediction=inc["prediction"],
    xai=inc["xai"],
    context=ctx,
    user_query="Por que este ciclo foi classificado como 3% (próximo à falha total)?",
)

# 3. Modo Pull (recuperação autônoma de dados pelos agentes via ferramentas registradas)
r_pull = layer.explain(
    incident_id=2,
    context=load_domain_context(include_baseline_thresholds=False),
    user_query="Diagnostique o incidente 2 comparando temperaturas e pressões com a operação normal.",
)

# 4. Modo Multi-Turno (reutilização do histórico e dos artefatos persistidos na sessão)
r_t1 = layer.explain(incident_id=2, session_id="op_01", context=ctx, user_query="Resuma a falha do incidente 2.")
r_t2 = layer.explain(session_id="op_01", user_query="Quanto o sensor TS1 subiu acima da média saudável?")

# 5. Inspeção do rastro de execução (execution trace)
print(r_push.narrative)
print(f"Nota: {r_push.validation_score}/10 | Tentativas: {r_push.attempts_used}")
for step in r_push.execution_trace:
    print(f"[{step.agent_name}] {step.status} ({step.execution_time_ms:.1f} ms): {step.summary}")
```

---

## 3. Gerenciamento Agentes, Prompts e Ferramentas

| Classe / Método | Parâmetros Principais | Finalidade e Contexto de Aplicação |
| :--- | :--- | :--- |
| `layer.inspect` | `(print_output=True)` | Permite auditar a topologia da camada, exibindo agentes ativos, ferramentas vinculadas e prompts em uso. |
| `layer.get_agent` | `(name)` | Recupera a instância de um agente (`"context_manager"`, `"artifacts"`, `"explainer"`, `"validator"` ou customizado) para configuração de ferramentas ou recuperadores. |
| `agent.register_tool` | `(tool_or_fn)` | Registra uma função Python devidamente tipada e documentada (*docstring*) como ferramenta executável pelo agente. |
| `layer.override_agent_prompt` | `(name, prompt)` | Substitui a instrução (*prompt*) de um agente por meio de uma `str` ou de um arquivo `Path("arquivo.md")`. |
| `layer.reset_agent_prompt` | `(name)` | Restaura a instrução padrão nativa do agente especificado. |
| `SpecializedAgent` | `(name, prompt, role=None, tools=None, temperature=None)` | Define um agente especialista customizado voltado a um subdomínio ou regra analítica específica. |
| `layer.register_agent` | `(agent, prompt_source=None)` | Incorpora um `SpecializedAgent` à camada ou substitui um agente previamente registrado. |
| `layer.remove_agent` | `(name)` | Remove um agente do *pipeline* de execução (por exemplo, desativar `"validator"` em experimentos controlados). |
| `cma.set_retriever` | `(retriever)` | Substitui o mecanismo padrão de recuperação do `ContextManagerAgent` por uma função `fn(query)` ou componente RAG customizado. |

### Exemplos de Implementação

```python
from eoa import SpecializedAgent
from tools import get_baseline_comparison, get_incident, get_sensor_readings

# 1. Registro de ferramentas para o Modo Pull e inspeção da topologia da camada
layer.get_agent("artifacts").register_tool(get_incident)
layer.get_agent("context_manager").register_tool(get_sensor_readings)
layer.get_agent("context_manager").register_tool(get_baseline_comparison)
layer.inspect()

# 2. Sobrescrita e posterior restauração do prompt do ExplainerAgent
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

# 3. Incorporação de um agente especialista de domínio (executado em paralelo na Fase 1)
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

| Método | Parâmetros Principais | Finalidade e Contexto de Aplicação |
| :--- | :--- | :--- |
| `layer.get_history` | `(session_id)` | Recupera cronologicamente todas as mensagens (`ChatMessage`) registradas na sessão especificada. |
| `layer.clear_session` | `(session_id)` | Remove o histórico conversacional e o cache de artefatos de uma sessão antes de uma nova execução. |
| `store.get_validation_logs` | `(session_id)` | Consulta as pontuações, aprovações e pareceres críticos persistidos pelo `ValidatorAgent` no SQLite. |
| `store.list_sessions` | `()` | Lista todos os identificadores de sessão (`session_id`) armazenados no banco de dados SQLite. |
| `store.export_dict` | `(session_id=None)` | Exporta sessões, mensagens e registros de validação em formato de dicionário Python/JSON. |

### Exemplos de Implementação

```python
# Inspeção do histórico conversacional persistido na sessão
for msg in layer_sqlite.get_history("op_01"):
    print(f"[{msg.role}] {msg.content}")

# Consulta aos registros de auditoria do ValidatorAgent no banco SQLite
for log in store.get_validation_logs("op_01"):
    print(f"Nota: {log['score']}/10 | Aprovado: {log['passed']} | Meta: {log['metadata']}")

# Exportação completa dos registros de auditoria e limpeza da sessão ao término
auditoria = store.export_dict(session_id="op_01")
layer_sqlite.clear_session("op_01")
```

---

## 5. Contratos de Dados (`PredictionArtifact`, `XAIArtifact`, `DomainContext`)

A API aceita tanto dicionários Python nativos (a exemplo dos retornados por `tools.py`) quanto instâncias diretas dos modelos Pydantic:

| Contrato | Principais campos | Especificação e Uso |
| :--- | :--- | :--- |
| `PredictionArtifact` | `target_variable`, `predicted_value`, `confidence_or_probability`, `status_label`, `model_name`, `metadata` | Encapsula a saída do modelo preditivo (Nível 1). Os atributos `target_variable` e `predicted_value` são obrigatórios. |
| `XAIArtifact` | `method`, `feature_attributions`, `baseline_values`, `counterfactuals`, `raw_artifacts` | Encapsula as atribuições de explicabilidade (Nível 2). Disponibiliza o método auxiliar `.top_features(n=5, absolute=True)`. |
| `DomainContext` | `variables`, `thresholds`, `documents`, `hierarchy`, `schemas` | Agrega o dicionário de sensores, limiares de referência e documentação técnica. Suporta composição via `.merge(...)`. |

### Exemplos de Implementação

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

## 6. Funções Auxiliares do Ambiente Experimental  ([`tools.py`](./tools.py))

Implementadas exclusivamente com a biblioteca padrão do Python (`csv`, `json`, `pathlib`), estas funções provêem acesso estruturado aos dados pré-processados no diretório `infrastructure/`:

| Função | Parâmetros | Retorno e Finalidade |
| :--- | :--- | :--- |
| `list_incidents` | `(condition_code=None)` | Retorna a listagem dos 9 incidentes selecionados (com filtragem opcional por `3`, `20` ou `100`), incluindo confiança, `stable_flag` e os três principais atributos no TreeSHAP. |
| `get_incident` | `(incident_id=None, *, cycle_id=None)` | Retorna o dicionário `{"prediction": ..., "xai": ...}` correspondente ao incidente (`1` a `9`) ou ciclo especificado. Adequado ao Modo Push ou como ferramenta do `ArtifactsAgent`. |
| `get_sensor_readings` | `(incident_id=None, *, cycle_id=None)` | Retorna as médias de 60 segundos dos 14 sensores físicos acompanhadas de unidades, indicador `stable_flag` e representação textual estruturada em `documents`. |
| `get_baseline_comparison` | `(incident_id=None, *, cycle_id=None)` | Compara as leituras dos 14 sensores do incidente frente ao regime nominal saudável de 100% (`N=593` ciclos), computando `delta`, `z_score` e intervalo interquantílico `p05..p95`. |
| `load_domain_context` | `(include_baseline_thresholds=True)` | Carrega os documentos [`description.md`](./infrastructure/docs/description.md) e [`documentation.md`](./infrastructure/docs/documentation.md), o dicionário dos 14 sensores e os limiares operacionais saudáveis. |

### Exemplos de Implementação

```python
from tools import (
    get_baseline_comparison,
    get_incident,
    get_sensor_readings,
    list_incidents,
    load_domain_context,
)

# Listagem filtrada de incidentes críticos (eficiência de 3%)
criticos = list_incidents(condition_code=3)

# Recuperação de artefatos de ML/XAI, leituras de sensores e desvios da linha de base para o incidente 2
inc2 = get_incident(2)
leituras = get_sensor_readings(2)
desvios = get_baseline_comparison(2)
print(desvios["comparisons"]["TS1"])

# # Carregamento da documentação técnica e do dicionário de sensores para o parâmetro context=
contexto_completo = load_domain_context()
```

---

## 7. Arquitetura Interna do Pacote `src/eoa/` (Referência Complementar)

 **Nota Metodológica:** Esta seção apresenta uma visão estrutural da organização interna do pacote distribuído no arquivo `.whl`. Trata-se de material estritamente complementar, não sendo necessária a consulta aos módulos internos para a condução do protocolo experimental no *notebook*.

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

*Reitera-se que esta visão arquitetural possui caráter exclusivamente informativo. Todos os recursos necessários para a instrumentação e avaliação empírica do framework EOA encontram-se especificados nas Seções 1 a 6 e exemplificados em [`exploration.ipynb`](./exploration.ipynb).*
