# EOA Framework - Technical Specification & Architecture Design

Este documento formaliza as decisões de arquitetura de software, contratos de dados, interfaces e diagramas do framework **EOA (Explanation-Oriented Agents)**, servindo como base técnica para a implementação e para o paper do **ICSE**.

---

## 1. Visão Geral da Arquitetura

O framework implementa a **Explanation Agent Layer** descrita no paper do SBES como uma camada de software autônoma, extensível e desacoplada (*unplugged*). 

### 1.1 Fluxo de Execução e Loop de Validação

#### A. Visão em Diagrama de Sequência Formal (Mermaid - Renderizável no GitHub/VS Code Preview)
```mermaid
sequenceDiagram
    autonumber
    actor DevApp as Developer Application (Host / Legacy)
    participant EOA as EOA Engine (Orchestrator)
    participant CMA as Context Manager Agent (CMA)
    participant AA as Artifacts Agent (AA)
    participant EA as Explainer Agent (EA)
    participant VA as Validator Agent (VA)
    participant Store as SessionStore (Memory/SQLite)

    DevApp->>EOA: explain(prediction, xai_artifacts, context, query)
    EOA->>Store: get_history(session_id)
    Store-->>EOA: chat_history
    
    par Query Context & Artifacts
        EOA->>CMA: resolve_context(context_contract)
        CMA-->>EOA: domain_context_summary
    and
        EOA->>AA: parse_artifacts(prediction, xai_artifacts)
        AA-->>EOA: technical_evidence_summary
    end

    loop Validation & Refinement Loop (Max Attempts: N)
        EOA->>EA: generate_explanation(evidence, domain_context, query, history, feedback)
        EA-->>EOA: candidate_explanation
        EOA->>VA: validate(candidate_explanation, evidence, domain_context)
        VA-->>EOA: ValidationResult (score, pass_flag, critique)
        break When score >= threshold OR attempts == max_attempts
            note over EOA: Score atingiu meta ou esgotou tentativas
        end
    end

    alt Validation Failed (excedeu tentativas)
        EOA-->>DevApp: ExplanationResponse(fallback_narrative, validation_passed=False, trace)
    else Validation Passed
        EOA->>Store: append(user_query, final_narrative)
        EOA-->>DevApp: ExplanationResponse(final_narrative, score, validation_passed=True, trace)
    end
```

#### B. Visão Rápida em Texto Monoespaçado (Leitura Direta no Editor)
```text
==========================================================================================================
FLUXO DE EXECUÇÃO DETALHADO DO EOA (ORQUESTRAÇÃO E QUALITY GATE)
==========================================================================================================

 [ APLICAÇÃO DO DEV ]
          │
          │ 1. explain(predição, artefatos_xai, contexto, pergunta)
          ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      ORCHESTRATOR ENGINE (Backbone)                    │
 └────────┬───────────────────────────────┬───────────────────────────────┘
          │                               │
          │ 2. Busca histórico do chat    │ 3. Dispara consultas em paralelo
          ▼                               ▼
 ┌───────────────────┐        ┌─────────────────────────┬─────────────────────────┐
 │   SessionStore    │        │ ContextManagerAgent     │ ArtifactsAgent (AA)     │
 │ (SQLite / Memory) │        │ (CMA)                   │                         │
 └───────────────────┘        │ Resolve manuais, regras │ Processa SHAP/LIME,     │
                              │ e esquemas de sensores  │ predição e métricas     │
                              └───────────┬─────────────┴───────────┬─────────────┘
                                          │                         │
                                          └────────────┬────────────┘
                                                       │ 4. Evidências + Contexto
                                                       ▼
                                      ┌───────────────────────────────────┐
                                      │ ExplainerAgent (EA)               │
                                      │ Redige rascunho da narrativa      │
                                      └────────────────┬──────────────────┘
                                                       │
                                                       │ 5. Avalia fidelidade e alucinação
                                                       ▼
                                      ┌───────────────────────────────────┐
                      ┌───────────────┤ ValidatorAgent (VA - Quality Gate)│◄────────────┐
                      │               │ Calcula nota (0 a 10) e feedback  │             │
                      │               └────────────────┬──────────────────┘             │
                      │                                │                                │
                      │ [Nota < 7.0] Reprovado         │ [Nota >= 7.0] Aprovado         │
                      │ (Tentativas < 3)               │                                │
                      ▼                                ▼                                │
      ┌───────────────────────────────┐     ┌───────────────────────────────────┐       │
      │ Envia crítica e erros do VA   │     │ Salva no SessionStore e entrega   │       │
      │ de volta ao Explainer (EA)    │     │ ExplanationResponse final         │       │
      └───────────────┬───────────────┘     └─────────────────┬─────────────────┘       │
                      │                                       │                         │
                      └───────────────────────────────────────┴─────────────────────────┘
                                       Loop de Refinamento (Máx 3 vezes)
==========================================================================================================
```

### Passo a Passo Explicado:
1. **Disparo**: O desenvolvedor passa os dados de predição e XAI brutos para `layer.explain(...)`.
2. **Histórico**: O orquestrador busca mensagens anteriores daquela sessão no `SessionStore` (arquivo `.db`).
3. **Consulta Paralela**: O `ContextManagerAgent` (CMA) e o `ArtifactsAgent` (AA) extraem as evidências e regras técnicas em paralelo.
4. **Síntese**: O `ExplainerAgent` (EA) redige uma primeira proposta de explicação contextualizada.
5. **Quality Gate (Validador)**: O `ValidatorAgent` (VA) audita se a explicação condiz com os dados ou se houve alucinação, atribuindo uma nota de 0 a 10.
   - **Se nota >= 7.0**: Aprovado! Salva a mensagem no histórico e entrega ao usuário.
   - **Se nota < 7.0**: O validador envia as correções de volta para o EA reescrever (até 3 tentativas). Se esgotar as tentativas, ativa o fallback gracioso informando que a nota foi abaixo do ideal sem travar o sistema host.

---

### 1.2 Padrões de Integração da Camada: Modo Direto (Push) vs. Modo Baseado em Consulta (Pull / Query-Based)

Para alinhar com a tese formal do paper do SBES ("*query-based integration where agents retrieve information without intrusive modifications*") e ao mesmo tempo oferecer simplicidade máxima ("Developer is King"), o framework suporta dois padrões complementares de integração:

```text
==========================================================================================================
PADRÕES DE INTEGRAÇÃO DO FRAMEWORK EOA
==========================================================================================================

 [ PADRÃO A: MODO DIRETO (Push) - Simplicidade Imediata ]
 ┌────────────────────────────────────────────────────────┐
 │ Aplicação do Desenvolvedor (Backend / FastAPI)         │
 │ Já executou o modelo de ML e tem o SHAP em mãos        │
 └──────────────────────────┬─────────────────────────────┘
                            │ Passa os dados brutos: explain(prediction={...}, xai={...})
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ EOA Layer valida os dados e entrega aos Agentes        │
 └────────────────────────────────────────────────────────┘

 [ PADRÃO B: MODO BASEADO EM CONSULTA (Pull / Autônomo do Paper) ]
 ┌────────────────────────────────────────────────────────┐
 │ Aplicação do Desenvolvedor (Backend Industrial)        │
 │ Passa apenas o identificador: explain(incident_id=...) │
 └──────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────┴───────────────────────────────┐
            ▼                                               ▼
 ┌──────────────────────────────┐               ┌──────────────────────────────────┐
 │ ArtifactsAgent (AA)          │               │ ContextManagerAgent (CMA)        │
 │ Executa Tool/Query no        │               │ Executa Tool/Query na Engenharia │
 │ Analytics Layer e busca XAI  │               │ de Dados e busca schemas/manuais │
 └──────────────┬───────────────┘               └────────────────┬─────────────────┘
                │                                                │
                └───────────────────────┬────────────────────────┘
                                        │ Montam internamente os contratos normalizados:
                                        ▼ PredictionArtifact, XAIArtifact e DomainContext
 ┌────────────────────────────────────────────────────────┐
 │ ExplainerAgent (EA) e ValidatorAgent (VA)              │
 └────────────────────────────────────────────────────────┘
==========================================================================================================
```

- **A diferença é quem instancia o contrato**:
  - No Modo Direto: O desenvolvedor passa o dicionário e o EOA preenche o contrato.
  - No Modo Autônomo (Query-Based): Os agentes (AA e CMA) executam suas Tools/Consultas e eles mesmos preenchem o contrato antes de enviar ao Explainer.
- **Decisão Arquitetural**: A implementação das ferramentas de consulta autônoma será feita no **Épico 06 (Agentes e Tools)** e **Épico 07 (Orquestração)**.

---

### 1.3 Defesa Arquitetural: EOA vs. Chamada Direta de LLM (Por que não apenas Prompt Engineering?)

Uma pergunta frequente em revisões de Engenharia de Software (ICSE) é: *"Se o cliente já tem a predição e o XAI no modo direto, por que ele não faz apenas uma chamada com prompt formatado para o ChatGPT/Gemini?"*

A resposta técnica baseia-se em 4 pilares de Engenharia de Sistemas Críticos:

1. **Prevenção de Alucinações com Quality Gate Fechado (`ValidatorAgent`)**:
   Uma chamada crua de LLM é estocástica e não supervisionada. O modelo pode inventar causas mecânicas inexistentes com alta convicção. No EOA, a narrativa nunca é entregue sem passar pela auditoria do Validador, que aplica nota e aciona refinamento autônomo com feedback delimitado.
2. **Ponte Semântica de Domínio (`ContextManagerAgent`)**:
   Modelos de linguagem não conhecem siglas proprietárias de plantas industriais (ex.: `PS1`, `EPS1`). O CMA ingere esquemas e manuais de engenharia, resolvendo a correlação física sem inchar a janela de contexto.
3. **Desacoplamento Fail-Safe ("Unplugged")**:
   Falhas de rede, timeouts ou rate limits na API de LLM derrubariam o backend do cliente. O EOA garante degradação graciosa: entrega uma resposta de fallback determinística com os dados brutos e flag `fallback_used = True`, mantendo a operação da planta intacta.
4. **Rastreabilidade de Auditoria (Compliance Industrial)**:
   O `AgentStepTrace` e o `SessionStore` (SQLite local auditável) mantêm registro formal de cada agente, tempo de execução e justificativas técnicas, atendendo a exigências de conformidade regulatória.

---

## 2. Mapa Estrutural de Classes e Protocolos

#### A. Diagrama Formal de Classes (Mermaid)
```mermaid
classDiagram
    direction TB

    class LLM {
        <<Protocol (PEP 544)>>
        +generate(prompt: str, system_prompt: str, **kwargs) str
        +agenerate(prompt: str, system_prompt: str, **kwargs) str
    }

    class ContextRetriever {
        <<Protocol / Abstract>>
        +retrieve_context(query_or_keys: list~str~) dict
    }

    class StructuredContextRetriever {
        -schemas: dict
        -dictionary: dict
        +retrieve_context(query_or_keys: list~str~) dict
    }

    class CustomRAGContextRetriever {
        <<User Pluggable>>
        +retrieve_context(query_or_keys: list~str~) dict
    }

    ContextRetriever <|.. StructuredContextRetriever
    ContextRetriever <|.. CustomRAGContextRetriever

    class Tool {
        <<Abstract>>
        +name: str
        +description: str
        +parameters_schema: dict
        +execute(**kwargs) Any
    }

    class BaseAgent {
        <<Abstract>>
        +name: str
        +role: str
        +prompt_template: str | Path | Callable
        +tools: list~Tool~
        +run(state: AgentState, llm: LLM) AgentResult
    }

    class ContextManagerAgent {
        -retriever: ContextRetriever
        +set_retriever(retriever: ContextRetriever) None
        +run(state: AgentState, llm: LLM) AgentResult
    }

    class ArtifactsAgent {
        +run(state: AgentState, llm: LLM) AgentResult
    }

    class ExplainerAgent {
        +run(state: AgentState, llm: LLM) AgentResult
    }

    class ValidatorAgent {
        +threshold: float
        +max_refinements: int
        +run(state: AgentState, llm: LLM) ValidationResult
    }

    class SpecializedAgent {
        +domain_identifier: str
        +run(state: AgentState, llm: LLM) AgentResult
    }

    class SessionStore {
        <<Protocol>>
        +get_history(session_id: str) list~Message~
        +append_message(session_id: str, message: Message) None
        +clear(session_id: str) None
    }

    class InMemorySessionStore {
        -sessions: dict
    }

    class SQLiteSessionStore {
        -db_path: Path
        +export_dict(session_id: str) dict
        +log_validation(...) None
        +get_validation_logs(session_id: str) list
    }

    class ExplanationLayer {
        -orchestrator: OrchestratorEngine
        -agents: dict~str, BaseAgent~
        -session_store: SessionStore
        -llm: LLM
        +create_default_agents() None
        +register_agent(agent: BaseAgent) None
        +remove_agent(name: str) None
        +override_agent(name: str, agent: BaseAgent) None
        +explain(...) ExplanationResponse
        +create_session(session_id: str) ChatSession
    }

    BaseAgent <|-- ContextManagerAgent
    BaseAgent <|-- ArtifactsAgent
    BaseAgent <|-- ExplainerAgent
    BaseAgent <|-- ValidatorAgent
    BaseAgent <|-- SpecializedAgent

    ContextManagerAgent o-- ContextRetriever

    SessionStore <|.. InMemorySessionStore
    SessionStore <|.. SQLiteSessionStore

    ExplanationLayer o-- BaseAgent
    ExplanationLayer o-- SessionStore
    ExplanationLayer o-- LLM
    BaseAgent o-- Tool
```

#### B. Mapa Estrutural Rápido (Texto Monoespaçado)
```text
==========================================================================================================
MAPA DE CLASSES DO FRAMEWORK
==========================================================================================================

 [ Interfaces Base (Protocolos) ]
 ├── LLM                       -> Contrato de chamada ao LLM (PEP 544): generate(prompt) -> str
 ├── ContextRetriever          -> Contrato de busca de contexto: retrieve_context(keys) -> dict
 └── SessionStore              -> Contrato de persistência de chat: get_history(), append_message()

 [ Implementações de Armazenamento (SessionStore) ]
 ├── InMemorySessionStore      -> Guarda o histórico em RAM (ideal para testes rápidos)
 └── SQLiteSessionStore        -> Salva em arquivo .db local nativo (o desenvolvedor audita via SQL)

 [ Agentes (Herfam de BaseAgent) ]
 ├── ContextManagerAgent (CMA) -> Consulta manuais e esquemas operacionais
 ├── ArtifactsAgent (AA)       -> Processa SHAP, LIME e predições numéricas
 ├── ExplainerAgent (EA)       -> Sintetiza a narrativa final contextualizada em linguagem natural
 ├── ValidatorAgent (VA)       -> Quality Gate (LLM-as-a-Judge com scoring 0-10 e crítica de refinamento)
 └── SpecializedAgent (SA)     -> Agente extra plugado pelo desenvolvedor para subsistemas específicos

 [ Camadas de Orquestração e Fachada ]
 ├── OrchestratorEngine        -> Motor interno que executa o grafo assíncrono e o loop de validação
 └── ExplanationLayer          -> Ponto de entrada amigável para o dev (Fachada DX)
==========================================================================================================
```

---

## 2.1 Os 3 Modos de Conexão com LLM (Flexibilidade Total)

Para garantir que o desenvolvedor tenha controle total ("Developer is King"), o framework suporta 3 modos de inicialização do LLM:

1. **Modo 1: Credenciais Diretas (Chave + Modelo)**:
   - **Variação A (Explícita)**: O desenvolvedor passa chave e modelo no construtor.
   - **Variação B (Zero-Config / Lazy via Env)**: Se omitidos, o framework resolve automaticamente de `GEMINI_API_KEY` e `GEMINI_MODEL` (fallback padrão: `gemini-3.5-flash-lite`).
   ```python
   # Variação A: Chave e modelo explícitos
   layer = eoa.ExplanationLayer(provider="gemini", api_key="AIzaSy...", model="gemini-3.5-flash-lite")

   # Variação B: Leitura automática do ambiente (export GEMINI_API_KEY="...")
   layer = eoa.ExplanationLayer(provider="gemini")
   # ou instanciação zero-config direta:
   layer = eoa.ExplanationLayer()
   ```

2. **Modo 2: Cliente Pronto (Bring Your Own Client)**:
   - O desenvolvedor injeta um cliente já autenticado com proxy corporativo, headers customizados ou certificados.
   - O framework não inspeciona nem exige a chave de API, respeitando o encapsulamento do cliente fornecido.
   ```python
   import google.genai as genai
   my_client = genai.Client(api_key="...", http_options={"proxy": "http://proxy.local:8080"})
   layer = eoa.ExplanationLayer(llm=my_client)
   ```

3. **Modo 3: Função Customizada (Custom Callable)**:
   - Suporte universal para qualquer IA (modelos locais em vLLM/Ollama, gateways internos, OpenAI ou Anthropic).
   - Aceita funções síncronas (`def`) ou assíncronas (`async def`), com inspeção dinâmica de assinatura para `prompt` e `system_prompt`.
   ```python
   # Exemplo síncrono para gateway local ou Ollama
   def my_custom_llm(prompt: str, system_prompt: str | None = None) -> str:
       return internal_ai_gateway.predict(prompt)

   layer = eoa.ExplanationLayer(llm=my_custom_llm)
   ```

---

## 2.2 Estratégias de Armazenamento e Recuperação de Informação

O framework separa claramente duas necessidades distintas de dados:

### A. Sessão de Chat e Mensagens (`SessionStore`)
- **Finalidade**: Armazenar o histórico de mensagens multi-turn entre operador e agentes.
- **Implementação Principal (`SQLiteSessionStore`)**:
  - Salva em um arquivo `.db` local via biblioteca padrão do Python (`sqlite3`, sem dependências externas).
  - **O Desenvolvedor Decide**: O banco é dele! Ele pode inspecionar com qualquer cliente SQL, fazer queries analíticas, criar backups ou auditar o que o operador conversou.
- **Implementação Efêmera (`InMemorySessionStore`)**: Mantém em RAM para APIs stateless ou testes unitários.
- **Implementação Customizada**: O dev pode plugar Redis/PostgreSQL via protocolo `SessionStore`.

### B. Recuperação de Manuais e Documentos Grandes (`ContextRetriever` no CMA)
- **Finalidade**: Municiar o `ContextManagerAgent` (CMA) com informações operacionais, manuais técnicos e significado de variáveis.
- **Modo Padrão do Paper (Estruturado - Zero Vetores)**:
  - `StructuredContextRetriever`: Ingere schemas JSON, tabelas e resumos de manuais mapeados. Leve, determinístico e sem dependência de vetores.
- **Modo Adaptável para Manuais Gigantes (RAG / Busca Livre)**:
  - Se a fábrica ou empresa tiver um manual de 500 páginas, o desenvolvedor pode plugar um `ContextRetriever` customizado (ex: usando LangChain Retriever, LlamaIndex, BM25 ou Elasticsearch):
  ```python
  cma = layer.get_agent("CMA")
  cma.set_retriever(my_rag_retriever_for_500_page_manual)
  ```
- **Conclusão de Design**: O core permanece leve e cumpre o princípio científico do paper (sem forçar 500MB de banco vetorial no `pip install`), mas o desenvolvedor tem a liberdade de plugar RAG se o caso de uso dele exigir!

---

## 2.3 Sistema de Prompts e Agentes Especializados

### A. Diretório de Prompts Default e Versionamento
Para garantir que as instruções dos agentes sejam transparentes, auditáveis e facilmente inspecionadas:
- O pacote empacota um diretório nativo `src/eoa/prompts/` contendo arquivos markdown (`.md`):
  - `cma_default.md`: Instruções de resolução e mapeamento de contexto do CMA.
  - `aa_default.md`: Instruções de análise de métricas XAI e predições do AA.
  - `ea_default.md`: Instruções de síntese e tom narrativo do EA.
  - `va_default.md`: Instruções do Quality Gate e critérios de scoring do VA.

### B. Mecanismo de Sobrescrita ("Developer is King")
O desenvolvedor pode customizar qualquer prompt através do método `override_agent_prompt`:
1. **Via Arquivo Externo**: `layer.override_agent_prompt("EA", source="meus_prompts/custom_ea.md")` (ou `.txt`).
2. **Via String Direta**: `layer.override_agent_prompt("EA", source="Você é um engenheiro sênior...")`.
3. **Via Função Geradora (Callable)**: `layer.override_agent_prompt("EA", source=minha_funcao_de_prompt)`.

### C. Agentes Especializados (`SpecializedAgent`)
O framework suporta a injeção de agentes dedicados a subsistemas industriais específicos:
- Classe `SpecializedAgent` herda de `BaseAgent`.
- Registro no fluxo: `layer.register_specialized_agent(name="TurbinaSpecialist", role="Analista de Mancais", prompt_source="...")`.
- No ciclo de orquestração, o `OrchestratorEngine` avalia metadados e aciona os agentes especializados pertinentes ao subsistema do alerta.

---

## 3. Contratos de Dados (Pydantic v2)

### 3.1 `PredictionArtifact`
Representa o output do modelo preditivo analítico (Level 1).
```python
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class PredictionArtifact(BaseModel):
    model_name: Optional[str] = None
    target_variable: str
    predicted_value: Any
    confidence_or_probability: Optional[float] = None
    status_label: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 3.2 `XAIArtifact`
Representa os artefatos técnicos de explicabilidade bruta (Level 2).
```python
class XAIArtifact(BaseModel):
    method: str = "feature_importance"  # ex: SHAP, LIME, counterfactual, integrated_gradients
    feature_attributions: Dict[str, float] = Field(default_factory=dict)
    baseline_values: Optional[Dict[str, Any]] = None
    counterfactuals: Optional[list] = None
    raw_artifacts: Dict[str, Any] = Field(default_factory=dict)
```

### 3.3 `DomainContext`
Representa o contexto operacional e metadados de domínio (sem vetor denso, puramente estruturado).
```python
class DomainContext(BaseModel):
    variables: Dict[str, str] = Field(default_factory=dict)  # Ex: {"PS1": "Pressão"} ou {"FICO": "Score"}
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    documents: Dict[str, str] = Field(default_factory=dict)  # Ex: {"manual": "Procedimentos"}
    hierarchy: Dict[str, Any] = Field(default_factory=dict)  # Taxonomia de subsistemas
    schemas: Dict[str, Any] = Field(default_factory=dict)    # Schemas de datasets/fontes de dados
```

### 3.4 `ExplanationResponse`
Resultado final entregue à aplicação cliente.
```python
class AgentStepTrace(BaseModel):
    agent_name: str
    execution_time_ms: float
    summary: str
    status: str

class ExplanationResponse(BaseModel):
    narrative: str
    validation_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    validation_critique: Optional[str] = None
    attempts_used: int = 1
    fallback_used: bool = False
    execution_trace: list[AgentStepTrace] = Field(default_factory=list)
```

---

## 4. Estratégia de Fallback e Degradação Graciosa

Em conformidade com o princípio de sistema crítico do paper:
1. **Falha de Conectividade com LLM**: Se o LLM falhar por timeout, rate limit ou erro de rede, o método `explain` captura a exceção e retorna um fallback determinístico formatado contendo os dados brutos de predição e XAI com aviso explícito de indisponibilidade da camada narrativa.
2. **Falha de Qualidade (Score < Threshold após N tentativas)**: Se o `ValidatorAgent` rejeitar a explicação após o número máximo de tentativas configurado, o framework retorna a melhor narrativa obtida acompanhada dos alertas gerados pelo validador, com `validation_passed = False`, permitindo que a interface cliente alerte o operador sobre a ressalva técnica.
