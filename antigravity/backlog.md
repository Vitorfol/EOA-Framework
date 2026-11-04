# EOA Framework - Backlog de Desenvolvimento & Git Workflow

Este documento serve como o **Quadro de Tarefas (Board Vivo)** do projeto **EOA (Explanation-Oriented Agents)**, organizado para permitir que a equipe (você, o outro desenvolvedor e os agentes) acompanhe o progresso de forma modular, transparente e dividida em fases claras.

---

## 🌿 1. Modelo de Git e Fluxo de Trabalho (Branching Strategy)

Adotamos um fluxo baseado em **Feature Branching** com foco em estabilidade e rastreabilidade:

```text
========================================================================================
FLUXO DE TRABALHO GIT (PASSO A PASSO VISUAL)
========================================================================================

  [ master ]  ============================================●===> [ Release v0.1.0 Estável ]
                                   ^                      ^
                                   | (Merge estável)      |
  [ develop ] ---------------------●----------------------●---> [ Integração Contínua ]
                     \                ^             /
                      \ (Cria branch) | (Abre PR)  / (Cria próxima feature)
                       ▼              |           ▼
  [ feature/* ]         ●─────────────●            ●─────────────●
                        (Implementa feature       (Próxima tarefa
                         + testes unitários)       do backlog)
========================================================================================
```

### Regras do Fluxo:
1. **`master`**: Código em estado de release / publicação no PyPI. Somente código testado e 100% estável entra aqui.
2. **`develop`**: Branch principal de trabalho diário. Todo desenvolvimento concluído converge para cá via Pull Request (PR).
3. **`feature/<id>-<nome-curto>`**: Branches de trabalho criadas a partir de `develop` (ex.: `feature/EP01-setup-and-packaging`).
4. **Definição de Pronto (Definition of Done - DoD)**:
   - Uma tarefa só é considerada concluída (`[x]`) se o código estiver implementado, com **testes unitários correspondentes passando**, tipagem estrita (type hints) e docstrings completas. Teste não é deixado para o final; é entregue junto com a feature!

---

## 🗺️ 2. Referências de Arquitetura Existentes no Repositório

Para não construir nada no escuro, as tarefas deste backlog respeitam estritamente as especificações já homologadas:
- **Diagramas de Fluxo e Classes**: [technical_specification.md](technical_specification.md)
- **Jornada do Desenvolvedor (Cenários Práticos)**: [user_journey_and_usage_scenarios.md](user_journey_and_usage_scenarios.md)
- **Glossário e Nomes Oficiais**: [glossary.md](glossary.md)
- **Requisitos Formais**: [requirements.txt](requirements.txt)

---

## 🔀 2.1 Matriz de Dependências Técnicas (O que é Bloqueante vs. O que é Destravado)

Abaixo está o mapa de dependências entre os épicos, deixando explícito o que precisa ser feito primeiro e o que pode ser desenvolvido de forma totalmente independente:

```text
========================================================================================================
MATRIZ DE DEPENDÊNCIAS DO PROJETO (BLOQUEANTE vs. DESTRAVADO)
========================================================================================================

 [ FASE 0: FUNDAÇÃO BLOQUEANTE ] (Pré-requisito obrigatório para destravar todo o restante)
 ┌────────────────────────────────────────────────────────┐
 │ ÉPICO 01: Setup, Packaging (pyproject.toml) & MockLLM  │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ ÉPICO 02: Contratos de Dados (Pydantic v2 Schemas)     │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            │ (Schemas congelados e MockLLM disponível -> Destrava a Fase 1)
                            ▼
 [ FASE 1: MÓDULOS INDEPENDENTES / DESTRAVADOS ] (Podem ser feitos em paralelo ou em qualquer ordem)
 ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
 │ ÉPICO 03: Conexões LLM       │ ÉPICO 04: Memória & Chat     │ ÉPICO 05 & 06: Prompts e     │
 │ - Adaptador Callable         │ - InMemorySessionStore       │ Agentes                      │
 │ - Adaptador Gemini           │ - SQLiteSessionStore (.db)   │ - Prompts Markdown (.md)     │
 │ - Adaptador OpenAI           │                              │ - CMA, AA, EA, VA            │
 │                              │                              │ - SpecializedAgent           │
 │                              │                              │ *(Agentes usam MockLLM p/    │
 │                              │                              │  testar sem depender do Ep03)*│
 └──────────────┬───────────────┴──────────────┬───────────────┴──────────────┬───────────────┘
                │                              │                              │
                └──────────────────────────────┼──────────────────────────────┘
                                               │ (Módulos prontos -> Destrava a integração)
                                               ▼
 [ FASE 2: INTEGRAÇÃO E EXPERIÊNCIA DO DESENVOLVEDOR ] (Depende dos módulos da Fase 1)
 ┌────────────────────────────────────────────────────────┐
 │ ÉPICO 07: Engine de Orquestração & Fail-Safe           │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ ÉPICO 08: Fachada Principal (ExplanationLayer) & CLI   │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 [ FASE 3: VALIDAÇÃO CIENTÍFICA & ICSE ]
 ┌────────────────────────────────────────────────────────┐
 │ ÉPICO 09: Casos Reais (Hidráulico + 2º Domínio) & ICSE │
 └────────────────────────────────────────────────────────┘
========================================================================================================
```

### 📌 Resumo Técnico de Travamento:
1. **O que é estritamente bloqueante?**
   - **Épicos 01 e 02**: Definem a estrutura de imports e os schemas de dados (`PredictionArtifact`, `XAIArtifact`, `DomainContext`, `ExplanationResponse`). Sem eles, nenhum módulo compila ou roda testes.
2. **O que fica destravado em seguida?**
   - **Épico 03 (LLM)**, **Épico 04 (Memória)**, **Épico 05 (Prompts)** e **Épico 06 (Agentes)** são **100% independentes entre si**.
    - O desenvolvimento do Épico 06 (Agentes) **não depende** do Épico 03 (Conexões reais de LLM), porque os agentes consom o protocolo `LLM` e são testados com o `MockLLM` entregue no Épico 01.
3. **O que depende de quem para integração?**
   - O **Épico 07 (Orchestrator)** integra os Agentes (Ep 06) com a Memória (Ep 04) e o LLM (Ep 03).
   - O **Épico 08 (ExplanationLayer)** expõe a API final amigável ao usuário.

---

### Legenda de Status
- `[ ]` Não Iniciado (To Do)
- `[/]` Em Progresso (In Progress)
- `[x]` Concluído (Done)
- `[!]` Bloqueado / Atenção (Blocked)

---

### 📦 ÉPICO 01: Setup do Projeto, Empacotamento e Infraestrutura de Testes
> **Objetivo**: Estabelecer a estrutura física de arquivos, packaging moderno via PEP 621 e pipeline local de testes.

- [x] **TASK-01.1: Configuração do `pyproject.toml`**
  - [x] Declarar metadados do pacote `eoa` com build backend `hatchling`.
  - [x] Declarar dependência estrita do core: `pydantic >= 2.0`.
  - [x] Declarar grupos opcionais de extras: `[gemini]`, `[openai]`, `[cli]`, `[test]`, `[all]`.
  - [x] Configurar ferramentas de lint/teste (`pytest`, `ruff` ou `black`/`flake8`).
- [x] **TASK-01.2: Inicialização da Estrutura Física de Diretórios**
  - [x] Criar estrutura `src/eoa/` com todos os submódulos vazios (`agents/`, `llm/`, `memory/`, `engine/`, `prompts/`).
  - [x] Criar pasta `tests/` com `conftest.py` contendo fixtures básicas.
- [x] **TASK-01.3: Pipeline de Testes e Mock Provider**
  - [x] Implementar `MockLLM` em `tests/mocks/` para testes determinísticos rápidos sem gastar tokens nem depender de internet.
  - [x] Validar execução do `pytest` com sucesso (zero falhas).

---

### 📜 ÉPICO 02: Contratos de Dados & Schemas (Pydantic v2)
> **Objetivo**: Materializar os contratos formais de entrada e saída definidos no paper e na especificação técnica.

- [x] **TASK-02.1: Modelos de Artefatos Analíticos e XAI (`src/eoa/contracts.py`)**
  - [x] Implementar `PredictionArtifact` (modelo preditivo, target, probabilidade/score, status).
  - [x] Implementar `XAIArtifact` (método, feature attributions/SHAP/LIME, baselines, metadados).
  - [x] Testes unitários para serialização/desserialização e validação de tipos inválidos.
- [x] **TASK-02.2: Modelos de Contexto e Resposta (`src/eoa/contracts.py`)**
  - [x] Implementar `DomainContext` (dicionário de sensores, limites operacionais, snippets de manuais).
  - [x] Implementar `ExplanationResponse` (narrativa, validation_score, validation_passed, execution_trace, fallback_used).
  - [x] Implementar `AgentStepTrace` para auditoria e observabilidade passo a passo.
  - [x] Testes unitários de validação e constraints de schema.

---

### 🔌 ÉPICO 03: Abstração de Conexão com LLM (Os 3 Modos)
> **Objetivo**: Permitir que o dev conecte qualquer IA sem dependências forçadas no core ("Developer is King").

- [x] **TASK-03.1: Protocolo Base `LLM` (`src/eoa/llm/base.py`)**
  - [x] Definir interface PEP 544 limpa com métodos assíncronos e síncronos: `generate(prompt, system_prompt=None, **kwargs)` e `agenerate(...)`.
  - [x] Implementar adaptador `CallableLLMAdapter` (Modo 3: aceitar funções customizadas em Python `def fn(prompt) -> str` ou async `async def fn(...) -> str`).
- [x] **TASK-03.2: Adaptadores Nativos de Provedores (`src/eoa/llm/gemini.py` & `factory.py`)**
  - [x] Implementar adaptador para Google Gemini (`GeminiProvider` com suporte a `google-genai`).
  - [x] Implementar fábrica `create_llm(...)` unificada para Modo 1 (chave + modelo), Modo 2 (cliente pronto) e Modo 3 (callables).
  - [x] Suporte a fallback e configuração flexível de modelo (`gemini-2.5-flash`, `gemini-3.8-flash`, etc.).
  - [x] Testes unitários com Mock e testes de injeção de dependência inválida com mensagens de erro educativas.
  - [x] Teste de integração assíncrono real via `GEMINI_API_KEY` opcional.

---

### 🧠 ÉPICO 04: Sistema de Memória e Histórico de Chat (`SessionStore`)
> **Objetivo**: Gestão de conversas multi-turn onde o desenvolvedor é dono dos dados sem dependência de banco vetorial no core.

- [x] **TASK-04.1: Protocolo `SessionStore` e Implementação InMemory (`src/eoa/memory/`)**
  - [x] Definir `SessionStore` (get_history, append_message, clear_session).
  - [x] Implementar `InMemorySessionStore` para testes rápidos e APIs sem persistência em disco.
  - [x] Testes unitários de append, ordenação temporal e expiração/limpeza.
- [x] **TASK-04.2: Implementação Persistente `SQLiteSessionStore` (`src/eoa/memory/sqlite.py`)**
  - [x] Implementar armazenamento relacional nativo usando `sqlite3` da biblioteca padrão do Python.
  - [x] Criação automática das tabelas (`sessions`, `messages`, `validation_logs`).
  - [x] Adicionar método utilitário para o dev inspecionar ou exportar dados em SQL/Dicionário.
  - [x] Testes unitários de persistência real em arquivo `.db` temporário com reabertura de conexão.

---

### 📝 ÉPICO 05: Gerenciador de Prompts Nativos e Sobrescrita
> **Objetivo**: Prompts em arquivos Markdown limpos, versionados e 100% customizáveis pelo dev.

- [x] **TASK-05.1: Criação dos Prompts Default (`src/eoa/prompts/*.md`)**
  - [x] Formato PTCF (Persona, Task, Context, Format) com placeholders `{variavel}`; nomes completos como chave canônica.
  - [x] `context_manager_default.md`: Instruções para o Context Manager Agent mapear variáveis e manuais.
  - [x] `artifacts_default.md`: Instruções para o Artifacts Agent interpretar relevância de sensores e métricas.
  - [x] `explainer_default.md`: Instruções para o Explainer Agent sintetizar narrativas acionáveis e claras.
  - [x] `validator_default.md`: Instruções para o Validator Agent auditar alucinações e calcular nota de fidelidade.
- [x] **TASK-05.2: Motor de Carregamento e Sobrescrita (`src/eoa/prompts/manager.py`)**
  - [x] Implementar `PromptManager` (`resolve` + `render`) com suporte a:
    - Carregar de pacote embutido (defaults) quando `source=None`.
    - Fonte explícita: `str` (texto do prompt) ou `Path` (arquivo `.md` local). Sem `callable`, sem `.txt`.
    - Interpolação segura de variáveis de template (`{variavel}`), preservando chaves JSON.
  - [x] Customização por um único caminho: `prompt=` no construtor do agente (consumido no EP06).
  - [x] Testes unitários de carregamento, fontes, interpolação e erro amigável se arquivo não for encontrado.

---

### 🤖 ÉPICO 06: Implementação dos Agentes Nucleares e Especializados
> **Objetivo**: Implementar as "caixas modulares" do paper com independência funcional.

- [x] **TASK-06.1: Classes Base (`BaseAgent`, `Tool`, `AgentState`) (`src/eoa/agents/base.py`)**
  - [x] Definir ciclo de vida do agente: `run(state: AgentState, llm: LLM) -> AgentResult` e `arun(...)` assíncrono.
  - [x] Implementar telemetria automática (`execution_time_ms`) e rastreabilidade (`AgentStepTrace`).
  - [x] Definir abstração limpa de `Tool` (`src/eoa/tools/base.py`) com introspecção de assinatura via `inspect.signature` e suporte síncrono/assíncrono.
- [x] **TASK-06.2: `ContextManagerAgent` (CMA) com `ContextRetriever` Plugável (`src/eoa/agents/context.py`)**
  - [x] Implementar `StructuredContextRetriever` padrão (lê dicionários e schemas JSON).
  - [x] Implementar hook `cma.set_retriever(...)` com protocolo PEP 544 permitindo busca vetorial/RAG customizada ou callables Python.
  - [x] Suporte à recuperação autônoma via `tools` casando com `query_params`.
  - [x] Testes unitários do CMA com recuperação estruturada e com mock de busca vetorial customizada.
- [x] **TASK-06.3: `ArtifactsAgent` (AA) (`src/eoa/agents/artifacts.py`)**
  - [x] Processar predições analíticas e importâncias XAI em resumos técnicos compreensíveis.
  - [x] Otimização Zero ML / Progressive Ramp: bypass imediato do LLM quando não houver predição, economizando 100% dos tokens de inferência.
  - [x] Suporte a ferramentas (tools) para recuperação autônoma de predições e XAI via `query_params` flexíveis e agnósticos.
  - [x] Testes unitários com fixtures de predições reais de manutenção preditiva e modo Zero ML.
- [x] **TASK-06.4: `ExplainerAgent` (EA) (`src/eoa/agents/explainer.py`)**
  - [x] Sintetizar evidências do AA + contexto do CMA + especialistas + histórico multi-turn de chat (`ChatMessage`) em narrativa clara.
  - [x] Incorporar crítica anterior do `ValidatorAgent` em loops de refinamento.
  - [x] Testes unitários validando geração da narrativa contextualizada com histórico.
- [x] **TASK-06.5: `ValidatorAgent` (VA) com Scoring e Refinamento (`src/eoa/agents/validator.py`)**
  - [x] Implementar auditoria com LLM-as-a-Judge: avaliar fidelidade, alucinação e conformidade técnica (nota de 0 a 10).
  - [x] Parser resiliente a blocos Markdown de código JSON (```json ... ```) e validação matemática de notas.
  - [x] Configuração dinâmica de `threshold` (padrão 7.0, validado entre 0.0 e 10.0) e geração de `ValidationResult` estruturado.
  - [x] Testes unitários de aprovação (nota >= 7.0) e reprovação com sugestão de correção.
- [x] **TASK-06.6: `SpecializedAgent` (SA) e Registro Dinâmico (`src/eoa/agents/specialized.py`)**
  - [x] Implementar agente especialista genérico com suporte a domínio específico e ferramentas próprias.
  - [x] Registro automático de insights no estado compartilhado `state.specialist_insights[self.name]`.
  - [x] Testes unitários de registro e execução de especialista de exemplo (ex.: especialista em turbinas).

---

### ⚙️ ÉPICO 07: Engine de Orquestração, Grafo Leve e Fail-Safe
> **Objetivo**: Conectar os agentes em uma máquina de estados/grafo nativo em Python, sem dependências externas, com loop de validação e degradação graciosa.

- [x] **TASK-07.1: Motor de Fluxo `OrchestratorEngine` (`src/eoa/engine/orchestrator.py`)**
  - [x] Implementar fluxo de execução assíncrono e síncrono: CMA, AA e Especialistas em paralelo -> EA -> VA.
  - [x] Suporte nativo aos 3 modos de operação (Push, Pull com hidratação por tools e Zero ML inicial) e sessões multi-turn persistentes ou stateless (`session_id` no `ExplanationResponse`).
  - [x] Implementar loop de refinamento delimitado: se score < threshold, retroalimenta o EA com a crítica do VA até atingir o threshold ou o teto de tentativas (max_attempts, padrão 3).
  - [x] Observabilidade estruturada com `logging.getLogger("eoa.orchestrator")` e consolidação de `AgentStepTrace`.
- [x] **TASK-07.2: Desacoplamento Fail-Safe e Degradação Graciosa ("Unplugged")**
  - [x] Tratar exceções de qualquer provedor LLM (Gemini, OpenAI, callables locais) ou falha de formato (`parse_error`): o orchestrator monta uma resposta de fallback determinística com os dados brutos e flag `fallback_used = True`, sem jamais derrubar o host.
  - [x] Testes unitários e de integração (`tests/test_orchestrator.py` e `tests/test_orchestrator_live.py`) simulando falha forçada do LLM, sessão progressiva (Zero ML -> Pull -> Push) e validação do fallback seguro.

---

### 🎯 ÉPICO 08: Fachada Principal do Desenvolvedor (Developer Experience - DX)
> **Objetivo**: A interface amigável que faz o desenvolvedor dizer *"Nossa, quem fez isso realmente se importou com a facilidade de uso"*.

- [x] **TASK-08.1: Implementação da `ExplanationLayer` (`src/eoa/layer.py` & `src/eoa/__init__.py`)**
  - [x] Métodos limpos: `explain(...)`, `aexplain(...)`, `create_default_agents(...)`, `register_agent(...)`, `override_agent_prompt(...)`, `reset_agent_prompt(...)`, `get_agent(...)`, `remove_agent(...)`, `get_history(...)`, `clear_session(...)`, `inspect()`.
  - [x] Suporte unificado aos modos Direto (Push), Autônomo (Pull via incident_id/query_params ou seleção/extração autônoma via LLM reasoning), Zero ML e Multi-Turn via `session_id` com cache de artefatos da sessão e gravação em `validation_logs` (SQLite).
  - [x] Carregamento desacoplado de `.env`: mantendo o core ultra-leve e Twelve-Factor (consultando `os.environ`), com leitura configurada em `tests/conftest.py` e exemplos interativos.
  - [x] Factory em 1 linha via `@classmethod ExplanationLayer.default_layer(...)` nos 3 modos de LLM.
  - [x] Testes de ponta a ponta determinísticos (`tests/test_layer.py`) e live com Gemini (`tests/test_layer_live.py`).
  - [x] Notebooks interativos separados em Mock/Offline (`examples/01_walkthrough_mock.ipynb`), Live Gemini (`examples/02_walkthrough_live_gemini.ipynb`) e Guia de Referência Completo da API (`examples/README.md`).
- [x] **TASK-08.2: Polimento da API Pública e Documentação da DX**
  - [x] Métodos de conveniência na camada (`layer.inspect()`, `layer.reset_agent_prompt(...)`, `layer.get_history(...)`, `layer.clear_session(...)`).
  - [x] Guia completo de referência da API pública em `examples/README.md`.

---

### 🔬 ÉPICO 09: Casos de Uso Reais, Validação Empírica & Preparação para o Artigo
> **Objetivo**: Construir os experimentos e evidências científicas de engenharia de software exigidos para o artigo (JSS / ICSE) e registro de software.

- [ ] **TASK-09.1: Reprodução do Caso Hidráulico do SBES como Exemplo Oficial**
  - [ ] Criar pasta `examples/predictive_maintenance/` reproduzindo os dados do dataset hidráulico da UCI (cenário da hidrelétrica do paper).
- [ ] **TASK-09.2: Construção de Segundo Caso de Uso Cross-Domain**
  - [ ] Criar exemplo em outro domínio crítico (ex.: diagnóstico médico ou risco financeiro) para comprovar a generalização arquitetural defendida no paper.
- [ ] **TASK-09.3: Instrumentação de Benchmarks de Engenharia de Software**
  - [ ] Medir overhead computacional da engine (latência em ms do framework excluindo a chamada do LLM).
  - [ ] Medir eficácia do loop do ValidatorAgent (taxa de alucinações corrigidas vs. tentativas).
  - [ ] Coleta de dados para a seção de avaliação empírica do artigo.

---

## 🚀 3. Roadmap Pós-v0.1.0 (Evoluções Futuras Mapeadas para v0.2.0)

Itens deliberadamente fora do escopo da `v0.1.0` para manter o core enxuto, estável e focado na validação científica e registro de software:

- [ ] **Edição e Customização Controlada do Grafo de Execução (Custom DAG)**:
  - Permitir que o desenvolvedor altere a topologia de execução entre agentes (encadeamento sequencial entre especialistas, hooks condicionais e grafos customizados).
- [ ] **Agente Roteador / Dispatcher Central Opcional**:
  - Opção de configurar um agente planejador inicial para selecionar seletivamente quais especialistas e ferramentas ativar em sistemas com dezenas de agentes.
- [ ] **CLI Externa de Linha de Comando (`eoa`)**:
  - Ferramenta de terminal (`eoa inspect`, `eoa init`) postergada para manter a `v0.1.0` focada 100% na experiência programática em Python (`layer.inspect()`).
