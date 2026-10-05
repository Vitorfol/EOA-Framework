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
   - O desenvolvimento do Épico 06 (Agentes) **não depende** do Épico 03 (Conexões reais de LLM), porque os agentes consom o protocolo `LLMProvider` e são testados com o `MockLLMProvider` entregue no Épico 01.
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
  - [x] Implementar `MockLLM` (`MockLLMProvider`) em `tests/mocks/` para testes determinísticos rápidos sem gastar tokens nem depender de internet.
  - [x] Validar execução do `pytest` com sucesso (zero falhas).

---

### 📜 ÉPICO 02: Contratos de Dados & Schemas (Pydantic v2)
> **Objetivo**: Materializar os contratos formais de entrada e saída definidos no paper e na especificação técnica.

- [ ] **TASK-02.1: Modelos de Artefatos Analíticos e XAI (`src/eoa/contracts.py`)**
  - [ ] Implementar `PredictionArtifact` (modelo preditivo, target, probabilidade/score, status).
  - [ ] Implementar `XAIArtifact` (método, feature attributions/SHAP/LIME, baselines, metadados).
  - [ ] Testes unitários para serialização/desserialização e validação de tipos inválidos.
- [ ] **TASK-02.2: Modelos de Contexto e Resposta (`src/eoa/contracts.py`)**
  - [ ] Implementar `DomainContext` (dicionário de sensores, limites operacionais, snippets de manuais).
  - [ ] Implementar `ExplanationResponse` (narrativa, validation_score, validation_passed, execution_trace, fallback_used).
  - [ ] Implementar `AgentStepTrace` para auditoria e observabilidade passo a passo.
  - [ ] Testes unitários de validação e constraints de schema.

---

### 🔌 ÉPICO 03: Abstração de Conexão com LLM (Os 3 Modos)
> **Objetivo**: Permitir que o dev conecte qualquer IA sem dependências forçadas no core ("Developer is King").

- [ ] **TASK-03.1: Protocolo Base `LLM` (`src/eoa/llm/base.py`)**
  - [ ] Definir interface PEP 544 limpa com métodos assíncronos e síncronos: `generate(prompt, system_prompt=None, **kwargs)`.
  - [ ] Implementar adaptador `CallableLLMAdapter` (Modo 3: aceitar funções customizadas em Python `def fn(prompt) -> str`).
- [ ] **TASK-03.2: Adaptadores Nativos de Provedores (`src/eoa/llm/providers.py`)**
  - [ ] Implementar adaptador para Google Gemini (`GeminiProvider` com suporte a `google-genai`).
  - [ ] Implementar adaptador para OpenAI (`OpenAIProvider`).
  - [ ] Implementar fábrica `create_llm(...)` / `create_llm_provider(...)` para o Modo 1 (chave + modelo) e Modo 2 (cliente pronto).
  - [ ] Testes unitários com Mock e testes de injeção de dependência inválida com mensagens de erro educativas.

---

### 🧠 ÉPICO 04: Sistema de Memória e Histórico de Chat (`SessionStore`)
> **Objetivo**: Gestão de conversas multi-turn onde o desenvolvedor é dono dos dados sem dependência de banco vetorial no core.

- [ ] **TASK-04.1: Protocolo `SessionStore` e Implementação InMemory (`src/eoa/memory/`)**
  - [ ] Definir `SessionStore` (get_history, append_message, clear_session).
  - [ ] Implementar `InMemorySessionStore` para testes rápidos e APIs sem persistência em disco.
  - [ ] Testes unitários de append, ordenação temporal e expiração/limpeza.
- [ ] **TASK-04.2: Implementação Persistente `SQLiteSessionStore` (`src/eoa/memory/sqlite.py`)**
  - [ ] Implementar armazenamento relacional nativo usando `sqlite3` da biblioteca padrão do Python.
  - [ ] Criação automática das tabelas (`sessions`, `messages`, `validation_logs`).
  - [ ] Adicionar método utilitário para o dev inspecionar ou exportar dados em SQL/Dicionário.
  - [ ] Testes unitários de persistência real em arquivo `.db` temporário com reabertura de conexão.

---

### 📝 ÉPICO 05: Gerenciador de Prompts Nativos e Sobrescrita
> **Objetivo**: Prompts em arquivos Markdown limpos, versionados e 100% customizáveis pelo dev.

- [ ] **TASK-05.1: Criação dos Prompts Default (`src/eoa/prompts/*.md`)**
  - [ ] `cma_default.md`: Instruções para o Context Manager Agent mapear variáveis e manuais.
  - [ ] `aa_default.md`: Instruções para o Artifacts Agent interpretar relevância de sensores e métricas.
  - [ ] `ea_default.md`: Instruções para o Explainer Agent sintetizar narrativas acionáveis e claras.
  - [ ] `va_default.md`: Instruções para o Validator Agent auditar alucinações e calcular nota de fidelidade.
- [ ] **TASK-05.2: Motor de Carregamento e Sobrescrita (`src/eoa/prompts/manager.py`)**
  - [ ] Implementar `PromptManager` com suporte a:
    - Carregar de pacote embutido (defaults).
    - Sobrescrever apontando para arquivos `.md` ou `.txt` locais do dev.
    - Sobrescrever com strings Python diretas ou funções geradoras (callables).
    - Interpolação segura de variáveis de template (`{variavel}`).
  - [ ] Testes unitários de carregamento, sobrescrita e erro amigável se arquivo não for encontrado.

---

### 🤖 ÉPICO 06: Implementação dos Agentes Nucleares e Especializados
> **Objetivo**: Implementar as "caixas modulares" do paper com independência funcional.

- [ ] **TASK-06.1: Classes Base (`BaseAgent`, `Tool`, `AgentState`) (`src/eoa/agents/base.py`)**
  - [ ] Definir ciclo de vida do agente: `run(state: AgentState, llm: LLMProvider) -> AgentResult`.
  - [ ] Definir abstração limpa de `Tool` para agentes que usam ferramentas.
- [ ] **TASK-06.2: `ContextManagerAgent` (CMA) com `ContextRetriever` Plugável (`src/eoa/agents/context.py`)**
  - [ ] Implementar `StructuredContextRetriever` padrão (lê dicionários e schemas JSON).
  - [ ] Implementar hook `cma.set_retriever(...)` para permitir que o dev plugue busca vetorial/RAG para manuais de 500 páginas.
  - [ ] Testes unitários do CMA com recuperação estruturada e com mock de busca vetorial customizada.
- [ ] **TASK-06.3: `ArtifactsAgent` (AA) (`src/eoa/agents/artifacts.py`)**
  - [ ] Processar predições analíticas e importâncias XAI em resumos técnicos compreensíveis.
  - [ ] Testes unitários com fixtures de predições reais de manutenção preditiva.
- [ ] **TASK-06.4: `ExplainerAgent` (EA) (`src/eoa/agents/explainer.py`)**
  - [ ] Sintetizar evidências do AA + contexto do CMA + histórico de chat em narrativa em linguagem natural.
  - [ ] Testes unitários validando geração da narrativa contextualizada.
- [ ] **TASK-06.5: `ValidatorAgent` (VA) com Scoring e Refinamento (`src/eoa/agents/validator.py`)**
  - [ ] Implementar auditoria com LLM-as-a-Judge: avaliar fidelidade, alucinação e conformidade técnica (nota de 0 a 10).
  - [ ] Configuração de threshold (padrão 7.0) e geração de crítica estruturada para direcionar correções.
  - [ ] Testes unitários de aprovação (nota >= 7.0) e reprovação com sugestão de correção.
- [ ] **TASK-06.6: `SpecializedAgent` (SA) e Registro Dinâmico (`src/eoa/agents/specialized.py`)**
  - [ ] Implementar agente especialista genérico com suporte a domínio específico e ferramentas próprias.
  - [ ] Testes unitários de registro e execução de um especialista de exemplo (ex.: especialista em turbinas).

---

### ⚙️ ÉPICO 07: Engine de Orquestração, Grafo Leve e Fail-Safe
> **Objetivo**: Conectar os agentes em uma máquina de estados/grafo nativo em Python, sem dependências externas, com loop de validação e degradação graciosa.

- [ ] **TASK-07.1: Motor de Fluxo `OrchestratorEngine` (`src/eoa/engine/orchestrator.py`)**
  - [ ] Implementar fluxo de execução assíncrono: CMA e AA em paralelo -> EA -> VA.
  - [ ] Implementar loop de refinamento delimitado: se score < threshold, retroalimenta o EA com a crítica do VA até atingir o threshold ou o teto de tentativas (max_attempts, padrão 3).
- [ ] **TASK-07.2: Desacoplamento Fail-Safe e Degradação Graciosa ("Unplugged")**
  - [ ] Tratar exceções de rede/timeout/tokens esgotados: se o LLM ou validação falhar, o orchestrator monta uma resposta de fallback determinística com os dados brutos e flag `fallback_used = True`, sem jamais derrubar o host.
  - [ ] Testes unitários de estresse simulando falha forçada do LLM e validação do fallback seguro.

---

### 🎯 ÉPICO 08: Fachada Principal do Desenvolvedor (Developer Experience - DX)
> **Objetivo**: A interface amigável que faz o desenvolvedor dizer *"Nossa, quem fez isso realmente se importou com a facilidade de uso"*.

- [ ] **TASK-08.1: Implementação da `ExplanationLayer` (`src/eoa/layer.py` & `src/eoa/__init__.py`)**
  - [ ] Métodos limpos: `explain(...)`, `create_session(...)`, `register_agent(...)`, `override_agent_prompt(...)`, `remove_agent(...)`.
  - [ ] Função helper `eoa.default_layer(...)` para inicialização imediata em 1 linha.
  - [ ] Testes de ponta a ponta dos 6 cenários da jornada do desenvolvedor.
- [ ] **TASK-08.2: Ferramenta de Linha de Comando (CLI) (`src/eoa/cli.py`)**
  - [ ] Comando `eoa list --agents --prompts` para inspecionar agentes ativos e caminhos de prompts.
  - [ ] Comando `eoa init` para criar pasta padrão de prompts e arquivo de configuração no projeto do dev.
  - [ ] Testes da CLI.

---

### 🔬 ÉPICO 09: Casos de Uso Reais, Validação Empírica & Preparação para o ICSE
> **Objetivo**: Construir os experimentos e evidências científicas de software engineering exigidos pelo ICSE.

- [ ] **TASK-09.1: Reprodução do Caso Hidráulico do SBES como Exemplo Oficial**
  - [ ] Criar pasta `examples/predictive_maintenance/` reproduzindo os dados do dataset hidráulico da UCI (cenário da hidrelétrica do paper).
- [ ] **TASK-09.2: Construção de Segundo Caso de Uso Cross-Domain**
  - [ ] Criar exemplo em outro domínio crítico (ex.: diagnóstico médico ou risco financeiro) para comprovar a generalização arquitetural defendida no paper.
- [ ] **TASK-09.3: Instrumentação de Benchmarks de Engenharia de Software**
  - [ ] Medir overhead computacional da engine (latência em ms do framework excluindo a chamada do LLM).
  - [ ] Medir eficácia do loop do ValidatorAgent (taxa de alucinações corrigidas vs. tentativas).
  - [ ] Coleta de dados para a seção de avaliação empírica do paper do ICSE.
