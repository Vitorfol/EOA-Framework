# EOA Framework - Glossário Oficial de Nomes, Conceitos e Arquitetura

Este documento estabelece a nomenclatura padronizada do framework **EOA**, definindo exatamente o papel e a responsabilidade de cada componente para evitar qualquer ambiguidade durante o desenvolvimento e na redação de artigos (SBES / ICSE).

---

## 1. Nomenclatura dos Componentes Centrais

| Nome da Classe / Componente | O que é de fato? (Propósito Prático) | Por que esse nome? |
| :--- | :--- | :--- |
| **`ExplanationLayer`** | A classe principal (o ponto de entrada do framework). É a camada que o usuário instancia, configura seus agentes e chama para obter explicações. | Representa a *Explanation Agent Layer* formalizada no artigo do SBES. |
| **`OrchestratorEngine`** | O motor interno de execução. Ele gerencia o fluxo de controle, a ordem em que os agentes são chamados, o histórico de mensagens e o loop de validação com o validador. | Atua como o "maestro" invisível que conecta os agentes sem que o usuário precise gerenciar estados manuais. |
| **`ContextManagerAgent` (CMA)** | O agente responsável por descobrir e fornecer o **contexto operacional** (o que significa a sigla de um sensor, limites operacionais normais, manuais técnicos ou documentos de domínio). | Nome canônico do paper: *Context Manager*. Gerencia o contexto do problema. |
| **`ArtifactsAgent` (AA)** | O agente responsável por digerir e interpretar os **artefatos técnicos** de IA (predição, probabilidade, importâncias SHAP/LIME, valores de sensores atuais). | Nome canônico do paper: *Artifacts Agent*. Trata os artefatos de predição e XAI. |
| **`ExplainerAgent` (EA)** | O agente que recebe as evidências do AA e o contexto do CMA para **escrever a narrativa final em linguagem natural**, explicando o alerta e sugerindo ações recomendadas. | Nome canônico do paper: *Explainer Agent*. É o sintetizador da explicação. |
| **`ValidatorAgent` (VA)** | O agente que atua como **portão de qualidade (Quality Gate)**. Ele audita a explicação gerada pelo EA comparando-a com as evidências técnicas do AA e do CMA, aplicando uma nota e apontando erros/alucinações. | Nome canônico do paper: *Validator Agent*. Garante a fidelidade técnica. |
| **`SpecializedAgent` (SA)** | Classe base para qualquer agente extra que o desenvolvedor queira criar para um subsistema específico (ex: especialista em vibração de mancais, especialista em trocadores de calor). | Nome canônico do paper: *Specialized Agents*. |

---

## 2. Nomenclatura de Conexão com LLM (Os 3 Modos de Conexão)

O framework adota o padrão **"Developer in Control"** e suporta exatamente 3 modos transparentes de conexão:

| Modo | Como o dev usa? | Quando usar? |
| :--- | :--- | :--- |
| **Modo 1: Chave + Modelo**<br>*(Direct Credentials)* | `eoa.ExplanationLayer(provider="gemini", api_key="...", model="gemini-2.5-flash")` | Início rápido, prototipagem, scripts diretos. O EOA cria a conexão internamente. |
| **Modo 2: Cliente Pronto**<br>*(Bring Your Own Client)* | `eoa.ExplanationLayer(llm=my_configured_client)` | Quando a empresa já tem um cliente com proxy corporativo, certificados TLS de rede industrial, LiteLLM ou headers customizados. |
| **Modo 3: Função Própria**<br>*(Custom Callable)* | `eoa.ExplanationLayer(llm=my_custom_function)` | Quando o modelo roda local (vLLM, TGI, Ollama) ou passa por um pipeline interno proprietário: `def my_fn(prompt: str) -> str`. |

---

## 3. Nomenclatura de Dados e Armazenamento

### 3.1 Sessões e Histórico de Chat (`SessionStore`)
- **O que é?**: O mecanismo que guarda as mensagens trocadas entre o operador humano e o sistema ao longo do tempo (diálogo multi-turn).
- **Abordagem padrão em arquivo `.db` (`SQLiteSessionStore`)**:
  - Salva em um arquivo SQLite local (ex: `eoa_sessions.db` ou caminho escolhido pelo dev).
  - **O desenvolvedor decide o que fazer com ele**: pode consultar com SQL puro (`SELECT * FROM messages WHERE session_id = 'alerta_82'`), exportar para relatórios de auditoria da fábrica, fazer backup ou apagar.
- **Alternativa em memória (`InMemorySessionStore`)**: Para quem quer rodar testes ou APIs REST com sessão efêmera sem gerar arquivos no disco.
- **Alternativa corporativa**: O dev pode criar sua própria classe implementando a interface `SessionStore` para salvar em PostgreSQL, Redis, MongoDB, etc.

---

### 3.2 Busca em Documentos Grandes e o Papel do CMA (`ContextRetriever`)
- **O que é?**: O mecanismo pelo qual o `ContextManagerAgent` (CMA) obtém manuais, diagramas e regras operacionais.
- **Abordagem Canônica do Paper (Estruturada / Zero Vetores)**:
  - `StructuredContextRetriever`: Lê esquemas JSON, dicionários de variáveis e resumos de manuais já mapeados por tags/chaves. É determinístico, leve e não requer bibliotecas externas.
- **Abordagem Adaptável para Manuais Grandes (RAG Opcional)**:
  - O paper original evitou RAG denso no core para provar a tese de contratos estruturados. Porém, seguindo a filosofia *"O usuário é rei / quase tudo é adaptável"*, o EOA **permite que o desenvolvedor plugue sua própria busca/RAG no CMA** se ele tiver manuais de 500 páginas!
  - Interface `ContextRetriever`:
    ```python
    # O dev pode passar sua própria função de busca ou indexador RAG se quiser:
    cma.set_retriever(my_custom_rag_retriever)
    ```
  - **Benefício de Engenharia**: O core do `eoa` continua pesando apenas poucos megabytes (sem ChromaDB, FAISS ou PyTorch forçados), mas quem precisa de busca vetorial em manuais longos tem a porta 100% aberta e documentada para plugar.
