# 📁 Estrutura do Projeto - PoC Agente de Explicabilidade

## Visão Geral da Arquitetura

```
PoC/
│
├── � layers/                        # Camadas da aplicação
│   │
│   ├── 📊 data/                      # CAMADA 1: Exploração de Dados
│   │   ├── notebooks/
│   │   │   └── data_explore.ipynb    # Notebook de análise exploratória
│   │   └── README.md
│   │
│   ├── 🤖 training/                  # CAMADA 2: Treinamento de Modelos
│   │   ├── explainers/
│   │   │   └── shap_explainer.py     # Cómputo de SHAP values e feature importance
│   │   ├── models/
│   │   │   └── MODELS.md             # Documentação de modelos planejados
│   │   ├── results/                  # Modelos e métricas (gerado)
│   │   ├── config.yaml               # Hiperparâmetros (placeholders)
│   │   ├── train.py                  # Script principal de treinamento
│   │   └── README.md
│   │
│   ├── 🧠 agent/                     # CAMADA 3: Explicabilidade (XAI + Gemini)
│   │   ├── prompts/
│   │   │   ├── system_prompt.md      # Prompt para geração de explicações
│   │   │   └── judge_prompt.md       # Prompt para LLM as a Judge
│   │   ├── gemini_adapter.py         # Google Gemini API adapter
│   │   ├── agent.py                  # Orchestrator principal
│   │   ├── judge.py                  # LLM as a Judge (avaliação de qualidade)
│   │   └── README.md
│   │
│   └── 🖥️  view/                     # CAMADA 4: Interface CLI
│       ├── cli.py                    # Lógica de interface/interação
│       ├── renderer.py               # Formatação visual com Rich
│       └── README.md
│
├── 📁 datasets/                      # Datasets (não versionados)
│   └── ...
│
├── 📁 picking_dataset/               # Análises de datasets candidatos
│   └── ...
│
├── 📄 Arquivos de Configuração
│   ├── requirements.txt              # Dependências Python
│   ├── .env.example                  # Template de variáveis de ambiente
│   ├── .gitignore                    # Arquivos ignorados pelo Git
│   ├── context.md                    # Especificação original do PoC
│   ├── readme.md                     # Documentação principal
│   └── structure.md                  # Este arquivo
│
└── 🐍 Ambiente Virtual (a criar)
    └── venv/                         # Ambiente isolado Python
```

## Fluxo de Dados

```
┌─────────────┐           ┌──────────────┐           ┌─────────────┐           ┌──────────┐
│ layers/     │  dataset  │ layers/        │  models   │ layers/     │  results  │ layers/  │
│ data/       │  ───────> │ training/      │  ───────> │ agent/      │  ───────> │ view/    │
│ (explore)   │   clean   │ (train)        │  predict  │ (explain)   │   format  │ (show)   │
└─────────────┘           └──────────────┘           └─────────────┘           └──────────┘
      │                          │                          │                        │
      │                          │                          │                        │
   Jupyter                     CLI                        CLI                      CLI
  Notebook                 train.py                    agent.py                  cli.py
```

## Responsabilidades por Camada

### 🔹 Camada 1: `layers/data/`
**Input**: Dados brutos (CSV, Parquet, etc)  
**Processo**: Exploração, validação, limpeza  
**Output**: Dataset limpo + documentação  
**Ferramentas**: Jupyter, pandas, matplotlib, seaborn  

### 🔹 Camada 2: `layers/training/`
**Input**: Dataset limpo  
**Processo**: Treino (XGBoost, Random Forest), avaliação, **cálculo de SHAP values**  
**Output**: Modelos serializados + métricas + **SHAP values salvos**  
**Ferramentas**: scikit-learn, XGBoost, SHAP, joblib  

### 🔹 Camada 3: `layers/agent/`
**Input**: Modelos + SHAP values de `layers/training/results/`  
**Processo**: Geração de narrativas via Gemini API + **avaliação de qualidade (LLM as a Judge)**  
**Output**: Explicações em JSON + score de qualidade  
**Ferramentas**: Google Gemini API (FREE tier)  

### 🔹 Camada 4: `layers/view/`
**Input**: Explicações geradas pelo agente  
**Processo**: Renderização formatada (cli.py = lógica, renderer.py = visual)  
**Output**: Terminal com explicações legíveis  
**Ferramentas**: Click, Rich  

## Pontos de Extensão

###  Adicionar Novo Modelo ML
1. Adicionar configuração em `layers/training/config.yaml`
2. Implementar lógica em `layers/training/models/pipeline.py`
3. Treinar: `python layers/training/train.py`

### 🎯 Customizar Prompt
1. Editar `layers/agent/prompts/system_prompt.md` (geração de explicações)
2. Editar `layers/agent/prompts/judge_prompt.md` (avaliação de qualidade)
3. Testar: `python layers/agent/agent.py ...`

### 🏆 Customizar Critérios de Avaliação (LLM as a Judge)
1. Modificar critérios em `layers/agent/prompts/judge_prompt.md`
2. Ajustar lógica em `layers/agent/judge.py`
3. Testar avaliação: `python layers/agent/judge.py ...`

### 📺 Adicionar Novo Formato de Output
1. Implementar em `layers/view/renderer.py`
2. Adicionar opção em `layers/view/cli.py --format novo_formato`

## Estado Atual vs. Implementação Futura

### ✅ Já Criado
- [x] Estrutura de pastas (4 camadas)
- [x] READMEs e documentação
- [x] Arquivos de configuração
- [x] Skeletons de código com TODOs
- [x] System prompt base
- [x] Interface de adapters LLM
- [x] Notebook template de exploração

### ⏭️ A Implementar
- [ ] Escolher dataset de `datasets/` ou `picking_dataset/`
- [ ] Análise exploratória completa (`data/notebooks/data_explore.ipynb`)
- [ ] Lógica de treinamento (`training/train.py` + preencher `config.yaml`)
- [ ] **Integração SHAP** (`training/explainers/shap_explainer.py`)
- [ ] Adapter Gemini (`agent/gemini_adapter.py` - TODOs)
- [ ] Orchestrator do agent (`agent/agent.py`)
- [ ] **LLM as a Judge** (`agent/judge.py`)
- [ ] CLI interativo (`view/cli.py`)
- [ ] Rendering com Rich (`view/renderer.py`)

## Convenções e Boas Práticas
- **Código executável**: Raiz de cada camada (ex: `train.py`, `agent.py`, `cli.py`)
- **Módulos reutilizáveis**: Subpastas (ex: `models/`, `adapters/`)
- **Resultados**: Subpasta `results/` (gitignored por padrão)
- **Configuração**: YAML para parametrização

### 🐍 Estilo Python
- Docstrings em todas as funções públicas
- Type hints quando possível
- TODOs marcam código a implementar
- Logging para debugging e progresso

### 📝 Documentação
- README.md em cada camada
- Comentários inline em lógica complexa
- Exemplos de uso em cada README

### 🔒 Segurança
- API keys via variáveis de ambiente
- `.env` no `.gitignore`
- Nunca commitar credenciais

## Dependências Principais

| Biblioteca | Uso | Camada |
|------------|-----|--------|
| pandas | Manipulação de dados | layers/data, layers/training |
| scikit-learn | ML pipeline | layers/training |
| xgboost | Modelo de classificação | layers/training |
| openai | OpenAI LLM API | layers/agent |
| anthropic | Anthropic LLM API | layers/agent |
| click | CLI interface | layers/view |
| rich | Terminal styling | layers/view |
| jupyter | Notebooks | layers/data |

## Próximos Marcos

1. **Milestone 1**: Implementar `training/` completo
2. **Milestone 2**: Implementar `agent/` com StubAdapter
3. **Milestone 3**: Integrar LLM real (OpenAI ou Anthropic)
4. **Milestone 4**: Implementar `view/` com Rich
5. **Milestone 5**: Pipeline end-to-end funcional
6. **Milestone 6**: Refinamento baseado em feedback
