# PoC: Agente de Explicabilidade para Usinas Hidrelétricas

## 🎯 Objetivo
Sistema modular para explicar predições de ML sobre operação de usinas hidrelétricas usando XAI (SHAP/Feature Importance) e LLM para narrativas em linguagem natural.

## 🏗️ Arquitetura

```
PoC/
├── layers/            # Camadas da aplicação
│   ├── data/          # Camada 1: Exploração e preparação de dados
│   ├── training/      # Camada 2: Treinamento de modelos ML
│   ├── agent/         # Camada 3: Agent de explicabilidade (XAI + LLM)
│   └── view/          # Camada 4: Interface de terminal
├── datasets/          # Datasets (não versionados)
└── picking_dataset/   # Análises de datasets 
```

### Camada 1: `layers/data/`
- **Responsabilidade**: Exploração, validação e preparação do dataset
- **Artefatos**: Notebooks de análise exploratória, schema de dados
- **Output**: Dataset limpo e documentado

### Camada 2: `layers/training/`
- **Responsabilidade**: Treino de modelos (XGBoost, Random Forest), avaliação **e computação de SHAP values**
- **Artefatos**: Modelos treinados, métricas, SHAP values, explainers
- **Output**: Modelos serializados + predições + SHAP values (CSV/JSON)

### Camada 3: `layers/agent/`
- **Responsabilidade**: Geração de narrativas (Gemini) a partir de SHAP + **avaliação de qualidade (LLM as a Judge)**
- **Artefatos**: Prompts, adaptador Gemini, judge de qualidade
- **Output**: Explicações em linguagem natural + score de qualidade (JSON/texto)

### Camada 4: `layers/view/`
- **Responsabilidade**: Interface CLI para visualizar explicações
- **Artefatos**: Scripts de rendering e formatação
- **Output**: Terminal com explicações formatadas

## 🚀 Setup

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente (opcional, para LLM APIs)
cp .env.example .env
# Editar .env com suas credenciais
```

## 📊 Workflow de Desenvolvimento

1. **Exploração de Dados** (`layers/data/`)
   ```bash
   jupyter notebook layers/data/notebooks/data_explore.ipynb
   ```

2. **Treinamento** (`layers/training/`)
   ```bash
   python layers/training/train.py --config layers/training/config.yaml
   ```

3. **Geração de Explicações** (`layers/agent/`)
   ```bash
   python layers/agent/agent.py --model layers/training/models/latest --output results/explanations.json
   ```

4. **Visualização** (`layers/view/`)
   ```bash
   python layers/view/cli.py --file results/explanations.json
   ```

## 🔌 LLM Integration

O sistema utiliza **Google Gemini API** (FREE tier) para geração de narrativas:
- Configuração via `.env` (variável `GEMINI_API_KEY`)
- Adapter implementado em `layers/agent/gemini_adapter.py`

## 📝 Notas
- **Read-Only Agent**: O agente NÃO atua; apenas explica
- **Execução Local**: Processa datasets locais em batch
- **Modular**: Cada camada pode ser desenvolvida/testada independentemente

## 📄 Licença
[Defina conforme necessário]
