# Camada 2: Training

## Objetivo
Treinamento e avaliação de modelos de Machine Learning.

## Responsabilidades
- Treinar modelos de classificação/regressão
- Avaliar performance
- **Computar explainers (SHAP values, feature importance)**
- Serializar modelos e artefatos
- Gerar predições + explicações técnicas para a camada agent

## Estrutura
```
training/
├── explainers/       # Técnicas de XAI (SHAP, feature importance)
├── models/           # Código dos pipelines de ML
├── results/          # Outputs: modelos, métricas, SHAP values
├── config.yaml       # Configuração de hiperparâmetros
└── train.py          # Script principal
```

## Fluxo
1. Treina modelo com dados limpos
2. **Computa SHAP values** (via `explainers/shap_explainer.py`)
3. Salva modelo + SHAP values em `results/`
4. Agent layer consome esses outputs para gerar narrativas

## TODO
- [ ] Implementar pipeline de treinamento baseado no dataset
- [ ] Definir modelos a serem testados (ver `models/MODELS.md`)
- [ ] Configurar métricas de avaliação
- [ ] **Implementar SHAP explainer** (`explainers/shap_explainer.py`)
- [ ] Estruturar outputs: modelo + SHAP values salvos em `results/`
