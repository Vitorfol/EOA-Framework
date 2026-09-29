# Camada 4: View (Console)

## Objetivo
Interface de terminal para navegar em dataframes, predições mockadas e
interagir com o agente de explicacao.

## Responsabilidades
- Exibir dataframes carregados pela camada de data
- Listar e visualizar predições (JSON) vindas da camada de training
- Permitir explain + ask question via ExplanationAgent

## Componentes

### `cli.py`
CLI principal. Faz:
- Carregamento de dataframes via `layers.data.src`
- Carregamento de predições via `AGENT_PAYLOAD_DIR`
- Interacao com `ExplanationAgent`

### `renderer.py`
Helpers de formatacao (console).

## Execucao

Ative o venv e rode a CLI:

```bash
source ./venv/bin/activate
python3 layers/view/cli.py
```

## Variaveis de ambiente

Usadas pela CLI e pela camada agent:

- `AGENT_PAYLOAD_DIR` -> diretório com JSONs de predição
- `AGENT_SYSTEM_PROMPT_PATH`
- `AGENT_JUDGE_PROMPT_PATH`
- `AGENT_JUDGE_MIN_SCORE`
- `AGENT_MAX_ATTEMPTS`
- `GEMINI_API_KEY` e `GEMINI_MODEL` (para execucao real)

## Template minimo de JSON de predicao

Estrutura generica que a view espera. Campos extras sao aceitos e
podem ser enviados ao LLM.

### Classificacao
```json
{
	"id": 1,
	"model_info": {
		"model_name": "nome_modelo",
		"model_type": "XGBoostClassifier",
		"category": "tree_based",
		"task_type": "classification",
		"target_variable": "target"
	},
	"prediction": {
		"instance_id": 123,
		"predicted_class": 2,
		"class_probabilities": {
			"0": 0.1,
			"1": 0.2,
			"2": 0.7
		}
	},
	"xai": {
		"method": "shap_tree_explainer",
		"top_features": []
	},
	"instance": {},
	"domain": {}
}
```

### Regressao / forecast
```json
{
	"id": 2,
	"model_info": {
		"model_name": "nome_modelo",
		"model_type": "ARIMA(2,1,1)",
		"category": "statistical",
		"task_type": "time_series_forecast",
		"target_variable": "target"
	},
	"prediction": {
		"instance_id": 456,
		"predicted_value": 12.34
	},
	"xai": {
		"method": "shap",
		"top_features": []
	},
	"instance": {},
	"domain": {}
}
```

## Observacoes

- A view nao tenta inferir `prediction_id` a partir do texto do usuario.
- Para perguntas sobre uma predicao, passe o id explicitamente no menu.
- Para conectar a dados reais, basta trocar o caminho de `AGENT_PAYLOAD_DIR`.

## Web frontend

There is a minimal Flask-based frontend at `layers/view/web` intended for quick visual inspection and demo.

To run (in the project root, with the venv activated):

```bash
pip install -r requirements.txt
python3 layers/view/web/app.py
```

Open http://127.0.0.1:5000 in your browser. The app lists available prediction JSONs from the directory configured in `AGENT_PAYLOAD_DIR` and allows clicking a prediction to view details, generate an explanation, or send a follow-up question.
