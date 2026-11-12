# EOA - Kit de Avaliação Experimental

Este pacote reúne tudo o que você precisa para testar o **EOA (*Explanation-Oriented Agents*)** na prática. Nosso objetivo é ouvir desenvolvedores e pesquisadores reais para avaliar a experiência de uso (DX), a clareza da API e o valor entregue pelo framework.

---

## 1. O que é o EOA?

Quem trabalha com Machine Learning sabe que entregar apenas uma predição (Nível 1, como *"eficiência em 3%, próxima à falha total"*) ou uma lista de pesos do SHAP (Nível 2, como `PS6: +0.1670`) raramente resolve a vida de quem precisa tomar uma decisão em cima do sistema. Os números existem, mas falta contexto de domínio, ligação com os dados reais e clareza para o usuário final.

Para atacar esse problema, publicamos no **SBES 2026** o artigo *[An Architecture for Explanation-Oriented Agents in Predictive Maintenance Systems](https://sol.sbc.org.br/index.php/sbes/article/view/47225/46981)*. Nele, propusemos uma arquitetura orientada a agentes em que a explicabilidade funciona como uma camada desacoplada do modelo preditivo.

Agora estamos dando o passo seguinte: generalizar aquela arquitetura em um **framework Python reutilizável, leve e independente de domínio**. A ideia do **EOA** é oferecer uma camada pronta (`ExplanationLayer`) que orquestra agentes especializados para transformar predições, artefatos XAI e documentação técnica em explicações contextualizadas, conversacionais e verificadas (Nível 3).

---

## 2. Por que estamos fazendo esta avaliação?

Este kit (`eoa-evaluation-kit`) foi montado para a etapa empírica do novo artigo. Queremos entender, na prática, como é desenvolver com o EOA:

- A API é intuitiva e fácil de integrar ao seu fluxo de trabalho?
- O código necessário para obter boas explicações e personalizar o comportamento dos agentes faz sentido?
- As explicações geradas, junto com o rastro de execução e a validação automática, agregam valor real?

Não existe resposta certa ou errada. Seu feedback sincero sobre o que funcionou bem, o que travou e o que poderia ser mais simples é exatamente o que procuramos.

---

## 3. O que já deixamos pronto (`infrastructure/` e `tools.py`)

Para que você não perca tempo limpando dados, treinando modelo ou escrevendo scripts auxiliares, já deixamos toda a infraestrutura do caso de uso montada aqui dentro:

- **Dataset e documentação ([`infrastructure/raw/`](./infrastructure/raw) e [`infrastructure/docs/`](./infrastructure/docs)):** Dados reais do sistema hidráulico da ZeMA (*Condition Monitoring of Hydraulic Systems*), com 2.205 ciclos de operação, 14 sensores físicos (pressão, vazão, temperatura, potência do motor e vibração) e a documentação original do conjunto de dados.
- **Modelo e artefatos XAI ([`cooler_incidents.csv`](./infrastructure/cooler_incidents.csv) e [`cooler_baseline_stats.json`](./infrastructure/cooler_baseline_stats.json)):** Predições de condição do refrigerador (`3%`, `20%` ou `100%` de eficiência) geradas por um `RandomForestClassifier` (99,77% de acurácia no conjunto de teste), junto com os valores **TreeSHAP** pré-calculados para **9 incidentes representativos** e as estatísticas da operação saudável (`100%`).
- **Ferramentas prontas ([`tools.py`](./tools.py)):** Funções em Python puro (sem depender de `scikit-learn`, `shap` ou `pandas`) para listar incidentes, consultar leituras dos sensores, comparar desvios em relação à operação normal e carregar o contexto do domínio.

Com isso tudo resolvido ao redor, você pode focar direto na experiência com o framework EOA.

---

## 4. Hands-On: instalação e primeiros passos

### 1. Confira sua versão do Python (`>= 3.11`)
O EOA requer **Python 3.11 ou superior**. Verifique no terminal:

```bash
python --version
```

### 2. Crie um ambiente virtual e instale o pacote
Dentro da pasta `eoa-evaluation-kit/`, crie e ative um ambiente virtual e instale a *wheel* do framework junto com o Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\activate

pip install "eoa-0.1.0-py3-none-any.whl[gemini]" jupyter ipykernel python-dotenv
```

*(Você não precisa instalar `scikit-learn` ou `shap`. Todos os resultados do modelo já estão pré-calculados nos arquivos de infraestrutura.)*

### 3. Chave da API do Gemini (`.env`)
Já deixamos um arquivo **`.env`** na raiz desta pasta com uma chave configurada para você rodar os testes sem precisar criar conta. O notebook carrega esse arquivo automaticamente.

Se a chave der erro de limite/cota ou você tiver qualquer problema com a API do Gemini durante os testes, mande uma mensagem para a gente que resolvemos rápido:
- **Vitor Fontenele:** `vitor_fontenele@atlantico.com.br`
- **Paulo Coelho:** `paulo_coelho@atlantico.com.br`

### 4. Siga a ordem sugerida de arquivos
Para aproveitar melhor a avaliação, recomendamos percorrer os arquivos nesta ordem:

1. **`README.md` (este arquivo):** visão geral, contexto e instalação do ambiente.
2. **[`GUIDE.md`](./GUIDE.md):** guia rápido de consulta com as funções, parâmetros e contratos do EOA.
3. **[`exploration.ipynb`](./exploration.ipynb):** notebook experimental com uma demonstração guiada e um espaço livre para você testar suas próprias ideias.

Para abrir o notebook no navegador:

```bash
jupyter notebook exploration.ipynb
```

Se preferir usar o VS Code ou Cursor, basta abrir o `exploration.ipynb` direto no editor e selecionar o ambiente `.venv` como kernel.

---

## 5. O que esperar do EOA durante os testes

Ao longo do [`exploration.ipynb`](./exploration.ipynb) e do [`GUIDE.md`](./GUIDE.md), você vai interagir com os principais recursos do framework:

- **Orquestração multi-agente voltada à explicabilidade:** Quatro agentes trabalham juntos no pipeline padrão (`ContextManagerAgent`, `ArtifactsAgent`, `ExplainerAgent` e `ValidatorAgent`), incluindo uma checagem automática que audita a resposta contra os dados originais antes de devolvê-la.
- **Pouquíssimas linhas para começar:** Dá para subir a camada com `ExplanationLayer.default_layer(...)` e gerar explicações com `layer.explain(...)`, passando os dados diretamente, deixando os agentes consultarem ferramentas Python por ID, ou até usando apenas documentação e contexto (sem modelo de ML).
- **Integração direta com suas funções:** Qualquer função Python tipada e documentada em `tools.py` pode ser registrada como ferramenta em um agente.
- **Personalização simples:** Você pode inspecionar os agentes ativos (`layer.inspect()`), alterar prompts pontualmente, adicionar um agente especialista próprio (`SpecializedAgent`) ou manter sessões multi-turno persistidas em SQLite nativo.
- **Execução segura:** Se houver falha de rede ou indisponibilidade do LLM, o pipeline degrada de forma controlada sem quebrar a aplicação.
