# EOA - Kit de Avaliação Experimental

Este pacote reúne as informações necessárias para testar o **EOA (*Explanation-Oriented Agents*)**. Nosso objetivo é ouvir desenvolvedores, pesquisadores e cientístas de dados para avaliar a experiência de uso, a clareza da API e o valor entregue pelo framework.

---

## 1. O que é o EOA?

Na aplicação de modelos de Aprendizado de Máquina (*Machine Learning*) em contextos críticos, o fornecimento isolado de saídas preditivas (Nível 1, como *"eficiência em 3%, próxima à falha total"*) ou de atribuições quantitativas de importância de atributos via SHAP (Nível 2, como `PS6: +0.1670`) frequentemente mostra-se insuficiente para subsidiar a tomada de decisão operacional. Embora tais métricas ofereçam indicadores numéricos, carecem de contextualização de domínio, rastreabilidade em relação aos dados empíricos e clareza interpretativa para o usuário final.

Com o objetivo de mitigar essa lacuna, propusemos no artigo *[An Architecture for Explanation-Oriented Agents in Predictive Maintenance Systems](https://sol.sbc.org.br/index.php/sbes/article/view/47225/46981)*, publicado no **SBES 2026**, uma arquitetura orientada a agentes na qual a explicabilidade opera como uma camada arquitetural desacoplada do modelo preditivo.

A presente pesquisa avança nessa direção ao generalizar a arquitetura original em um **framework Python reutilizável, leve e agnóstico ao domínio**. O **EOA (*Explanation-Oriented Agents*)** disponibiliza uma abstração central (`ExplanationLayer`) responsável por orquestrar agentes especializados, convertendo saídas preditivas, artefatos de Inteligência Artificial Explicável (XAI) e documentação técnica em explicações contextualizadas, conversacionais e formalmente verificadas (Nível 3).

---

## 2. Por que estamos fazendo esta avaliação?

ste pacote (`eoa-evaluation-kit`) foi estruturado para viabilizar a avaliação empírica do framework no âmbito de uma investigação científica em curso. O estudo busca analisar a experiência de desenvolvimento e a adequação prática do EOA sob três dimensões principais:

- **Usabilidade e integração da API:** A interface de programação demonstra-se intuitiva e de fácil incorporação a fluxos de trabalho existentes?
- **Expressividade e extensibilidade:** A estrutura de código requerida para gerar explicações e customizar o comportamento dos agentes é clara e coerente?
- **Utilidade e confiabilidade das saídas:** As explicações geradas, acompanhadas do rastro de execução (*execution trace*) e dos mecanismos de validação automática, agregam valor efetivo à interpretação do sistema?

Ressalta-se que não há respostas esperadas ou critérios de desempenho individual em avaliação. Relatos objetivos acerca de facilidades encontradas, limitações técnicas, gargalos de implementação e oportunidades de simplificação constituem os dados centrais desta investigação.

---

## 3. O que já deixamos pronto (`infrastructure/` e `tools.py`)

A fim de isolar a variável de interesse (a utilização do framework EOA) e eliminar o esforço prévio de pré-processamento de dados, treinamento de modelos e implementação de rotinas auxiliares, o ambiente experimental contempla integralmente os artefatos do estudo de caso:

- **Conjunto de dados e documentação técnica ([`infrastructure/raw/`](./infrastructure/raw) e [`infrastructure/docs/`](./infrastructure/docs)):** Dados reais do sistema hidráulico da ZeMA (*Condition Monitoring of Hydraulic Systems*), com 2.205 ciclos de operação, 14 sensores físicos (pressão, vazão, temperatura, potência do motor e vibração) e a documentação original do conjunto de dados.
- **Modelo preditivo e artefatos XAI ([`cooler_incidents.csv`](./infrastructure/cooler_incidents.csv) e [`cooler_baseline_stats.json`](./infrastructure/cooler_baseline_stats.json)):** Predições de condição do refrigerador (`3%`, `20%` ou `100%` de eficiência) geradas por um `RandomForestClassifier` (99,77% de acurácia no conjunto de teste), junto com os valores **TreeSHAP** pré-calculados para **9 incidentes representativos** e as estatísticas da operação saudável (`100%`).
- **Biblioteca de ferramentas auxiliares ([`tools.py`](./tools.py)):** Funções em Python puro (sem depender de `scikit-learn`, `shap` ou `pandas`) para listar incidentes, consultar leituras dos sensores, comparar desvios em relação à operação normal e carregar o contexto do domínio.

Dessa forma, o participante pode concentrar-se integralmente na instrumentação, configuração e avaliação do framework EOA.

---

## 4. Hands-On: instalação e primeiros passos


### 1. Verificação da versão do Python (`>= 3.11`)
O EOA requer **Python 3.11 ou superior**. Verifique no terminal:

```bash
python --version
```

### 2. Criação do ambiente virtual e instalação das dependências
Dentro da pasta `eoa-evaluation-kit/`, crie e ative um ambiente virtual e instale a *wheel* do framework junto com o Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\activate

pip install "eoa-0.1.0-py3-none-any.whl[gemini]" jupyter ipykernel python-dotenv
```

*(Não é necessária a instalação das bibliotecas `scikit-learn` ou `shap`. Todos os resultados do modelo já estão pré-calculados nos arquivos de infraestrutura.)*

### 3.Configuração da chave de API do Gemini (`.env`)
Um arquivo **`.env`** contendo uma chave de API previamente configurada foi disponibilizado na raiz deste diretório para viabilizar a execução imediata dos experimentos, dispensando a criação de contas externas. O *notebook* experimental carrega essa variável de ambiente automaticamente.

Em caso de excedente de cota (*rate limit*) ou eventuais falhas de autenticação junto à API do Gemini durante a execução do protocolo, solicita-se entrar em contato com os pesquisadores responsáveis para imediata substituição da credencial ou aguardar 1 hora para realizar os testes novamente.
- **Vitor Fontenele:** `vitor_fontenele@atlantico.com.br`
- **Paulo Coelho:** `paulo_coelho@atlantico.com.br`

### 4. Siga a ordem sugerida de arquivos
Para a condução adequada do protocolo experimental, recomenda-se percorrer os artefatos na seguinte ordem:

1. **`README.md` (este arquivo):** visão geral, contexto e instalação do ambiente.
2. **[`GUIDE.md`](./GUIDE.md):** guia rápido de consulta com as funções, parâmetros e contratos do EOA.
3. **[`exploration.ipynb`](./exploration.ipynb):** notebook experimental com uma demonstração guiada e um espaço livre para você testar suas próprias ideias.

Para inicializar o *notebook* via navegador web, execute:

```bash
jupyter notebook exploration.ipynb
```

Alternativamente, caso utilize ambientes integrados como VS Code ou Cursor, basta abrir o arquivo `exploration.ipynb` diretamente no editor e selecionar o ambiente virtual `.venv` como *kernel* de execução.
---

## 5. Funcionalidades e Arquitetura Observáveis Durante a Avaliação

Ao longo da interação com [`exploration.ipynb`](./exploration.ipynb) e [`GUIDE.md`](./GUIDE.md), serão exercitados os principais mecanismos arquiteturais do framework:

- **Orquestração multiagente orientada à explicabilidade:** Quatro agentes especializados operam de forma coordenada no *pipeline* padrão (`ContextManagerAgent`, `ArtifactsAgent`, `ExplainerAgent` e `ValidatorAgent`), incorporando uma etapa automatizada de auditoria que verifica a consistência da explicação gerada frente aos dados empíricos originais antes de retorná-la ao usuário.
- **Baixa sobrecarga de configuração (*low-code integration*):** A camada de explicabilidade pode ser instanciada de forma sucinta via `ExplanationLayer.default_layer(...)` e invocada por meio de `layer.explain(...)`, suportando tanto a injeção direta dos dados quanto a consulta autônoma a ferramentas Python por identificador, ou mesmo a operação baseada exclusivamente em documentação e contexto de domínio (na ausência de um modelo preditivo).
- **Acoplamento nativo de funções (*tool binding*):** Qualquer função Python devidamente tipada e documentada, a exemplo das implementadas em `tools.py`, pode ser registrada como ferramenta (*tool*) executável por um agente.
- **Inspecionabilidade e extensibilidade arquitetural:** O framework permite auditar a topologia dos agentes ativos (`layer.inspect()`), refinar instruções (*prompts*) de forma granular, incorporar agentes especialistas customizados (`SpecializedAgent`) e manter o histórico de interações em sessões multiturno persistidas nativamente em SQLite.
- **Tolerância a falhas e degradação graciosa (*graceful degradation*):** Na ocorrência de falhas de rede ou indisponibilidade do Modelo de Linguagem Grande (LLM), o *pipeline* de execução degrada de maneira controlada, preservando a estabilidade da aplicação hospedeira sem interrupções abruptas.
