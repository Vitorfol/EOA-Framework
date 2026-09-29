# EOA Framework - Guia do Fluxo de Uso Idealizado (Developer Journey)

> *"Nossa, quem desenvolveu isso realmente se importou em fazer algo limpo, intuitivo e fácil de usar."*  
> — Esse é o nosso padrão de qualidade e a sensação que todo desenvolvedor deve ter ao importar a biblioteca.

Este documento detalha **como o desenvolvedor vai usar o framework no dia a dia**, cobrindo do caso mais simples e rápido até as customizações mais avançadas de sistemas industriais e legados.

---

## 🧭 Visão Geral dos Cenários de Uso

| Cenário | Descrição do Caso de Uso | Esforço do Dev |
| :--- | :--- | :--- |
| **Cenário 1** | **Plug & Play (3 Linhas)**: Explicar um alerta básico sem configurar nada. | ⚡ < 1 min |
| **Cenário 2** | **Conexão com LLM Flexível**: 3 formas diferentes de conectar a IA. | 🔌 1 min |
| **Cenário 3** | **"O Desenvolvedor é Rei"**: Remover agentes, trocar prompts por `.md` ou `.txt`. | 🛠️ 3 min |
| **Cenário 4** | **Chat Multi-Turn e Banco SQLite**: Conversar com o agente e auditar o `.db`. | 💬 2 min |
| **Cenário 5** | **Manuais Técnicos Gigantes**: Busca estruturada padrão vs. Plugando RAG. | 📚 4 min |
| **Cenário 6** | **Integração num Sistema Real (FastAPI / Backend Legado)**: Degradação graciosa. | 🏭 5 min |

---

## ⚡ Cenário 1: Uso Plug & Play (Início Imediato em Poucas Linhas)

O desenvolvedor acabou de instalar via `pip install eoa` e só quer explicar uma anomalia detectada pelo modelo XGBoost dele.

```python
import eoa

# 1. Cria a camada padrão com o modelo desejado (o EOA gerencia tudo)
layer = eoa.default_layer(provider="gemini", api_key="AIzaSy...", model="gemini-2.5-flash")

# 2. Pede a explicação passando os dados brutos da predição
response = layer.explain(
    prediction={
        "target": "Condição do Resfriador",
        "status": "Eficiência Reduzida (20%)",
        "probabilidade": 0.82
    },
    xai={
        "top_features": {"PS1_pressao": 0.45, "EPS1_potencia_motor": 0.35, "TS2_temperatura": 0.20}
    },
    user_query="Explique por que esse alerta disparou e o que devo verificar."
)

# 3. Usa o resultado limpo e validado
print(response.narrative)
# -> "O alerta de risco de 82% decorre do aumento simultâneo na pressão axial (PS1) e potência..."

print(f"Nota da Auditoria Técnica: {response.validation_score}/10")
# -> "Nota da Auditoria Técnica: 9.6/10"
```

---

## 🔌 Cenário 2: Flexibilidade na Conexão com LLM (3 Modos)

O desenvolvedor não é obrigado a usar nossa abstração se não quiser. Ele tem controle total de como a IA é chamada:

```python
import eoa

# -------------------------------------------------------------
# Modo 2A: Chave e Modelo Direto (Para scripts rápidos)
# -------------------------------------------------------------
layer_facil = eoa.ExplanationLayer(
    provider="gemini", 
    api_key="MINHA_CHAVE", 
    model="gemini-2.5-flash"
)

# -------------------------------------------------------------
# Modo 2B: Cliente que ele JÁ tem instanciado no sistema dele
# (Ótimo para quem usa proxy corporativo, certificados TLS de usina ou LiteLLM)
# -------------------------------------------------------------
from google import genai
cliente_corporativo = genai.Client(api_key="...", http_options={"proxy": "http://proxy.fabrica:8080"})

layer_cliente = eoa.ExplanationLayer(llm=cliente_corporativo)

# -------------------------------------------------------------
# Modo 2C: Função Própria (Para modelos locais como Ollama, vLLM ou IA interna)
# -------------------------------------------------------------
def meu_llm_local(prompt: str) -> str:
    # Chama o Ollama, vLLM ou uma API interna da empresa
    return servidor_interno_ia.predict(prompt)

layer_custom = eoa.ExplanationLayer(llm=meu_llm_local)
```

---

## 🛠️ Cenário 3: "O Desenvolvedor é Rei" (Customização Total de Agentes)

O desenvolvedor não quer usar o agente de contexto (`CMA`) porque o sistema dele é muito simples, mas quer criar um agente especialista em turbinas hidráulicas usando um arquivo `.md` que a equipe de engenharia escreveu.

```python
import eoa

# Cria uma camada limpa
layer = eoa.ExplanationLayer(provider="gemini", api_key="...")

# 1. Carrega os agentes padrão (CMA, AA, EA, VA)
layer.create_default_agents()

# 2. Não precisa do CMA neste projeto? Desliga com uma linha:
layer.remove_agent("CMA")

# 3. O prompt padrão do Explainer não tem o tom desejado?
# Sobrescreve apontando para um arquivo .md ou .txt no repositório:
layer.override_agent_prompt("EA", source="meus_prompts/explainer_industrial.md")

# 4. Quer adicionar um Especialista Próprio?
layer.register_specialized_agent(
    name="EspecialistaTurbinaFrancis",
    role="Analista mecânico de cavitação e mancais",
    prompt_source="meus_prompts/turbina_francis.txt"
)

# 5. Lista no terminal para inspecionar como a camada ficou configurada
layer.inspect()
# Exibe no terminal uma tabela limpa e formatada:
# [Agente]        [Status]   [Origem do Prompt]
# Orchestrator    Ativo      Built-in
# AA (Artifacts)  Ativo      Built-in
# EA (Explainer)  Ativo      meus_prompts/explainer_industrial.md
# VA (Validator)  Ativo      Built-in (Threshold: 7.0/10)
# Especialista..  Ativo      meus_prompts/turbina_francis.txt
```

---

## 💬 Cenário 4: Chat Multi-Turn e Banco SQLite (O Dev é Dono do `.db`)

O operador quer tirar dúvidas contínuas sobre o alerta em um chat interativo. Todas as perguntas e respostas ficam guardadas num arquivo SQLite local.

```python
import eoa

# Configura o framework apontando para o arquivo de banco desejado
layer = eoa.ExplanationLayer(
    provider="gemini",
    api_key="...",
    # O arquivo .db é criado automaticamente pelo Python (zero dependências extras)
    session_store=eoa.SQLiteSessionStore(db_path="meus_alertas.db")
)

# Inicia ou retoma uma sessão com o ID do incidente
chat = layer.create_session(session_id="incidente_turbina_04")

# Turno 1
res1 = chat.ask("O que significa essa vibração no sensor PS1?")
print(res1.narrative)

# Turno 2 (O agente lembra do contexto do turno anterior!)
res2 = chat.ask("Isso tem risco iminente de cavitação se mantivermos a carga atual?")
print(res2.narrative)

# O desenvolvedor pode abrir o arquivo 'meus_alertas.db' no DBeaver / VSCode e auditar:
# SELECT * FROM eoa_messages WHERE session_id = 'incidente_turbina_04';
```

---

## 📚 Cenário 5: Manuais Técnicos Gigantes (Estruturado vs. Plugando RAG)

Como lidar com manuais de equipamentos industriais de centenas de páginas?

```python
import eoa

layer = eoa.default_layer(provider="gemini", api_key="...")

# -------------------------------------------------------------
# Opção A: Padrão do Paper (Estruturado - Leve, Rápido e Seguro)
# O dev passa apenas os capítulos ou esquemas mapeados do subsistema afetado:
# -------------------------------------------------------------
response = layer.explain(
    prediction=pred_dados,
    xai=xai_dados,
    context={
        "sistema": "Resfriamento Hidráulico",
        "manual_snippet": "Capítulo 4.2: Trocas térmicas anormais ocorrem quando EPS1 > 3.0kW...",
        "sensor_dictionary": {"PS1": "Pressão de entrada", "EPS1": "Potência consumida"}
    },
    user_query="Explique o alerta."
)

# -------------------------------------------------------------
# Opção B: O Dev TEM um PDF gigante de 500 páginas e QUER usar RAG?
# Ele não fica travado! Ele pluga o próprio mecanismo de busca dele:
# -------------------------------------------------------------
def meu_buscador_vetorial(query: str) -> str:
    # Aqui o dev pode usar LangChain, LlamaIndex, ChromaDB, Elasticsearch, etc.
    trechos_relevantes = meu_indice_faiss.similarity_search(query, k=3)
    return "\n".join([t.page_content for t in trechos_relevantes])

# Pluga no ContextManagerAgent (CMA):
cma = layer.get_agent("CMA")
cma.set_retriever(meu_buscador_vetorial)

# Pronto! Agora o CMA consulta o manual de 500 páginas sob demanda
```

---

## 🏭 Cenário 6: Sistema Real em Produção (FastAPI e Fail-Safe)

Exemplo de como plugar o EOA numa rota de API existente de monitoramento. Se o LLM cair ou a internet falhar, **o sistema principal de monitoramento não para**!

```python
from fastapi import FastAPI, HTTPException
import eoa

app = FastAPI(title="Planta Hidrelétrica - Monitoramento")
explanation_layer = eoa.default_layer(provider="gemini", api_key="...")

@app.post("/api/v1/monitorar")
async def monitorar_sensor(leitura: dict):
    # 1. Pipeline preditivo principal (Nível 1 & 2)
    predicao = meu_modelo_ml.predict(leitura)
    xai_importancias = calcular_shap(meu_modelo_ml, leitura)

    # 2. Camada de Explicação (Nível 3) - Desacoplada e Segura
    explicacao = explanation_layer.explain(
        prediction=predicao,
        xai=xai_importancias,
        user_query="Resuma a saúde do equipamento"
    )

    # 3. Retorna a resposta completa
    return {
        "status_operacional": predicao["status"],
        "risco": predicao["probabilidade"],
        # Se a IA falhar ou o validador reprovar, o fallback_used será True
        # mas o operador ainda vê os dados essenciais!
        "explicacao_texto": explicacao.narrative,
        "validacao_aprovada": explicacao.validation_passed,
        "auditoria_score": explicacao.validation_score,
        "fallback_utilizado": explicacao.fallback_used
    }
```
