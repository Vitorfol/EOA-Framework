# Diretrizes de Desenvolvimento e Engenharia de Software (EOA Framework)

Este documento define os princípios sagrados que devem guiar toda linha de código, refatoração, teste e documentação do framework **EOA**.

---

## 1. Princípio da Empatia com o Desenvolvedor ("Developer Experience First")
> *"Nossa, quem desenvolveu isso realmente se importou em fazer algo limpo, intuitivo e fácil de usar."*

- **APIs Autoexplicativas**: Nomes de métodos devem ser verbos de ação claros (`explain`, `create_session`, `remove_agent`, `override_agent_prompt`).
- **Zero Configuração Obrigatória para Começar**: O caso de uso mais simples deve funcionar em 3 linhas com valores padrão inteligentes.
- **Mensagens de Erro Educativas**: Quando algo falhar (ex: schema inválido, chave de API faltando, agente não encontrado), a mensagem de erro deve dizer **exatamente o que aconteceu e como resolver** em 1 linha de exemplo.
- **Tipagem Estrita com Autocomplete Perfeito**: Usar type hints completos em todas as assinaturas públicas. O desenvolvedor no VSCode/PyCharm deve ver todas as opções no autocomplete sem precisar abrir a documentação.

---

## 2. Princípio da Postura Crítica e Comunicação Transparente
- **Não Pressupor**: Avaliar criticamente os impactos de design antes de implementar. Se um requisito for ambíguo ou houver um trade-off arquitetural relevante, pergunte e proponha alternativas fundamentadas.
- **Rigor para o ICSE**: Pensar sempre na defensabilidade acadêmica. Código limpo, desacoplamento real, benchmarks reprodutíveis e ausência de dependências inúteis ou obscuras.

---

## 3. Qualidade de Código e Modularidade
- **PEP 8 & Padrões Python Modernos**: Código limpo, modular, altamente legível (*readable*), com docstrings em linguagem natural clara.
- **Princípio da Responsabilidade Única (SRP)**: Cada módulo e agente tem uma única fronteira de responsabilidade.
- **Dependências Mínimas no Core**: O core do `eoa` deve depender **estritamente apenas do Pydantic (>= 2.0)**. Tudo o que for provedor externo (Google GenAI, OpenAI, etc.) ou CLI deve ser opcional via *extras* no `pyproject.toml`.
- **Desacoplamento Fail-Safe ("Unplugged")**: A camada de agentes é um aditivo inteligente, nunca um ponto único de falha (*Single Point of Failure*). Exceções em agentes ou LLM são capturadas e tratadas com degradação graciosa.
