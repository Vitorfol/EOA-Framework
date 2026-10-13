# EOA Development Directives & Guidelines (Skill Reference)

Este arquivo é um espelho de referência da skill agêntica `.agents/skills/eoa-dev/SKILL.md`.

## Filosofia Central
1. **Developer is King / Almost Everything is Adaptable**:
   - Fornecer defaults sensatos, mas nunca engessados.
   - Qualquer agente padrão (CMA, AA, EA, VA) pode ser desativado, removido, sobrescrito ou reconfigurado.
   - Prompts vêm do default embutido, de uma `str` (texto) ou de um `Path` para arquivo `.md`.
   - Core ultra-enxuto: zero dependências pesadas de IA além de `pydantic`.

2. **Desacoplamento e Fail-Safe ("Unplugged")**:
   - A camada é não-intrusiva. Falhas de LLM ou de validação geram degradação graciosa com fallback sem quebrar a aplicação principal.

3. **Contratos Estruturados vs RAG Denso**:
   - Respeito integral ao paper do SBES: contratos JSON previsíveis, esquemas de sensores, metadados e tabelas de XAI.
   - Zero dependência obrigatória de bancos vetoriais no core.

4. **Padrão ICSE**:
   - Código tipado com PEP 544 Protocols, async nativo, benchmarks de overhead e estudos empíricos reproduzíveis.

## Arquivos de Controle no Projeto
- `antigravity/requirements.txt`: Especificação de requisitos funcionais e não-funcionais.
- `antigravity/rascunho.txt`: Bloco de rascunhos, snippets de API e ideias de sintaxe.
- `antigravity/technical_specification.md`: Especificação técnica, diagramas Mermaid e contratos Pydantic.
