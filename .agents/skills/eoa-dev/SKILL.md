---
name: eoa-dev
description: >-
  Expert guide and directives for developing the EOA (Explanation-Oriented Agents) Python framework
  and preparing its empirical research paper for ICSE (International Conference on Software Engineering).
---

# EOA (Explanation-Oriented Agents) Development Skill

This skill guides the design, implementation, evaluation, and empirical study of the **EOA** Python framework.

## Core Philosophy
1. **Developer is King / Almost Everything is Adaptable**:
   - Provide clean, non-binding defaults (Orchestrator, CMA, AA, EA, VA).
   - Allow developers to add, override, customize, or remove agents at will.
   - Accept prompt definitions from `.md` files (`Path`) or inline strings (`str`), via the agent constructor.
   - Zero mandatory heavy dependencies in core (only `pydantic >= 2.0`).

2. **Decoupled & Fail-Safe ("Unplugged")**:
   - Explanation layer acts as a sidecar/layer.
   - In case of failure or LLM unavailability, the host predictive system degrades gracefully without breaking.

3. **Structured Contracts vs. Dense Vector RAG**:
   - Adhere to the SBES paper foundation: structured JSON schemas, domain dictionaries, and tabular XAI outputs.
   - Avoid mandatory vector database dependencies in the framework core.

4. **ICSE Standards**:
   - Maintain rigorous software engineering standards (high test coverage, clean abstractions, asynchronous support, PEP 544 protocols, reproducible benchmarks).

## Key Files & References
- `antigravity/requirements.txt`: Living specification of functional and non-functional requirements.
- `antigravity/rascunho.txt`: Shared scratchpad for developer experience syntax, snippets, and quick brainstorms.
- `antigravity/technical_specification.md`: Class diagrams, sequence diagrams, and Pydantic schemas.
- `papers/An Architecture for Explanation-Oriented Agents in Predictive Maintenance Systems.pdf`: Foundational SBES paper.
