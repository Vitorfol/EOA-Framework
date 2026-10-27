"""Validator Agent (VA) acting as an LLM-as-a-Judge quality gate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eoa.agents.base import AgentState, BaseAgent, ValidationResult
from eoa.agents.context import StructuredContextRetriever
from eoa.contracts import AgentStepTrace
from eoa.llm.base import LLM
from eoa.prompts.manager import PromptManager


class ValidatorAgent(BaseAgent):
    """Quality gate auditing explanations for fidelity, consistency, and hallucinations.

    Evaluates candidate explanations produced by the ExplainerAgent against
    provided ground-truth evidence, assigning a score (0.0 to 10.0) and
    generating constructive critique if score < threshold.

    Args:
        threshold: Acceptance score cutoff (0.0 to 10.0). Defaults to 7.0.
        prompt: Custom prompt text, Path to .md file, or None.
        role: Descriptive role for telemetry. Defaults to "Quality Gate Auditor".
        tools: Optional tools for external auditing utilities.
    """

    def __init__(
        self,
        threshold: float = 7.0,
        prompt: str | Path | None = None,
        role: str | None = "Quality Gate Auditor",
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__(
            name="validator",
            prompt=prompt,
            role=role,
            tools=tools,
        )
        self._validate_threshold(threshold)
        self.threshold: float = threshold

    @staticmethod
    def _validate_threshold(val: float) -> None:
        if not (0.0 <= val <= 10.0):
            raise ValueError(
                f"Validation threshold must be between 0.0 and 10.0, got {val}"
            )

    @property
    def threshold(self) -> float:
        """Acceptance score threshold (0.0 to 10.0)."""
        return self._threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        self._validate_threshold(val)
        self._threshold = float(val)

    def _parse_llm_json(self, raw_output: str) -> dict[str, Any]:
        """Extract and parse JSON from the LLM response, stripping code fences."""
        cleaned = raw_output.strip()

        # Strip markdown fences ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "score" in parsed:
                return {**parsed, "parse_error": False}
        except Exception:
            pass

        # Fallback heuristic: attempt regex extraction of score, issues, critique
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', cleaned)
        if score_match:
            try:
                score = float(score_match.group(1))
                return {
                    "score": score,
                    "issues": [
                        "Recovered score via regex fallback from malformed JSON."
                    ],
                    "critique": cleaned,
                    "parse_error": False,
                }
            except ValueError:
                pass

        return {
            "score": 0.0,
            "issues": ["Failed to parse strict JSON evaluation from model output."],
            "critique": "",
            "parse_error": True,
        }

    def _format_ground_truth_domain(self, state: AgentState) -> str:
        parts: list[str] = []
        if state.context is not None and not state.context.is_empty():
            raw_gt = StructuredContextRetriever().retrieve(
                query=state.user_query,
                context=state.context,
            )
            parts.append(f"Ground-Truth Domain Context:\n{raw_gt}")

        if state.context_summary:
            if parts:
                parts.append(
                    f"Retrieved Context Digest (CMA):\n{state.context_summary}"
                )
            else:
                parts.append(state.context_summary)

        if not parts:
            return "No domain context provided."

        return "\n\n".join(parts)

    def _build_prompt(self, state: AgentState) -> str:
        pred_text = (
            str(state.prediction.model_dump())
            if state.prediction
            else "No prediction artifact provided."
        )
        xai_text = (
            str(state.xai.model_dump())
            if state.xai
            else "No feature attribution (XAI) artifact provided."
        )
        domain_text = self._format_ground_truth_domain(state)
        explanation_text = state.draft_explanation or ""

        return PromptManager.render(
            self.prompt,
            prediction=pred_text,
            xai=xai_text,
            domain_context=domain_text,
            explanation=explanation_text,
            user_query=state.user_query,
        )

    def _process_output(self, raw_output: str, state: AgentState) -> ValidationResult:
        parsed = self._parse_llm_json(raw_output)
        parse_error = bool(parsed.get("parse_error", False))

        raw_score = parsed.get("score", 0.0)
        try:
            score = max(0.0, min(10.0, float(raw_score)))
        except (ValueError, TypeError):
            score = 0.0
            parse_error = True

        passed = (score >= self.threshold) and not parse_error

        raw_issues = parsed.get("issues", [])
        issues = (
            [str(i) for i in raw_issues]
            if isinstance(raw_issues, list)
            else [str(raw_issues)]
        )
        critique = str(parsed.get("critique", ""))

        if parse_error:
            status = "error"
            summary = "Validation failed: Unable to parse JSON score from judge output."
        elif not passed:
            issues_summary = ", ".join(issues)
            fallback_critique = (
                f"Score {score:.1f} is below threshold {self.threshold:.1f}. "
                f"Issues: {issues_summary}"
            )
            state.validation_critique = critique or fallback_critique
            status = "refinement"
            summary = f"Validated: Refine (Score {score:.1f}/{self.threshold:.1f})."
        else:
            state.validation_critique = None
            status = "success"
            summary = f"Validated: Approved (Score {score:.1f}/{self.threshold:.1f})."

        result = ValidationResult(
            output=raw_output,
            score=score,
            passed=passed,
            issues=issues,
            critique=critique,
            parse_error=parse_error,
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary=summary,
                status=status,
                details={
                    "score": score,
                    "passed": passed,
                    "issues_count": len(issues),
                    "parse_error": parse_error,
                },
            ),
        )
        state.validation_result = result
        return result

    def _execute_sync(self, state: AgentState, llm: LLM) -> ValidationResult:
        rendered_prompt = self._build_prompt(state)
        output = llm.generate(rendered_prompt)
        return self._process_output(output, state)

    async def _execute_async(self, state: AgentState, llm: LLM) -> ValidationResult:
        rendered_prompt = self._build_prompt(state)
        output = await llm.agenerate(rendered_prompt)
        return self._process_output(output, state)
