"""Tests for EOA prompt resolution, stateful overrides, and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from eoa import PromptManager

AGENTS = ("context_manager", "artifacts", "explainer", "validator")
PTCF = ("## Persona", "## Task", "## Context", "## Format")


class TestDefaults:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_default_prompt_follows_ptcf(self, agent: str) -> None:
        prompt = PromptManager.resolve(agent)
        positions = [prompt.index(section) for section in PTCF]
        assert positions == sorted(positions)

    def test_unknown_agent_without_source_explains_how_to_fix(self) -> None:
        with pytest.raises(ValueError, match="prompt="):
            PromptManager.resolve("turbine_specialist")


class TestSources:
    def test_str_is_the_prompt_text_even_if_it_looks_like_a_path(self) -> None:
        assert PromptManager.resolve("validator", "my_prompt.md") == "my_prompt.md"

    def test_path_is_read_from_markdown_file(self, tmp_path: Path) -> None:
        file = tmp_path / "custom.md"
        file.write_text("You are a turbine expert.", encoding="utf-8")
        assert PromptManager.resolve("turbine", file) == "You are a turbine expert."

    def test_missing_file_raises_friendly_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="custom.md"):
            PromptManager.resolve("validator", tmp_path / "custom.md")

    def test_non_markdown_file_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "custom.txt"
        file.write_text("text", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.md"):
            PromptManager.resolve("validator", file)


class TestStatefulManager:
    def test_get_returns_default_when_no_override(self) -> None:
        mgr = PromptManager()
        default_val = mgr.get("validator")
        assert "## Persona" in default_val

    def test_override_with_string_and_reset(self) -> None:
        mgr = PromptManager()
        mgr.override("explainer", "Custom Explainer string template.")
        assert mgr.get("explainer") == "Custom Explainer string template."

        # Reset specific agent
        mgr.reset("explainer")
        assert "## Persona" in mgr.get("explainer")

    def test_override_with_path(self, tmp_path: Path) -> None:
        mgr = PromptManager()
        custom_file = tmp_path / "custom_explainer.md"
        custom_file.write_text("Custom prompt from markdown file.", encoding="utf-8")

        mgr.override("explainer", custom_file)
        assert mgr.get("explainer") == "Custom prompt from markdown file."

        # Reset all
        mgr.reset()
        assert "## Persona" in mgr.get("explainer")

    def test_render_agent_interpolates_active_template(self) -> None:
        mgr = PromptManager()
        mgr.override("validator", "Audit {target} with criteria {rule}.")
        rendered = mgr.render_agent("validator", target="HydraulicPump", rule="R-101")
        assert rendered == "Audit HydraulicPump with criteria R-101."


class TestRender:
    def test_fills_provided_variables(self) -> None:
        assert PromptManager.render("Q: {q} / {q}", q="why?") == "Q: why? / why?"

    def test_unknown_placeholders_are_left_untouched(self) -> None:
        assert PromptManager.render("{a} {b}", a="x") == "x {b}"

    def test_json_braces_are_safe(self) -> None:
        template = 'Reply {"score": 1} about {topic}'
        rendered = PromptManager.render(template, topic="PS1")
        assert rendered == 'Reply {"score": 1} about PS1'

    def test_default_validator_prompt_keeps_its_json_example(self) -> None:
        template = PromptManager.resolve("validator")
        rendered = PromptManager.render(
            template, prediction="p", xai="x", domain_context="c", explanation="e"
        )
        assert '{"score":' in rendered
        assert "{explanation}" not in rendered

    def test_default_context_manager_prompt_renders_all_placeholders(self) -> None:
        mgr = PromptManager()
        rendered = mgr.render_agent(
            "context_manager",
            context="Sensors: PS1 (Pressure), EPS1 (Power)",
            user_query="Why did the alert fire?",
        )
        assert "Sensors: PS1 (Pressure), EPS1 (Power)" in rendered
        assert "Why did the alert fire?" in rendered
        assert "{context}" not in rendered
        assert "{user_query}" not in rendered

    def test_default_artifacts_prompt_renders_all_placeholders(self) -> None:
        mgr = PromptManager()
        rendered = mgr.render_agent(
            "artifacts",
            prediction="Risk 0.87 (High Failure Probability)",
            xai="EPS1: +0.42, PS1: +0.31",
        )
        assert "Risk 0.87 (High Failure Probability)" in rendered
        assert "EPS1: +0.42, PS1: +0.31" in rendered
        assert "{prediction}" not in rendered
        assert "{xai}" not in rendered

    def test_default_explainer_prompt_renders_all_placeholders(self) -> None:
        mgr = PromptManager()
        rendered = mgr.render_agent(
            "explainer",
            artifacts="Model indicates failure via EPS1",
            domain_context="EPS1 is power consumed",
            history="Operator: Check status",
            critique="Refine next action",
            user_query="What happened?",
        )
        assert "Model indicates failure via EPS1" in rendered
        assert "EPS1 is power consumed" in rendered
        assert "Operator: Check status" in rendered
        assert "Refine next action" in rendered
        assert "What happened?" in rendered
        assert "{artifacts}" not in rendered
        assert "{domain_context}" not in rendered
        assert "{history}" not in rendered
        assert "{critique}" not in rendered
        assert "{user_query}" not in rendered

