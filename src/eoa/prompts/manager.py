"""Prompt resolution, storage, and rendering for EOA agents."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Any

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class PromptManager:
    """Manages default and custom prompt templates for EOA agents.

    Supports resolving built-in default prompts, overriding templates via
    in-memory strings or local Markdown (.md) files, resetting to defaults,
    and safely interpolating template variables without corrupting JSON structures.

    Can be used as a stateful registry (e.g., inside an ExplanationLayer) or
    via its static resolution and rendering utilities.

    Example:
        >>> # 1. Stateful usage (inside ExplanationLayer)
        >>> manager = PromptManager()
        >>> default_va = manager.get("validator")
        >>> manager.override("validator", "Custom criteria: {criteria}")
        >>> active_va = manager.get("validator")
        >>> rendered = manager.render_agent("validator", criteria="High Precision")
        >>> manager.reset("validator")  # Restores default
        >>>
        >>> # 2. Stateless / static utilities (used directly by agent constructors)
        >>> prompt_text = PromptManager.resolve("validator")
        >>> rendered_text = PromptManager.render("Alert on {sensor}", sensor="PS1")
    """

    def __init__(self) -> None:
        self._overrides: dict[str, str] = {}

    @staticmethod
    def resolve(agent: str, source: str | Path | None = None) -> str:
        """Resolve a prompt template from an explicit source or fallback to default.

        Args:
            agent: Canonical agent name (e.g., "context_manager", "artifacts",
                "explainer", "validator").
            source: None to load the package's default prompt, a direct `str`
                containing the prompt text, or a `Path` pointing to a local `.md` file.

        Returns:
            The resolved prompt template text.

        Raises:
            ValueError: If `Path` is not a `.md` file, or if `agent` has no
                built-in default.
            FileNotFoundError: If the specified `Path` does not exist.
        """
        if isinstance(source, str):
            return source
        if isinstance(source, Path):
            if source.suffix != ".md":
                raise ValueError(f"Prompt files must be Markdown (.md): {source}")
            if not source.is_file():
                raise FileNotFoundError(f"Prompt file not found: {source}")
            return source.read_text(encoding="utf-8")

        default_file = files("eoa.prompts").joinpath(f"{agent}_default.md")
        if not default_file.is_file():
            raise ValueError(
                f"No default prompt for agent '{agent}'. "
                "Pass `prompt=` as a str or a Path to a .md file."
            )
        return default_file.read_text(encoding="utf-8")

    @staticmethod
    def render(template: str, **variables: Any) -> str:
        """Replace `{name}` placeholders with variables safely.

        Only matching placeholders present in `variables` are replaced. Other
        braces (such as JSON schema examples or curly braces in code blocks)
        remain intact.

        Args:
            template: The template string containing `{placeholder}` tags.
            **variables: Keyword arguments matching the placeholder names to fill.

        Returns:
            The interpolated string with provided placeholders replaced.

        Example:
            >>> PromptManager.render("Hello {name}!", name="Engineer")
            'Hello Engineer!'
        """
        return _PLACEHOLDER.sub(lambda m: str(variables.get(m[1], m[0])), template)

    def get(self, agent: str) -> str:
        """Get the active prompt template for an agent (override if set, else default).

        Args:
            agent: Canonical name of the agent.

        Returns:
            The active prompt template.

        Example:
            >>> prompt = manager.get("explainer")
        """
        if agent in self._overrides:
            return self._overrides[agent]
        return self.resolve(agent)

    def override(self, agent: str, source: str | Path) -> None:
        """Override the prompt template for a specific agent.

        Args:
            agent: Canonical name of the agent to override.
            source: Direct prompt text (str) or path to a local .md file (Path).

        Example:
            >>> manager.override("explainer", "You are an industrial engineer.")
            >>> manager.override("explainer", Path("prompts/custom.md"))
        """
        resolved_text = self.resolve(agent=agent, source=source)
        self._overrides[agent] = resolved_text

    def reset(self, agent: str | None = None) -> None:
        """Reset agent prompt(s) back to built-in default(s).

        Args:
            agent: If specified, resets only this agent. If None, resets all overrides.

        Example:
            >>> manager.reset("explainer")  # Reset single agent
            >>> manager.reset()             # Reset all overrides
        """
        if agent is None:
            self._overrides.clear()
        else:
            self._overrides.pop(agent, None)

    def render_agent(self, agent: str, **variables: Any) -> str:
        """Render the active prompt of an agent using the provided variables.

        Fetches the active template for ``agent`` (override if present, else default)
        and substitutes matching ``{placeholders}`` with ``variables``.

        Args:
            agent: Canonical name of the agent (e.g. "validator", "explainer").
            **variables: Keyword arguments corresponding to placeholders in the prompt.

        Returns:
            The rendered prompt string with placeholders populated.
        """
        template = self.get(agent)
        return self.render(template, **variables)
