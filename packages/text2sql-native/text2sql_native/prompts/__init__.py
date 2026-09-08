"""Prompt template loading.

Templates are editable ``.txt`` files, kept outside the code. Each file has a
``SYSTEM:`` section and a ``USER:`` section. Both support ``str.format``-style
``{placeholder}`` substitution.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from ..errors import ConfigError

#: Package that holds the built-in prompt templates.
_PACKAGE = "text2sql_native.prompts"

_SYSTEM_MARKER = "SYSTEM:"
_USER_MARKER = "USER:"


class PromptTemplate:
    """A parsed template with a system and a user section."""

    def __init__(self, system: str, user: str) -> None:
        self.system = system
        self.user = user

    def render(self, **kwargs: object) -> tuple[str, str]:
        """Return ``(system, user)`` with placeholders filled.

        Raises:
            ConfigError: If the template uses a placeholder that is not given.
        """
        try:
            return (
                self.system.format(**kwargs),
                self.user.format(**kwargs),
            )
        except KeyError as exc:
            raise ConfigError(
                f"Prompt template uses an unknown placeholder {exc}."
            ) from exc


class PromptTemplates:
    """Loads named templates.

    This is the DI seam for prompts: pass a different directory (or subclass) to
    swap templates without touching code.

    Args:
        directory: Directory with the ``.txt`` files. When ``None`` (default),
            the templates packaged inside ``text2sql_native.prompts`` are used, so the
            library works when installed as a package, from any working
            directory.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = Path(directory).expanduser() if directory is not None else None

    def load(self, filename: str) -> PromptTemplate:
        """Load and parse a template by name.

        Raises:
            ConfigError: If the template is missing or lacks the two sections.
        """
        if self._dir is not None:
            path = self._dir / filename
            if not path.is_file():
                raise ConfigError(f"Prompt template not found: '{path}'.")
            return self._parse(path.read_text(encoding="utf-8"), str(path))

        # Fall back to the templates packaged with the library.
        try:
            resource = resources.files(_PACKAGE).joinpath(filename)
            text = resource.read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise ConfigError(
                f"Packaged prompt template '{filename}' not found in {_PACKAGE}."
            ) from exc
        return self._parse(text, f"{_PACKAGE}/{filename}")

    @staticmethod
    def _parse(text: str, path: str) -> PromptTemplate:
        if _SYSTEM_MARKER not in text or _USER_MARKER not in text:
            raise ConfigError(
                f"Prompt template '{path}' must have a '{_SYSTEM_MARKER}' and a "
                f"'{_USER_MARKER}' section."
            )
        _, _, remainder = text.partition(_SYSTEM_MARKER)
        system_part, _, user_part = remainder.partition(_USER_MARKER)
        return PromptTemplate(system=system_part.strip(), user=user_part.strip())


__all__ = ["PromptTemplate", "PromptTemplates"]
