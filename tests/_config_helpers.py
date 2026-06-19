"""Shared helpers for building valid pyddock TOML configs in tests.

`_parse_config` fails closed when a policy-bearing section is missing
(`[execution]`, `[imports]`, `[filesystem]`, `[ast]`, `[audit]`). Tests that
write configs to disk should build them through these helpers so every config
carries the required sections, and so the "valid baseline" lives in one place.

Usage:
    write_workspace_config(tmp_path)                       # all defaults
    write_workspace_config(tmp_path, imports="[imports]\\nre = true\\n")
    make_config_toml(extra='[shell.p4]\\nmode = "deny"\\n')  # add a section
    make_config_toml(ast="")                               # omit a section (negative test)
"""

from __future__ import annotations

from pathlib import Path

# Minimal valid content for each required section.
DEFAULT_SECTIONS: dict[str, str] = {
    "execution": "[execution]\ntimeout = 30\nmax_timeout = 3600\n",
    "imports": "[imports]\njson = true\n",
    "filesystem": '[filesystem]\nwritable_paths = ["."]\nreadable_paths = ["."]\n',
    "ast": "[ast]\nblock_calls = []\nblock_attributes = []\n",
    "audit": '[audit]\n"open" = "fs"\n',
}

_ORDER = ("execution", "imports", "filesystem", "ast", "audit")


def make_config_toml(*, extra: str = "", **overrides: str) -> str:
    """Build a complete config TOML string.

    Each required section can be overridden by keyword (pass the full section
    text, or "" to omit it entirely for negative tests). `extra` is appended
    verbatim for additive sections like [restrictions]/[shell]/[deny_messages].
    """
    parts: list[str] = []
    for name in _ORDER:
        section = overrides.get(name, DEFAULT_SECTIONS[name])
        if section:
            parts.append(section)
    if extra:
        parts.append(extra)
    return "\n".join(parts) + "\n"


def write_workspace_config(tmp_path: Path, *, extra: str = "", **overrides: str) -> Path:
    """Write a complete .pyddock/pyddock.toml under *tmp_path* and return its path."""
    cfg = tmp_path / ".pyddock" / "pyddock.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(make_config_toml(extra=extra, **overrides), encoding="utf-8")
    return cfg
