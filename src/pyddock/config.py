"""Configuration loader for pyddock.

Resolution order:
1. Base config: .pyddock/pyddock.toml in the workspace (CWD), or default_config.toml bundled with the package
2. Overlay (optional): .pyddock/pyddock.override.toml — deep-merged on top of the base
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when the config file has an invalid structure."""


@dataclass
class ExecutionConfig:
    """Execution settings."""

    timeout: float = 30.0
    max_timeout: float = 3600.0


@dataclass
class ImportsConfig:
    """Import allowlist configuration."""

    allowed: list[str] = field(default_factory=list)
    workspace: dict[str, str] = field(default_factory=dict)
    pip_packages: dict[str, str] = field(default_factory=dict)


@dataclass
class GuardRule:
    """A single filesystem guard rule (regex → disposition)."""

    pattern: str  # regex pattern matched against resolved path (forward slashes)
    disposition: str  # "deny", "workspace", or "allow"


@dataclass
class FilesystemConfig:
    """Filesystem scoping configuration."""

    writable_paths: list[str] = field(default_factory=lambda: ["."])
    readable_paths: list[str] = field(default_factory=lambda: ["."])
    guards: list[GuardRule] = field(default_factory=list)


@dataclass
class ASTConfig:
    """AST validation configuration."""

    block_calls: list[str] = field(default_factory=list)
    block_attributes: list[str] = field(default_factory=list)


@dataclass
class RestrictionConfig:
    """Per-module restriction configuration."""

    mode: str  # "allow" or "deny"
    module_allow: list[str] = field(default_factory=list)
    module_deny: list[str] = field(default_factory=list)
    class_allow: list[str] = field(default_factory=list)
    class_deny: list[str] = field(default_factory=list)


@dataclass
class ShellPolicyConfig:
    """Per-command shell execution policy.

    Each [shell.<name>] section in pyddock.toml becomes one of these.
    """

    command: str  # regex matched against the command string
    mode: str  # "allow" — permit all except deny; "deny" — block all except allow
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    arg_paths: str = "workspace"  # "workspace" | "protected" | "none"


@dataclass
class PyddockConfig:
    """Top-level pyddock configuration."""

    execution: ExecutionConfig
    imports: ImportsConfig
    filesystem: FilesystemConfig
    ast: ASTConfig
    restrictions: dict[str, RestrictionConfig] = field(default_factory=dict)
    shell: dict[str, ShellPolicyConfig] = field(default_factory=dict)


def _parse_execution(data: dict) -> ExecutionConfig:
    """Parse the [execution] section."""
    section = data.get("execution", {})
    if not isinstance(section, dict):
        raise ConfigError("[execution] must be a table")
    timeout = section.get("timeout", 30.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError("[execution].timeout must be a positive number")
    max_timeout = section.get("max_timeout", 3600.0)
    if not isinstance(max_timeout, (int, float)) or max_timeout <= 0:
        raise ConfigError("[execution].max_timeout must be a positive number")
    return ExecutionConfig(timeout=float(timeout), max_timeout=float(max_timeout))


def _parse_imports(data: dict) -> ImportsConfig:
    """Parse the [imports] section (table of booleans/strings).

    Values:
      - true: allowed, import name == pip name
      - false: revoked / not allowed
      - string starting with '.', '/', or '\\': workspace package (editable install)
      - other non-empty string: pip package name (e.g. dateutil = "python-dateutil")
      - empty string: excluded (treated like false)
    """
    section = data.get("imports", {})
    if not isinstance(section, dict):
        raise ConfigError("[imports] must be a table")

    allowed: list[str] = []
    workspace: dict[str, str] = {}
    pip_packages: dict[str, str] = {}

    for name, value in section.items():
        if isinstance(value, bool):
            if value:
                allowed.append(name)
        elif isinstance(value, str):
            if not value:  # empty string → excluded (treated like false)
                continue
            allowed.append(name)
            # Distinguish workspace paths from pip package names:
            # paths start with '.', '/', or '\' (or contain path separators)
            if value.startswith((".", "/", "\\")) or "/" in value or "\\" in value:
                workspace[name] = value
            else:
                pip_packages[name] = value
        else:
            raise ConfigError(
                f"[imports].{name} must be a bool or string, got {type(value).__name__}"
            )

    return ImportsConfig(allowed=sorted(allowed), workspace=workspace, pip_packages=pip_packages)


def _parse_filesystem(data: dict) -> FilesystemConfig:
    """Parse the [filesystem] section."""
    section = data.get("filesystem", {})
    if not isinstance(section, dict):
        raise ConfigError("[filesystem] must be a table")
    writable = section.get("writable_paths", ["."])
    readable = section.get("readable_paths", ["."])
    if not isinstance(writable, list) or not all(isinstance(s, str) for s in writable):
        raise ConfigError("[filesystem].writable_paths must be a list of strings")
    if not isinstance(readable, list) or not all(isinstance(s, str) for s in readable):
        raise ConfigError("[filesystem].readable_paths must be a list of strings")

    # Parse [filesystem.guards] — ordered table of regex → disposition
    guards_section = section.get("guards", {})
    if not isinstance(guards_section, dict):
        raise ConfigError("[filesystem.guards] must be a table")
    guards: list[GuardRule] = []
    _valid_dispositions = ("deny", "workspace", "allow")
    for pattern, disposition in guards_section.items():
        if not isinstance(disposition, str):
            raise ConfigError(
                f"[filesystem.guards].'{pattern}' must be a string "
                f"('deny', 'workspace', or 'allow'), got {type(disposition).__name__}"
            )
        if disposition not in _valid_dispositions:
            raise ConfigError(
                f"[filesystem.guards].'{pattern}' must be 'deny', 'workspace', "
                f"or 'allow', got '{disposition}'"
            )
        guards.append(GuardRule(pattern=pattern, disposition=disposition))

    return FilesystemConfig(writable_paths=writable, readable_paths=readable, guards=guards)


def _parse_ast(data: dict) -> ASTConfig:
    """Parse the [ast] section."""
    section = data.get("ast", {})
    if not isinstance(section, dict):
        raise ConfigError("[ast] must be a table")
    block_calls = section.get("block_calls", [])
    block_attributes = section.get("block_attributes", [])
    if not isinstance(block_calls, list) or not all(
        isinstance(s, str) for s in block_calls
    ):
        raise ConfigError("[ast].block_calls must be a list of strings")
    if not isinstance(block_attributes, list) or not all(
        isinstance(s, str) for s in block_attributes
    ):
        raise ConfigError("[ast].block_attributes must be a list of strings")
    return ASTConfig(block_calls=block_calls, block_attributes=block_attributes)


def _parse_restrictions(data: dict) -> dict[str, RestrictionConfig]:
    """Parse the [restrictions] section (table of tables)."""
    section = data.get("restrictions", {})
    if not isinstance(section, dict):
        raise ConfigError("[restrictions] must be a table")

    restrictions: dict[str, RestrictionConfig] = {}
    for name, value in section.items():
        if not isinstance(value, dict):
            raise ConfigError(f"[restrictions.{name}] must be a table")

        mode = value.get("mode")
        if mode is None:
            raise ConfigError(
                f"[restrictions.{name}].mode is required (must be 'allow' or 'deny')"
            )
        if mode not in ("allow", "deny"):
            raise ConfigError(
                f"[restrictions.{name}].mode must be 'allow' or 'deny', got '{mode}'"
            )

        module_allow = value.get("module_allow", [])
        if not isinstance(module_allow, list) or not all(isinstance(s, str) for s in module_allow):
            raise ConfigError(
                f"[restrictions.{name}].module_allow must be a list of strings"
            )

        module_deny = value.get("module_deny", [])
        if not isinstance(module_deny, list) or not all(isinstance(s, str) for s in module_deny):
            raise ConfigError(
                f"[restrictions.{name}].module_deny must be a list of strings"
            )

        class_allow = value.get("class_allow", [])
        if not isinstance(class_allow, list) or not all(isinstance(s, str) for s in class_allow):
            raise ConfigError(
                f"[restrictions.{name}].class_allow must be a list of strings"
            )

        class_deny = value.get("class_deny", [])
        if not isinstance(class_deny, list) or not all(isinstance(s, str) for s in class_deny):
            raise ConfigError(
                f"[restrictions.{name}].class_deny must be a list of strings"
            )

        restrictions[name] = RestrictionConfig(
            mode=mode,
            module_allow=module_allow,
            module_deny=module_deny,
            class_allow=class_allow,
            class_deny=class_deny,
        )

    return restrictions


def _parse_shell(data: dict) -> dict[str, ShellPolicyConfig]:
    """Parse the [shell.*] sections (table of tables)."""
    section = data.get("shell", {})
    if not isinstance(section, dict):
        raise ConfigError("[shell] must be a table")

    shell: dict[str, ShellPolicyConfig] = {}
    for name, value in section.items():
        if not isinstance(value, dict):
            raise ConfigError(f"[shell.{name}] must be a table")

        mode = value.get("mode")
        if mode is None:
            raise ConfigError(
                f"[shell.{name}].mode is required (must be 'allow' or 'deny')"
            )
        if mode not in ("allow", "deny"):
            raise ConfigError(
                f"[shell.{name}].mode must be 'allow' or 'deny', got '{mode}'"
            )

        # Default command regex to ^<name>$ when not specified
        command = value.get("command")
        if command is None:
            command = f"^{name}$"
        elif not isinstance(command, str):
            raise ConfigError(f"[shell.{name}].command must be a string")

        allow = value.get("allow", [])
        if not isinstance(allow, list) or not all(isinstance(s, str) for s in allow):
            raise ConfigError(
                f"[shell.{name}].allow must be a list of strings"
            )

        deny = value.get("deny", [])
        if not isinstance(deny, list) or not all(isinstance(s, str) for s in deny):
            raise ConfigError(
                f"[shell.{name}].deny must be a list of strings"
            )

        arg_paths = value.get("arg_paths", "workspace")
        if arg_paths not in ("workspace", "protected", "none"):
            raise ConfigError(
                f"[shell.{name}].arg_paths must be 'workspace', 'protected', "
                f"or 'none', got '{arg_paths}'"
            )

        shell[name] = ShellPolicyConfig(
            command=command,
            mode=mode,
            allow=allow,
            deny=deny,
            arg_paths=arg_paths,
        )

    return shell


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* on top of *base* (neither is mutated).

    For each key in overlay:
      - If the key exists in base and both values are dicts → recurse.
      - Otherwise → overlay value wins.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_config(data: dict) -> PyddockConfig:
    """Parse a raw TOML dict into a PyddockConfig."""
    return PyddockConfig(
        execution=_parse_execution(data),
        imports=_parse_imports(data),
        filesystem=_parse_filesystem(data),
        ast=_parse_ast(data),
        restrictions=_parse_restrictions(data),
        shell=_parse_shell(data),
    )


def _get_default_config_path() -> Path:
    """Get the path to the package-bundled default config."""
    pkg_files = resources.files("pyddock")
    config_resource = pkg_files / "default_config.toml"
    # For file-based packages, this returns a Path directly.
    # For zipped packages, we'd need resources.as_file(), but for
    # a standard wheel install this works fine.
    return Path(str(config_resource))


def resolve_config_path(workspace: Path | None = None) -> Path:
    """Resolve which config file to use.

    Resolution order:
    1. .pyddock/pyddock.toml in the workspace (CWD if workspace is None)
    2. default_config.toml bundled with the pyddock package

    Args:
        workspace: The workspace root directory. Defaults to CWD.

    Returns:
        Path to the resolved config file.

    Raises:
        ConfigError: If no config file can be found.
    """
    if workspace is None:
        workspace = Path.cwd()

    # 1. Workspace config
    workspace_config = workspace / ".pyddock" / "pyddock.toml"
    if workspace_config.is_file():
        return workspace_config

    # 2. Package-bundled default
    default_path = _get_default_config_path()
    if default_path.is_file():
        return default_path

    raise ConfigError(
        "No pyddock config found. Expected .pyddock/pyddock.toml in workspace "
        "or default_config.toml in the pyddock package."
    )


def load_config(workspace: Path | None = None) -> PyddockConfig:
    """Load and parse the pyddock configuration.

    Resolves the base config (workspace pyddock.toml or bundled default), then
    checks for an optional overlay file (.pyddock/pyddock.override.toml). If the
    overlay exists it is deep-merged on top of the base before parsing.

    Args:
        workspace: The workspace root directory. Defaults to CWD.

    Returns:
        Parsed PyddockConfig.

    Raises:
        ConfigError: If the config file is missing or has an invalid structure.
    """
    if workspace is None:
        workspace = Path.cwd()

    config_path = resolve_config_path(workspace)

    try:
        raw_bytes = config_path.read_bytes()
    except OSError as e:
        raise ConfigError(f"Cannot read config file {config_path}: {e}") from e

    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            f"Invalid TOML in {config_path}: {e}"
        ) from e

    logger.debug("Loaded base config from %s", config_path)

    # Check for optional overlay
    override_path = workspace / ".pyddock" / "pyddock.override.toml"
    if override_path.is_file():
        try:
            override_bytes = override_path.read_bytes()
        except OSError as e:
            raise ConfigError(
                f"Cannot read override config file {override_path}: {e}"
            ) from e

        try:
            override_data = tomllib.loads(override_bytes.decode("utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(
                f"Invalid TOML in {override_path}: {e}"
            ) from e

        data = _deep_merge(data, override_data)
        logger.debug("Applied overlay from %s", override_path)
    else:
        logger.debug("No overlay found (checked %s)", override_path)

    try:
        return _parse_config(data)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(
            f"Invalid config structure in {config_path}: {e}"
        ) from e
