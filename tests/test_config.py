"""Unit tests for pyddock config loader — resolution order and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.config import (
    ConfigError,
    PyddockConfig,
    load_config,
    resolve_config_path,
)


class TestConfigResolution:
    """Tests for config file resolution order."""

    def test_workspace_config_takes_priority(self, tmp_path: Path) -> None:
        """Workspace .pyddock/pyddock.toml is used when present."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 99\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n"
        )

        config = load_config(tmp_path)
        assert config.execution.timeout == 99

    def test_fallback_to_package_default(self, tmp_path: Path) -> None:
        """When no workspace config exists, package default is used."""
        # tmp_path has no .pyddock/ directory
        config = load_config(tmp_path)
        # Should load the package default (timeout=30)
        assert config.execution.timeout == 30
        assert "json" in config.imports.allowed

    def test_workspace_config_fully_replaces_default(self, tmp_path: Path) -> None:
        """Workspace config replaces default entirely — no merging."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        # Minimal config with only one allowed import
        workspace_cfg.write_text(
            "[execution]\ntimeout = 5\n\n[imports]\nre = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n"
        )

        config = load_config(tmp_path)
        # Only "re" is allowed — default list is NOT merged in
        assert config.imports.allowed == ["re"]
        assert "json" not in config.imports.allowed

    def test_resolve_returns_workspace_path(self, tmp_path: Path) -> None:
        """resolve_config_path returns workspace config when it exists."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text("[execution]\ntimeout = 1\n")

        result = resolve_config_path(tmp_path)
        assert result == workspace_cfg


class TestConfigValidation:
    """Tests for config structure validation."""

    def test_invalid_toml_raises(self, tmp_path: Path) -> None:
        """Malformed TOML raises ConfigError."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text("this is not valid toml [[[")

        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(tmp_path)

    def test_negative_timeout_raises(self, tmp_path: Path) -> None:
        """Negative timeout raises ConfigError."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = -1\n\n[imports]\n"
        )

        with pytest.raises(ConfigError, match="positive"):
            load_config(tmp_path)

    def test_invalid_restriction_mode_raises(self, tmp_path: Path) -> None:
        """Invalid restriction mode raises ConfigError."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n"
            '[restrictions.boto3]\nmode = "invalid"\n'
        )

        with pytest.raises(ConfigError, match="'allow' or 'deny'"):
            load_config(tmp_path)

    def test_restrictions_parsed_correctly(self, tmp_path: Path) -> None:
        """Restrictions with all fields are parsed into RestrictionConfig."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\nboto3 = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n"
            '[restrictions.boto3]\nmode = "deny"\n'
            'module_allow = ["client"]\n'
            'class_allow = ["list_.*", "describe_.*"]\n'
        )

        config = load_config(tmp_path)
        assert "boto3" in config.restrictions
        r = config.restrictions["boto3"]
        assert r.mode == "deny"
        assert r.module_allow == ["client"]
        assert r.class_allow == ["list_.*", "describe_.*"]



class TestShellConfigParsing:
    """Tests for [shell.*] config section parsing."""

    def test_valid_shell_section_parsed(self, tmp_path: Path) -> None:
        """Valid [shell.*] section is parsed into ShellPolicyConfig."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
            '[shell.p4]\nmode = "deny"\nallow = ["filelog.*", "files.*"]\n'
        )

        config = load_config(tmp_path)
        assert "p4" in config.shell
        p = config.shell["p4"]
        assert p.mode == "deny"
        assert p.allow == ["filelog.*", "files.*"]
        assert p.command == "^p4$"  # default regex

    def test_shell_missing_mode_raises(self, tmp_path: Path) -> None:
        """Shell section without mode raises ConfigError."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
            '[shell.p4]\nallow = ["filelog.*"]\n'
        )

        with pytest.raises(ConfigError, match="mode is required"):
            load_config(tmp_path)

    def test_shell_invalid_mode_raises(self, tmp_path: Path) -> None:
        """Shell section with invalid mode raises ConfigError."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
            '[shell.p4]\nmode = "invalid"\n'
        )

        with pytest.raises(ConfigError, match="'allow' or 'deny'"):
            load_config(tmp_path)

    def test_shell_default_command_regex(self, tmp_path: Path) -> None:
        """Shell section without command field defaults to ^<name>$."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
            '[shell.git]\nmode = "allow"\ndeny = ["push.*"]\n'
        )

        config = load_config(tmp_path)
        assert config.shell["git"].command == "^git$"

    def test_shell_custom_command_regex(self, tmp_path: Path) -> None:
        """Shell section with explicit command field uses that regex."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
            '[shell.scripts]\ncommand = "\\\\.kiro/scripts/.*\\\\.ps1"\nmode = "allow"\ndeny = []\n'
        )

        config = load_config(tmp_path)
        assert config.shell["scripts"].command == r"\.kiro/scripts/.*\.ps1"

    def test_empty_shell_section(self, tmp_path: Path) -> None:
        """Empty [shell] section (no sub-tables) produces empty dict."""
        workspace_cfg = tmp_path / ".pyddock" / "pyddock.toml"
        workspace_cfg.parent.mkdir(parents=True)
        workspace_cfg.write_text(
            "[execution]\ntimeout = 30\n\n[imports]\njson = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['.']\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n[restrictions]\n\n"
        )

        config = load_config(tmp_path)
        assert config.shell == {}


class TestConfigOverride:
    """Tests for .pyddock/pyddock.override.toml overlay merging."""

    # Minimal base config template used across tests
    _BASE_TEMPLATE = (
        "[execution]\ntimeout = {timeout}\nmax_timeout = {max_timeout}\n\n"
        "[imports]\n{imports}\n\n"
        "[filesystem]\n{filesystem}\n\n"
        "[ast]\nblock_calls = []\nblock_attributes = []\n\n"
        "[restrictions]\n{restrictions}\n\n"
        "{shell}"
    )

    def _write_base(self, tmp_path: Path, content: str) -> None:
        cfg = tmp_path / ".pyddock" / "pyddock.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(content)

    def _write_override(self, tmp_path: Path, content: str) -> None:
        override = tmp_path / ".pyddock" / "pyddock.override.toml"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text(content)

    def test_override_adds_import(self, tmp_path: Path) -> None:
        """Override adds a new import on top of the base imports."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, "[imports]\nrequests = true\n")

        config = load_config(tmp_path)
        assert "json" in config.imports.allowed
        assert "requests" in config.imports.allowed

    def test_override_revokes_import(self, tmp_path: Path) -> None:
        """Override can revoke an import by setting it to false."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true\nboto3 = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, "[imports]\nboto3 = false\n")

        config = load_config(tmp_path)
        assert "json" in config.imports.allowed
        assert "boto3" not in config.imports.allowed

    def test_override_changes_scalar(self, tmp_path: Path) -> None:
        """Override can change a scalar value while leaving siblings intact."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, "[execution]\ntimeout = 120\n")

        config = load_config(tmp_path)
        assert config.execution.timeout == 120
        assert config.execution.max_timeout == 3600  # inherited from base

    def test_override_adds_restriction(self, tmp_path: Path) -> None:
        """Override can add a new restriction entry alongside existing ones."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions='[restrictions.polars]\nmode = "allow"\ndeny = ["write_.*"]',
                shell="",
            ),
        )
        self._write_override(
            tmp_path,
            '[restrictions.requests]\nmode = "allow"\ndeny = ["delete"]\n',
        )

        config = load_config(tmp_path)
        assert "polars" in config.restrictions
        assert "requests" in config.restrictions

    def test_override_adds_shell_policy(self, tmp_path: Path) -> None:
        """Override can add a new shell policy alongside existing ones."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell='[shell.git]\nmode = "deny"\nallow = ["status.*"]\n',
            ),
        )
        self._write_override(
            tmp_path,
            '[shell.npm]\nmode = "deny"\nallow = ["run build.*"]\n',
        )

        config = load_config(tmp_path)
        assert "git" in config.shell
        assert "npm" in config.shell

    def test_override_replaces_filesystem_list(self, tmp_path: Path) -> None:
        """Override replaces a list value wholesale; unmentioned siblings are inherited."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["*"]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, '[filesystem]\nreadable_paths = ["."]\n')

        config = load_config(tmp_path)
        assert config.filesystem.readable_paths == ["."]
        assert config.filesystem.writable_paths == ["."]  # inherited from base

    def test_no_override_file(self, tmp_path: Path) -> None:
        """Without an override file, config loads normally from base only."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        # No override file created

        config = load_config(tmp_path)
        assert isinstance(config, PyddockConfig)
        assert config.execution.timeout == 30
        assert "json" in config.imports.allowed

    def test_override_invalid_toml(self, tmp_path: Path) -> None:
        """Invalid TOML in override file raises ConfigError."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=30,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, "this is not valid toml [[[")

        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(tmp_path)

    def test_override_applies_on_workspace_config(self, tmp_path: Path) -> None:
        """Override merges on top of workspace config, not the bundled default."""
        self._write_base(
            tmp_path,
            self._BASE_TEMPLATE.format(
                timeout=5,
                max_timeout=3600,
                imports="json = true",
                filesystem='writable_paths = ["."]\nreadable_paths = ["."]',
                restrictions="",
                shell="",
            ),
        )
        self._write_override(tmp_path, "[execution]\ntimeout = 99\n")

        config = load_config(tmp_path)
        assert config.execution.timeout == 99
