"""Tests for filesystem guards ([filesystem.guards] config section).

Tests config parsing and runtime enforcement of regex-based path guards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock.config import (
    ConfigError,
    GuardRule,
    load_config,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager


# =============================================================================
# Config parsing tests
# =============================================================================


class TestGuardConfigParsing:
    """Tests for [filesystem.guards] config parsing."""

    def _write_config(self, tmp_path: Path, guards_toml: str) -> None:
        cfg = tmp_path / ".pyddock" / "pyddock.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            "[execution]\ntimeout = 30\nmax_timeout = 60\n\n"
            "[imports]\nos = true\nsys = true\npathlib = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['*']\n\n"
            f"{guards_toml}\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n"
            "[restrictions]\n",
            encoding="utf-8",
        )

    def test_guards_parsed_in_order(self, tmp_path: Path) -> None:
        """Guards are parsed in TOML insertion order with correct types."""
        self._write_config(tmp_path, (
            "[filesystem.guards]\n"
            "'/\\.ssh/' = 'deny-all'\n"
            "'/\\.env$' = 'workspace'\n"
            "'/var/data/' = 'allow'\n"
        ))
        config = load_config(tmp_path)
        assert len(config.filesystem.guards) == 3
        assert config.filesystem.guards[0] == GuardRule(pattern="/\\.ssh/", disposition="deny-all")
        assert config.filesystem.guards[1] == GuardRule(pattern="/\\.env$", disposition="workspace")
        assert config.filesystem.guards[2] == GuardRule(pattern="/var/data/", disposition="allow")

    def test_invalid_disposition_raises(self, tmp_path: Path) -> None:
        """Invalid disposition value raises ConfigError."""
        self._write_config(tmp_path, (
            "[filesystem.guards]\n"
            "'/\\.ssh/' = 'block'\n"
        ))
        with pytest.raises(ConfigError, match="'deny-agent', 'deny-all', 'workspace', or 'allow'"):
            load_config(tmp_path)


# =============================================================================
# Runtime enforcement tests
# =============================================================================


class TestGuardEnforcement:
    """Tests for runtime enforcement of filesystem guards."""

    def _make_executor(self, tmp_path: Path, guards_toml: str) -> tuple[SubprocessExecutor, Path]:
        """Create a workspace with guards and return (executor, workspace_root)."""
        ws = tmp_path / "ws"
        ws.mkdir()
        cfg_dir = ws / ".pyddock"
        cfg_dir.mkdir()
        (cfg_dir / "tmp").mkdir()
        (cfg_dir / "pyddock.toml").write_text(
            "[execution]\ntimeout = 10\nmax_timeout = 30\n\n"
            "[imports]\nos = true\nsys = true\npathlib = true\nio = true\n\n"
            "[filesystem]\nwritable_paths = ['.']\nreadable_paths = ['*']\n\n"
            f"{guards_toml}\n\n"
            "[ast]\nblock_calls = []\nblock_attributes = []\n\n"
            "[restrictions]\n",
            encoding="utf-8",
        )
        config = load_config(ws)
        vm = VenvManager(venv_path=ws / ".pyddock" / "venv", allowed_imports=config.imports.allowed)
        vm.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
        executor = SubprocessExecutor(config, vm)
        return executor, ws

    def test_deny_all_blocks_read(self, tmp_path: Path) -> None:
        """Guard with disposition 'deny-all' blocks reads unconditionally."""
        executor, ws = self._make_executor(tmp_path, (
            "[filesystem.guards]\n"
            "'/secret\\.txt$' = 'deny-all'\n"
        ))
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")

        code = f"print(open(r'{secret}').read())"
        result = executor.execute(code, [], timeout=10, workspace_root=ws)
        assert result.exit_code != 0 or "PermissionError" in result.stderr
        assert "top secret" not in result.stdout

    def test_workspace_allows_inside(self, tmp_path: Path) -> None:
        """Guard with disposition 'workspace' allows access inside workspace."""
        executor, ws = self._make_executor(tmp_path, (
            "[filesystem.guards]\n"
            "'/\\.env$' = 'workspace'\n"
        ))
        env_file = ws / ".env"
        env_file.write_text("DB_HOST=localhost", encoding="utf-8")

        code = f"print(open(r'{env_file}').read())"
        result = executor.execute(code, [], timeout=10, workspace_root=ws)
        assert "DB_HOST=localhost" in result.stdout

    def test_workspace_blocks_outside(self, tmp_path: Path) -> None:
        """Guard with disposition 'workspace' blocks access outside workspace."""
        executor, ws = self._make_executor(tmp_path, (
            "[filesystem.guards]\n"
            "'/\\.env$' = 'workspace'\n"
        ))
        outside_env = tmp_path / ".env"
        outside_env.write_text("SECRET_KEY=abc123", encoding="utf-8")

        code = f"print(open(r'{outside_env}').read())"
        result = executor.execute(code, [], timeout=10, workspace_root=ws)
        assert result.exit_code != 0 or "PermissionError" in result.stderr
        assert "SECRET_KEY" not in result.stdout

    def test_allow_permits_outside_workspace(self, tmp_path: Path) -> None:
        """Guard with disposition 'allow' permits access outside workspace."""
        executor, ws = self._make_executor(tmp_path, (
            "[filesystem.guards]\n"
            "'/allowed_data\\.txt$' = 'allow'\n"
        ))
        outside = tmp_path / "allowed_data.txt"
        outside.write_text("public info", encoding="utf-8")

        code = f"print(open(r'{outside}').read())"
        result = executor.execute(code, [], timeout=10, workspace_root=ws)
        assert "public info" in result.stdout

    def test_first_match_wins(self, tmp_path: Path) -> None:
        """When multiple guards match, the first one wins."""
        executor, ws = self._make_executor(tmp_path, (
            "[filesystem.guards]\n"
            "'/special\\.env$' = 'allow'\n"
            "'/\\.env' = 'deny-all'\n"
        ))
        special = tmp_path / "special.env"
        special.write_text("allowed", encoding="utf-8")

        code = f"print(open(r'{special}').read())"
        result = executor.execute(code, [], timeout=10, workspace_root=ws)
        assert "allowed" in result.stdout
