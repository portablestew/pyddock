"""Unit tests for ShellExecutor arg path scanning (_check_arg_paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    ShellPolicyConfig,
)
from pyddock.shell_executor import ShellExecutor


def _make_config(
    shell: dict[str, ShellPolicyConfig] | None = None,
    workspace_imports: dict[str, str] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    imports = ImportsConfig(
        allowed=["json"],
        workspace=workspace_imports or {},
    )
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=imports,
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
    )


class TestArgPathsBlocksPyddockDir:
    """Args targeting .pyddock/ are blocked (both workspace and protected modes)."""

    def test_blocks_direct_pyddock_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", ".pyddock/pwned.txt"])
        assert result is not None
        assert ".pyddock/" in result

    def test_blocks_nested_pyddock_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="protected"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", ".pyddock/venv/evil.py"])
        assert result is not None
        assert ".pyddock/" in result

    def test_allows_pyddock_tmp(self, tmp_path: Path) -> None:
        """Writes to .pyddock/tmp/ are allowed (used by tempfile)."""
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", ".pyddock/tmp/output.txt"])
        assert result is None


class TestArgPathsBlocksWorkspaceModules:
    """Args targeting workspace module directories are blocked."""

    def test_blocks_workspace_module_dir(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(
            shell={"p4": policy},
            workspace_imports={"my_tool": ".kiro/scripts/my-tool"},
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["-o", ".kiro/scripts/my-tool/hack.py"]
        )
        assert result is not None
        assert "workspace module" in result

    def test_allows_path_outside_workspace_module(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(
            shell={"p4": policy},
            workspace_imports={"my_tool": ".kiro/scripts/my-tool"},
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "output/result.txt"])
        assert result is None


class TestArgPathsBlocksScriptDirs:
    """Args targeting shell-executable script directories are blocked."""

    def test_blocks_script_dir_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        scripts_policy = ShellPolicyConfig(
            command=r"\.kiro/scripts/.*", mode="allow"
        )
        config = _make_config(shell={"p4": policy, "kiro-scripts": scripts_policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["-o", ".kiro/scripts/evil.ps1"]
        )
        assert result is not None
        assert "script" in result.lower()

    def test_allows_non_script_dir(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        scripts_policy = ShellPolicyConfig(
            command=r"\.kiro/scripts/.*", mode="allow"
        )
        config = _make_config(shell={"p4": policy, "kiro-scripts": scripts_policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "src/output.txt"])
        assert result is None


class TestArgPathsWorkspaceMode:
    """'workspace' mode blocks paths resolving outside the workspace."""

    def test_blocks_absolute_path_outside_workspace(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "C:/Windows/System32/evil.dll"])
        assert result is not None
        assert "outside" in result

    def test_blocks_relative_path_escaping_workspace(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "../../etc/passwd"])
        assert result is not None
        assert "outside" in result

    def test_allows_workspace_relative_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "output/result.txt"])
        assert result is None

    def test_allows_dot_relative_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "./subdir/file.txt"])
        assert result is None


class TestArgPathsProtectedMode:
    """'protected' mode only blocks protected dirs, allows outside workspace."""

    def test_allows_path_outside_workspace(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="protected"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "C:/other/place/file.txt"])
        assert result is None

    def test_still_blocks_pyddock(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["print.*"], arg_paths="protected"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", ".pyddock/config.toml"])
        assert result is not None
        assert ".pyddock/" in result


class TestArgPathsNoneMode:
    """'none' mode skips all path scanning."""

    def test_allows_pyddock_path(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^trusted$", mode="allow", deny=[], arg_paths="none"
        )
        config = _make_config(shell={"trusted": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", ".pyddock/whatever.txt"])
        assert result is None

    def test_allows_outside_workspace(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^trusted$", mode="allow", deny=[], arg_paths="none"
        )
        config = _make_config(shell={"trusted": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(policy, ["-o", "C:/anywhere/file.txt"])
        assert result is None


class TestArgPathsNonPathArgsIgnored:
    """Non-path-like args are not scanned."""

    def test_plain_args_ignored(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["filelog.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["filelog", "//depot/main/..."]
        )
        assert result is None

    def test_flags_without_paths_ignored(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(
            command="^git$", mode="allow", deny=[], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["log", "--oneline", "-n", "10"]
        )
        assert result is None


class TestArgPathsEmbeddedFlagValues:
    """Args with --flag=path patterns are scanned for embedded paths."""

    def test_blocks_output_equals_pyddock(self, tmp_path: Path) -> None:
        """The git --output=.pyddock/file bypass."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "--output=.pyddock/pwned_git.txt"]
        )
        assert result is not None
        assert ".pyddock/" in result

    def test_blocks_short_flag_equals_pyddock(self, tmp_path: Path) -> None:
        """-o=.pyddock/file is also caught."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "-o=.pyddock/pwned.txt"]
        )
        assert result is not None
        assert ".pyddock/" in result

    def test_blocks_flag_equals_outside_workspace(self, tmp_path: Path) -> None:
        """--output=C:/Windows/evil.txt is blocked in workspace mode."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "--output=C:/Windows/evil.txt"]
        )
        assert result is not None
        assert "outside" in result

    def test_allows_flag_equals_safe_workspace_path(self, tmp_path: Path) -> None:
        """--output=./output/diff.txt is fine."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "--output=./output/diff.txt"]
        )
        assert result is None

    def test_allows_flag_equals_non_path_value(self, tmp_path: Path) -> None:
        """--format=%H is not a path, should be ignored."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["log.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["log", "--format=%H %s"]
        )
        assert result is None

    def test_allows_flag_equals_pyddock_tmp(self, tmp_path: Path) -> None:
        """--output=.pyddock/tmp/file.txt is allowed (tmp is exempt)."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "--output=.pyddock/tmp/diff.txt"]
        )
        assert result is None

    def test_blocks_flag_equals_workspace_module(self, tmp_path: Path) -> None:
        """--output targeting a workspace module dir is blocked."""
        policy = ShellPolicyConfig(
            command="^git$", mode="deny", allow=["diff.*"], arg_paths="workspace"
        )
        config = _make_config(
            shell={"git": policy},
            workspace_imports={"my_tool": "src/my_tool"},
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_arg_paths(
            policy, ["diff", "--output=src/my_tool/injected.py"]
        )
        assert result is not None
        assert "workspace module" in result


class TestArgPathsIntegration:
    """End-to-end: execute() rejects commands with bad path args."""

    def test_p4_print_to_pyddock_blocked(self, tmp_path: Path) -> None:
        """The original exploit: p4 print -o .pyddock/pwned.txt."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(
                command="^p4$", mode="deny", allow=["print.*"],
                arg_paths="workspace",
            ),
        })
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute(
            "p4", ["print", "-o", ".pyddock/pwned.txt", "//depot/file"], 10.0
        )
        assert result.exit_code == 1
        assert ".pyddock/" in result.stderr

    def test_p4_print_to_workspace_allowed(self, tmp_path: Path) -> None:
        """p4 print -o to a workspace path is fine."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(
                command="^p4$", mode="deny", allow=["print.*"],
                arg_paths="workspace",
            ),
        })
        executor = ShellExecutor(config, tmp_path)
        # Mock the actual execution since p4 isn't available in test env
        from unittest.mock import patch, MagicMock

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"file content", b"")
        mock_proc.returncode = 0

        with patch("pyddock.shell_executor.subprocess.Popen", return_value=mock_proc):
            result = executor.execute(
                "p4", ["print", "-o", "output/file.txt", "//depot/file"], 10.0
            )
        assert result.exit_code == 0
