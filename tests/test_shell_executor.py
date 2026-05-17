"""Unit tests for pyddock shell executor — policy matching, args validation, execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    ShellPolicyConfig,
)
from pyddock.shell_executor import RunShellOutput, ShellExecutor


def _make_config(
    shell: dict[str, ShellPolicyConfig] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["json"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
    )


class TestFindMatchingPolicy:
    """Tests for ShellExecutor._find_matching_policy()."""

    def test_exact_match(self, tmp_path: Path) -> None:
        """Exact command name matches policy with default regex."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["filelog.*"]),
        })
        executor = ShellExecutor(config, tmp_path)
        policy = executor._find_matching_policy("p4")
        assert policy is not None
        assert policy.command == "^p4$"

    def test_regex_match(self, tmp_path: Path) -> None:
        """Regex pattern matches command."""
        config = _make_config(shell={
            "scripts": ShellPolicyConfig(
                command=r"\.kiro/scripts/.*\.ps1", mode="allow"
            ),
        })
        executor = ShellExecutor(config, tmp_path)
        policy = executor._find_matching_policy(".kiro/scripts/build.ps1")
        assert policy is not None

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        """Command not matching any policy returns None."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["filelog.*"]),
        })
        executor = ShellExecutor(config, tmp_path)
        policy = executor._find_matching_policy("rm")
        assert policy is None

    def test_first_match_wins(self, tmp_path: Path) -> None:
        """When multiple policies could match, first one wins."""
        config = _make_config(shell={
            "p4-strict": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
            "p4-loose": ShellPolicyConfig(command="^p4$", mode="allow"),
        })
        executor = ShellExecutor(config, tmp_path)
        policy = executor._find_matching_policy("p4")
        assert policy is not None
        assert policy.allow == ["info"]

    def test_re_match_anchored_at_start(self, tmp_path: Path) -> None:
        """re.match() is anchored at start — partial match at end doesn't work."""
        config = _make_config(shell={
            "git": ShellPolicyConfig(command="^git$", mode="allow"),
        })
        executor = ShellExecutor(config, tmp_path)
        # "evil-git" should NOT match "^git$"
        policy = executor._find_matching_policy("evil-git")
        assert policy is None


class TestCheckArgsPolicy:
    """Tests for ShellExecutor._check_args_policy()."""

    def test_deny_mode_allows_matching_args(self, tmp_path: Path) -> None:
        """Deny-mode permits args matching an allow pattern."""
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["filelog.*", "files.*"]
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["filelog", "//depot/..."])
        assert result is None

    def test_deny_mode_rejects_non_matching_args(self, tmp_path: Path) -> None:
        """Deny-mode rejects args not matching any allow pattern."""
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["filelog.*", "files.*"]
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["submit", "-d", "hack"])
        assert result is not None
        assert "not permitted" in result
        assert "run_python" in result

    def test_deny_mode_empty_allow_rejects_all(self, tmp_path: Path) -> None:
        """Deny-mode with no allow patterns rejects everything."""
        policy = ShellPolicyConfig(command="^locked$", mode="deny", allow=[])
        config = _make_config(shell={"locked": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["anything"])
        assert result is not None
        assert "No argument patterns" in result

    def test_allow_mode_permits_non_matching_args(self, tmp_path: Path) -> None:
        """Allow-mode permits args not matching any deny pattern."""
        policy = ShellPolicyConfig(
            command="^git$", mode="allow", deny=["push.*", "force.*"]
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["status"])
        assert result is None

    def test_allow_mode_rejects_matching_deny(self, tmp_path: Path) -> None:
        """Allow-mode rejects args matching a deny pattern."""
        policy = ShellPolicyConfig(
            command="^git$", mode="allow", deny=["push.*", "force.*"]
        )
        config = _make_config(shell={"git": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["push", "origin", "main"])
        assert result is not None
        assert "deny pattern" in result
        assert "run_python" in result

    def test_allow_mode_empty_deny_permits_all(self, tmp_path: Path) -> None:
        """Allow-mode with no deny patterns permits everything."""
        policy = ShellPolicyConfig(command="^echo$", mode="allow", deny=[])
        config = _make_config(shell={"echo": policy})
        executor = ShellExecutor(config, tmp_path)
        result = executor._check_args_policy(policy, ["hello", "world"])
        assert result is None

    def test_args_joined_with_space(self, tmp_path: Path) -> None:
        """Args are joined with space for pattern matching."""
        policy = ShellPolicyConfig(
            command="^p4$", mode="deny", allow=["filelog //depot/.*"]
        )
        config = _make_config(shell={"p4": policy})
        executor = ShellExecutor(config, tmp_path)
        # "filelog //depot/main/..." joined = "filelog //depot/main/..."
        result = executor._check_args_policy(policy, ["filelog", "//depot/main/..."])
        assert result is None


class TestResolveCommand:
    """Tests for ShellExecutor._resolve_command()."""

    def test_ps1_extension(self, tmp_path: Path) -> None:
        """PowerShell scripts get pwsh or powershell prefix."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("build.ps1")
        # pwsh preferred, powershell fallback
        assert result[0] in ("pwsh", "powershell")
        assert result[1:] == ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "build.ps1"]

    def test_py_extension(self, tmp_path: Path) -> None:
        """Python scripts get python prefix."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("script.py")
        assert result == ["python", "script.py"]

    def test_sh_extension(self, tmp_path: Path) -> None:
        """Shell scripts get bash prefix."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("deploy.sh")
        assert result == ["bash", "deploy.sh"]

    def test_bat_extension(self, tmp_path: Path) -> None:
        """Batch files get cmd /c prefix."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("setup.bat")
        assert result == ["cmd", "/c", "setup.bat"]

    def test_no_extension(self, tmp_path: Path) -> None:
        """Commands without recognized extension run directly."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("p4")
        assert result == ["p4"]

    def test_unknown_extension(self, tmp_path: Path) -> None:
        """Commands with unrecognized extension run directly."""
        config = _make_config()
        executor = ShellExecutor(config, tmp_path)
        result = executor._resolve_command("tool.exe")
        assert result == ["tool.exe"]


class TestExecute:
    """Tests for ShellExecutor.execute() end-to-end."""

    def test_command_not_in_policy_rejected(self, tmp_path: Path) -> None:
        """Command not matching any policy returns error."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
        })
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute("rm", ["-rf", "/"], 10.0)
        assert result.exit_code == 1
        assert "not allowed" in result.stderr
        assert "run_python" in result.stderr

    def test_args_rejected_by_policy(self, tmp_path: Path) -> None:
        """Args not matching policy returns error."""
        config = _make_config(shell={
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
        })
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute("p4", ["submit"], 10.0)
        assert result.exit_code == 1
        assert "not permitted" in result.stderr

    def test_timeout_handling(self, tmp_path: Path) -> None:
        """Timeout returns structured error."""
        config = _make_config(shell={
            "python": ShellPolicyConfig(command="^python$", mode="allow", deny=[]),
        })
        executor = ShellExecutor(config, tmp_path)

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=1.0)
        mock_proc.pid = 12345
        mock_proc.wait.return_value = None

        with patch("pyddock.shell_executor.subprocess.Popen", return_value=mock_proc):
            with patch("pyddock.shell_executor.subprocess.run"):  # for taskkill
                result = executor.execute("python", ["-c", "import time; time.sleep(99)"], 1.0)

        assert result.exit_code == 1
        assert "TimeoutError" in result.stderr
        assert "run_python" in result.stderr

    def test_command_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError returns structured error."""
        config = _make_config(shell={
            "nonexistent": ShellPolicyConfig(command="^nonexistent$", mode="allow", deny=[]),
        })
        executor = ShellExecutor(config, tmp_path)

        with patch("pyddock.shell_executor.subprocess.Popen") as mock_popen:
            mock_popen.side_effect = FileNotFoundError()
            result = executor.execute("nonexistent", [], 10.0)

        assert result.exit_code == 1
        assert "not found" in result.stderr
        assert "run_python" in result.stderr

    def test_successful_execution(self, tmp_path: Path) -> None:
        """Successful command returns stdout/stderr/exit_code."""
        config = _make_config(shell={
            "python": ShellPolicyConfig(command="^python$", mode="allow", deny=[]),
        })
        executor = ShellExecutor(config, tmp_path)

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"hello\r\n", b"")
        mock_proc.returncode = 0

        with patch("pyddock.shell_executor.subprocess.Popen", return_value=mock_proc):
            result = executor.execute("python", ["-c", "print('hello')"], 10.0)

        assert result.exit_code == 0
        assert result.stdout == "hello\n"  # \r\n normalized
        assert result.stderr == ""

    def test_output_truncation(self, tmp_path: Path) -> None:
        """Output exceeding 64KB is truncated."""
        config = _make_config(shell={
            "echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[]),
        })
        executor = ShellExecutor(config, tmp_path)

        big_output = b"x" * 100_000
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (big_output, b"")
        mock_proc.returncode = 0

        with patch("pyddock.shell_executor.subprocess.Popen", return_value=mock_proc):
            result = executor.execute("echo", ["big"], 10.0)

        assert len(result.stdout) < 100_000
        assert "[truncated" in result.stdout



from pyddock.shell_executor import _derive_write_protected_paths


class TestDeriveWriteProtectedPaths:
    """Tests for _derive_write_protected_paths()."""

    def test_path_like_regex_with_slash(self) -> None:
        """Regex containing '/' is classified as path-like."""
        config = {
            "scripts": ShellPolicyConfig(
                command=r"\.kiro/scripts/.*\.ps1", mode="allow"
            ),
        }
        result = _derive_write_protected_paths(config)
        assert len(result) == 1
        assert ".kiro/scripts" in result[0]

    def test_path_like_regex_with_backslash(self) -> None:
        """Regex containing '\\\\' is classified as path-like."""
        config = {
            "scripts": ShellPolicyConfig(
                command=r"scripts\\\\build\.ps1", mode="allow"
            ),
        }
        result = _derive_write_protected_paths(config)
        assert len(result) == 1

    def test_path_like_regex_starting_with_dot(self) -> None:
        """Regex starting with '\\.' is classified as path-like."""
        config = {
            "scripts": ShellPolicyConfig(
                command=r"\./scripts/run\.sh", mode="allow"
            ),
        }
        result = _derive_write_protected_paths(config)
        assert len(result) == 1

    def test_non_path_regex_ignored(self) -> None:
        """Regex without path separators is NOT classified as path-like."""
        config = {
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
            "git": ShellPolicyConfig(command="^git$", mode="deny", allow=["status.*"]),
        }
        result = _derive_write_protected_paths(config)
        assert result == []

    def test_mixed_config(self) -> None:
        """Mix of path-like and non-path-like regexes."""
        config = {
            "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
            "scripts": ShellPolicyConfig(
                command=r"\.kiro/scripts/.*\.ps1", mode="allow"
            ),
        }
        result = _derive_write_protected_paths(config)
        assert len(result) == 1
        assert ".kiro/scripts" in result[0]
