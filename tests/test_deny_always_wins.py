"""Tests for the shared "deny always wins" shell-policy matching.

The allow/deny decision is centralized in shell_executor (args_match_deny /
args_match_allow / evaluate_arg_policy) and reused by all three enforcement
sites: run_shell (ShellExecutor), subprocess.run inside run_python
(_subprocess_patch), and the GitPython guard (_gitpython_patch).

Key invariants under test:
  - deny is checked first, in BOTH modes (deny wins over allow)
  - deny uses re.search (matches a token anywhere, incl. after a newline)
  - allow uses re.match (start-anchored to the leading verb)
  - the three sites agree (the git RCE token `ext::` is blocked on each)
"""

from __future__ import annotations

import sys
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
from pyddock.executor import SubprocessExecutor
from pyddock.shell_executor import (
    ShellExecutor,
    args_match_allow,
    args_match_deny,
    evaluate_arg_policy,
)
from pyddock._gitpython_patch import build_git_command_validator
from pyddock.venv_manager import VenvManager


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class TestArgsMatchDeny:
    def test_matches_anywhere(self) -> None:
        assert args_match_deny(["ext::"], "fetch ext::sh -c evil") == "ext::"

    def test_matches_at_start(self) -> None:
        assert args_match_deny(["ext::"], "ext::sh -c evil") == "ext::"

    def test_matches_across_newline(self) -> None:
        # re.search (not re.match(".*tok")) — a token hidden after a newline must
        # still be caught, since "." does not match newlines.
        assert args_match_deny(["ext::"], "fetch \next::sh -c evil") == "ext::"

    def test_no_match_returns_none(self) -> None:
        assert args_match_deny(["ext::"], "fetch origin main") is None

    def test_returns_first_matching_pattern(self) -> None:
        assert args_match_deny(["--upload-pack", "ext::"], "x --upload-pack=y") == "--upload-pack"


class TestArgsMatchAllow:
    def test_anchored_at_start(self) -> None:
        assert args_match_allow(["status.*"], "status --short") is True

    def test_does_not_match_midstring(self) -> None:
        # Anchored: an allowed verb appearing later must NOT pass.
        assert args_match_allow(["status.*"], "frobnicate --then status") is False

    def test_any_of_multiple(self) -> None:
        assert args_match_allow(["log.*", "status.*"], "log --oneline") is True

    def test_empty_allow_is_false(self) -> None:
        assert args_match_allow([], "status") is False


class TestEvaluateArgPolicy:
    def test_deny_wins_in_deny_mode(self) -> None:
        reason = evaluate_arg_policy(
            "fetch ext::x", mode="deny", allow=["fetch.*"], deny=["ext::"]
        )
        assert reason is not None and "deny pattern" in reason

    def test_deny_wins_in_allow_mode(self) -> None:
        # Even in allow-by-default mode, a deny match rejects.
        reason = evaluate_arg_policy(
            "run ext::x", mode="allow", allow=[], deny=["ext::"]
        )
        assert reason is not None and "deny pattern" in reason

    def test_allowed_passes(self) -> None:
        assert evaluate_arg_policy(
            "status --short", mode="deny", allow=["status.*"], deny=["ext::"]
        ) is None

    def test_not_in_allowlist_rejected(self) -> None:
        reason = evaluate_arg_policy(
            "frob status", mode="deny", allow=["status.*"], deny=[]
        )
        assert reason is not None and "allow-list" in reason

    def test_empty_allow_in_deny_mode(self) -> None:
        reason = evaluate_arg_policy("anything", mode="deny", allow=[], deny=[])
        assert reason is not None and "no argument patterns" in reason

    def test_allow_mode_without_deny_passes(self) -> None:
        assert evaluate_arg_policy(
            "literally anything", mode="allow", allow=[], deny=[]
        ) is None


# ---------------------------------------------------------------------------
# Site 1: run_shell (ShellExecutor)
# ---------------------------------------------------------------------------


def _git_policy() -> ShellPolicyConfig:
    return ShellPolicyConfig(
        command="^git$",
        mode="deny",
        allow=["fetch.*", "status.*", "ls-remote.*"],
        deny=["ext::", "--upload-pack", "--receive-pack"],
    )


def _shell_config() -> PyddockConfig:
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["json"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell={"git": _git_policy()},
    )


class TestRunShellPath:
    def test_ext_transport_denied(self, tmp_path: Path) -> None:
        ex = ShellExecutor(_shell_config(), tmp_path)
        err = ex._check_args_policy(_git_policy(), ["fetch", "ext::sh -c evil"])
        assert err is not None and "deny pattern" in err

    def test_upload_pack_denied(self, tmp_path: Path) -> None:
        ex = ShellExecutor(_shell_config(), tmp_path)
        err = ex._check_args_policy(_git_policy(), ["fetch", "--upload-pack=/bin/sh", "origin"])
        assert err is not None and "deny pattern" in err

    def test_ls_remote_ext_denied(self, tmp_path: Path) -> None:
        ex = ShellExecutor(_shell_config(), tmp_path)
        err = ex._check_args_policy(_git_policy(), ["ls-remote", "ext::sh -c evil"])
        assert err is not None and "deny pattern" in err

    def test_plain_fetch_allowed(self, tmp_path: Path) -> None:
        ex = ShellExecutor(_shell_config(), tmp_path)
        assert ex._check_args_policy(_git_policy(), ["fetch", "origin", "main"]) is None

    def test_deny_wins_in_allow_mode(self, tmp_path: Path) -> None:
        policy = ShellPolicyConfig(command="^tool$", mode="allow", deny=["ext::"])
        ex = ShellExecutor(_shell_config(), tmp_path)
        # mid-string deny token rejected even in allow-by-default mode
        assert ex._check_args_policy(policy, ["run", "ext::x"]) is not None
        assert ex._check_args_policy(policy, ["run", "safe"]) is None


# ---------------------------------------------------------------------------
# Site 2: subprocess.run inside run_python (end-to-end through the executor)
# ---------------------------------------------------------------------------


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


class TestSubprocessPath:
    def test_ext_transport_denied_end_to_end(
        self, tmp_path: Path, venv_manager: VenvManager
    ) -> None:
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["subprocess", "os"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            shell={"git": _git_policy()},
        )
        executor = SubprocessExecutor(config, venv_manager)
        # The deny check rejects before any process is spawned, so this needs no
        # real git binary.
        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['git', 'fetch', 'ext::sh -c evil'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 15, tmp_path)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout
        assert "deny pattern" in result.stdout


# ---------------------------------------------------------------------------
# Site 3: GitPython guard
# ---------------------------------------------------------------------------


class TestGitPythonGuardDeny:
    @pytest.fixture
    def validate(self):
        config = {
            "imports": {"allowed": ["git"]},
            "shell": {
                "git": {
                    "mode": "deny",
                    "allow": ["fetch.*", "status.*", "ls-remote.*"],
                    "deny": ["ext::", "--upload-pack", "--receive-pack"],
                }
            },
        }
        return build_git_command_validator(config, deny_messages=[])

    @pytest.mark.parametrize("command", [
        pytest.param(["git", "fetch", "ext::sh -c evil"], id="fetch_ext"),
        pytest.param(["git", "ls-remote", "ext::sh -c evil"], id="ls_remote_ext"),
        pytest.param(["git", "fetch", "--upload-pack=/bin/sh", "origin"], id="upload_pack"),
        pytest.param(["git", "fetch", "--receive-pack=/bin/sh", "origin"], id="receive_pack"),
        pytest.param(["git", "pull", "ext::sh -c evil"], id="pull_ext"),
    ])
    def test_dangerous_tokens_denied(self, validate, command) -> None:
        with pytest.raises(PermissionError):
            validate(command)

    def test_plain_fetch_allowed(self, validate) -> None:
        validate(["git", "fetch", "origin"])  # should not raise
