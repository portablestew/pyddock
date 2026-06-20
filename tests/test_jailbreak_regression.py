"""Regression tests for two jailbreaks found during a black-box pentest.

1. `git rebase --exec`/`-x` — an allowed git subcommand that runs an arbitrary
   per-commit shell command in git's own (unsandboxed) child process. The deny
   list previously stopped `ext::`/`--upload-pack`/`--receive-pack`/`-c` but not
   `--exec`/`-x`. Fixed by adding deny patterns to [shell.git].

2. NTFS alternate data streams — `open(".pyddock:pwned", "w")` writes a data
   stream onto the protected `.pyddock` directory while the lexical leaf
   `.pyddock:pwned` evades the `relative_to('.pyddock')` containment check.
   Fixed by rejecting any path component carrying a `:` stream reference
   (has_ntfs_stream), wired into the shared _check_read/_check_write closures
   and the shell arg-path / cwd scanners.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock._base import has_ntfs_stream
from pyddock.config import load_config
from pyddock.executor import SubprocessExecutor
from pyddock.shell_executor import evaluate_arg_policy, evaluate_arg_paths
from pyddock.venv_manager import VenvManager

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


def _git_policy():
    """The real, shipped [shell.git] policy from the bundled default config."""
    config = load_config()
    policy = config.shell["git"]
    return policy


# ---------------------------------------------------------------------------
# 1. git rebase --exec / -x
# ---------------------------------------------------------------------------


class TestGitExecDenied:
    @pytest.mark.parametrize(
        "args_str",
        [
            "rebase --exec sh foo.sh HEAD~1",
            "rebase --autostash --exec sh C:/ws/foo.sh HEAD~1",
            "rebase --exec=make-test HEAD~1",
            "rebase -x make-test HEAD~1",
            "rebase -i --exec cmd HEAD~1",
            "pull --rebase -x cmd",
            "pull --rebase --exec cmd",
        ],
    )
    def test_exec_forms_are_denied(self, args_str: str) -> None:
        p = _git_policy()
        reason = evaluate_arg_policy(
            args_str, mode=p.mode, allow=p.allow, deny=p.deny
        )
        assert reason is not None, f"expected denial for: {args_str}"
        assert "deny" in reason

    @pytest.mark.parametrize(
        "args_str",
        [
            "clean -x",           # remove ignored files — NOT command execution
            "clean -xfd",
            "clean -X",
            "status",
            "rebase --continue",  # ordinary rebase without an exec command
            "rebase --abort",
            "log --oneline -10",
        ],
    )
    def test_legitimate_forms_are_allowed(self, args_str: str) -> None:
        p = _git_policy()
        reason = evaluate_arg_policy(
            args_str, mode=p.mode, allow=p.allow, deny=p.deny
        )
        assert reason is None, f"unexpectedly denied: {args_str} -> {reason}"

    def test_exec_path_still_distinct_from_exec(self) -> None:
        """--exec-path (subprogram lookup dir) and --exec (per-commit cmd) are
        both denied, but via their own patterns — neither masks the other."""
        p = _git_policy()
        assert evaluate_arg_policy("rev-parse --exec-path", mode=p.mode, allow=p.allow, deny=p.deny) is not None
        assert evaluate_arg_policy("rebase --exec cmd HEAD~1", mode=p.mode, allow=p.allow, deny=p.deny) is not None


# ---------------------------------------------------------------------------
# 2. NTFS alternate data streams
# ---------------------------------------------------------------------------


class TestHasNtfsStream:
    def test_normal_paths_are_not_streams(self) -> None:
        for ok in [
            "file.txt",
            ".pyddock/pwned.txt",
            "C:/Git/pyddock/.pyddock/x.txt",
            "C:\\Git\\pyddock\\file",
            "sub/dir/file.log",
        ]:
            assert has_ntfs_stream(ok) is False, ok

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_stream_refs_detected_on_windows(self) -> None:
        for bad in [
            ".pyddock:pwned",
            ".pyddock:pwned.txt",
            "file.txt:stream",
            "C:/Git/pyddock/.pyddock:pwned",
            ".pyddock/x.txt::$DATA",
            "dir\\file:stream",
        ]:
            assert has_ntfs_stream(bad) is True, bad

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_drive_relative_not_flagged(self) -> None:
        # 'C:rest' (drive-relative) and 'C:\\...' are the only legal colons.
        assert has_ntfs_stream("C:relative\\file") is False
        assert has_ntfs_stream("C:\\abs\\file") is False


class TestAdsWriteBlockedEndToEnd:
    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_ads_write_to_pyddock_dir_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """open('.pyddock:pwned', 'w') must be rejected — it would otherwise
        write a data stream onto the protected .pyddock directory."""
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["pathlib", "io"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=["__subclasses__", "__globals__"],
            ),
            restrictions={},
        )
        executor = SubprocessExecutor(config, venv_manager)
        source = (
            "import io, pathlib\n"
            "blocked = 0\n"
            "for target in ['.pyddock:pwned', '.pyddock:pwned.txt']:\n"
            "    try:\n"
            "        open(target, 'w').write('x')\n"
            "        print(f'SHOULD NOT REACH: {target}')\n"
            "    except PermissionError as e:\n"
            "        blocked += 1\n"
            "        assert 'alternate data stream' in str(e), str(e)\n"
            "    # also via the leaked real FileIO type (audit-hook path)\n"
            "    try:\n"
            "        io.FileIO('.pyddock:pwned2', 'w')\n"
            "        print('SHOULD NOT REACH: FileIO')\n"
            "    except PermissionError:\n"
            "        blocked += 1\n"
            "print(f'BLOCKED={blocked}')\n"
        )
        result = executor.execute(source, [], 10, workspace)
        assert result.exit_code == 0, result.stderr
        assert "SHOULD NOT REACH" not in result.stdout
        assert "BLOCKED=4" in result.stdout
        assert not (workspace / ".pyddock:pwned").exists()


class TestAdsArgPathScanBlocked:
    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_shell_output_arg_with_stream_rejected(self, tmp_path: Path) -> None:
        """A shell arg like --output=.pyddock:x must be rejected by the arg
        scanner (same containment gap as the fs guard)."""
        reason = evaluate_arg_paths(
            ["--output=.pyddock:pwned"],
            arg_paths="workspace",
            workspace_root=tmp_path,
            workspace_module_dirs={},
            shell_command_patterns=["^git$"],
        )
        assert reason is not None
        assert "alternate data stream" in reason

    def test_switch_value_is_parsed_not_raw_token(self, tmp_path: Path) -> None:
        """The path is parsed from the switch VALUE, so a drive colon in the
        value (C:/...) is judged on its own merits (outside-workspace), and the
        raw '--output=C:/...' token's '=C:' is never mistaken for a stream."""
        reason = evaluate_arg_paths(
            ["--output=C:/Windows/evil.txt"],
            arg_paths="workspace",
            workspace_root=tmp_path,
            workspace_module_dirs={},
            shell_command_patterns=["^git$"],
        )
        assert reason is not None
        assert "outside" in reason
        assert "alternate data stream" not in reason

    def test_switch_value_inside_workspace_allowed(self, tmp_path: Path) -> None:
        """A clean switch value inside the workspace passes."""
        reason = evaluate_arg_paths(
            ["--output=./out/diff.txt"],
            arg_paths="workspace",
            workspace_root=tmp_path,
            workspace_module_dirs={},
            shell_command_patterns=["^git$"],
        )
        assert reason is None
