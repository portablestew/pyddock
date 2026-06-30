"""Regression tests for two jailbreaks found during a black-box pentest.

1. `git rebase --exec`/`-x` — an allowed git subcommand that runs an arbitrary
   per-commit shell command in git's own (unsandboxed) child process. The deny
   list previously stopped `ext::`/`--upload-pack`/`--receive-pack`/`-c` but not
   `--exec`/`-x`. Fixed by adding deny patterns to [shell.git].

2. NTFS alternate data streams — `open(".pyddock:pwned", "w")` writes a data
   stream onto the protected `.pyddock` directory while the lexical leaf
   `.pyddock:pwned` evades the `relative_to('.pyddock')` containment check.
   Fixed by detecting a `:` stream reference and rejecting it when the stream's
   BASE object exists on disk (ntfs_stream_base + an existence check), wired
   into the shared _check_read/_check_write closures and the shell arg-path /
   cwd scanners. The existence gate is sound because every protected target
   (.pyddock, workspace module dirs, the stdlib) always exists, so the attack is
   still blocked — while a colon in a string that merely looks path-like (e.g. a
   commit message) has a non-existent base and is no longer a false positive.
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
    AuditConfig,
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
    def test_stream_base_extraction(self) -> None:
        # ntfs_stream_base returns the object the stream attaches to (or None).
        from pyddock._base import ntfs_stream_base

        assert ntfs_stream_base(".pyddock:pwned") == ".pyddock"
        assert ntfs_stream_base(".pyddock:pwned.txt") == ".pyddock"
        assert ntfs_stream_base("C:/Git/pyddock/.pyddock:pwned") == "C:\\Git\\pyddock\\.pyddock"
        assert ntfs_stream_base("dir/file.txt:s") == "dir\\file.txt"
        assert ntfs_stream_base("file.txt") is None
        assert ntfs_stream_base("C:\\abs\\file") is None

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_drive_relative_not_flagged(self) -> None:
        # 'C:rest' (drive-relative) and 'C:\\...' are the only legal colons.
        assert has_ntfs_stream("C:relative\\file") is False
        assert has_ntfs_stream("C:\\abs\\file") is False

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_primitive_stays_strict_for_url_disguised_stream(self) -> None:
        # has_ntfs_stream is a low-level filesystem primitive: it must NOT make
        # an exception for URL-looking strings. A scheme:// prefix slapped onto a
        # stream-bearing path (e.g. open("https://.pyddock:pwned", "w")) resolves
        # to an in-workspace lexical path and would otherwise slip past the
        # writable_paths containment check. The primitive must still reject it;
        # URL filtering belongs to the shell-arg layer (_looks_like_path), not
        # here. See the matching regression in TestUrlArgsNotPathCandidates.
        assert has_ntfs_stream("https://.pyddock:pwned") is True
        assert has_ntfs_stream("http://host/file.txt:stream") is True


class TestUrlArgsNotPathCandidates:
    """URLs passed as shell CLI arguments must not be treated as local paths.

    Regression: a URL argument (e.g. `-Url https://host/a/b`) tripped the
    NTFS-stream check because the scheme's `:` (and any `host:port`) looked like
    an alternate-data-stream reference. URLs are filtered out at the
    `_looks_like_path` layer so they never reach the stream check, while the
    primitive itself stays strict for real filesystem paths.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.example.com/projects/abc/designs/xyz",
            "http://example.com/path",
            "https://host:8443/path",  # host:port colon
            "ssh://git@github.com/org/repo.git",
            "ftp://files.example.com/data.txt",
        ],
    )
    def test_urls_are_not_path_candidates(self, url: str) -> None:
        from pyddock.shell_executor import _extract_path_candidates, _looks_like_path

        assert _looks_like_path(url) is False, url
        assert _extract_path_candidates(url) == [], url

    def test_url_arg_passes_arg_path_scan(self, tmp_path: Path) -> None:
        # A URL argument must not be rejected by the arg-path scanner in either
        # "protected" or "workspace" mode.
        for mode in ("protected", "workspace"):
            reason = evaluate_arg_paths(
                ["-Url", "https://api.example.com/projects/abc/designs/xyz", "-Raw"],
                arg_paths=mode,
                workspace_root=tmp_path,
                workspace_module_dirs={},
                shell_command_patterns=[],
            )
            assert reason is None, (mode, reason)

    def test_stream_in_flag_value_still_blocked(self, tmp_path: Path) -> None:
        # The URL exclusion must NOT weaken the output-flag stream attack: a
        # value that is a stream-bearing path (not a scheme://) is still scanned.
        (tmp_path / ".pyddock").mkdir()  # base object the stream targets
        reason = evaluate_arg_paths(
            ["-o=.pyddock:pwned"],
            arg_paths="protected",
            workspace_root=tmp_path,
            workspace_module_dirs={},
            shell_command_patterns=[],
        )
        if sys.platform == "win32":
            assert reason is not None and "alternate data stream" in reason
        else:
            # NTFS ADS is Windows-only; on POSIX ':' is a legal filename char.
            assert reason is None or "alternate data stream" not in reason


class TestAdsWriteBlockedEndToEnd:
    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_ads_write_to_pyddock_dir_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """open('.pyddock:pwned', 'w') must be rejected — it would otherwise
        write a data stream onto the protected .pyddock directory."""
        # The server always creates .pyddock at startup; mirror that here so the
        # stream's base object exists (the existence gate keys off this).
        (workspace / ".pyddock").mkdir(exist_ok=True)
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


class TestFsTOCTOU:
    """Path-resolution TOCTOU: a path arg must be resolved once and that exact
    value used by both the check and the syscall.

    Name-patch layer: open-family functions materialize the path once and
    forward the frozen value (closing the old check-via-Path(file) /
    use-via-open(file) double resolution). Audit layer: needs no special
    handling — every fs audit event already carries the single resolved value
    the syscall uses (see _enforce_fs), so the backstop checks the true target.
    """

    def _exec(self, workspace: Path, venv_manager: VenvManager, source: str):
        # Build an explicit config WITH an [audit] table so the audit backstop
        # is actually installed (mirrors test_audit_hook_backstop.py). A config
        # without [audit] installs no audit hook, which would make the
        # leaked-FileIO backstop tests pass vacuously.
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["pathlib", "io", "sys", "os", "types"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=["__subclasses__", "__globals__"],
            ),
            restrictions={},
            audit=AuditConfig(rules=[
                ("open", "fs"),
                ("os.rename", "fs-write-pair"), ("os.link", "fs-write-pair"),
                ("os.symlink", "fs-write-pair"),
                ("os.remove", "fs-write"), ("os.unlink", "fs-write"),
                ("os.mkdir", "fs-write"), ("os.rmdir", "fs-write"),
                ("os.chmod", "fs-write"), ("os.truncate", "fs-write"),
            ]),
        )
        return SubprocessExecutor(config, venv_manager).execute(source, [], 10, workspace)

    def test_fspath_returning_protected_is_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """A __fspath__ that yields the protected path is caught (the check sees
        the real resolved value)."""
        src = (
            "class P:\n"
            "    def __fspath__(self):\n"
            "        return '.pyddock/pwned.txt'\n"
            "try:\n"
            "    open(P(), 'w').write('x')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print('BLOCKED')\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
        assert not (workspace / ".pyddock" / "pwned.txt").exists()

    def test_call_counting_fspath_cannot_swap_to_protected(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """A __fspath__ that returns a benign path first and the protected path
        on later calls cannot land a write in .pyddock: the value is resolved
        once and that same value is what the kernel opens."""
        (workspace / ".pyddock").mkdir(exist_ok=True)
        src = (
            "class P:\n"
            "    def __init__(self): self.n = 0\n"
            "    def __fspath__(self):\n"
            "        self.n += 1\n"
            "        return 'benign_target.txt' if self.n == 1 else '.pyddock/pwned.txt'\n"
            "p = P()\n"
            "open(p, 'w').write('data')\n"
            "import pathlib\n"
            "print('benign_exists=', pathlib.Path('benign_target.txt').exists())\n"
            "print('pyddock_exists=', pathlib.Path('.pyddock/pwned.txt').exists())\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        # The kernel opened the checked (benign) value, not the swapped one.
        assert "benign_exists= True" in result.stdout
        assert "pyddock_exists= False" in result.stdout
        assert not (workspace / ".pyddock" / "pwned.txt").exists()

    def test_bytearray_path_rejected(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """A mutable path buffer (bytearray) is rejected outright."""
        src = (
            "try:\n"
            "    open(bytearray(b'x.txt'), 'w').write('x')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print('BLOCKED:', 'mutable' in str(e))\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED: True" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_audit_backstop_catches_leaked_fileio_protected(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """The audit backstop catches a leaked (C-created, unpatched) _io.FileIO
        writing to a protected path. FileIO resolves __fspath__ once and raises
        the 'open' event with that resolved string, so the hook checks the true
        target — no name-patch materialization is involved on this path. The
        real FileIO class is leaked via open().buffer.raw (the C open builds a
        genuine _io.FileIO, not our _PatchedFileIO subclass)."""
        (workspace / ".pyddock").mkdir(exist_ok=True)
        src = (
            "import pathlib\n"
            "f = open('seed.txt', 'w')\n"
            "RealFileIO = type(f.buffer.raw)\n"
            "f.close()\n"
            "class P:\n"
            "    def __fspath__(self):\n"
            "        return '.pyddock/pwned.txt'\n"
            "try:\n"
            "    RealFileIO(P(), 'w')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError:\n"
            "    print('BLOCKED')\n"
            "print('pyddock_exists=', pathlib.Path('.pyddock/pwned.txt').exists())\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
        assert "pyddock_exists= False" in result.stdout
        assert not (workspace / ".pyddock" / "pwned.txt").exists()

    def test_leaked_fileio_counting_fspath_cannot_reach_pyddock(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """A leaked FileIO with a call-counting __fspath__ (benign first,
        protected later) can never land a write in .pyddock: FileIO resolves
        once and both the syscall and the 'open' event observe that same
        resolved value, so the check matches what is opened."""
        (workspace / ".pyddock").mkdir(exist_ok=True)
        src = (
            "import pathlib\n"
            "f = open('seed.txt', 'w')\n"
            "RealFileIO = type(f.buffer.raw)\n"
            "f.close()\n"
            "class P:\n"
            "    def __init__(self): self.n = 0\n"
            "    def __fspath__(self):\n"
            "        self.n += 1\n"
            "        return 'benign.txt' if self.n == 1 else '.pyddock/pwned.txt'\n"
            "try:\n"
            "    RealFileIO(P(), 'w')\n"
            "    print('opened')\n"
            "except PermissionError:\n"
            "    print('blocked')\n"
            "print('pyddock_exists=', pathlib.Path('.pyddock/pwned.txt').exists())\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        # Either outcome is safe; the invariant is the protected file is never made.
        assert "pyddock_exists= False" in result.stdout
        assert not (workspace / ".pyddock" / "pwned.txt").exists()

    def test_os_proxy_mkdir_resolves_once(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """The os proxy (makedirs/mkdir/chmod) resolves a path-like once and
        forwards that frozen value, so a __fspath__ targeting a protected dir is
        caught and a mutable buffer is rejected."""
        (workspace / ".pyddock").mkdir(exist_ok=True)
        src = (
            "import os, pathlib\n"
            "class P:\n"
            "    def __fspath__(self): return '.pyddock/evil_dir'\n"
            "try:\n"
            "    os.makedirs(P())\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError:\n"
            "    print('BLOCKED_PATHLIKE')\n"
            "try:\n"
            "    os.mkdir(bytearray(b'd'))\n"
            "    print('SHOULD NOT REACH 2')\n"
            "except PermissionError as e:\n"
            "    print('BLOCKED_BYTEARRAY:', 'mutable' in str(e))\n"
            "print('evil_exists=', pathlib.Path('.pyddock/evil_dir').exists())\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED_PATHLIKE" in result.stdout
        assert "BLOCKED_BYTEARRAY: True" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
        assert "evil_exists= False" in result.stdout
        assert not (workspace / ".pyddock" / "evil_dir").exists()

    def test_genuine_pathlib_still_works(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Regression guard: a genuine pathlib.Path write/read still works."""
        src = (
            "import pathlib\n"
            "p = pathlib.Path('ok.txt')\n"
            "p.write_text('hello')\n"
            "print('roundtrip=', p.read_text())\n"
            "print('via_open=', open(pathlib.Path('ok.txt')).read())\n"
        )
        result = self._exec(workspace, venv_manager, src)
        assert result.exit_code == 0, result.stderr
        assert "roundtrip= hello" in result.stdout
        assert "via_open= hello" in result.stdout


class TestAdsArgPathScanBlocked:
    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS ADS is Windows-only")
    def test_shell_output_arg_with_stream_rejected(self, tmp_path: Path) -> None:
        """A shell arg like --output=.pyddock:x must be rejected by the arg
        scanner (same containment gap as the fs guard)."""
        (tmp_path / ".pyddock").mkdir()  # base object the stream targets
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

    def test_stream_value_with_nonexistent_base_allowed(self, tmp_path: Path) -> None:
        """Existence gate (false-positive fix): a colon-bearing, path-like value
        whose base object does NOT exist is not treated as a stream. This is what
        keeps colons in non-path strings (e.g. a 'feat: ...' commit message that
        happens to contain a '/') from being rejected as ADS references."""
        reason = evaluate_arg_paths(
            ["-o=notes:draft/today.txt"],
            arg_paths="workspace",
            workspace_root=tmp_path,
            workspace_module_dirs={},
            shell_command_patterns=[],
        )
        assert reason is None, reason
