"""Regression tests for the sys.addaudithook filesystem backstop.

The monkeypatches in `_fs_enforcement` rebind names (`builtins.open`,
`io.FileIO`, ...). They cannot cover the genuine `_io.FileIO` C class, which an
agent can re-derive from any live stream object:

    type(sys.stdout.buffer.raw)(".pyddock/pwned.txt", "wb")

`install_audit_enforcement` closes that hole by enforcing the *same*
`_check_read`/`_check_write` policy at the `open` audit event, which fires
beneath the Python name layer. These tests assert:

  * the `_io.FileIO` bypass is now blocked for protected / out-of-workspace
    targets (and the file is never created),
  * legitimate writes inside the workspace still succeed via that same class
    (the backstop is path-scoped, not a blanket FileIO ban),
  * trusted-library temp files and ordinary imports are unaffected (the
    import-machinery exemption and per-path policy keep dependencies working).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pyddock.config import (
    ASTConfig,
    AuditConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    ShellPolicyConfig,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager

DEFAULT_BLOCK_ATTRS = [
    "__subclasses__", "__globals__", "__code__", "__bases__",
    "__mro__", "__closure__",
]

# A representative audit table for the fixtures below. The engine has no hidden
# fallback — [audit] is the single source of truth — so tests state their policy.
AUDIT_RULES = [
    ("open", "fs"),
    ("os.rename", "fs-write-pair"), ("os.link", "fs-write-pair"),
    ("os.symlink", "fs-write-pair"),
    ("os.remove", "fs-write"), ("os.unlink", "fs-write"),
    ("os.mkdir", "fs-write"), ("os.rmdir", "fs-write"),
    ("os.chmod", "fs-write"), ("os.truncate", "fs-write"),
    ("ctypes.*", "agent-deny"), ("marshal.loads", "agent-deny"),
    ("subprocess.Popen", "observe"),
]


@pytest.fixture
def config() -> PyddockConfig:
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(
            allowed=["os", "sys", "io", "json", "csv", "pathlib", "tempfile", "types"]
        ),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(
            block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
            block_attributes=list(DEFAULT_BLOCK_ATTRS),
        ),
        restrictions={},
        audit=AuditConfig(rules=AUDIT_RULES),
    )


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


@pytest.fixture
def executor(config: PyddockConfig, venv_manager: VenvManager) -> SubprocessExecutor:
    return SubprocessExecutor(config, venv_manager)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # Pre-create the temp area the way the server does at startup, so tempfile
    # (which is redirected to .pyddock/tmp) has somewhere to write.
    (tmp_path / ".pyddock" / "tmp").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _run(executor: SubprocessExecutor, workspace: Path, source: str):
    return executor.execute(source, args=[], timeout=30, workspace_root=workspace)


# Re-derive the genuine _io.FileIO class the way the exploit does.
_REAL_FILEIO = "type(__import__('sys').stdout.buffer.raw)"


class TestAuditBackstopBlocksBypass:
    """The _io.FileIO bypass is caught at the audit layer."""

    def test_fileio_write_to_pyddock_blocked(self, executor, workspace) -> None:
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            "f = F('.pyddock/pwned.txt', 'wb')\n"
            "f.write(b'pwned'); f.close()\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code != 0, result.stdout
        assert "PermissionError" in result.stderr
        assert not (workspace / ".pyddock" / "pwned.txt").exists()

    def test_fileio_write_outside_workspace_blocked(self, executor, workspace) -> None:
        target = workspace.parent / "pyddock_audit_escape.txt"
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            f"f = F({str(target)!r}, 'wb')\n"
            "f.write(b'x'); f.close()\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code != 0, result.stdout
        assert "PermissionError" in result.stderr
        assert not target.exists()

    def test_fileio_write_to_pyddock_tmp_via_traversal_blocked(self, executor, workspace) -> None:
        # .pyddock/tmp is writable, but a traversal back into .pyddock must not be.
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            "f = F('.pyddock/tmp/../escalated.txt', 'wb')\n"
            "f.write(b'x'); f.close()\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code != 0, result.stdout
        assert "PermissionError" in result.stderr
        assert not (workspace / ".pyddock" / "escalated.txt").exists()


class TestAuditBackstopAllowsLegitimate:
    """The backstop is path-scoped: it must not over-block."""

    def test_fileio_write_inside_workspace_allowed(self, executor, workspace) -> None:
        # The same genuine class, writing to a permitted workspace path, works.
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            "f = F('ok.bin', 'wb')\n"
            "f.write(b'data'); f.close()\n"
            "open('ok.bin', 'rb').read()\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "b'data'"
        assert (workspace / "ok.bin").read_bytes() == b"data"

    def test_agent_workspace_write_via_open_still_works(self, executor, workspace) -> None:
        src = "open('plain.txt', 'w').write('hi')"
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert (workspace / "plain.txt").read_text() == "hi"

    def test_tempfile_named_temporary_file_works(self, executor, workspace) -> None:
        # Trusted-library internal write (redirected to .pyddock/tmp) must pass.
        src = (
            "import tempfile\n"
            "f = tempfile.NamedTemporaryFile(delete=False)\n"
            "f.write(b'x'); f.close()\n"
            "import os\nos.path.basename(f.name).startswith('tmp') or bool(f.name)\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "True"

    def test_imports_and_stdlib_usage_unaffected(self, executor, workspace) -> None:
        # Import machinery + library file reads must not trip the hook.
        src = (
            "import json, csv, pathlib\n"
            "json.dumps({'a': 1}) and pathlib.Path('.').exists()\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "True"

    def test_workspace_file_read_allowed(self, executor, workspace) -> None:
        (workspace / "data.txt").write_text("hello")
        src = "open('data.txt').read()"
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "'hello'"


class TestDebugAuditLog:
    """`--debug` writes a JSONL trail of observed events with caller class."""

    @pytest.fixture
    def debug_executor(self, config, venv_manager) -> SubprocessExecutor:
        return SubprocessExecutor(config, venv_manager, debug=True)

    @staticmethod
    def _read_log(workspace: Path) -> list[dict]:
        log = workspace / ".pyddock" / "tmp" / "audit.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    def test_records_agent_allow_and_deny(self, debug_executor, workspace) -> None:
        src = (
            "import sys\n"
            "F = type(sys.stdout.buffer.raw)\n"
            "open('allowed.txt', 'w').write('ok')\n"          # allow (workspace)
            "try:\n"
            "    F('.pyddock/blocked.txt', 'wb')\n"           # deny (.pyddock bypass)
            "except PermissionError:\n"
            "    pass\n"
        )
        result = _run(debug_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        recs = self._read_log(workspace)
        assert recs, "expected a debug audit trail"

        allows = [r for r in recs
                  if r["decision"] == "allow" and "allowed.txt" in (r["detail"] or "")]
        assert allows and allows[0]["caller"] == "AGENT", recs

        denies = [r for r in recs
                  if r["decision"] == "deny" and "blocked.txt" in (r["detail"] or "")]
        assert denies and denies[0]["caller"] == "AGENT", recs

    def test_trusted_library_write_classified_trusted(self, debug_executor, workspace) -> None:
        src = (
            "import tempfile\n"
            "f = tempfile.NamedTemporaryFile(delete=False)\n"
            "f.write(b'x'); f.close()\n"
        )
        result = _run(debug_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        recs = self._read_log(workspace)
        trusted = [r for r in recs if r["decision"] == "allow" and r["caller"] == "TRUSTED"]
        assert trusted, f"expected a TRUSTED allow record; got {[(r['decision'], r['caller']) for r in recs]}"

    def test_no_log_file_without_debug(self, executor, workspace) -> None:
        _run(executor, workspace, "open('x.txt', 'w').write('hi')")
        assert not (workspace / ".pyddock" / "tmp" / "audit.jsonl").exists()


class TestDispositionEngine:
    """The config-driven disposition table: agent-deny + caller scoping.

    Uses os.scandir (reachable via the safe os proxy) mapped to `agent-deny` to
    exercise the mechanism deterministically — most real agent-deny primitives
    (ctypes, marshal) aren't reachable by compliant agent code by design.
    """

    @pytest.fixture
    def scandir_executor(self, venv_manager) -> SubprocessExecutor:
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["os", "sys", "glob", "io"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=list(DEFAULT_BLOCK_ATTRS),
            ),
            restrictions={},
            audit=AuditConfig(rules=[("open", "fs"), ("os.scandir", "agent-deny")]),
        )
        return SubprocessExecutor(cfg, venv_manager)

    def test_agent_deny_blocks_agent_caller(self, scandir_executor, workspace) -> None:
        result = _run(scandir_executor, workspace, "import os\nlist(os.scandir('.'))")
        assert result.exit_code != 0, result.stdout
        assert "PermissionError" in result.stderr
        assert "agent-deny" in result.stderr

    def test_agent_deny_allows_trusted_library(self, scandir_executor, workspace) -> None:
        # glob.glob() calls os.scandir from stdlib (a trusted frame) -> allowed,
        # even though the agent initiated it.
        (workspace / "a.txt").write_text("x")
        result = _run(scandir_executor, workspace, "import glob\nsorted(glob.glob('*.txt'))")
        assert result.exit_code == 0, result.stderr
        assert "a.txt" in (result.result or "")

    def test_fs_disposition_still_blocks_fileio_bypass(self, scandir_executor, workspace) -> None:
        # The explicit `open=fs` rule routes the FileIO bypass to _check_write.
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            "try:\n"
            "    F('.pyddock/x.txt', 'wb')\n"
            "except PermissionError as e:\n"
            "    print('blocked')\n"
        )
        result = _run(scandir_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "blocked" in result.stdout
        assert not (workspace / ".pyddock" / "x.txt").exists()


class TestDefaultConfigShipsProtections:
    """The bundled default_config.toml carries the audit protections.

    There is no code-level fallback — [audit] is the single source of truth —
    so the shipped default must actually contain the fs + agent-deny rules.
    """

    def test_default_config_has_fs_and_agent_deny_and_network(self, tmp_path) -> None:
        from pyddock.config import load_config

        # tmp_path has no .pyddock/pyddock.toml, so this resolves to the bundled
        # default_config.toml (with no override).
        cfg = load_config(tmp_path)
        rules = dict(cfg.audit.rules)
        assert rules.get("open") == "fs"
        assert rules.get("os.rename") == "fs-write-pair"
        assert any(d == "agent-deny" for d in rules.values()), rules
        assert rules.get("ctypes.*") == "agent-deny"
        assert any(d == "network" for d in rules.values()), rules
        # shell disposition for subprocess spawn
        assert rules.get("subprocess.Popen") == "shell"
        # os.exec/system/spawn are agent-deny (defense-in-depth)
        assert rules.get("os.exec") == "agent-deny"
        assert rules.get("os.system") == "agent-deny"
        assert rules.get("os.posix_spawn") == "agent-deny"
        assert rules.get("os.spawn") == "agent-deny"

    def test_empty_audit_means_no_audit_hook(self, venv_manager, workspace) -> None:
        # With [audit] empty and debug off, the audit hook isn't installed, so the
        # _io.FileIO bypass is NOT caught at the audit layer (the monkeypatches
        # remain first-line, but they cannot see the genuine FileIO). This
        # documents that the table is authoritative — omit it and you lose the
        # backstop.
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["os", "sys", "io"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=list(DEFAULT_BLOCK_ATTRS),
            ),
            restrictions={},
            audit=AuditConfig(rules=[]),
        )
        executor = SubprocessExecutor(cfg, venv_manager)
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            "F('.pyddock/leak.txt', 'wb').write(b'x')\n"
            "print('wrote')\n"
        )
        result = _run(executor, workspace, src)
        # No audit hook -> the genuine FileIO write succeeds (this is the
        # documented consequence of omitting [audit]).
        assert result.exit_code == 0, result.stderr
        assert (workspace / ".pyddock" / "leak.txt").exists()


class TestNetworkDisposition:
    """The `network` disposition denies socket egress from agent code."""

    @pytest.fixture
    def net_executor(self, venv_manager) -> SubprocessExecutor:
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["os", "sys", "io", "socket"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=list(DEFAULT_BLOCK_ATTRS),
            ),
            restrictions={},
            audit=AuditConfig(rules=[("open", "fs"), ("socket.connect", "network")]),
        )
        return SubprocessExecutor(cfg, venv_manager)

    def test_agent_socket_connect_denied(self, net_executor, workspace) -> None:
        # The socket.connect audit event fires (in C) before the OS connect, so
        # the agent never reaches the network.
        src = (
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(1)\n"
            "try:\n"
            "    s.connect(('127.0.0.1', 9))\n"
            "    print('CONNECTED')\n"
            "except PermissionError as e:\n"
            "    print('blocked', 'network' in str(e))\n"
        )
        result = _run(net_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "blocked True" in result.stdout
        assert "CONNECTED" not in result.stdout


class TestNullDeviceAllowed:
    """Writing/reading the OS null device must be permitted (subprocess/stdio
    redirection opens it via low-level os.open, which the audit hook now sees)."""

    def test_open_null_device_write_allowed(self, executor, workspace) -> None:
        import os as _os
        src = (
            "import os\n"
            f"open({_os.devnull!r}, 'w').write('discard')\n"
            "print('ok')\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "ok" in result.stdout

    def test_fileio_null_device_allowed(self, executor, workspace) -> None:
        # Mirrors how subprocess/GitPython reach the null device (genuine FileIO).
        import os as _os
        src = (
            "import sys\n"
            f"F = {_REAL_FILEIO}\n"
            f"f = F({_os.devnull!r}, 'wb'); f.write(b'x'); f.close()\n"
            "print('ok')\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "ok" in result.stdout


class TestAllowDispositionSilent:
    """`allow` is a silent no-op: no enforcement AND no debug log (vs `observe`)."""

    @staticmethod
    def _read_log(workspace: Path) -> list[dict]:
        log = workspace / ".pyddock" / "tmp" / "audit.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    @pytest.fixture
    def executor(self, venv_manager) -> SubprocessExecutor:
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["os", "sys", "io"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=list(DEFAULT_BLOCK_ATTRS),
            ),
            restrictions={},
            audit=AuditConfig(rules=[
                ("open", "fs"),
                ("os.scandir", "allow"),    # silent
                ("os.listdir", "observe"),  # logged under --debug
            ]),
        )
        return SubprocessExecutor(cfg, venv_manager, debug=True)

    def test_allow_silent_observe_logs(self, executor, workspace) -> None:
        src = "import os\nlist(os.scandir('.'))\nos.listdir('.')\n"
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        events = {r["event"] for r in self._read_log(workspace)}
        assert "os.listdir" in events, events          # observe -> logged
        assert "os.scandir" not in events, events      # allow -> silent


# ---------------------------------------------------------------------------
# Shell disposition: subprocess.Popen validated against [shell.*] at audit layer
# ---------------------------------------------------------------------------


GIT_POLICY = {
    "command": "^git$",
    "mode": "deny",
    "allow": ["status(\\s|$)", "log(\\s|$)", "fetch(\\s|$)", "add(\\s|$)"],
    "deny": ["ext::", "--upload-pack", "--receive-pack"],
    "arg_paths": "workspace",
}


class TestEvaluateSpawnCommand:
    """Direct unit tests of the audit-layer shell validator.

    `evaluate_spawn_command` is the logic the `shell` disposition runs for the
    `subprocess.Popen` audit event. It is the authoritative backstop for spawns
    that bypass the subprocess proxy (a captured `from subprocess import Popen`
    reference, as GitPython does). Tested directly because, by design, the hook
    skips spawns routed through the proxy, so an end-to-end run exercises the
    proxy rather than this code.

    Both platform event shapes are covered: POSIX passes argv as a LIST; Windows
    passes a list2cmdline STRING.
    """

    @staticmethod
    def _ev(executable, raw_args, **kw):
        from pyddock._audit_enforcement import evaluate_spawn_command
        return evaluate_spawn_command(executable, raw_args, **kw)

    @staticmethod
    def _shell():
        return {"git": dict(GIT_POLICY)}

    # --- POSIX shape (list) ---
    def test_posix_allowed(self) -> None:
        self._ev("git", ["git", "status", "--short"], shell_policies=self._shell())

    def test_posix_denied_subcommand(self) -> None:
        with pytest.raises(PermissionError, match="allow-list"):
            self._ev("git", ["git", "push", "origin"], shell_policies=self._shell())

    def test_posix_deny_pattern_wins(self) -> None:
        with pytest.raises(PermissionError, match="deny pattern"):
            self._ev("git", ["git", "fetch", "ext::sh -c evil"], shell_policies=self._shell())

    # --- Windows shape (list2cmdline string) ---
    def test_windows_allowed(self) -> None:
        self._ev(None, "git status --short", shell_policies=self._shell())

    def test_windows_denied_subcommand(self) -> None:
        with pytest.raises(PermissionError, match="allow-list"):
            self._ev(None, "git push origin", shell_policies=self._shell())

    def test_windows_deny_pattern_in_quoted_value(self) -> None:
        # list2cmdline quotes the arg with spaces; the deny token must still be
        # found inside the quoted value (re.search), and quotes are stripped.
        import subprocess
        cmdline = subprocess.list2cmdline(["git", "fetch", "ext::sh -c evil"])
        with pytest.raises(PermissionError, match="deny pattern"):
            self._ev(None, cmdline, shell_policies=self._shell())

    # --- command identification / spoofing ---
    def test_unknown_command_denied(self) -> None:
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev("curl", ["curl", "http://evil"], shell_policies=self._shell())

    def test_planted_path_executable_cannot_impersonate_git(self) -> None:
        # The spoof: an agent-written workspace/hax/git.exe. The FULL token is
        # matched against ^git$ (never a basename), so a path cannot match.
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev(
                "workspace/hax/git.exe",
                ["workspace/hax/git.exe", "status"],
                shell_policies=self._shell(),
            )

    def test_windows_planted_path_executable_denied(self) -> None:
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev(None, r'"C:\hax\git.exe" status', shell_policies=self._shell())

    def test_absolute_path_to_real_git_denied_like_proxy(self) -> None:
        # Consistent with the proxy: an absolute path doesn't match ^git$.
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev(None, r'"C:\Program Files\Git\cmd\git.exe" status',
                     shell_policies=self._shell())

    # --- executable= redirect spoof ---
    def test_executable_redirect_spoof_denied(self) -> None:
        # argv[0]='git' (allowed) but executable points at a planted binary.
        with pytest.raises(PermissionError, match="does not match"):
            self._ev(
                "workspace/hax/evil.exe",
                ["git", "status"],
                shell_policies=self._shell(),
            )

    def test_executable_redirect_spoof_denied_windows_shape(self) -> None:
        with pytest.raises(PermissionError, match="does not match"):
            self._ev(r"C:\hax\evil.exe", "git status", shell_policies=self._shell())

    def test_executable_equal_to_argv0_allowed(self) -> None:
        # POSIX defaults executable to argv[0]; an equal value is the normal case.
        self._ev("git", ["git", "status"], shell_policies=self._shell())

    def test_executable_bytes_equal_to_argv0_allowed(self) -> None:
        # bytes executable matching bytes argv[0] (both decode to 'git').
        self._ev(b"git", [b"git", b"status"], shell_policies=self._shell())

    def test_honest_full_path_executable_denied_failclosed(self) -> None:
        # Documented trade-off: an honest full-path executable with a bare argv[0]
        # is denied (fail-closed). No PATH resolution is performed.
        with pytest.raises(PermissionError, match="does not match"):
            self._ev(
                "/usr/bin/git", ["git", "status"], shell_policies=self._shell()
            )

    # --- shell=True forms are denied because cmd/sh isn't an allowed command ---
    def test_windows_shell_true_form_denied(self) -> None:
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev(None, r'C:\WINDOWS\system32\cmd.exe /c "git status"',
                     shell_policies=self._shell())

    def test_posix_shell_true_form_denied(self) -> None:
        with pytest.raises(PermissionError, match="no matching shell policy"):
            self._ev("/bin/sh", ["/bin/sh", "-c", "git status"], shell_policies=self._shell())

    # --- arg_paths scanning (parity with the proxy) ---
    def test_arg_paths_blocks_pyddock_write(self, tmp_path) -> None:
        with pytest.raises(PermissionError, match=r"\.pyddock"):
            self._ev(
                "git", ["git", "add", ".pyddock/pwned.txt"],
                shell_policies=self._shell(),
                workspace_root=str(tmp_path),
            )

    def test_arg_paths_blocks_outside_workspace(self, tmp_path) -> None:
        with pytest.raises(PermissionError, match="outside"):
            self._ev(
                "git", ["git", "add", "../../etc/passwd"],
                shell_policies=self._shell(),
                workspace_root=str(tmp_path),
            )

    def test_arg_paths_allows_workspace_relative(self, tmp_path) -> None:
        self._ev(
            "git", ["git", "add", "output/result.txt"],
            shell_policies=self._shell(),
            workspace_root=str(tmp_path),
        )

    def test_arg_paths_skipped_without_workspace_root(self) -> None:
        # No workspace_root -> path scan disabled (back-compat for unit use).
        self._ev("git", ["git", "add", ".pyddock/whatever.txt"], shell_policies=self._shell())

    # --- cwd scanning (parity with the proxy; covers the bypass path) ---
    def test_cwd_blocks_pyddock(self, tmp_path) -> None:
        with pytest.raises(PermissionError, match=r"\.pyddock"):
            self._ev(
                "git", ["git", "status"], cwd=".pyddock",
                shell_policies=self._shell(), workspace_root=str(tmp_path),
            )

    def test_cwd_blocks_outside_workspace(self, tmp_path) -> None:
        with pytest.raises(PermissionError, match="outside"):
            self._ev(
                "git", ["git", "status"], cwd="../../somewhere",
                shell_policies=self._shell(), workspace_root=str(tmp_path),
            )

    def test_cwd_allows_workspace_subdir(self, tmp_path) -> None:
        self._ev(
            "git", ["git", "status"], cwd="subdir",
            shell_policies=self._shell(), workspace_root=str(tmp_path),
        )

    def test_cwd_none_allowed(self, tmp_path) -> None:
        self._ev(
            "git", ["git", "status"], cwd=None,
            shell_policies=self._shell(), workspace_root=str(tmp_path),
        )

    # --- proxy_validated: only the executable-redirect check runs ---
    def test_proxy_validated_skips_policy(self) -> None:
        # A subcommand the policy would reject is allowed through when the proxy
        # already validated it (the proxy handles resolve_command rewriting).
        self._ev("pwsh", ["pwsh", "-File", "x.ps1"], shell_policies=self._shell(),
                 proxy_validated=True)

    def test_proxy_validated_still_blocks_executable_redirect(self) -> None:
        # The executable= spoof is the one thing the proxy doesn't check, so the
        # audit layer enforces it even for proxy-routed spawns.
        with pytest.raises(PermissionError, match="does not match"):
            self._ev("workspace/hax/evil.exe", ["git", "status"],
                     shell_policies=self._shell(), proxy_validated=True)

    def test_proxy_validated_normal_spawn_allowed(self) -> None:
        # Normal proxy spawn (executable defaults to argv[0]) passes.
        self._ev("git", ["git", "push", "origin"], shell_policies=self._shell(),
                 proxy_validated=True)

    # --- malformed / empty ---
    def test_empty_args_is_noop(self) -> None:
        self._ev(None, [], shell_policies=self._shell())  # nothing runnable
        self._ev(None, "", shell_policies=self._shell())


class TestShellDispositionIntegration:
    """End-to-end: a config with `subprocess.Popen = shell` enforces policy.

    These go through the subprocess proxy (the audit hook skips proxy-routed
    spawns), so they confirm the integrated config is wired correctly and a
    denied command never spawns a process.
    """

    @pytest.fixture
    def shell_executor(self, venv_manager) -> SubprocessExecutor:
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["os", "sys", "subprocess"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=list(DEFAULT_BLOCK_ATTRS),
            ),
            restrictions={},
            shell={"git": ShellPolicyConfig(
                command="^git$",
                mode="deny",
                allow=["status(\\s|$)", "log(\\s|$)", "fetch(\\s|$)"],
                deny=["ext::", "--upload-pack", "--receive-pack"],
            )},
            audit=AuditConfig(rules=[("open", "fs"), ("subprocess.Popen", "shell")]),
        )
        return SubprocessExecutor(cfg, venv_manager)

    def test_denied_command_blocked(self, shell_executor, workspace) -> None:
        src = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['git', 'push', 'origin'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = _run(shell_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout

    def test_deny_pattern_wins(self, shell_executor, workspace) -> None:
        src = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['git', 'fetch', 'ext::sh -c evil'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = _run(shell_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout
        assert "deny pattern" in result.stdout

    def test_executable_redirect_blocked_through_proxy(self, shell_executor, workspace) -> None:
        # The proxy validates argv[0]='git' and forwards executable= without
        # inspecting it; the audit layer enforces the executable-redirect check
        # even for proxy-routed spawns. Before the fix this produced a
        # FileNotFoundError (the redirect was attempted); now it's blocked.
        src = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['git', 'status'], executable='zzz_marker', capture_output=True)\n"
            "    print('NOT BLOCKED')\n"
            "except PermissionError as e:\n"
            "    print('BLOCKED:', e)\n"
            "except FileNotFoundError:\n"
            "    print('REDIRECT ATTEMPTED')\n"
        )
        result = _run(shell_executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert "BLOCKED" in result.stdout
        assert "does not match" in result.stdout
        assert "audit policy: shell" in result.stdout

