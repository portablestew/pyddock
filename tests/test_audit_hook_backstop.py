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
