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

import sys
from pathlib import Path

import pytest

from pyddock.config import (
    ASTConfig,
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
