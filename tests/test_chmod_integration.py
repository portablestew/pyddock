"""Integration tests for guarded chmod support.

Verifies that Path.chmod() and os.chmod():
- Allow any standard permission bits (owner/group/other rwx)
- Block special bits (setuid, setgid, sticky)
- Enforce path guards (_check_write)
- Reject non-integer modes
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


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)
    return manager


def _make_config() -> PyddockConfig:
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(
            allowed=["os", "sys", "pathlib"],
        ),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(
            block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
            block_attributes=["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__"],
        ),
        restrictions={},
    )


class TestPathChmodAllowed:
    """Path.chmod() allows standard permission modes."""

    def test_common_modes(self, workspace: Path, venv_manager: VenvManager) -> None:
        """Standard Unix modes (644, 755, 777, etc.) all work."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "for mode in [0o644, 0o755, 0o777, 0o600, 0o440, 0o750]:\n"
            "    p.chmod(mode)\n"
            "print('OK')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0, result.stderr
        assert "OK" in result.stdout

    def test_set_readonly_then_writable(self, workspace: Path, venv_manager: VenvManager) -> None:
        """Can toggle read-only and then write to the file."""
        target = workspace / "test_file.txt"
        target.write_text("original", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "p.chmod(0o444)\n"
            "# File is now read-only, write should fail\n"
            "try:\n"
            "    p.write_text('nope')\n"
            "    print('write_while_readonly=succeeded')\n"
            "except PermissionError:\n"
            "    print('write_while_readonly=blocked')\n"
            "# Make writable again\n"
            "p.chmod(0o644)\n"
            "p.write_text('modified')\n"
            "print(f'content={p.read_text()}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0, result.stderr
        assert "write_while_readonly=blocked" in result.stdout
        assert "content=modified" in result.stdout


class TestPathChmodBlocked:
    """Path.chmod() blocks special bits and invalid inputs."""

    def test_setuid_rejected(self, workspace: Path, venv_manager: VenvManager) -> None:
        """setuid bit raises PermissionError."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "try:\n"
            "    p.chmod(0o4755)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "setuid/setgid/sticky" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_setgid_rejected(self, workspace: Path, venv_manager: VenvManager) -> None:
        """setgid bit raises PermissionError."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "try:\n"
            "    p.chmod(0o2755)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_sticky_rejected(self, workspace: Path, venv_manager: VenvManager) -> None:
        """Sticky bit raises PermissionError."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "try:\n"
            "    p.chmod(0o1755)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_non_integer_rejected(self, workspace: Path, venv_manager: VenvManager) -> None:
        """Non-integer mode raises PermissionError."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "try:\n"
            "    p.chmod('0o644')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestPathChmodPathGuards:
    """Path.chmod() respects path guards."""

    def test_pyddock_dir_blocked(self, workspace: Path, venv_manager: VenvManager) -> None:
        """Cannot chmod files in .pyddock/ directory."""
        pyddock_dir = workspace / ".pyddock"
        pyddock_dir.mkdir(exist_ok=True)
        target = pyddock_dir / "pyddock.toml"
        target.write_text("# config", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{target}')\n"
            "try:\n"
            "    p.chmod(0o644)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_outside_workspace_blocked(
        self, tmp_path: Path, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Cannot chmod files outside the workspace."""
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{outside}')\n"
            "try:\n"
            "    p.chmod(0o644)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestOsChmod:
    """os.chmod() has the same behavior as Path.chmod()."""

    def test_allowed(self, workspace: Path, venv_manager: VenvManager) -> None:
        """os.chmod() works for standard modes."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import os\n"
            f"os.chmod(r'{target}', 0o755)\n"
            "print('OK')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0, result.stderr
        assert "OK" in result.stdout

    def test_special_bits_rejected(self, workspace: Path, venv_manager: VenvManager) -> None:
        """os.chmod() blocks special bits."""
        target = workspace / "test_file.txt"
        target.write_text("hello", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import os\n"
            "try:\n"
            f"    os.chmod(r'{target}', 0o4755)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_outside_workspace_blocked(
        self, tmp_path: Path, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """os.chmod() blocks paths outside workspace."""
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import os\n"
            "try:\n"
            f"    os.chmod(r'{outside}', 0o644)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
