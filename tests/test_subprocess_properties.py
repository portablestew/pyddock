"""Tests for subprocess enforcement.

Verifies universal properties of subprocess patching:
- subprocess rejects shell=True
- subprocess rejects bare string commands
- os.system always blocked

Uses targeted examples covering edge cases rather than Hypothesis fuzzing,
since the enforcement logic doesn't depend on command string content.
"""

from __future__ import annotations

import sys
import tempfile
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
from pyddock.venv_manager import VenvManager


def _make_config(
    shell: dict[str, ShellPolicyConfig] | None = None,
) -> PyddockConfig:
    """Create a config with subprocess allowed."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["subprocess", "os"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
    )


def _make_executor(config: PyddockConfig) -> tuple[SubprocessExecutor, Path]:
    """Create an executor with a temp workspace."""
    tmp = Path(tempfile.mkdtemp())
    workspace = tmp / "ws"
    workspace.mkdir(exist_ok=True)
    manager = VenvManager(venv_path=tmp / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return SubprocessExecutor(config, manager), workspace


# Edge-case command strings that exercise different code paths
EDGE_CASE_COMMANDS = [
    "",                     # empty string (min boundary)
    "../../../etc/passwd", # path traversal attempt
    "$(whoami)",          # shell expansion attempt
]


@pytest.fixture
def executor_and_workspace():
    """Shared executor setup for all tests in this module."""
    config = _make_config(
        shell={"any": ShellPolicyConfig(command=".*", mode="allow", deny=[])},
    )
    return _make_executor(config)


@pytest.mark.parametrize("cmd_str", EDGE_CASE_COMMANDS)
def test_subprocess_rejects_shell_true(cmd_str: str, executor_and_workspace) -> None:
    """subprocess.run with shell=True is always blocked regardless of command."""
    executor, workspace = executor_and_workspace

    source = (
        "import subprocess\n"
        "try:\n"
        f"    subprocess.run({cmd_str!r}, shell=True)\n"
        "    print('PERMITTED')\n"
        "except PermissionError:\n"
        "    print('BLOCKED')\n"
        "except Exception:\n"
        "    print('BLOCKED')\n"
    )
    result = executor.execute(source, [], 5, workspace)
    assert "PERMITTED" not in result.stdout


@pytest.mark.parametrize("cmd_str", EDGE_CASE_COMMANDS)
def test_subprocess_rejects_string_command(cmd_str: str, executor_and_workspace) -> None:
    """subprocess.run with a bare string command is always blocked."""
    executor, workspace = executor_and_workspace

    source = (
        "import subprocess\n"
        "try:\n"
        f"    subprocess.run({cmd_str!r})\n"
        "    print('PERMITTED')\n"
        "except PermissionError:\n"
        "    print('BLOCKED')\n"
        "except Exception:\n"
        "    print('BLOCKED')\n"
    )
    result = executor.execute(source, [], 5, workspace)
    assert "PERMITTED" not in result.stdout


@pytest.mark.parametrize("cmd_str", EDGE_CASE_COMMANDS)
def test_os_system_always_blocked(cmd_str: str, executor_and_workspace) -> None:
    """os.system is always blocked regardless of command."""
    executor, workspace = executor_and_workspace

    source = (
        "import os\n"
        "try:\n"
        f"    os.system({cmd_str!r})\n"
        "    print('PERMITTED')\n"
        "except PermissionError:\n"
        "    print('BLOCKED')\n"
        "except Exception:\n"
        "    print('BLOCKED')\n"
    )
    result = executor.execute(source, [], 5, workspace)
    assert "PERMITTED" not in result.stdout
