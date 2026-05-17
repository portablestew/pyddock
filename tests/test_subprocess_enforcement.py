"""Unit tests for subprocess/os.system runtime enforcement.

Tests run snippets through the full executor pipeline to verify that
subprocess.run, subprocess.Popen, and os.system are properly patched.
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
from pyddock.venv_manager import VenvManager


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    """VenvManager using the current Python interpreter."""
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


def _make_config(
    allowed_imports: list[str] | None = None,
    shell: dict[str, ShellPolicyConfig] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=allowed_imports or ["subprocess", "os", "json", "tempfile", "codecs", "encodings"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
    )


class TestSubprocessShellTrue:
    """Tests that shell=True is always rejected."""

    def test_subprocess_run_shell_true_rejected(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.run with shell=True raises PermissionError."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run('echo hello', shell=True)\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "shell=True" in result.stdout


class TestSubprocessStringCommand:
    """Tests that string commands are rejected."""

    def test_subprocess_run_string_cmd_rejected(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.run with string command raises PermissionError."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run('echo hello')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "list" in result.stdout


class TestSubprocessNoPolicy:
    """Tests that subprocess is blocked when no shell policies exist."""

    def test_no_policy_blocks_all(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """When no [shell.*] config exists, all subprocess calls are blocked."""
        config = _make_config(shell={})  # No shell policies
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['echo', 'hello'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "No shell policies" in result.stdout


class TestSubprocessAllowedCommand:
    """Tests that allowed list-form commands succeed."""

    def test_allowed_list_form_succeeds(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """List-form command matching a shell policy is permitted."""
        config = _make_config(
            shell={"python": ShellPolicyConfig(command="^python$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "result = subprocess.run(['python', '-c', 'print(42)'], capture_output=True, text=True)\n"
            "print(f'OUTPUT: {result.stdout.strip()}')\n"
            "print(f'EXIT: {result.returncode}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "OUTPUT: 42" in result.stdout
        assert "EXIT: 0" in result.stdout


class TestSubprocessDeniedCommand:
    """Tests that denied list-form commands fail."""

    def test_denied_command_fails(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """List-form command not matching any shell policy is rejected."""
        config = _make_config(
            shell={"python": ShellPolicyConfig(command="^python$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['rm', '-rf', '/'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "not permitted" in result.stdout


class TestOsSystem:
    """Tests that os.system is always blocked."""

    def test_os_system_always_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """os.system always raises PermissionError."""
        config = _make_config(
            allowed_imports=["subprocess", "os", "json"],
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import os\n"
            "try:\n"
            "    os.system('echo hello')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "subprocess.run" in result.stdout


class TestSubprocessArgsValidation:
    """Tests that args are validated against policy in subprocess calls."""

    def test_denied_args_rejected_in_subprocess(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Args not matching policy are rejected in subprocess.run."""
        config = _make_config(
            shell={"p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info", "filelog.*"])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['p4', 'submit', '-d', 'hack'])\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "not permitted" in result.stdout



class TestWriteProtection:
    """Tests for write protection of shell-executable paths."""

    def test_path_like_regex_blocks_writes(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Path-like shell command regex blocks writes to that directory."""
        # Create the scripts directory
        scripts_dir = workspace / ".kiro" / "scripts"
        scripts_dir.mkdir(parents=True)

        config = _make_config(
            allowed_imports=["pathlib", "subprocess", "os"],
            shell={
                "kiro-scripts": ShellPolicyConfig(
                    command=r"\.kiro/scripts/.*\.ps1",
                    mode="allow",
                    deny=[],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # Try to write to the protected scripts directory
        target = (scripts_dir / "evil.ps1").as_posix()
        source = (
            "import pathlib\n"
            "try:\n"
            f"    pathlib.Path('{target}').write_text('malicious')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "write-protected" in result.stdout

    def test_non_path_regex_does_not_block_writes(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Non-path-like shell command regex does NOT block writes."""
        config = _make_config(
            allowed_imports=["pathlib", "subprocess", "os", "tempfile", "codecs", "encodings"],
            shell={
                "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["info"]),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # Writing to workspace should still work (p4 is not path-like)
        source = (
            "import pathlib\n"
            "pathlib.Path('output.txt').write_text('ok')\n"
            "'done'"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'done'"
        assert (workspace / "output.txt").read_text() == "ok"

    def test_write_outside_protected_path_succeeds(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to a non-protected path within workspace still works."""
        config = _make_config(
            allowed_imports=["pathlib", "subprocess", "os", "tempfile", "codecs", "encodings"],
            shell={
                "kiro-scripts": ShellPolicyConfig(
                    command=r"\.kiro/scripts/.*\.ps1",
                    mode="allow",
                    deny=[],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # Writing to workspace root (not .kiro/scripts/) should work
        source = (
            "import pathlib\n"
            "pathlib.Path('safe.txt').write_text('ok')\n"
            "'done'"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'done'"
        assert (workspace / "safe.txt").read_text() == "ok"


class TestSubprocessProxySurface:
    """Tests that the subprocess proxy only exposes subprocess.run() and subprocess.Popen()."""

    def test_popen_is_available(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen is accessible through the proxy as a validated wrapper."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "has_popen = hasattr(subprocess, 'Popen')\n"
            "print(f'HAS_POPEN: {has_popen}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "HAS_POPEN: True" in result.stdout

    def test_popen_validates_command(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen rejects commands not matching shell policy."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    proc = subprocess.Popen(['curl', 'http://evil.com'])\n"
            "    print('SHOULD_NOT_REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout
        assert "SHOULD_NOT_REACH" not in result.stdout

    def test_popen_rejects_shell_true(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen rejects shell=True."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    proc = subprocess.Popen(['echo', 'hi'], shell=True)\n"
            "    print('SHOULD_NOT_REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout
        assert "shell=True" in result.stdout

    def test_popen_rejects_string_command(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen rejects string commands."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "try:\n"
            "    proc = subprocess.Popen('echo hi')\n"
            "    print('SHOULD_NOT_REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout
        assert "SHOULD_NOT_REACH" not in result.stdout

    def test_popen_allowed_command_works(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen works for allowed commands with communicate()."""
        config = _make_config(
            shell={"python": ShellPolicyConfig(command="^python$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "proc = subprocess.Popen(\n"
            "    ['python', '-c', 'print(\"hello from popen\")'],\n"
            "    stdout=subprocess.PIPE, stderr=subprocess.PIPE\n"
            ")\n"
            "stdout, stderr = proc.communicate()\n"
            "print(f'OUTPUT: {stdout.decode().strip()}')\n"
            "print(f'RC: {proc.returncode}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "OUTPUT: hello from popen" in result.stdout
        assert "RC: 0" in result.stdout

    def test_popen_context_manager(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """subprocess.Popen works as a context manager."""
        config = _make_config(
            shell={"python": ShellPolicyConfig(command="^python$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "with subprocess.Popen(\n"
            "    ['python', '-c', 'print(42)'],\n"
            "    stdout=subprocess.PIPE\n"
            ") as proc:\n"
            "    stdout, _ = proc.communicate()\n"
            "    print(f'OUTPUT: {stdout.decode().strip()}')\n"
            "print(f'RC: {proc.returncode}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "OUTPUT: 42" in result.stdout
        assert "RC: 0" in result.stdout

    def test_call_not_available(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Only subprocess.run and subprocess.Popen are exposed; legacy APIs are removed."""
        config = _make_config(
            shell={"python": ShellPolicyConfig(command="^python$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import subprocess\n"
            "has_run = hasattr(subprocess, 'run')\n"
            "has_popen = hasattr(subprocess, 'Popen')\n"
            "has_pipe = hasattr(subprocess, 'PIPE')\n"
            "has_devnull = hasattr(subprocess, 'DEVNULL')\n"
            "has_call = hasattr(subprocess, 'call')\n"
            "has_check_call = hasattr(subprocess, 'check_call')\n"
            "has_check_output = hasattr(subprocess, 'check_output')\n"
            "has_getoutput = hasattr(subprocess, 'getoutput')\n"
            "print(f'run={has_run} Popen={has_popen} PIPE={has_pipe} DEVNULL={has_devnull}')\n"
            "print(f'call={has_call} check_call={has_check_call} check_output={has_check_output} getoutput={has_getoutput}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "run=True" in result.stdout
        assert "Popen=True" in result.stdout
        assert "PIPE=True" in result.stdout
        assert "DEVNULL=True" in result.stdout
        assert "call=False" in result.stdout
        assert "check_call=False" in result.stdout
        assert "check_output=False" in result.stdout
        assert "getoutput=False" in result.stdout


class TestPyddockSelfImportBlocked:
    """Tests that user code cannot import pyddock internals to bypass the sandbox."""

    def test_import_pyddock_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """All pyddock imports are blocked — prevents access to unpatched internals."""
        config = _make_config(
            shell={"echo": ShellPolicyConfig(command="^echo$", mode="allow", deny=[])},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "# Direct import\n"
            "try:\n"
            "    import pyddock\n"
            "    print('LEAKED_DIRECT')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED_DIRECT: {e}')\n"
            "\n"
            "# Submodule import\n"
            "try:\n"
            "    from pyddock._runtime import RuntimeEnforcement\n"
            "    print('LEAKED_RUNTIME')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED_RUNTIME: {e}')\n"
            "\n"
            "# Shell executor import\n"
            "try:\n"
            "    from pyddock.shell_executor import subprocess\n"
            "    print('LEAKED_SHELL')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED_SHELL: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_DIRECT" in result.stdout
        assert "BLOCKED_RUNTIME" in result.stdout
        assert "BLOCKED_SHELL" in result.stdout
        assert "LEAKED" not in result.stdout
