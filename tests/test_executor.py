"""Tests for the subprocess executor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock import SNIPPET_FILENAME
from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
)
from pyddock.executor import RunPythonOutput, SubprocessExecutor, _RESULT_SENTINEL
from pyddock.venv_manager import VenvManager


@pytest.fixture
def config() -> PyddockConfig:
    """Minimal config for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["json", "math", "pathlib", "sys"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["."]),
        ast=ASTConfig(block_calls=["eval", "exec"], block_attributes=["__globals__"]),
        restrictions={},
    )


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    """VenvManager that uses the current Python interpreter (no real venv needed)."""
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    # Patch get_python_path to return the current interpreter
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


@pytest.fixture
def executor(config: PyddockConfig, venv_manager: VenvManager) -> SubprocessExecutor:
    return SubprocessExecutor(config, venv_manager)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


class TestExecuteBasic:
    """Basic execution tests."""

    def test_simple_expression(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Last expression is captured as result."""
        result = executor.execute("2 + 2", args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert result.result == "4"
        assert result.stderr == ""

    def test_simple_print(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Print output goes to stdout, no result when last stmt is not expr."""
        result = executor.execute(
            "print('hello')", args=[], timeout=10, workspace_root=workspace
        )
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.result is None

    def test_print_and_expression(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Both stdout and result are captured."""
        source = "print('hello')\n42"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.result == "42"

    def test_multiline_code(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Multi-line code with last expression."""
        source = "x = 10\ny = 20\nx + y"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert result.result == "30"

    def test_no_expression_no_result(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Code without a trailing expression has result=None."""
        source = "x = 42"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert result.result is None


class TestSysArgv:
    """Tests for sys.argv handling."""

    def test_args_available(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Provided args appear as sys.argv[1:]."""
        source = "import sys\nsys.argv[1:]"
        result = executor.execute(
            source, args=["foo", "bar"], timeout=10, workspace_root=workspace
        )
        assert result.exit_code == 0
        assert result.result == "['foo', 'bar']"

    def test_argv_zero_is_snippet(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """sys.argv[0] is set to SNIPPET_FILENAME."""
        source = "import sys\nsys.argv[0]"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert result.result == repr(SNIPPET_FILENAME)


class TestTimeout:
    """Tests for timeout enforcement."""

    def test_timeout_kills_process(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Infinite loop is killed after timeout."""
        source = "while True: pass"
        result = executor.execute(source, args=[], timeout=2, workspace_root=workspace)
        assert result.exit_code != 0
        assert "TimeoutError" in result.stderr
        assert "2" in result.stderr


class TestStderr:
    """Tests for stderr capture."""

    def test_runtime_error_in_stderr(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Runtime errors appear in stderr."""
        source = "raise ValueError('oops')"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code != 0
        assert "ValueError" in result.stderr
        assert "oops" in result.stderr


class TestWorkingDirectory:
    """Tests for cwd handling."""

    def test_cwd_is_workspace(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Subprocess cwd is set to workspace_root."""
        source = "import pathlib\nstr(pathlib.Path.cwd())"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        # result is repr() of the string, so it includes quotes and escaped backslashes
        # Evaluate the repr to get the actual path string
        actual_path = eval(result.result)
        assert Path(actual_path).resolve() == workspace.resolve()


class TestParseResult:
    """Tests for the sentinel parsing logic."""

    def test_no_sentinel(self) -> None:
        stdout = "hello world\n"
        cleaned, result = SubprocessExecutor._parse_result(stdout)
        assert cleaned == "hello world\n"
        assert result is None

    def test_sentinel_only(self) -> None:
        stdout = f"\n{_RESULT_SENTINEL}42\n"
        cleaned, result = SubprocessExecutor._parse_result(stdout)
        assert cleaned == ""
        assert result == "42"

    def test_stdout_plus_sentinel(self) -> None:
        stdout = f"hello\n{_RESULT_SENTINEL}'world'\n"
        cleaned, result = SubprocessExecutor._parse_result(stdout)
        assert cleaned == "hello"
        assert result == "'world'"

    def test_sentinel_at_start(self) -> None:
        stdout = f"{_RESULT_SENTINEL}99\n"
        cleaned, result = SubprocessExecutor._parse_result(stdout)
        assert cleaned == ""
        assert result == "99\n" or result == "99"


class TestFileExecution:
    """Tests for executing .py files via the executor."""

    def test_execute_py_file(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """A .py file is read and executed, with last expression captured."""
        script = workspace / "script.py"
        script.write_text("x = 10\ny = 20\nx + y\n")

        source = script.read_text()
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert result.result == "30"

    def test_execute_py_file_with_args(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """A .py file can access args via sys.argv."""
        script = workspace / "args_script.py"
        script.write_text("import sys\n' '.join(sys.argv[1:])\n")

        source = script.read_text()
        result = executor.execute(
            source, args=["hello", "world"], timeout=10, workspace_root=workspace
        )
        assert result.exit_code == 0
        assert result.result == "'hello world'"

    def test_execute_py_file_with_print(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """A .py file's print output goes to stdout."""
        script = workspace / "print_script.py"
        script.write_text("for i in range(3):\n    print(f'line {i}')\n")

        source = script.read_text()
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)
        assert result.exit_code == 0
        assert "line 0" in result.stdout
        assert "line 2" in result.stdout
        assert result.result is None


class TestOutputTruncation:
    """Tests for stdout/stderr size limits."""

    def test_large_stdout_is_truncated(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Output exceeding 64 KB is truncated with a helpful message."""
        # Generate ~100 KB of output
        source = "print('x' * 100_000)"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)

        assert result.exit_code == 0
        assert len(result.stdout) < 70_000  # truncated below 100K
        assert "[truncated:" in result.stdout
        assert "write to a file" in result.stdout

    def test_small_stdout_not_truncated(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Output under 64 KB is returned in full."""
        source = "print('hello ' * 100)"
        result = executor.execute(source, args=[], timeout=10, workspace_root=workspace)

        assert result.exit_code == 0
        assert "[truncated:" not in result.stdout
        assert "hello" in result.stdout
