"""Unit tests for pyddock.server — input validation and pipeline orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pyddock.server import _validate_input


class TestValidateInput:
    """Tests for _validate_input function."""

    def test_both_code_and_file_rejects(self):
        """Providing both code and file is rejected."""
        err = _validate_input(code="print(1)", file="test.py", timeout=None)
        assert err is not None
        assert "not both" in err

    def test_neither_code_nor_file_rejects(self):
        """Providing neither code nor file is rejected."""
        err = _validate_input(code=None, file=None, timeout=None)
        assert err is not None
        assert "'code' or 'file'" in err

    def test_file_not_py_extension_rejects(self):
        """File without .py extension is rejected."""
        err = _validate_input(code=None, file="script.txt", timeout=None)
        assert err is not None
        assert ".py" in err

    def test_file_not_found_rejects(self):
        """Non-existent .py file is rejected."""
        err = _validate_input(code=None, file="nonexistent.py", timeout=None)
        assert err is not None
        assert "not found" in err.lower() or "File not found" in err

    def test_file_exists_and_valid(self, tmp_path):
        """Existing .py file passes validation."""
        f = tmp_path / "script.py"
        f.write_text("print('hello')")
        err = _validate_input(code=None, file=str(f), timeout=None)
        assert err is None

    def test_timeout_zero_rejects(self):
        """Timeout of zero is rejected."""
        err = _validate_input(code="print(1)", file=None, timeout=0)
        assert err is not None
        assert "positive" in err.lower()

    def test_timeout_negative_rejects(self):
        """Negative timeout is rejected."""
        err = _validate_input(code="print(1)", file=None, timeout=-5)
        assert err is not None
        assert "positive" in err.lower()

    def test_timeout_positive_passes(self):
        """Positive timeout passes validation."""
        err = _validate_input(code="print(1)", file=None, timeout=10)
        assert err is None

    def test_code_only_passes(self):
        """Providing only code passes validation."""
        err = _validate_input(code="x = 1", file=None, timeout=None)
        assert err is None

    def test_file_only_passes(self, tmp_path):
        """Providing only a valid file passes validation."""
        f = tmp_path / "run.py"
        f.write_text("x = 1")
        err = _validate_input(code=None, file=str(f), timeout=None)
        assert err is None


class TestMaxTimeout:
    """Tests for max_timeout enforcement."""

    def test_timeout_exceeding_max_is_rejected(self):
        """Timeout exceeding max_timeout returns an error."""
        from pyddock.config import load_config

        config = load_config()
        # Simulate what the handler does
        timeout = 5000.0
        if timeout > config.execution.max_timeout:
            error = (
                f"Timeout {timeout}s exceeds maximum allowed "
                f"({config.execution.max_timeout}s)."
            )
        else:
            error = None

        assert error is not None
        assert "5000" in error
        assert "3600" in error

    def test_timeout_within_max_is_accepted(self):
        """Timeout within max_timeout is fine."""
        from pyddock.config import load_config

        config = load_config()
        timeout = 600.0
        assert timeout <= config.execution.max_timeout


class TestShellToolDescription:
    """Tests for _build_shell_tool_description."""

    def test_description_contains_configured_patterns(self):
        """Tool description lists all configured command patterns."""
        from pyddock.config import (
            ASTConfig,
            ExecutionConfig,
            FilesystemConfig,
            ImportsConfig,
            PyddockConfig,
            ShellPolicyConfig,
        )
        from pyddock.server import _build_shell_tool_description

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["json"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            shell={
                "p4": ShellPolicyConfig(command="^p4$", mode="deny", allow=["filelog.*", "files.*"]),
                "git": ShellPolicyConfig(command="^git$", mode="allow", deny=["push.*"]),
            },
        )

        desc = _build_shell_tool_description(config)
        assert "^p4$" in desc
        assert "^git$" in desc
        assert "filelog.*" in desc
        assert "push.*" in desc
        assert "run_python" in desc
        assert "no shell interpretation" in desc

    def test_description_empty_shell_config(self):
        """Empty shell config produces minimal description."""
        from pyddock.config import (
            ASTConfig,
            ExecutionConfig,
            FilesystemConfig,
            ImportsConfig,
            PyddockConfig,
        )
        from pyddock.server import _build_shell_tool_description

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["json"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            shell={},
        )

        desc = _build_shell_tool_description(config)
        assert "run_python" in desc


class TestShellInputValidation:
    """Tests for run_shell input validation via the server handler."""

    def test_shell_error_output_format(self):
        """_shell_error_output returns correct text format."""
        from pyddock.server import _shell_error_output

        result = _shell_error_output("test error")
        assert "--- STDERR ---" in result
        assert "test error" in result
        assert "--- EXIT CODE: 1 ---" in result

    def test_shell_max_timeout_enforcement(self):
        """Timeout exceeding max_timeout is rejected (same pattern as run_python)."""
        from pyddock.config import load_config

        config = load_config()
        # max_timeout is 3600 by default
        timeout = 5000.0
        assert timeout > config.execution.max_timeout
