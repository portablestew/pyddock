"""Subprocess executor for pyddock.

Forks a subprocess to run validated Python code with runtime enforcement
applied. Captures stdout, stderr, and last-expression repr separately.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pyddock._runtime import SNIPPET_FILENAME
from pyddock.config import PyddockConfig
from pyddock._process_utils import get_startupinfo, kill_and_drain, make_child_env, truncate_output
from pyddock.venv_manager import VenvManager

# Sentinel used to extract the last-expression repr from stdout.
_RESULT_SENTINEL = "__PYDDOCK_RESULT__"


@dataclass
class RunPythonOutput:
    """Structured output from a subprocess execution."""

    stdout: str
    stderr: str
    result: str | None  # repr of last expression, if any
    exit_code: int


class SubprocessExecutor:
    """Executes validated Python code in a sandboxed subprocess.

    Uses the venv Python interpreter, injects a bootstrap script that
    applies runtime enforcement and captures the last expression's repr.

    Args:
        config: The loaded pyddock configuration.
        venv_manager: The venv manager providing the Python interpreter path.
    """

    def __init__(self, config: PyddockConfig, venv_manager: VenvManager, debug: bool = False) -> None:
        self._config = config
        self._venv_manager = venv_manager
        self._debug = debug

    def execute(
        self,
        source: str,
        args: list[str],
        timeout: float,
        workspace_root: Path,
    ) -> RunPythonOutput:
        """Execute validated source in a sandboxed subprocess.

        Preconditions:
            - source has passed AST validation
            - All required packages are installed in venv
            - timeout is positive
            - workspace_root exists and is a directory

        Postconditions:
            - Returns RunPythonOutput with captured stdout, stderr, result, exit_code
            - If timeout exceeded: process is killed, stderr contains timeout message,
              exit_code is non-zero
            - Subprocess inherits full parent environment
            - Subprocess cwd is workspace_root
            - Runtime enforcement is applied before user code runs

        Args:
            source: Validated Python source code to execute.
            args: Arguments to make available as sys.argv[1:].
            timeout: Execution timeout in seconds.
            workspace_root: Working directory for the subprocess.

        Returns:
            RunPythonOutput with stdout, stderr, result, and exit_code.
        """
        bootstrap = self._build_bootstrap(source, args, workspace_root)
        python_path = self._venv_manager.get_python_path()

        # Write bootstrap to a temp file and execute it
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="_pyddock_bootstrap_",
                delete=False,
                encoding="utf-8",
            )
            tmp_file.write(bootstrap)
            tmp_file.close()

            env = make_child_env()
            proc = subprocess.Popen(
                [str(python_path), tmp_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(workspace_root),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
                startupinfo=get_startupinfo() if os.name == "nt" else None,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                stdout_partial, stderr_partial = kill_and_drain(proc)
                timeout_msg = (
                    f"TimeoutError: Execution exceeded {timeout}s limit. "
                    f"Increase the timeout parameter if this task needs more time."
                )
                combined_stderr = (
                    f"{timeout_msg}\n{stderr_partial}" if stderr_partial else timeout_msg
                )
                return RunPythonOutput(
                    stdout=stdout_partial,
                    stderr=combined_stderr,
                    result=None,
                    exit_code=1,
                )

            stdout_str = stdout_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")

            # Truncate large outputs to prevent memory/transport issues
            stdout_str = truncate_output(stdout_str, "output")
            stderr_str = truncate_output(stderr_str, "stderr")
            stdout, result = self._parse_result(stdout_str)

            return RunPythonOutput(
                stdout=stdout,
                stderr=stderr_str,
                result=result,
                exit_code=proc.returncode,
            )
        finally:
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass

    def _build_bootstrap(
        self, source: str, args: list[str], workspace_root: Path
    ) -> str:
        """Generate the bootstrap script that runs inside the subprocess.

        The bootstrap:
        1. Sets sys.argv from the provided args
        2. Conditionally imports and applies RuntimeEnforcement
        3. Parses the user source, detects if last statement is an expression
        4. If last statement is expression: pops it, execs the rest, evals the
           expression, prints with sentinel
        5. If not: just execs the whole thing
        """
        # Serialize config for runtime enforcement
        config_dict = self._serialize_config()
        sentinel = _RESULT_SENTINEL

        # Find the path to pyddock's source for the subprocess
        import pyddock
        pyddock_src_path = str(Path(pyddock.__file__).parent.parent)

        lines = [
            "# -*- coding: utf-8 -*-",
            "import sys",
            "import ast as _ast",
            "import types as _pyddock_types",
            f"sys.path.insert(0, {pyddock_src_path!r})",
            f"sys.argv = [{SNIPPET_FILENAME!r}] + {args!r}",
            "",
            "# Apply runtime enforcement",
            "from pyddock._runtime import RuntimeEnforcement as _RE",
            f"_enforcement = _RE(",
            f"    config={config_dict!r},",
            f"    workspace_root={str(workspace_root)!r},",
            f"    debug={self._debug!r},",
            f")",
            "_enforcement.apply_all()",
            "",
            "# Execute user code in an ISOLATED namespace.",
            "# SECURITY: the snippet must never see bootstrap/enforcement internals",
            "# (_enforcement, _RE, the real `sys` module, etc.). Running the snippet",
            "# in this module's own globals (the historical behavior) leaked all of",
            "# those names — e.g. `_enforcement._real_os` handed back the real os, and",
            "# the bootstrap-global `sys` was the real (unproxied) sys module.",
            "#",
            "# We back the namespace with a real module object registered as",
            "# `__main__` so typing/dataclasses forward-reference resolution keeps",
            "# working, and we use a SINGLE dict for both exec and the last-expression",
            "# eval so comprehensions, class bodies, and nested functions resolve",
            "# names exactly like ordinary module-level execution.",
            f"_source = {source!r}",
            "_tree = _ast.parse(_source)",
            "_last_expr = None",
            "",
            "if _tree.body and isinstance(_tree.body[-1], _ast.Expr):",
            "    _last_expr_node = _tree.body.pop()",
            "    _last_expr = _ast.Expression(_last_expr_node.value)",
            "    _ast.fix_missing_locations(_last_expr)",
            "",
            "_snippet_mod = _pyddock_types.ModuleType('__main__')",
            "_ns = _snippet_mod.__dict__",
            "_ns['__name__'] = '__main__'",
            f"_ns['__file__'] = {SNIPPET_FILENAME!r}",
            "_ns['__doc__'] = None",
            "_ns['__package__'] = None",
            "_ns['__spec__'] = None",
            "_ns['__loader__'] = None",
            "_ns['__builtins__'] = __builtins__",
            "# Register so library code (typing.get_type_hints, dataclasses) can",
            "# resolve annotations against the snippet's own globals via sys.modules.",
            "sys.modules['__main__'] = _snippet_mod",
            "",
            f"_code = compile(_tree, {SNIPPET_FILENAME!r}, 'exec')",
            "exec(_code, _ns)",
            "",
            "if _last_expr is not None:",
            f"    _result = eval(compile(_last_expr, {SNIPPET_FILENAME!r}, 'eval'), _ns)",
            f"    if _result is not None:",
            f"        print(f'\\n{sentinel}{{repr(_result)}}')",
        ]

        return "\n".join(lines) + "\n"

    def _serialize_config(self) -> dict:
        """Serialize PyddockConfig to a plain dict for passing to subprocess."""
        return {
            "execution": {"timeout": self._config.execution.timeout},
            "imports": {
                "allowed": self._config.imports.allowed,
                "workspace": self._config.imports.workspace,
            },
            "filesystem": {
                "writable_paths": self._config.filesystem.writable_paths,
                "readable_paths": self._config.filesystem.readable_paths,
                "guards": [
                    {"pattern": g.pattern, "disposition": g.disposition}
                    for g in self._config.filesystem.guards
                ],
            },
            "ast": {
                "block_calls": self._config.ast.block_calls,
                "block_attributes": self._config.ast.block_attributes,
            },
            "restrictions": {
                name: {
                    "mode": r.mode,
                    "module_allow": r.module_allow,
                    "module_deny": r.module_deny,
                    "class_allow": r.class_allow,
                    "class_deny": r.class_deny,
                }
                for name, r in self._config.restrictions.items()
            },
            "shell": {
                name: {
                    "command": s.command,
                    "mode": s.mode,
                    "allow": s.allow,
                    "deny": s.deny,
                    "arg_paths": s.arg_paths,
                }
                for name, s in self._config.shell.items()
            },
            "deny_messages": [
                {"pattern": rule.pattern.pattern, "message": rule.message}
                for rule in self._config.deny_messages
            ],
            "audit": [
                {"pattern": p, "disposition": d}
                for p, d in self._config.audit.rules
            ],
        }

    @staticmethod
    def _parse_result(stdout: str) -> tuple[str, str | None]:
        """Parse stdout to extract the sentinel-wrapped result.

        Returns a tuple of (cleaned_stdout, result_or_none).
        The sentinel line is stripped from stdout.
        """
        sentinel_prefix = f"\n{_RESULT_SENTINEL}"

        # Look for the sentinel in stdout
        idx = stdout.rfind(sentinel_prefix)
        if idx == -1:
            # Also check if stdout starts with the sentinel (no leading newline)
            if stdout.startswith(_RESULT_SENTINEL):
                result_str = stdout[len(_RESULT_SENTINEL):]
                return "", result_str
            return stdout, None

        # Split: everything before the sentinel is user stdout,
        # everything after (on the same line) is the result
        user_stdout = stdout[:idx]
        result_str = stdout[idx + len(sentinel_prefix):]

        # Strip trailing newline from result if present
        if result_str.endswith("\n"):
            result_str = result_str[:-1]

        return user_stdout, result_str
