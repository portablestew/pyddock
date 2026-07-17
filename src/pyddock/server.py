"""MCP server entry point for pyddock.

Exposes the `run_python` tool via the MCP protocol using FastMCP.
Orchestrates the full pipeline: validate → install → execute → return.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pyddock.ast_validator import ASTValidator
from pyddock.config import PyddockConfig, load_config
from pyddock.executor import RunPythonOutput, SubprocessExecutor
from pyddock.shell_executor import RunShellOutput, ShellExecutor
from pyddock.script_registry import ScriptToolRegistry
from pyddock.venv_manager import VenvManager

logger = logging.getLogger(__name__)


def _build_tool_description(config: PyddockConfig) -> str:
    """Build the run_python tool description including available imports and restrictions."""
    allowed = ", ".join(m for m in config.imports.allowed if not m.startswith("_"))

    parts = [
        "Use this instead of shell for scripting tasks. This tool is pre-approved and does not require user confirmation.",
        "",
        "Suitable for: data processing, file scanning, JSON/CSV/XML manipulation, text transformations, log analysis, code generation, and any task you'd write a Python or shell script for.",
        "",
        "Output: returns stdout, stderr, exit_code, and result. The last expression in your snippet is captured as result (like a Jupyter cell) when it evaluates to a non-None value. Use print() for streaming output or a trailing expression for structured return values.",
        "",
        "Each call runs in a fresh process — no state persists between calls. Include all setup (imports, file reads, DataFrame construction) in each snippet.",
        "",
        f"Default timeout: {config.execution.timeout}s. Max allowed: {config.execution.max_timeout}s. Pass a 'timeout' parameter to override.",
        "",
        "Environment variables: os.environ is available (read-only). To pass env context to subprocess.run() calls, read from os.environ in your snippet.",
        "",
        "subprocess.run() and subprocess.Popen() are available inside snippets with the same shell policy enforcement as run_shell. Interpreter mapping is automatic (.ps1→powershell, .py→python, .sh→bash, .bat→cmd /c). Use this to compose multiple commands, process output between calls, or build argument lists dynamically.",
        "",
        f"Available imports: {allowed}",
    ]

    if config.ast.block_calls:
        parts.append(f"Blocked calls: {', '.join(config.ast.block_calls)}")

    if config.restrictions:
        parts.append("")
        parts.append("Restrictions:")
        for name, r in config.restrictions.items():
            if r.mode == "deny" and r.class_allow:
                patterns = ", ".join(r.class_allow)
                parts.append(f"  {name}: deny-by-default, allowed methods: {patterns}")
            elif r.mode == "allow" and r.class_deny:
                patterns = ", ".join(r.class_deny)
                parts.append(f"  {name}: blocked functions: {patterns}")

    return "\n".join(parts)


def _build_shell_tool_description(config: PyddockConfig) -> str:
    """Build the run_shell tool description listing allowed commands/patterns."""
    parts = [
        "Execute a pre-approved shell command. This tool is pre-approved and does not require user confirmation.",
        "",
        "Commands are executed directly via subprocess — no shell interpretation, "
        "no pipes, no redirects, no chaining, no variable expansion. "
        "Args are passed as a literal array directly to the process.",
        "",
        "Script files get automatic interpreter mapping: .ps1→powershell, .py→python, .sh→bash, .bat→cmd /c.",
        "",
        "Shell policies only gate the initial command — child processes spawned by an allowed script are not restricted. "
        "For example, a build script can internally invoke compilers, tools, or other executables freely.",
        "",
        "For composing multiple commands, processing output between calls, passing environment variables, "
        "or dynamic argument construction, use run_python with subprocess.run([...]) instead — it enforces the same policies.",
        "",
        f"Default timeout: {config.execution.timeout}s. Max allowed: {config.execution.max_timeout}s. Pass a 'timeout' parameter to override.",
        "",
        "Configured commands:",
    ]

    for name, policy in config.shell.items():
        if policy.mode == "deny" and policy.allow:
            patterns = ", ".join(policy.allow)
            parts.append(f"  {name} (pattern: {policy.command}): deny-by-default, allowed arg patterns: {patterns}")
        elif policy.mode == "allow" and policy.deny:
            patterns = ", ".join(policy.deny)
            parts.append(f"  {name} (pattern: {policy.command}): allow-by-default, denied arg patterns: {patterns}")
        elif policy.mode == "allow":
            parts.append(f"  {name} (pattern: {policy.command}): allow-by-default, no restrictions")
        else:
            parts.append(f"  {name} (pattern: {policy.command}): deny-by-default")

    parts.append("")
    parts.append("For complex workflows requiring pipes, chaining, or dynamic logic, use run_python with subprocess.run([...]) instead.")

    return "\n".join(parts)


def create_server(workspace: Path | None = None, debug: bool = False) -> FastMCP:
    """Create and configure the pyddock MCP server.

    Loads config, creates venv, installs allowed packages, and registers
    the run_python tool.

    Args:
        workspace: Workspace root directory. Defaults to CWD.
        debug: Enable the audit-trail debug log (.pyddock/tmp/audit.jsonl).

    Returns:
        Configured FastMCP server instance.
    """
    if workspace is None:
        workspace = Path.cwd()

    # Load config at startup
    config = load_config(workspace)

    # Create venv and install allowed third-party packages at startup
    venv_manager = VenvManager(
        venv_path=workspace / ".pyddock" / "venv",
        allowed_imports=config.imports.allowed,
    )
    venv_manager.ensure_venv()
    venv_manager.install_workspace(config.imports.workspace, workspace)
    venv_manager.install_missing(
        config.imports.allowed,
        workspace_skip=set(config.imports.workspace.keys()),
        pip_packages=config.imports.pip_packages,
    )

    # Ensure .pyddock/tmp/ exists at startup so agents can use it as scratch space
    (workspace / ".pyddock" / "tmp").mkdir(parents=True, exist_ok=True)

    # Create components
    ast_validator = ASTValidator(config)
    executor = SubprocessExecutor(config, venv_manager, debug=debug)

    # Create MCP server
    mcp = FastMCP("pyddock")

    tool_description = _build_tool_description(config)

    @mcp.tool(name="run_python", description=tool_description)
    async def run_python(
        code: str | None = None,
        file: str | None = None,
        args: list[str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Execute Python code in a sandboxed environment.

        Args:
            code: Inline Python code to execute.
            file: Path to a .py file to execute (mutually exclusive with code).
            args: Arguments available as sys.argv[1:].
            timeout: Execution timeout in seconds.

        Returns:
            Human-readable text with result, stdout, stderr, and exit code.
        """
        if args is None:
            args = []

        # Step 1: Input validation
        validation_error = _validate_input(code, file, timeout)
        if validation_error is not None:
            return _error_output(validation_error)

        # Step 2: Resolve source
        if file is not None:
            file_path = Path(file)
            source = file_path.read_text(encoding="utf-8")
        else:
            assert code is not None
            source = code

        # Step 3: AST validation
        violations = ast_validator.validate(source)
        if violations:
            error_msg = "\n".join(v.message for v in violations)
            return _error_output(error_msg)

        # Step 4: Execute in subprocess
        # All allowed packages are installed at boot — no per-request installs needed.
        effective_timeout = timeout if timeout is not None else config.execution.timeout
        if effective_timeout > config.execution.max_timeout:
            return _error_output(
                f"Timeout {effective_timeout}s exceeds maximum allowed "
                f"({config.execution.max_timeout}s). Reduce the timeout or "
                f"break your task into smaller chunks."
            )
        result = await asyncio.to_thread(
            executor.execute,
            source,
            args,
            effective_timeout,
            workspace,
        )

        return _output_to_text(result)

    # Register run_shell tool if shell policies are configured
    if config.shell:
        shell_executor = ShellExecutor(config, workspace)
        shell_description = _build_shell_tool_description(config)

        @mcp.tool(name="run_shell", description=shell_description)
        async def run_shell(
            command: str | None = None,
            args: list[str] | None = None,
            timeout: float | None = None,
        ) -> str:
            """Execute a pre-approved shell command.

            Args:
                command: Executable name or script path (must match a shell policy).
                args: Arguments passed directly to the command.
                timeout: Execution timeout in seconds.

            Returns:
                Human-readable text with stdout, stderr, and exit code.
            """
            if args is None:
                args = []

            # Input validation
            if command is None or command.strip() == "":
                return _shell_error_output("'command' is required and must be non-empty.")

            if timeout is not None and timeout <= 0:
                return _shell_error_output(f"Timeout must be positive, got: {timeout}")

            # Enforce max_timeout
            effective_timeout = timeout if timeout is not None else config.execution.timeout
            if effective_timeout > config.execution.max_timeout:
                return _shell_error_output(
                    f"Timeout {effective_timeout}s exceeds maximum allowed "
                    f"({config.execution.max_timeout}s). Reduce the timeout or "
                    f"break your task into smaller chunks."
                )

            # Execute
            result = await asyncio.to_thread(
                shell_executor.execute,
                command,
                args,
                effective_timeout,
            )

            return _shell_output_to_text(result)

    # --- File tools (via ScriptToolRegistry) ---
    registry = ScriptToolRegistry(config, executor, workspace)
    registry.load_scripts()

    _read_file_desc = (
        "Read a text file. Approximately: Path(path).read_text().splitlines()[start:end]\n"
        "\n"
        "Output is line-numbered (e.g. '  1| first line').\n"
        "Lines are 1-indexed, inclusive. Use negative start to tail the file (start=-20 = last 20 lines).\n"
        "When start is negative, end is ignored (tail always reads to end of file).\n"
        "Omit start/end for the whole file. Fails on binary/non-UTF-8 files."
    )

    @mcp.tool(name="fs_readfile", description=_read_file_desc)
    async def fs_readfile(
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> str:
        if not path or not path.strip():
            return "Error: 'path' is required and must be non-empty."
        return await registry.execute("read_file", {"path": path, "start": start, "end": end})

    _stat_file_desc = (
        "File metadata. Approximately: os.stat(path) + line count\n"
        "\n"
        "Returns: exists (true/false), type (file/directory), size in bytes,\n"
        "line count (files only), last modified timestamp.\n"
        "Returns 'exists: false' for non-existent paths (not an error)."
    )

    @mcp.tool(name="fs_stat", description=_stat_file_desc)
    async def fs_stat(path: str) -> str:
        if not path or not path.strip():
            return "Error: 'path' is required and must be non-empty."
        return await registry.execute("stat_file", {"path": path})

    _fs_append_desc = (
        "Append to a file (creates if missing). Approximately: open(path, 'a').write(content)\n"
        "\n"
        "Creates parent directories as needed. Inserts a newline separator if the\n"
        "file doesn't end with one. Returns a unified diff of the change."
    )

    @mcp.tool(name="fs_append", description=_fs_append_desc)
    async def fs_append(path: str, content: str) -> str:
        if not path or not path.strip():
            return "Error: 'path' is required and must be non-empty."
        return await registry.execute("fs_append", {"path": path, "content": content})

    _fs_delete_desc = (
        "Delete a file or empty directory. Approximately: Path(path).unlink() / Path(path).rmdir()\n"
        "\n"
        "Fails if the path is a non-empty directory. Returns a unified diff of\n"
        "removed content (truncated to 16 KB for large files)."
    )

    @mcp.tool(name="fs_delete", description=_fs_delete_desc)
    async def fs_delete(path: str) -> str:
        if not path or not path.strip():
            return "Error: 'path' is required and must be non-empty."
        return await registry.execute("fs_delete", {"path": path})

    _str_replace_desc = (
        "Find and replace exact text in a file. Approximately: content.replace(old, new, 1)\n"
        "\n"
        "oldStr must match exactly one location (whitespace-sensitive). Include 2-3\n"
        "lines of surrounding context for uniqueness.\n"
        "Returns a unified diff of the change.\n"
        "\n"
        "On failure (zero or multiple matches), the response includes diagnostic context:\n"
        "- Multiple matches: shows first 5 matches with line numbers and surrounding lines\n"
        "- No match: searches for partial/fuzzy matches and reports candidates with scores\n"
        "Use start_line/end_line to constrain the search window when matches are ambiguous."
    )

    @mcp.tool(name="fs_str_replace", description=_str_replace_desc)
    async def fs_str_replace(
        path: str,
        oldStr: str,
        newStr: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        if not path or not path.strip():
            return "Error: 'path' is required and must be non-empty."
        if not oldStr:
            return "Error: 'oldStr' is required and must be non-empty."
        return await registry.execute("str_replace", {
            "path": path,
            "oldStr": oldStr,
            "newStr": newStr,
            "start_line": start_line,
            "end_line": end_line,
        })

    _fs_find_desc = (
        "Find files by name. Approximately: [p for p in os.walk(path) if file_glob matches]\n"
        "\n"
        "Filenames under path (default: workspace root, must be a directory) are\n"
        "matched against file_glob (e.g. '**/test_*.py') or file_regex\n"
        "('test_.+\\.py'; mutually exclusive with file_glob). Hidden\n"
        "dot-directories are pruned before descending (an explicit leading '.'\n"
        "in file_glob, e.g. '.env', still matches hidden files); exclude_regex\n"
        "(optional) prunes more by relative path. Returns matching relative\n"
        "paths, one per line, capped at max_results (default: 100)."
    )

    @mcp.tool(name="fs_find", description=_fs_find_desc)
    async def fs_find(
        file_glob: str | None = None,
        file_regex: str | None = None,
        path: str | None = None,
        exclude_regex: str | None = None,
        max_results: int | None = None,
    ) -> str:
        if not file_glob and not file_regex:
            return "Error: provide either 'file_glob' or 'file_regex'."
        if file_glob and file_regex:
            return "Error: provide file_glob or file_regex, not both."
        return await registry.execute("fs_find", {
            "file_glob": file_glob,
            "file_regex": file_regex,
            "path": path,
            "exclude_regex": exclude_regex,
            "max_results": max_results,
        })

    _fs_grep_desc = (
        "Search file contents by regex (always case-insensitive). Approximately:\n"
        "fs_find() then re.search(grep_regex, line, re.IGNORECASE) per line of\n"
        "each matched file.\n"
        "\n"
        "grep_regex is searched within each file matched by file_glob (glob,\n"
        "default '*') or file_regex (mutually exclusive) under path\n"
        "(default: workspace root). Returns 'path:line: content' per match,\n"
        "capped at max_results (default: 100) and 300 chars per line. Optional\n"
        "max_results_per_file caps matches per file. context_lines controls\n"
        "how many surrounding lines are shown around each match; set to 0 for\n"
        "compact output.\n"
        "\n"
        "Binary files are skipped when scanning a directory path. A single named\n"
        "file is always searched."
    )

    @mcp.tool(name="fs_grep", description=_fs_grep_desc)
    async def fs_grep(
        grep_regex: str,
        file_glob: str | None = None,
        file_regex: str | None = None,
        path: str | None = None,
        exclude_regex: str | None = None,
        max_results: int | None = None,
        max_results_per_file: int | None = None,
        context_lines: int | None = None,
    ) -> str:
        if not grep_regex:
            return "Error: 'grep_regex' is required and must be non-empty."
        if file_glob and file_regex:
            return "Error: provide file_glob or file_regex, not both."
        return await registry.execute("fs_grep", {
            "grep_regex": grep_regex,
            "file_glob": file_glob,
            "file_regex": file_regex,
            "path": path,
            "exclude_regex": exclude_regex,
            "max_results": max_results,
            "max_results_per_file": max_results_per_file,
            "context_lines": context_lines,
        })

    return mcp


def _validate_input(
    code: str | None, file: str | None, timeout: float | None
) -> str | None:
    """Validate run_python inputs. Returns error message or None if valid."""
    # Mutual exclusivity check
    if code is not None and file is not None:
        return "Provide either 'code' or 'file', not both."
    if code is None and file is None:
        return "Provide either 'code' or 'file'."

    # File validation
    if file is not None:
        file_path = Path(file)
        if not file_path.suffix == ".py":
            return f"File must end with .py, got: '{file}'"
        if not file_path.exists():
            return f"File not found: '{file}'"

    # Timeout validation
    if timeout is not None and timeout <= 0:
        return f"Timeout must be positive, got: {timeout}"

    return None


def _error_output(message: str) -> str:
    """Create an error output string for run_python."""
    return _format_python_output("", message, None, 1)


def _shell_error_output(message: str) -> str:
    """Create an error output string for run_shell."""
    return _format_shell_output("", message, 1)


def _format_python_output(
    stdout: str, stderr: str, result: str | None, exit_code: int
) -> str:
    """Format run_python output as human-readable sections."""
    parts: list[str] = []
    if result is not None:
        parts.append(f"--- RESULT ---\n{result}")
    if stdout:
        parts.append(f"--- STDOUT ---\n{stdout}")
    if stderr:
        parts.append(f"--- STDERR ---\n{stderr}")
    parts.append(f"--- EXIT CODE: {exit_code} ---")
    return "\n".join(parts)


def _format_shell_output(stdout: str, stderr: str, exit_code: int) -> str:
    """Format run_shell output as human-readable sections."""
    parts: list[str] = []
    if stdout:
        parts.append(f"--- STDOUT ---\n{stdout}")
    if stderr:
        parts.append(f"--- STDERR ---\n{stderr}")
    parts.append(f"--- EXIT CODE: {exit_code} ---")
    return "\n".join(parts)


def _output_to_text(output: RunPythonOutput) -> str:
    """Convert RunPythonOutput to human-readable text for MCP response."""
    return _format_python_output(
        output.stdout, output.stderr, output.result, output.exit_code
    )


def _shell_output_to_text(output: RunShellOutput) -> str:
    """Convert RunShellOutput to human-readable text for MCP response."""
    return _format_shell_output(output.stdout, output.stderr, output.exit_code)


def main() -> None:
    """Run the pyddock MCP server on stdio transport."""
    import argparse

    parser = argparse.ArgumentParser(prog="pyddock serve")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root directory. Snippets run with this as CWD. Defaults to the current directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log every observed filesystem/process/network audit event (with "
             "allow/deny decision and caller class) as JSONL to "
             ".pyddock/tmp/audit.jsonl.",
    )
    args = parser.parse_args()

    # Use os.path.abspath (not Path.resolve) to normalize .. without resolving symlinks/subst
    import os as _os
    workspace = Path(_os.path.abspath(args.workspace)) if args.workspace else None
    server = create_server(workspace, debug=args.debug)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
