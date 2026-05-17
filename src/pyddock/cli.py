"""CLI entry point for pyddock.

Provides two subcommands:
  pyddock serve   — run the MCP server (stdio transport)
  pyddock run     — execute a snippet or .py file directly and print output

Uses argparse (stdlib only). The ``run`` subcommand reuses the same pipeline
as the MCP tool handler: load config → validate → AST check → install → execute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyddock.ast_validator import ASTValidator
from pyddock.config import load_config
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="pyddock",
        description="Policy-controlled Python execution for AI agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- serve subcommand ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the pyddock MCP server (stdio transport).",
    )
    serve_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root directory. Snippets run with this as CWD.",
    )

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a Python snippet or .py file directly.",
    )
    run_parser.add_argument(
        "code_or_file",
        help="Inline Python code or path to a .py file.",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Execution timeout in seconds (overrides config default).",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config_path",
        help="Path to a pyddock TOML config file (overrides resolution).",
    )
    # Everything after -- is passed as args to the snippet
    run_parser.add_argument(
        "snippet_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the snippet as sys.argv[1:] (use -- to separate).",
    )

    return parser


def _cmd_serve(workspace: str | None) -> None:
    """Run the MCP server on stdio transport."""
    from pyddock.server import create_server

    # Use os.path.abspath (not Path.resolve) to normalize .. without resolving symlinks/subst
    import os as _os
    ws = Path(_os.path.abspath(workspace)) if workspace else None
    server = create_server(ws)
    server.run(transport="stdio")


def _cmd_run(
    code_or_file: str,
    timeout: float | None,
    config_path: str | None,
    snippet_args: list[str],
) -> int:
    """Execute a snippet or file using the same pipeline as the MCP tool.

    Returns the exit code from the subprocess.
    """
    workspace = Path.cwd()

    # Load config — use explicit path if provided, otherwise normal resolution
    if config_path is not None:
        from pyddock.config import _parse_config, ConfigError
        import tomllib

        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            return 1
        try:
            data = tomllib.loads(cfg_path.read_bytes().decode("utf-8"))
            config = _parse_config(data)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
    else:
        config = load_config(workspace)

    # Resolve source: if it ends with .py and the file exists, treat as file
    if code_or_file.endswith(".py") and Path(code_or_file).is_file():
        source = Path(code_or_file).read_text(encoding="utf-8")
    else:
        source = code_or_file

    # AST validation
    validator = ASTValidator(config)
    violations = validator.validate(source)
    if violations:
        for v in violations:
            print(v.message, file=sys.stderr)
        return 1

    # Auto-install missing packages
    venv_manager = VenvManager(
        venv_path=workspace / ".pyddock" / "venv",
        allowed_imports=config.imports.allowed,
    )
    venv_manager.ensure_venv()
    imports = validator.extract_imports(source)
    venv_manager.install_missing(imports)

    # Execute
    effective_timeout = timeout if timeout is not None else config.execution.timeout
    executor = SubprocessExecutor(config, venv_manager)

    # Strip leading '--' from snippet_args if present
    args = snippet_args
    if args and args[0] == "--":
        args = args[1:]

    result = executor.execute(source, args, effective_timeout, workspace)

    # Output: print stdout, then stderr (to stderr), then result if present
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.result is not None:
        print(result.result)

    return result.exit_code


def main() -> None:
    """CLI entry point (wired via pyproject.toml [project.scripts])."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _cmd_serve(workspace=args.workspace)
    elif args.command == "run":
        exit_code = _cmd_run(
            code_or_file=args.code_or_file,
            timeout=args.timeout,
            config_path=args.config_path,
            snippet_args=args.snippet_args,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
