"""Shell command executor for pyddock.

Validates commands against shell policies and executes them via subprocess.
Commands are always executed with shell=False — no shell interpretation,
pipes, redirects, chaining, or variable expansion.
"""

from __future__ import annotations

import os as _os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pyddock.config import PyddockConfig, ShellPolicyConfig
from pyddock._process_utils import get_startupinfo, kill_and_drain, make_child_env, truncate_output

import shutil


def _abspath(p: Path) -> Path:
    """Normalize path without resolving symlinks/subst drives."""
    return Path(_os.path.abspath(str(p)))


def _looks_like_path(arg: str) -> bool:
    """Heuristic: does this argument look like a filesystem path?

    Returns True if the arg contains path separators, starts with '.',
    or starts with a drive letter (Windows). Excludes UNC-style paths
    (//server/...) and Perforce depot paths (//depot/...) which are not
    local filesystem targets.
    """
    # Exclude //... paths (UNC paths, Perforce depot paths)
    if arg.startswith("//"):
        return False
    if "/" in arg or "\\" in arg:
        return True
    if arg.startswith("."):
        return True
    # Windows drive letter (e.g. C:\...)
    if len(arg) >= 2 and arg[1] == ":" and arg[0].isalpha():
        return True
    return False


def _extract_path_candidates(arg: str) -> list[str]:
    """Extract all path-like substrings from an argument.

    Handles --flag=value and -flag=value patterns where the value portion
    may be a filesystem path that the command will write to. This prevents
    bypasses like --output=.pyddock/file where the scanner would otherwise
    treat the entire "--output=.pyddock/file" as a single (non-matching) path.

    Returns a list of path candidates to check. Always includes the raw arg
    itself if it looks like a path, plus any extracted value portions.
    """
    candidates: list[str] = []

    # Always check the raw arg
    if _looks_like_path(arg):
        candidates.append(arg)

    # Extract value from --flag=value or -flag=value patterns
    if arg.startswith("-") and "=" in arg:
        _, _, value = arg.partition("=")
        if value and _looks_like_path(value):
            candidates.append(value)

    return candidates


def resolve_command(command: str) -> list[str]:
    """Resolve interpreter prefix based on file extension.

    .ps1 → pwsh (preferred) or powershell (fallback)
    .py  → python (system python, not pyddock venv)
    .sh  → bash
    .bat → cmd /c
    Otherwise → direct execution as [command]
    """
    if command.endswith(".ps1"):
        ps = "pwsh" if shutil.which("pwsh") else "powershell"
        return [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", command]
    elif command.endswith(".py"):
        return ["python", command]
    elif command.endswith(".sh"):
        return ["bash", command]
    elif command.endswith(".bat"):
        return ["cmd", "/c", command]
    else:
        return [command]


@dataclass
class RunShellOutput:
    """Structured output from a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int


class ShellExecutor:
    """Validates commands against shell policies and executes them via subprocess.

    Args:
        config: The loaded pyddock configuration.
        workspace_root: The workspace root directory (used as cwd for commands).
    """

    def __init__(self, config: PyddockConfig, workspace_root: Path) -> None:
        self._config = config
        self._workspace_root = workspace_root

    def execute(
        self,
        command: str,
        args: list[str],
        timeout: float,
    ) -> RunShellOutput:
        """Execute a command after validating against shell policies.

        Preconditions:
            - command is a non-empty string
            - args is a list of strings (may be empty)
            - timeout is positive

        Postconditions:
            - Returns RunShellOutput with captured stdout, stderr, exit_code
            - If no policy matches: returns error output (exit_code=1)
            - If args rejected: returns error output (exit_code=1)
            - Command is NEVER executed with shell=True
        """
        # Step 1: Find matching policy
        policy = self._find_matching_policy(command)
        if policy is None:
            configured = ", ".join(
                p.command for p in self._config.shell.values()
            )
            return RunShellOutput(
                stdout="",
                stderr=(
                    f"Command '{command}' is not allowed. "
                    f"No matching [shell.*] policy found.\n"
                    f"Configured command patterns: {configured}\n"
                    f"Tip: Use run_python for complex workflows."
                ),
                exit_code=1,
            )

        # Step 2: Check args against policy
        rejection = self._check_args_policy(policy, args)
        if rejection is not None:
            return RunShellOutput(stdout="", stderr=rejection, exit_code=1)

        # Step 3: Check args for path-like values targeting protected dirs
        path_rejection = self._check_arg_paths(policy, args)
        if path_rejection is not None:
            return RunShellOutput(stdout="", stderr=path_rejection, exit_code=1)

        # Step 4: Resolve interpreter prefix
        cmd_parts = self._resolve_command(command)

        # Step 5: Execute
        full_cmd = cmd_parts + args
        try:
            env = make_child_env()

            proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(self._workspace_root),
                env=env,
                shell=False,  # NEVER shell=True
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
                if _os.name == "nt"
                else 0,
                startupinfo=get_startupinfo() if _os.name == "nt" else None,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                stdout_partial, stderr_partial = kill_and_drain(proc)
                timeout_msg = (
                    f"TimeoutError: Command exceeded {timeout}s limit. "
                    f"Increase the timeout parameter if this task needs more time.\n"
                    f"Tip: Use run_python for complex workflows."
                )
                combined_stderr = (
                    f"{timeout_msg}\n{stderr_partial}" if stderr_partial else timeout_msg
                )
                return RunShellOutput(
                    stdout=stdout_partial,
                    stderr=combined_stderr,
                    exit_code=1,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            # Normalize line endings (Windows)
            stdout = stdout.replace("\r\n", "\n")
            stderr = stderr.replace("\r\n", "\n")
            # Truncate large outputs to prevent memory/transport issues
            stdout = truncate_output(stdout, "output")
            stderr = truncate_output(stderr, "stderr")
            return RunShellOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return RunShellOutput(
                stdout="",
                stderr=(
                    f"Command not found: '{command}'. "
                    f"Ensure it is installed and on PATH.\n"
                    f"Tip: Use run_python for complex workflows."
                ),
                exit_code=1,
            )

    def _find_matching_policy(self, command: str) -> ShellPolicyConfig | None:
        """Find the first shell policy whose command regex matches.

        Iterates policies in dict insertion order (TOML section order).
        Returns None if no policy matches.
        """
        for policy in self._config.shell.values():
            if re.match(policy.command, command):
                return policy
        return None

    def _check_args_policy(
        self, policy: ShellPolicyConfig, args: list[str]
    ) -> str | None:
        """Validate args against the policy's allow/deny patterns.

        Returns None if args are permitted, or an error message string if rejected.
        """
        args_str = " ".join(args)

        if policy.mode == "deny":
            # Deny-by-default: args must match at least one allow pattern
            if not policy.allow:
                return (
                    "No argument patterns are allowed for this command.\n"
                    "Tip: Use run_python for complex workflows."
                )
            if not any(re.match(pattern, args_str) for pattern in policy.allow):
                allowed = ", ".join(policy.allow)
                return (
                    f"Arguments '{args_str}' not permitted. "
                    f"Allowed patterns: {allowed}\n"
                    f"Tip: Use run_python for complex workflows."
                )
            return None

        elif policy.mode == "allow":
            # Allow-by-default: args must NOT match any deny pattern
            for pattern in policy.deny:
                if re.match(pattern, args_str):
                    return (
                        f"Arguments '{args_str}' matched deny pattern '{pattern}'.\n"
                        f"Tip: Use run_python for complex workflows."
                    )
            return None

        return None

    def _check_arg_paths(
        self, policy: ShellPolicyConfig, args: list[str]
    ) -> str | None:
        """Scan args for path-like values and validate against arg_paths policy.

        Modes:
          "workspace" — block any path-like arg that resolves outside the workspace
                        or into a protected directory (.pyddock/, workspace modules,
                        shell script dirs).
          "protected" — only block paths resolving into protected directories.
          "none"      — no path scanning.

        Also extracts embedded paths from --flag=value style arguments to prevent
        bypasses where a command flag embeds a target path (e.g. --output=.pyddock/file).

        Returns None if all args pass, or an error message string if blocked.
        """
        if policy.arg_paths == "none":
            return None

        for arg in args:
            # Extract all path candidates from this arg (raw + embedded values)
            candidates = _extract_path_candidates(arg)
            if not candidates:
                continue

            for candidate in candidates:
                # Resolve relative to workspace root (same as command cwd)
                resolved = _abspath(self._workspace_root / candidate)

                # Check protected directories (both "workspace" and "protected" modes)
                # 1. .pyddock/ (excluding .pyddock/tmp/)
                pyddock_dir = _abspath(self._workspace_root / ".pyddock")
                try:
                    rel = resolved.relative_to(pyddock_dir)
                    if not str(rel).startswith("tmp"):
                        return (
                            f"Argument '{arg}' targets the protected .pyddock/ directory. "
                            f"Shell commands cannot write to .pyddock/ "
                            f"(self-modification protection)."
                        )
                except ValueError:
                    pass

                # 2. Workspace module directories
                workspace_imports = self._config.imports.workspace
                for mod_name, rel_path in workspace_imports.items():
                    ws_mod_dir = _abspath(self._workspace_root / rel_path)
                    try:
                        resolved.relative_to(ws_mod_dir)
                        return (
                            f"Argument '{arg}' targets workspace module directory "
                            f"'{mod_name}' ({rel_path}). Shell commands cannot write "
                            f"to workspace module directories."
                        )
                    except ValueError:
                        continue

                # 3. Shell script directories (derived from path-like command regexes)
                for _name, other_policy in self._config.shell.items():
                    cmd_regex = other_policy.command
                    if "/" in cmd_regex or "\\\\" in cmd_regex or cmd_regex.startswith("\\."):
                        path_pattern = cmd_regex.lstrip("^").rstrip("$")
                        if "/" in path_pattern:
                            dir_part = path_pattern.rsplit("/", 1)[0]
                        elif "\\\\" in path_pattern:
                            dir_part = path_pattern.rsplit("\\\\", 1)[0]
                        else:
                            dir_part = path_pattern
                        if dir_part:
                            clean_dir = dir_part.replace("\\.", ".").replace("\\/", "/")
                            script_dir = _abspath(self._workspace_root / clean_dir)
                            try:
                                resolved.relative_to(script_dir)
                                return (
                                    f"Argument '{arg}' targets a shell-executable script "
                                    f"directory ({clean_dir}). Shell commands cannot write "
                                    f"to script directories (write-then-execute prevention)."
                                )
                            except ValueError:
                                continue

                # 4. "workspace" mode: block paths outside the workspace entirely
                if policy.arg_paths == "workspace":
                    try:
                        resolved.relative_to(_abspath(self._workspace_root))
                    except ValueError:
                        return (
                            f"Argument '{arg}' resolves to '{resolved}' which is outside "
                            f"the workspace. Shell commands are restricted to workspace-"
                            f"relative paths (arg_paths = \"workspace\")."
                        )

        return None

    def _resolve_command(self, command: str) -> list[str]:
        """Resolve interpreter prefix based on file extension.

        .ps1 → powershell, .py → python, .sh → bash, .bat → cmd /c
        Otherwise → direct execution as [command].
        """
        return resolve_command(command)



def _derive_write_protected_paths(
    shell_config: dict[str, ShellPolicyConfig],
) -> list[str]:
    """Extract path patterns from shell command regexes for write protection.

    A regex is "path-like" if it contains '/', '\\\\', or starts with '\\.'.
    For path-like regexes, the directory portion is extracted and returned
    as a write-protected path pattern.

    Args:
        shell_config: The parsed shell policies dict.

    Returns:
        List of path patterns that should be write-denied in run_python.
    """
    protected: list[str] = []
    for _name, policy in shell_config.items():
        cmd_regex = policy.command
        # Heuristic: regex is path-like if it contains path separators or starts with \.
        if "/" in cmd_regex or "\\\\" in cmd_regex or cmd_regex.startswith("\\."):
            # Strip regex anchors and convert to a directory pattern
            path_pattern = cmd_regex.lstrip("^").rstrip("$")
            # Extract the directory portion
            if "/" in path_pattern:
                dir_part = path_pattern.rsplit("/", 1)[0]
            elif "\\\\" in path_pattern:
                dir_part = path_pattern.rsplit("\\\\", 1)[0]
            else:
                dir_part = path_pattern
            if dir_part:
                # Clean up regex escapes to get a usable filesystem path
                clean_dir = dir_part.replace("\\.", ".").replace("\\/", "/")
                protected.append(clean_dir)
    return protected
