"""Subprocess and shell enforcement patch.

This module contains the subprocess/shell enforcement logic extracted from
RuntimeEnforcement.apply_subprocess_patch. It replaces the subprocess module
in sys.modules with a safe proxy that validates all commands against configured
shell policies, and blocks os.system().
"""
from __future__ import annotations

import pathlib
import re
import sys
from typing import Any

from pyddock._base import _ORIGINALS, _find_deny_hint, canonical_path
from pyddock.shell_executor import evaluate_arg_policy, evaluate_arg_paths


def apply_subprocess_patch(
    config: dict,
    workspace_root: pathlib.Path,
    real_os: Any,
    subprocess_module: Any,
    types_module: Any,
    resolve_command: Any,
    looks_like_path: Any,
    extract_path_candidates: Any,
    deny_messages: list,
) -> None:
    """Replace subprocess with a safe proxy module and block os.system.

    Instead of patching individual functions (whack-a-mole), we replace
    the entire subprocess module in sys.modules with a proxy that only
    exposes subprocess.run() and subprocess.Popen() — both validated
    against shell policies.

    Exposed on the proxy:
    - subprocess.run() — validated against shell policies
    - subprocess.Popen() — validated proxy class (same policy checks at construction)
    - subprocess.PIPE, DEVNULL, STDOUT — constants for run()/Popen() calls
    - subprocess.CompletedProcess — return type
    - subprocess.CalledProcessError, TimeoutExpired, SubprocessError — exceptions

    NOT exposed (no bypass surface):
    - call, check_call, check_output, getoutput, getstatusoutput
    """
    types = types_module

    _real_os = real_os
    _resolve_cmd = resolve_command
    shell_policies = config.get("shell", {})
    _deny_msgs_sp = deny_messages

    # Build example command for error messages
    if shell_policies:
        first_name = next(iter(shell_policies))
        first_policy = shell_policies[first_name]
        example_cmd = first_policy.get("command", first_name).lstrip("^").rstrip("$")
        allowed_commands_str = ", ".join(
            p.get("command", name) for name, p in shell_policies.items()
        )
    else:
        example_cmd = "command"
        allowed_commands_str = "(none configured)"

    def _find_matching_policy(command: str) -> dict | None:
        """Find first matching shell policy for a command."""
        for _name, policy in shell_policies.items():
            if re.match(policy["command"], command):
                return policy
        return None

    def _check_args_policy(policy: dict, cmd_args: list[str]) -> str | None:
        """Validate args against policy. Returns error message or None.

        Delegates to the shared evaluate_arg_policy() so this path stays in
        lockstep with run_shell (ShellExecutor) and the GitPython guard.
        """
        args_str = " ".join(cmd_args)
        reason = evaluate_arg_policy(
            args_str,
            mode=policy.get("mode", "deny"),
            allow=policy.get("allow", []),
            deny=policy.get("deny", []),
        )
        if reason is None:
            return None
        if "deny pattern" in reason:
            return f"Arguments '{args_str}' {reason}."
        if "no argument patterns" in reason:
            return "No argument patterns are allowed for this command."
        allowed = ", ".join(policy.get("allow", []))
        return (
            f"Arguments '{args_str}' not permitted. Allowed patterns: {allowed}"
        )

    # Pre-compute protected paths for arg scanning.
    # canonical_path (realpath) resolves 8.3 short names / symlinks / junctions /
    # subst drives so cwd containment checks cannot be bypassed by aliasing — the
    # same hardening applied in _fs_enforcement and shell_executor.
    _ws_root = workspace_root
    _pyddock_dir = canonical_path(_ws_root / ".pyddock")
    _workspace_imports = config.get("imports", {}).get("workspace", {})
    _ws_module_dirs: list[tuple[str, pathlib.Path]] = [
        (mod_name, canonical_path(_ws_root / rel_path))
        for mod_name, rel_path in _workspace_imports.items()
    ]
    _shell_protected_dirs: list[tuple[str, pathlib.Path]] = []
    for _sp_name, _sp_policy in shell_policies.items():
        _sp_cmd = _sp_policy.get("command", "")
        if "/" in _sp_cmd or "\\\\" in _sp_cmd or _sp_cmd.startswith("\\."):
            _sp_pattern = _sp_cmd.lstrip("^").rstrip("$")
            if "/" in _sp_pattern:
                _sp_dir = _sp_pattern.rsplit("/", 1)[0]
            elif "\\\\" in _sp_pattern:
                _sp_dir = _sp_pattern.rsplit("\\\\", 1)[0]
            else:
                _sp_dir = _sp_pattern
            if _sp_dir:
                _sp_clean = _sp_dir.replace("\\.", ".").replace("\\/", "/")
                _shell_protected_dirs.append(
                    (_sp_clean, canonical_path(_ws_root / _sp_clean))
                )
    _ws_root_abs = canonical_path(_ws_root)

    _looks_like_path_rt = looks_like_path
    _extract_path_candidates_rt = extract_path_candidates

    def _check_arg_paths(policy: dict, cmd_args: list[str]) -> str | None:
        """Scan args for path-like values and validate against arg_paths policy.

        Thin wrapper over the shared evaluate_arg_paths() so subprocess.run stays
        in lockstep with run_shell (ShellExecutor) and the GitPython guard.
        """
        return evaluate_arg_paths(
            cmd_args,
            arg_paths=policy.get("arg_paths", "workspace"),
            workspace_root=_ws_root,
            workspace_module_dirs=_workspace_imports,
            shell_command_patterns=[
                p.get("command", "") for p in shell_policies.values()
            ],
        )

    def _check_cwd(policy: dict, cwd: Any) -> str | None:
        """Validate cwd kwarg against the same rules as arg paths.

        Applies the policy's arg_paths mode to the cwd directory:
        - "none": no restriction
        - "protected": block cwd inside protected directories
        - "workspace": block cwd outside the workspace or inside protected dirs

        Returns None if cwd is permitted, or an error message string if blocked.
        """
        if cwd is None:
            return None

        arg_paths_mode = policy.get("arg_paths", "workspace")
        if arg_paths_mode == "none":
            return None

        resolved = canonical_path(cwd)

        # Check .pyddock/ (excluding .pyddock/tmp/)
        try:
            rel = resolved.relative_to(_pyddock_dir)
            if not str(rel).startswith("tmp"):
                return (
                    f"cwd '{cwd}' targets the protected .pyddock/ directory. "
                    f"Subprocess cwd cannot be set to .pyddock/ "
                    f"(self-modification protection)."
                )
        except ValueError:
            pass

        # Check workspace module directories
        for mod_name, ws_dir in _ws_module_dirs:
            try:
                resolved.relative_to(ws_dir)
                return (
                    f"cwd '{cwd}' targets workspace module directory "
                    f"'{mod_name}'. Subprocess cwd cannot be set to workspace "
                    f"module directories."
                )
            except ValueError:
                continue

        # Check shell script directories
        for dir_label, script_dir in _shell_protected_dirs:
            try:
                resolved.relative_to(script_dir)
                return (
                    f"cwd '{cwd}' targets a shell-executable script "
                    f"directory ({dir_label}). Subprocess cwd cannot be set "
                    f"to script directories (write-then-execute prevention)."
                )
            except ValueError:
                continue

        # "workspace" mode: block cwd outside the workspace
        if arg_paths_mode == "workspace":
            try:
                resolved.relative_to(_ws_root_abs)
            except ValueError:
                return (
                    f"cwd '{cwd}' resolves to '{resolved}' which is outside "
                    f"the workspace. Subprocess cwd is restricted to the "
                    f"workspace directory (arg_paths = \"workspace\")."
                )

        return None

    def _validated_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        """subprocess.run replacement that validates against shell policies."""
        # Reject shell=True
        if kwargs.get("shell", False):
            raise PermissionError(
                "PermissionError: shell=True is not permitted in subprocess.run(). "
                "Pass command as a list instead: "
                f"subprocess.run(['{example_cmd}', 'arg1', 'arg2'])"
            )
        # Reject string commands
        if isinstance(cmd, str):
            raise PermissionError(
                "PermissionError: String commands are not permitted in subprocess.run(). "
                "Pass command as a list instead: "
                f"subprocess.run(['{example_cmd}', 'arg1', 'arg2'])"
            )
        # If no shell policies configured, block entirely
        if not shell_policies:
            raise PermissionError(
                "PermissionError: No shell policies configured. "
                "Add [shell.*] sections to pyddock.toml to enable command execution, "
                "or use run_shell directly."
            )
        # Validate command against shell policy
        if not cmd:
            raise PermissionError(
                "PermissionError: Empty command list is not permitted."
            )
        command = str(cmd[0])
        cmd_args = [str(a) for a in cmd[1:]]
        policy = _find_matching_policy(command)
        if policy is None:
            msg = (
                f"PermissionError: Command '{command}' is not permitted. "
                f"No matching shell policy found. "
                f"Allowed commands: {allowed_commands_str}"
            )
            hint = _find_deny_hint(command, _deny_msgs_sp)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        rejection = _check_args_policy(policy, cmd_args)
        if rejection is not None:
            msg = f"PermissionError: {rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        # Check arg paths against protected directories
        path_rejection = _check_arg_paths(policy, cmd_args)
        if path_rejection is not None:
            msg = f"PermissionError: {path_rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        # Check cwd against the same path rules
        cwd_rejection = _check_cwd(policy, kwargs.get("cwd"))
        if cwd_rejection is not None:
            msg = f"PermissionError: {cwd_rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        # Apply interpreter mapping (same as run_shell) and execute
        resolved = _resolve_cmd(command)
        full_cmd = resolved + cmd_args
        kwargs["shell"] = False
        return _ORIGINALS["subprocess.run"](full_cmd, *args, **kwargs)

    def _validate_command(cmd: Any, caller: str, cwd: Any = None) -> tuple[list[str], list[str]]:
        """Shared validation for run() and Popen(). Returns (resolved_cmd, cmd_args).

        Raises PermissionError if the command is not permitted.
        """
        # Reject shell=True handled by caller (kwargs not passed here)
        # Reject string commands
        if isinstance(cmd, str):
            raise PermissionError(
                f"PermissionError: String commands are not permitted in subprocess.{caller}(). "
                "Pass command as a list instead: "
                f"subprocess.{caller}(['{example_cmd}', 'arg1', 'arg2'])"
            )
        # If no shell policies configured, block entirely
        if not shell_policies:
            raise PermissionError(
                "PermissionError: No shell policies configured. "
                "Add [shell.*] sections to pyddock.toml to enable command execution, "
                "or use run_shell directly."
            )
        if not cmd:
            raise PermissionError(
                "PermissionError: Empty command list is not permitted."
            )
        command = str(cmd[0])
        cmd_args = [str(a) for a in cmd[1:]]
        policy = _find_matching_policy(command)
        if policy is None:
            msg = (
                f"PermissionError: Command '{command}' is not permitted. "
                f"No matching shell policy found. "
                f"Allowed commands: {allowed_commands_str}"
            )
            hint = _find_deny_hint(command, _deny_msgs_sp)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        rejection = _check_args_policy(policy, cmd_args)
        if rejection is not None:
            msg = f"PermissionError: {rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        path_rejection = _check_arg_paths(policy, cmd_args)
        if path_rejection is not None:
            msg = f"PermissionError: {path_rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        cwd_rejection = _check_cwd(policy, cwd)
        if cwd_rejection is not None:
            msg = f"PermissionError: {cwd_rejection}"
            hint = _find_deny_hint(
                f"{command} {' '.join(cmd_args)}", _deny_msgs_sp
            )
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        resolved = _resolve_cmd(command)
        return resolved + cmd_args, cmd_args

    class _SafePopen:
        """Proxy around subprocess.Popen that validates commands against shell policies.

        Validates the command at construction time (same checks as subprocess.run),
        then delegates all safe operations to the real Popen instance.
        """

        def __init__(self, cmd: Any, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("shell", False):
                raise PermissionError(
                    "PermissionError: shell=True is not permitted in subprocess.Popen(). "
                    "Pass command as a list instead: "
                    f"subprocess.Popen(['{example_cmd}', 'arg1', 'arg2'])"
                )
            full_cmd, _ = _validate_command(cmd, "Popen", cwd=kwargs.get("cwd"))
            kwargs["shell"] = False
            self._proc = _ORIGINALS["subprocess.Popen"](full_cmd, *args, **kwargs)

        # --- Process control ---
        def communicate(self, *args: Any, **kwargs: Any) -> tuple:
            return self._proc.communicate(*args, **kwargs)

        def wait(self, *args: Any, **kwargs: Any) -> int:
            return self._proc.wait(*args, **kwargs)

        def poll(self) -> int | None:
            return self._proc.poll()

        def terminate(self) -> None:
            return self._proc.terminate()

        def kill(self) -> None:
            return self._proc.kill()

        def send_signal(self, signal: int) -> None:
            return self._proc.send_signal(signal)

        # --- Properties ---
        @property
        def stdout(self) -> Any:
            return self._proc.stdout

        @property
        def stderr(self) -> Any:
            return self._proc.stderr

        @property
        def stdin(self) -> Any:
            return self._proc.stdin

        @property
        def pid(self) -> int:
            return self._proc.pid

        @property
        def returncode(self) -> int | None:
            return self._proc.returncode

        @property
        def args(self) -> Any:
            return self._proc.args

        # --- Context manager ---
        def __enter__(self) -> "_SafePopen":
            return self

        def __exit__(self, *args: Any) -> None:
            self._proc.__exit__(*args)

        def __repr__(self) -> str:
            return f"<SafePopen pid={self.pid} returncode={self.returncode}>"

    # Build the safe subprocess proxy
    _subprocess_module = subprocess_module
    if _subprocess_module is not None:
        _ORIGINALS["subprocess.run"] = _subprocess_module.run
        _ORIGINALS["subprocess.Popen"] = _subprocess_module.Popen

        safe_subprocess = types.ModuleType("subprocess")
        safe_subprocess.__doc__ = "Safe subprocess proxy provided by pyddock. Only subprocess.run() and subprocess.Popen() are available, validated against shell policies."

        # Allowed entry points
        safe_subprocess.run = _validated_run
        safe_subprocess.Popen = _SafePopen

        # Constants needed for run()/Popen() calls
        safe_subprocess.PIPE = _subprocess_module.PIPE
        safe_subprocess.DEVNULL = _subprocess_module.DEVNULL
        safe_subprocess.STDOUT = _subprocess_module.STDOUT

        # Types needed for return values and error handling
        safe_subprocess.CompletedProcess = _subprocess_module.CompletedProcess
        safe_subprocess.CalledProcessError = _subprocess_module.CalledProcessError
        safe_subprocess.TimeoutExpired = _subprocess_module.TimeoutExpired
        safe_subprocess.SubprocessError = _subprocess_module.SubprocessError

        # Replace in sys.modules so 'import subprocess' finds the proxy
        sys.modules["subprocess"] = safe_subprocess

    # Always patch os.system
    def _blocked_os_system(cmd: Any) -> None:
        raise PermissionError(
            "PermissionError: os.system() is not available. "
            f"Use subprocess.run(['{example_cmd}', 'arg1', 'arg2']) instead, "
            "which validates commands against the shell policy."
        )

    # Patch os.system on the real os module
    _real_os.system = _blocked_os_system

    # Also patch it on the safe os proxy if it exists in sys.modules
    if "os" in sys.modules:
        safe_os = sys.modules["os"]
        safe_os.system = _blocked_os_system
