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
from typing import Any

from pyddock.config import PyddockConfig, ShellPolicyConfig, find_deny_hint
from pyddock._base import canonical_path
from pyddock._process_utils import get_startupinfo, kill_and_drain, make_child_env, truncate_output

import shutil


def _abspath(p: Path) -> Path:
    """Canonicalize a path for shell-arg containment checks.

    Uses realpath (via canonical_path) so OS-level aliases — Windows 8.3 short
    names (PYDDOC~1 -> .pyddock), symlinks, junctions, subst drives — are
    resolved the same way the OS resolves them when the command runs. A purely
    lexical abspath would leave such aliases intact and let a path-like arg slip
    a protected directory (e.g. .pyddock/) past the arg_paths scanner.
    """
    return canonical_path(p)


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


# ---------------------------------------------------------------------------
# Shared shell-policy argument matching
# ---------------------------------------------------------------------------
#
# These primitives are the single source of truth for how shell-policy patterns
# are matched. They are used by all enforcement sites — run_shell
# (ShellExecutor), subprocess.run inside run_python (_subprocess_patch), and the
# audit-layer shell disposition (_audit_enforcement) — so the paths cannot drift.
#
# The match/search asymmetry is deliberate and security-relevant:
#   - ALLOW patterns authorize an operation by its leading verb (position 0 of
#     the args string), so they are start-anchored via re.match. Using search
#     would let an approved word appearing anywhere wrongly pass.
#   - DENY patterns forbid a token wherever it appears (e.g. "fetch ext::..."),
#     so they use re.search. (A re.match(".*token") idiom would miss a token
#     placed after a newline, since "." doesn't match newlines — search does.)
# Deny always wins: it is checked first, in both modes.


def args_match_deny(deny_patterns: list[str], args_str: str) -> str | None:
    """Return the first deny pattern that matches anywhere in args_str, else None."""
    for pattern in deny_patterns:
        if re.search(pattern, args_str):
            return pattern
    return None


def args_match_allow(allow_patterns: list[str], args_str: str) -> bool:
    """True if args_str matches (start-anchored) at least one allow pattern."""
    return any(re.match(pattern, args_str) for pattern in allow_patterns)


def evaluate_arg_policy(
    args_str: str,
    *,
    mode: str,
    allow: list[str],
    deny: list[str],
) -> str | None:
    """Evaluate an args string against allow/deny patterns.

    Returns None if permitted, or a short reason string if rejected. Deny is
    checked first (and in both modes); for mode="deny" the args must then match
    an allow pattern. Callers format the reason into a user-facing message.
    """
    hit = args_match_deny(deny, args_str)
    if hit is not None:
        return f"matched deny pattern '{hit}'"

    if mode == "deny":
        if not allow:
            return "no argument patterns are allowed for this command"
        if not args_match_allow(allow, args_str):
            return "not permitted by the allow-list"
    return None


# ---------------------------------------------------------------------------
# Environment-variable policy for subprocess spawns
# ---------------------------------------------------------------------------
#
# The child's environment is always derived from a known-good SNAPSHOT captured
# at startup (before agent code runs). Agent-supplied `env=` is treated as a set
# of OVERRIDES layered on that snapshot — never a wholesale replacement — so a
# child always keeps the trusted base.
#
# These primitives are the single source of truth shared by the two enforcement
# sites that can see an agent-controlled env: the subprocess proxy
# (_subprocess_patch, which rewrites env via filter_child_env) and the audit
# shell disposition (_audit_enforcement, which can only deny, via
# assert_env_locks, for captured-reference spawns like GitPython).

# A scheme://… URI prefix (tcp://, http://, ssh://, …). Distinct from a Windows
# drive (C:\) which is caught by the separator/drive checks below.
_ENV_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def is_unsafe_env_value(value: str, resolve_bases: tuple[Path, ...] = ()) -> bool:
    """True if an env value looks like a filesystem path or a URI.

    Tuned for ENV VALUES, deliberately NOT reusing `_looks_like_path` (which is
    for shell *args* and excludes `//…` as Perforce/UNC). For an env value,
    `\\\\server\\share` and `scheme://host` are exactly the dangerous redirects
    we want to catch. False positives are acceptable (the agent merely can't set
    that var to a path); false negatives for the path/URI class are the only real
    risk.

    The separator/drive/URI checks are the cheap, deterministic half — they need
    no filesystem access. The optional `resolve_bases` adds the one I/O-bearing
    case they miss: a value with NO separators that nonetheless names a real file
    or directory. A tool can resolve such a bare token as a path relative to its
    working directory (e.g. p4 reads `P4ENVIRO=z.txt` relative to cwd and loads
    P4EDITOR/P4DIFF from it — a command-execution redirect carrying no separator).
    When `resolve_bases` is supplied (the spawn's cwd / workspace root), a value
    that resolves to an existing entry under any base is treated as unsafe.
    Command-execution vars that need no existing file (e.g. GIT_SSH_COMMAND=calc)
    are covered by the per-command hard-locks instead.
    """
    if not value:
        return False
    if "/" in value or "\\" in value:        # separators, incl. UNC \\server
        return True
    if value[0] in ".~":                      # ./  ../  ~  (relative / home)
        return True
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True                           # C:\...  drive-qualified
    if _ENV_URI_RE.match(value):              # tcp://  http://  ssh://  ...
        return True
    # Separator-less but names a real file/dir → a tool may resolve it as a path.
    for base in resolve_bases:
        try:
            if (base / value).exists():
                return True
        except (OSError, ValueError):
            continue
    return False


def _coerce_env_item(raw_key: Any, raw_value: Any) -> tuple[str, str]:
    """Normalize an env (key, value) pair to text (handles bytes / None)."""
    key = (
        raw_key.decode("utf-8", "surrogateescape")
        if isinstance(raw_key, bytes)
        else str(raw_key)
    )
    if raw_value is None:
        value = ""
    elif isinstance(raw_value, bytes):
        value = raw_value.decode("utf-8", "surrogateescape")
    else:
        value = str(raw_value)
    return key, value


def _env_key_locked(key: str, deny_patterns: list[str]) -> bool:
    """True if `key` matches (full-match) any hard-lock deny pattern.

    Matching is case-INSENSITIVE. Windows environment variable names are
    case-insensitive, so a case-sensitive lock on ``PATH``/``HOME`` would be
    trivially bypassed by supplying ``Path``/``home``. Locking is fail-safe (the
    child keeps the snapshot value), so the rare POSIX case where a genuinely
    distinct lowercase variable gets locked is an acceptable trade-off. This
    applies to both the global ``[env]`` deny list and the per-command
    ``[shell.<cmd>.env]`` deny lists, which share this helper.
    """
    return any(re.fullmatch(p, key, re.IGNORECASE) for p in deny_patterns)


def _policy_env(policy: Any) -> dict | None:
    """Read a policy's per-command env block from dict or dataclass form."""
    if policy is None:
        return None
    if isinstance(policy, dict):
        return policy.get("env") or None
    return getattr(policy, "env", None) or None


def resolve_env_policy(global_env: Any, command_policy: Any) -> tuple[list[str], str]:
    """Merge the global [env] base with a per-command env block.

    Returns (deny_patterns, default). Deny always wins: the command's deny
    patterns are appended to the global ones. The command may override the
    global default disposition. Accepts both the serialized dict form (proxy /
    audit) and the EnvConfig/ShellPolicyConfig dataclass form (callers may pass
    either) so the two enforcement sites cannot drift.
    """
    if isinstance(global_env, dict):
        deny = list(global_env.get("deny", []))
        default = global_env.get("default", "inert")
    elif global_env is None:
        deny, default = [], "inert"
    else:  # EnvConfig dataclass
        deny = list(getattr(global_env, "deny", []))
        default = getattr(global_env, "default", "inert")
    cmd_env = _policy_env(command_policy)
    if cmd_env:
        deny = deny + list(cmd_env.get("deny", []))
        default = cmd_env.get("default", default)
    return deny, default


def filter_child_env(
    agent_env: Any,
    snapshot: dict[str, str],
    *,
    deny_patterns: list[str],
    default: str,
    resolve_bases: tuple[Path, ...] = (),
) -> dict[str, str]:
    """Return the env a child may receive, or raise PermissionError.

    Baseline is the snapshot. Each agent-supplied entry is an override:
      * value == snapshot[key]   → allowed (no-op)
      * value == ""              → allowed (explicit removal of the var)
      * key locked (deny match)  → rejected (only snapshot/empty allowed above)
      * default == "snapshot"    → rejected (only snapshot/empty permitted)
      * default == "inert"       → allowed iff the value is not path/URI-like and
                                   does not name an existing file/dir under any of
                                   `resolve_bases` (the spawn's cwd / workspace)

    Keys the agent does not mention keep their snapshot value. Used by the
    subprocess proxy, which both rewrites (the returned dict) and rejects.
    """
    effective = dict(snapshot)
    if not agent_env:
        return effective
    items = agent_env.items() if hasattr(agent_env, "items") else agent_env
    for raw_key, raw_value in items:
        key, value = _coerce_env_item(raw_key, raw_value)
        snap_value = snapshot.get(key)
        if snap_value is not None and value == snap_value:
            continue  # unchanged from the known-good value
        if value == "":
            effective.pop(key, None)  # explicit removal
            continue
        if _env_key_locked(key, deny_patterns):
            raise PermissionError(
                f"PermissionError: environment variable {key!r} is locked to its "
                f"known-good value for this command and cannot be overridden "
                f"(env policy)."
            )
        if default == "snapshot":
            raise PermissionError(
                f"PermissionError: environment variable {key!r} cannot be "
                f"overridden (env policy: snapshot — only the known-good value or "
                f"removal is permitted)."
            )
        if is_unsafe_env_value(value, resolve_bases):
            raise PermissionError(
                f"PermissionError: refusing environment override {key}={value!r} — "
                f"the value looks like a path or URI, which can redirect a tool to "
                f"attacker-controlled code or locations (env policy: inert)."
            )
        effective[key] = value
    return effective


def assert_env_locks(
    agent_env: Any,
    snapshot: dict[str, str],
    deny_patterns: list[str],
) -> None:
    """Audit-layer backstop: reject a LOCKED key set to a non-default value.

    Unlike filter_child_env (the proxy: rewrites + full inert filter), the audit
    hook cannot mutate the env it observes, so it enforces only the non-
    negotiable hard-locks — the keys whose values cause code execution / library
    loading. The inert-default filtering of unknown keys is a proxy-only
    convenience (it needs the snapshot merge that only the proxy can perform).
    This keeps the backstop from false-positiving on benign inert vars that a
    captured-reference spawn (e.g. GitPython) forwards from the environment.

    Raises PermissionError on the first violating locked key; returns None
    otherwise (including when agent_env is None — an env=None spawn inherits the
    process environment, which the audit hook cannot rewrite; keeping
    os.putenv/os.unsetenv denied is the defense-in-depth for that path).
    """
    if not agent_env:
        return
    items = agent_env.items() if hasattr(agent_env, "items") else agent_env
    for raw_key, raw_value in items:
        key, value = _coerce_env_item(raw_key, raw_value)
        if not _env_key_locked(key, deny_patterns):
            continue
        if value == "" or value == snapshot.get(key):
            continue
        raise PermissionError(
            f"PermissionError: environment variable {key!r} is locked and cannot "
            f"be set to a non-default value for this command (audit policy: env)."
        )


def find_matching_policy(
    shell_policies: dict[str, Any], command: str
) -> tuple[str | None, Any | None]:
    """Find the first [shell.*] policy whose command regex matches `command`.

    The single command→policy lookup shared by all three enforcement sites:
    run_shell (ShellExecutor, ``ShellPolicyConfig`` dataclasses), the subprocess
    proxy (_subprocess_patch, serialized dicts), and the audit-layer shell
    disposition (_audit_enforcement, serialized dicts). Accepts either the
    dataclass or the dict form (see `_policy_command`).

    The command token is matched in FULL via re.match against the policy's
    `command` regex (default `^<name>$`) — never a basename — so a path like
    `workspace/hax/git.exe` cannot match `^git$` and impersonate an allowed
    tool. This mirrors every site's matching exactly.

    Returns (policy_name, policy) in the caller's original form, or
    (None, None) if nothing matches.
    """
    for name, policy in shell_policies.items():
        pattern = _policy_command(policy) or f"^{re.escape(name)}$"
        if re.match(pattern, command):
            return name, policy
    return None, None


def _policy_command(policy: Any) -> str | None:
    """Read a policy's command regex from either the dict or dataclass form."""
    if isinstance(policy, dict):
        return policy.get("command")
    return getattr(policy, "command", None)


def derive_protected_dir(cmd_regex: str) -> str | None:
    """Extract the clean directory portion from a path-like shell command regex.

    A command regex is "path-like" if it contains '/' or '\\\\', or starts with
    '\\.'. For those, the directory portion (everything before the final path
    component) is extracted and regex escapes are cleaned up into a usable
    filesystem path. Returns None for non-path-like regexes (e.g. '^git$').

    This is the single source of truth for deriving write-protected script
    directories from shell command patterns — used by both the write-protection
    derivation (_derive_write_protected_paths) and the arg-path scanner
    (evaluate_arg_paths) so the two cannot drift.
    """
    if not ("/" in cmd_regex or "\\\\" in cmd_regex or cmd_regex.startswith("\\.")):
        return None
    path_pattern = cmd_regex.lstrip("^").rstrip("$")
    if "/" in path_pattern:
        dir_part = path_pattern.rsplit("/", 1)[0]
    elif "\\\\" in path_pattern:
        dir_part = path_pattern.rsplit("\\\\", 1)[0]
    else:
        dir_part = path_pattern
    if not dir_part:
        return None
    # Clean up regex escapes to get a usable filesystem path
    return dir_part.replace("\\.", ".").replace("\\/", "/")


def evaluate_arg_paths(
    args: list[str],
    *,
    arg_paths: str,
    workspace_root: Path,
    workspace_module_dirs: dict[str, str],
    shell_command_patterns: list[str],
) -> str | None:
    """Scan command args for path-like values and validate against arg_paths.

    This is the single source of truth for path-argument scanning, shared by all
    enforcement sites: run_shell (ShellExecutor), subprocess.run inside
    run_python (_subprocess_patch), and the audit-layer shell disposition.
    Keeping it here guarantees the paths cannot drift.

    Args:
        args: The command arguments (excluding the command/subcommand-launching
            token itself; callers pass whatever set should be path-scanned).
        arg_paths: "none" (skip), "protected" (block protected dirs only), or
            "workspace" (also block anything resolving outside the workspace).
        workspace_root: The workspace root; relative args resolve against it.
        workspace_module_dirs: Map of module name -> relative path for editable
            workspace packages (write-protected).
        shell_command_patterns: Every shell policy's command regex. Path-like
            ones yield write-protected script directories (write-then-execute
            prevention).

    Returns:
        None if all args pass, else a user-facing rejection message.
    """
    if arg_paths == "none":
        return None

    ws_root_abs = _abspath(workspace_root)
    pyddock_dir = _abspath(workspace_root / ".pyddock")
    script_dirs = [
        d for d in (derive_protected_dir(p) for p in shell_command_patterns)
        if d is not None
    ]

    for arg in args:
        # Extract all path candidates from this arg (raw + embedded --flag=value)
        candidates = _extract_path_candidates(arg)
        if not candidates:
            continue

        for candidate in candidates:
            # Resolve relative to workspace root (same as command cwd)
            resolved = _abspath(workspace_root / candidate)

            # 1. .pyddock/ (excluding .pyddock/tmp/)
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
            for mod_name, rel_path in workspace_module_dirs.items():
                ws_mod_dir = _abspath(workspace_root / rel_path)
                try:
                    resolved.relative_to(ws_mod_dir)
                    return (
                        f"Argument '{arg}' targets workspace module directory "
                        f"'{mod_name}' ({rel_path}). Shell commands cannot write "
                        f"to workspace module directories."
                    )
                except ValueError:
                    continue

            # 3. Shell-executable script directories
            for clean_dir in script_dirs:
                script_dir = _abspath(workspace_root / clean_dir)
                try:
                    resolved.relative_to(script_dir)
                    return (
                        f"Argument '{arg}' targets a shell-executable script "
                        f"directory ({clean_dir}). Shell commands cannot write "
                        f"to script directories (write-then-execute prevention)."
                    )
                except ValueError:
                    continue

            # 4. "workspace" mode: block paths resolving outside the workspace
            if arg_paths == "workspace":
                try:
                    resolved.relative_to(ws_root_abs)
                except ValueError:
                    return (
                        f"Argument '{arg}' resolves to '{resolved}' which is outside "
                        f"the workspace. Shell commands are restricted to workspace-"
                        f"relative paths (arg_paths = \"workspace\")."
                    )

    return None


def evaluate_cwd(
    cwd: Any,
    *,
    arg_paths: str,
    workspace_root: Path,
    workspace_module_dirs: dict[str, str],
    shell_command_patterns: list[str],
) -> str | None:
    """Validate a subprocess working directory against the arg_paths policy.

    Unlike `evaluate_arg_paths` (which filters path-*looking* args), a cwd is
    always a directory and is therefore always resolved and checked. Blocks a
    cwd inside `.pyddock/` (except `.pyddock/tmp/`), a workspace module dir, a
    shell-executable script dir, or — in "workspace" mode — anywhere outside the
    workspace. Mirrors the subprocess proxy's cwd check so the proxy and the
    audit-layer shell disposition stay aligned.

    Returns None if the cwd is permitted, else a rejection message.
    """
    if cwd is None or arg_paths == "none":
        return None

    cwd_str = cwd.decode("utf-8", "surrogateescape") if isinstance(cwd, bytes) else str(cwd)
    ws_root_abs = _abspath(workspace_root)
    pyddock_dir = _abspath(workspace_root / ".pyddock")
    resolved = _abspath(workspace_root / Path(cwd_str))

    # 1. .pyddock/ (excluding .pyddock/tmp/)
    try:
        rel = resolved.relative_to(pyddock_dir)
        if not str(rel).startswith("tmp"):
            return (
                f"cwd '{cwd_str}' targets the protected .pyddock/ directory. "
                f"Subprocess cwd cannot be set to .pyddock/ "
                f"(self-modification protection)."
            )
    except ValueError:
        pass

    # 2. Workspace module directories
    for mod_name, rel_path in workspace_module_dirs.items():
        try:
            resolved.relative_to(_abspath(workspace_root / rel_path))
            return (
                f"cwd '{cwd_str}' targets workspace module directory "
                f"'{mod_name}' ({rel_path}). Subprocess cwd cannot be set to "
                f"workspace module directories."
            )
        except ValueError:
            continue

    # 3. Shell-executable script directories
    for cmd_regex in shell_command_patterns:
        clean_dir = derive_protected_dir(cmd_regex)
        if clean_dir is None:
            continue
        try:
            resolved.relative_to(_abspath(workspace_root / clean_dir))
            return (
                f"cwd '{cwd_str}' targets a shell-executable script directory "
                f"({clean_dir}). Subprocess cwd cannot be set to script "
                f"directories (write-then-execute prevention)."
            )
        except ValueError:
            continue

    # 4. "workspace" mode: block cwd outside the workspace
    if arg_paths == "workspace":
        try:
            resolved.relative_to(ws_root_abs)
        except ValueError:
            return (
                f"cwd '{cwd_str}' resolves to '{resolved}' which is outside the "
                f"workspace. Subprocess cwd is restricted to the workspace "
                f"directory (arg_paths = \"workspace\")."
            )

    return None


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
            msg = (
                f"Command '{command}' is not allowed. "
                f"No matching [shell.*] policy found.\n"
                f"Configured command patterns: {configured}\n"
                f"Tip: Use run_python for complex workflows."
            )
            hint = find_deny_hint(command, self._config.deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            return RunShellOutput(
                stdout="",
                stderr=msg,
                exit_code=1,
            )

        # Step 2: Check args against policy
        rejection = self._check_args_policy(policy, args)
        if rejection is not None:
            hint = find_deny_hint(
                f"{command} {' '.join(args)}", self._config.deny_messages
            )
            if hint:
                rejection += f"\n[{hint}]"
            return RunShellOutput(stdout="", stderr=rejection, exit_code=1)

        # Step 3: Check args for path-like values targeting protected dirs
        path_rejection = self._check_arg_paths(policy, args)
        if path_rejection is not None:
            hint = find_deny_hint(
                f"{command} {' '.join(args)}", self._config.deny_messages
            )
            if hint:
                path_rejection += f"\n[{hint}]"
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

        Delegates to the shared find_matching_policy() so run_shell, the
        subprocess proxy, and the audit shell disposition use one command→policy
        lookup and cannot drift. Returns None if no policy matches.
        """
        _name, policy = find_matching_policy(self._config.shell, command)
        return policy

    def _check_args_policy(
        self, policy: ShellPolicyConfig, args: list[str]
    ) -> str | None:
        """Validate args against the policy's allow/deny patterns.

        Returns None if args are permitted, or an error message string if rejected.
        Delegates the allow/deny decision to the shared evaluate_arg_policy() so
        run_shell, subprocess.run, and the audit shell disposition stay in lockstep.
        """
        args_str = " ".join(args)
        reason = evaluate_arg_policy(
            args_str, mode=policy.mode, allow=policy.allow, deny=policy.deny
        )
        if reason is None:
            return None
        if "deny pattern" in reason:
            return (
                f"Arguments '{args_str}' {reason}.\n"
                f"Tip: Use run_python for complex workflows."
            )
        if "no argument patterns" in reason:
            return (
                "No argument patterns are allowed for this command.\n"
                "Tip: Use run_python for complex workflows."
            )
        allowed = ", ".join(policy.allow)
        return (
            f"Arguments '{args_str}' not permitted. "
            f"Allowed patterns: {allowed}\n"
            f"Tip: Use run_python for complex workflows."
        )

    def _check_arg_paths(
        self, policy: ShellPolicyConfig, args: list[str]
    ) -> str | None:
        """Scan args for path-like values and validate against arg_paths policy.

        Thin wrapper over the shared evaluate_arg_paths() so run_shell stays in
        lockstep with subprocess.run (_subprocess_patch) and the audit-layer
        shell disposition.

        Modes:
          "workspace" — block any path-like arg that resolves outside the workspace
                        or into a protected directory (.pyddock/, workspace modules,
                        shell script dirs).
          "protected" — only block paths resolving into protected directories.
          "none"      — no path scanning.

        Returns None if all args pass, or an error message string if blocked.
        """
        return evaluate_arg_paths(
            args,
            arg_paths=policy.arg_paths,
            workspace_root=self._workspace_root,
            workspace_module_dirs=self._config.imports.workspace,
            shell_command_patterns=[
                p.command for p in self._config.shell.values()
            ],
        )

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
        clean_dir = derive_protected_dir(policy.command)
        if clean_dir is not None:
            protected.append(clean_dir)
    return protected
