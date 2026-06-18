"""GitPython command enforcement.

GitPython bypasses pyddock's subprocess proxy: at import time it does
`from subprocess import Popen` (a captured function reference), so swapping
`sys.modules["subprocess"]` for the safe proxy never affects it. Every git
operation — `repo.git.X()`, `repo.iter_commits()`, `repo.index.commit()`,
`repo.remotes.origin.push()` — instead funnels through a single chokepoint:
`git.cmd.Git.execute(self, command, ...)`, where `command` is the fully built
argv list (`[git_exe, *global_opts, subcommand, *args]`).

We wrap that chokepoint and validate the command against the `[shell.git]`
policy before the real `execute` spawns the process. This gives GitPython the
same command allow-listing as `run_shell` / `subprocess.run(["git", ...])`,
plus extra hardening for git's option-injection surface.

Note: GitPython uses lower-level plumbing than a human at the CLI
(`rev-list`, `cat-file --batch-check`, `remote get-url`), so the `[shell.git]`
allow-list must include those read-only plumbing verbs for ordinary read
operations to work. See default_config.toml.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

from pyddock._base import _find_deny_hint
from pyddock.shell_executor import args_match_allow, args_match_deny, evaluate_arg_paths

# Global git options (before the subcommand) that enable command/config
# injection — effectively arbitrary code execution — and must never be allowed
# from agent code. None of these appear in GitPython's normal operation; they
# only show up if an agent explicitly injects them (e.g. repo.git(c=...) or
# repo.git.execute([...])).
_REJECT_GLOBAL_OPTS = frozenset({
    "-c",              # -c core.sshCommand=<cmd>, -c protocol.ext.allow=always, ...
    "--config-env",    # same as -c but value from env var
    "--exec-path",     # relocate git's helper binaries
    "--upload-pack",   # arbitrary command on fetch/clone
    "--receive-pack",  # arbitrary command on push
})

# Global options that take a value in a separate token (e.g. `-C <path>`).
_GLOBAL_VALUE_OPTS = frozenset({
    "-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix",
})

# Global flags that take no value.
_GLOBAL_FLAGS = frozenset({
    "-p", "--paginate", "-P", "--no-pager", "--bare", "--no-replace-objects",
    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
    "--icase-pathspecs", "--no-optional-locks",
    "--html-path", "--man-path", "--info-path",
})


def _basename_noext(path: str) -> str:
    """Return the lowercase executable basename without a .exe suffix."""
    base = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base


def build_git_command_validator(
    config: dict,
    workspace_root: str | Path | None = None,
    deny_messages: list[tuple[re.Pattern[str], str]] | None = None,
) -> Callable[[Any], None]:
    """Build a validator for GitPython command vectors.

    Returns a callable that takes the `command` argument GitPython passes to
    `Git.execute` and raises PermissionError if it is not permitted. The
    validator enforces, in order:

    1. command must be a non-empty list (string commands → shell mode → reject)
    2. command[0] must be the git executable (blocks repo.git.execute([...])
       being used to launch a non-git program)
    3. leading global options are vetted: injection-class options are rejected,
       known value/flag options are consumed, unknown ones are rejected
    4. the subcommand + its args must match an allow pattern from [shell.git]
    5. path-like args are scanned against the [shell.git] arg_paths policy
       (same scan as run_shell / subprocess.run) so a permitted subcommand
       cannot write into .pyddock/, workspace module dirs, script dirs, or
       (in "workspace" mode) anywhere outside the workspace.

    Args:
        config: Serialized pyddock config dict.
        workspace_root: Workspace root for arg-path resolution. When None, the
            path scan (step 5) is skipped — this keeps the validator usable in
            unit tests that only exercise command/allow validation. Production
            callers (apply_all) always pass it.
        deny_messages: Pre-compiled (pattern, message) deny hints.
    """
    deny_msgs = deny_messages or []
    policy = config.get("shell", {}).get("git")
    allow_list = policy.get("allow", []) if policy else []
    # Deny-always-wins tokens (e.g. ext::, --upload-pack) — searched anywhere in
    # the command, regardless of subcommand. Catches RCE/transport tokens that a
    # per-subcommand allow pattern would otherwise wave through.
    deny_list = policy.get("deny", []) if policy else []
    # arg_paths scanning config (enforced at the policy's configured level —
    # default "workspace"). Mirrors run_shell / subprocess.run path scanning.
    arg_paths_mode = policy.get("arg_paths", "workspace") if policy else "workspace"
    _ws_root = Path(workspace_root) if workspace_root is not None else None
    _workspace_imports = config.get("imports", {}).get("workspace", {})
    _shell_command_patterns = [
        p.get("command", "") for p in config.get("shell", {}).values()
    ]

    def _reject(msg: str, attempted: str = "") -> None:
        full = f"PermissionError: {msg}"
        hint = _find_deny_hint(attempted, deny_msgs) if attempted else None
        if hint:
            full += f"\n[{hint}]"
        raise PermissionError(full)

    def _validate(command: Any) -> None:
        if isinstance(command, str):
            _reject(
                "GitPython string commands are not permitted under pyddock "
                "enforcement. (This usually means shell mode.)"
            )
        if not command:
            _reject("Empty git command is not permitted.")
        cmd = [str(x) for x in command]

        # 2. The executable must be git.
        if _basename_noext(cmd[0]) != "git":
            _reject(
                f"'{cmd[0]}' is not the git executable; only git commands are "
                f"permitted via GitPython."
            )

        # 3. Deny always wins: reject dangerous tokens (e.g. ext:: transport,
        # --upload-pack) anywhere in the command, regardless of subcommand. This
        # shares args_match_deny() with the shell paths and closes RCE vectors
        # that an allowed verb (fetch/pull/ls-remote) would otherwise carry.
        post_exe = " ".join(cmd[1:])
        hit = args_match_deny(deny_list, post_exe)
        if hit is not None:
            _reject(
                f"git '{post_exe}' matched a denied pattern "
                f"('{hit}') — this token is not permitted.",
                attempted=f"git {post_exe}",
            )

        # 4. Walk leading global options.
        i, n = 1, len(cmd)
        while i < n:
            tok = cmd[i]
            if not tok.startswith("-"):
                break  # reached the subcommand
            base = tok.split("=", 1)[0]
            if base in _REJECT_GLOBAL_OPTS:
                _reject(
                    f"Global git option '{base}' is not permitted "
                    f"(command/config injection risk).",
                    attempted=f"git {base}",
                )
            if base in _GLOBAL_VALUE_OPTS:
                # `--opt=value` is one token; `-C value` consumes the next token.
                i += 1 if "=" in tok else 2
                continue
            if tok in _GLOBAL_FLAGS:
                i += 1
                continue
            _reject(
                f"Unrecognized global git option '{tok}' is not permitted.",
                attempted=f"git {tok}",
            )

        if i >= n:
            _reject("No git subcommand found in command.")

        # 5. Validate subcommand + args against the allow-list (re.match via the
        # shared args_match_allow, applied to the post-global-options subcommand).
        args_str = " ".join(cmd[i:])
        if not args_match_allow(allow_list, args_str):
            allowed_str = ", ".join(allow_list) or "(none)"
            _reject(
                f"git '{args_str}' is not permitted. Allowed subcommand "
                f"patterns: {allowed_str}",
                attempted=f"git {args_str}",
            )

        # 6. Scan path-like args against the arg_paths policy (parity with
        # run_shell / subprocess.run). Without this, an allowed subcommand could
        # carry a path argument into a protected directory — e.g.
        # `git add .pyddock/x` or a write into a workspace module dir — because
        # git is a native subprocess that bypasses the Python filesystem patches.
        # Scans the subcommand + its args (cmd[i:]); the subcommand token itself
        # is not path-like so it is harmlessly ignored.
        if _ws_root is not None:
            path_rejection = evaluate_arg_paths(
                cmd[i:],
                arg_paths=arg_paths_mode,
                workspace_root=_ws_root,
                workspace_module_dirs=_workspace_imports,
                shell_command_patterns=_shell_command_patterns,
            )
            if path_rejection is not None:
                _reject(path_rejection, attempted=f"git {args_str}")

    return _validate


def apply_gitpython_patch(
    config: dict,
    workspace_root: str | Path | None = None,
    deny_messages: list[tuple[re.Pattern[str], str]] | None = None,
) -> bool:
    """Wrap git.cmd.Git.execute to validate commands against the shell policy.

    No-op (returns False) if git is not an allowed import or GitPython is not
    installed. Returns True if the guard was installed.
    """
    allowed = config.get("imports", {}).get("allowed", [])
    if "git" not in allowed:
        return False

    # GitPython is pre-imported during install_import_hook, so git.cmd is in
    # sys.modules and still the real module at this point (install_module_proxies
    # runs later). Fall back to importing it if needed.
    gitcmd = sys.modules.get("git.cmd")
    if gitcmd is None:
        try:
            import importlib
            gitcmd = importlib.import_module("git.cmd")
        except ImportError:
            return False

    Git = getattr(gitcmd, "Git", None)
    if Git is None:
        return False

    _orig_execute = Git.execute
    if getattr(_orig_execute, "_pyddock_guarded", False):
        return True  # already patched

    _validate = build_git_command_validator(config, workspace_root, deny_messages)

    def _guarded_execute(self: Any, command: Any, *args: Any, **kwargs: Any) -> Any:
        _validate(command)
        return _orig_execute(self, command, *args, **kwargs)

    _guarded_execute._pyddock_guarded = True  # type: ignore[attr-defined]
    Git.execute = _guarded_execute
    return True
