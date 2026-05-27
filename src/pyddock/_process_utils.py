"""Shared process utilities for pyddock executors.

Contains helpers used by both executor.py (run_python) and shell_executor.py (run_shell)
to avoid circular imports.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows registry environment snapshot
# ---------------------------------------------------------------------------
# Launchers like `uv run` construct a minimal environment that omits most
# machine-level and user-level registry variables (JAVA_HOME, P4CLIENT, etc.).
# We snapshot the registry once at import time and merge missing vars into
# child environments so tools behave as if launched from a normal shell.
#
# PATH is handled specially: the registry Machine + User paths are merged
# (machine first, user appended) and used as a fallback only if PATH is
# missing entirely — we never override an existing PATH since the launcher
# may have prepended venv/tool paths we need to preserve.
# ---------------------------------------------------------------------------

_REGISTRY_SNAPSHOT: dict[str, str] | None = None
_REGISTRY_PATH: str | None = None  # merged Machine;User PATH from registry


def _read_registry_env() -> tuple[dict[str, str], str | None]:
    """Read Machine + User environment variables from the Windows registry.

    Returns:
        (env_dict, merged_path) where env_dict contains all non-PATH vars
        (user overrides machine) and merged_path is the combined PATH string
        (machine paths ; user paths), or None if neither has PATH.
    """
    import winreg

    def _read_key(root: int, subkey: str) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            with winreg.OpenKey(root, subkey) as key:
                i = 0
                while True:
                    try:
                        name, value, vtype = winreg.EnumValue(key, i)
                        # Expand REG_EXPAND_SZ values (e.g. %SystemRoot%\Temp)
                        if vtype == winreg.REG_EXPAND_SZ:
                            value = winreg.ExpandEnvironmentStrings(value)
                        result[name] = str(value)
                        i += 1
                    except OSError:
                        break
        except OSError:
            pass
        return result

    machine = _read_key(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
    )
    user = _read_key(
        winreg.HKEY_CURRENT_USER,
        r"Environment",
    )

    # Merge PATH separately: machine paths first, then user paths appended
    machine_path = machine.pop("Path", machine.pop("PATH", None))
    user_path = user.pop("Path", user.pop("PATH", None))

    merged_path: str | None = None
    if machine_path and user_path:
        merged_path = machine_path.rstrip(";") + ";" + user_path.rstrip(";")
    elif machine_path:
        merged_path = machine_path
    elif user_path:
        merged_path = user_path

    # Merge remaining vars: user overrides machine
    env = {}
    env.update(machine)
    env.update(user)

    return env, merged_path


def _ensure_registry_snapshot() -> None:
    """Lazily populate the registry snapshot (called once on first use)."""
    global _REGISTRY_SNAPSHOT, _REGISTRY_PATH
    if _REGISTRY_SNAPSHOT is not None:
        return
    if os.name != "nt":
        _REGISTRY_SNAPSHOT = {}
        _REGISTRY_PATH = None
        return
    try:
        _REGISTRY_SNAPSHOT, _REGISTRY_PATH = _read_registry_env()
        logger.debug(
            "Registry env snapshot: %d vars, PATH %s",
            len(_REGISTRY_SNAPSHOT),
            "present" if _REGISTRY_PATH else "absent",
        )
    except Exception as exc:
        logger.warning("Failed to read registry environment: %s", exc)
        _REGISTRY_SNAPSHOT = {}
        _REGISTRY_PATH = None


def make_child_env() -> dict[str, str]:
    """Build an environment dict for child processes.

    Merges the current process environment with the Windows registry snapshot
    (registry vars fill gaps only — never override what's already set).
    Removes pyddock internals that shouldn't leak into child processes.
    """
    _ensure_registry_snapshot()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    if os.name == "nt":
        # Backfill from registry snapshot — only vars not already present
        assert _REGISTRY_SNAPSHOT is not None
        for key, value in _REGISTRY_SNAPSHOT.items():
            if key not in env:
                env[key] = value

        # Ensure SystemRoot is present under both casings — some tools expect
        # "SystemRoot" while others look for "SYSTEMROOT".
        _sysroot = (
            env.get("SystemRoot")
            or env.get("SYSTEMROOT")
            or env.get("windir")
            or r"C:\Windows"
        )
        env.setdefault("SystemRoot", _sysroot)
        env.setdefault("SYSTEMROOT", _sysroot)

        # PATH: don't override (launcher may have prepended venv paths),
        # but if completely missing, use registry merged path
        if "PATH" not in env and "Path" not in env and _REGISTRY_PATH:
            env["Path"] = _REGISTRY_PATH

    env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    env.pop("PYTHONPATH", None)
    return env


def get_startupinfo() -> subprocess.STARTUPINFO:
    """Create STARTUPINFO that hides the process window (Windows only)."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


# 64 KB ≈ 16K tokens — enough for useful output without dominating agent context
MAX_OUTPUT_BYTES = 65_536


def truncate_output(text: str, label: str = "output", limit: int = MAX_OUTPUT_BYTES) -> str:
    """Truncate text to *limit* characters, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[truncated: {label} exceeded 64 KB. "
        f"For large results, write to a file instead of printing.]"
    )


def kill_and_drain(proc: subprocess.Popen) -> tuple[str, str]:
    """Kill a timed-out process tree and return its buffered stdout/stderr (truncated).

    Performs a forceful kill (taskkill on Windows, SIGKILL on Unix), then drains
    whatever output was already buffered in the OS pipes before the process died.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    # Drain any partial output buffered in the pipes
    stdout_bytes, stderr_bytes = proc.communicate()
    stdout = truncate_output(
        stdout_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n"), "output"
    )
    stderr = truncate_output(
        stderr_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n"), "stderr"
    )
    return stdout, stderr
