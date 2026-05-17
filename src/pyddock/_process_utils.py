"""Shared process utilities for pyddock executors.

Contains helpers used by both executor.py (run_python) and shell_executor.py (run_shell)
to avoid circular imports.
"""

from __future__ import annotations

import os
import subprocess


def make_child_env() -> dict[str, str]:
    """Build an environment dict for child processes.

    Starts from os.environ, ensures essential Windows variables are present
    (some launchers like uv run strip them), and removes pyddock internals
    that shouldn't leak into child processes.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt":
        _temp_dir = os.environ.get("TEMP", os.environ.get("TMP", r"C:\Windows\Temp"))
        _win_defaults = {
            "PATHEXT": ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.CPL",
            "SystemRoot": r"C:\Windows",
            "SYSTEMROOT": r"C:\Windows",
            "TMP": _temp_dir,
            "TEMP": _temp_dir,
            "COMSPEC": r"C:\Windows\system32\cmd.exe",
            "NUMBER_OF_PROCESSORS": str(os.cpu_count() or 1),
            "OS": "Windows_NT",
        }
        for key, default in _win_defaults.items():
            if key not in env:
                env[key] = default
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
