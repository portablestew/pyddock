"""Registry of per-library enforcement guards.

Most third-party libraries are constrained purely declaratively:
  - [imports]            — allow/deny the module
  - [restrictions.<mod>] — attribute/method-name filtering via proxies
  - [shell.<name>]       — command policy enforced by the subprocess proxy

A *library guard* is the rare escape hatch for a library that falls outside
those declarative tiers — typically because it shells out through an
execution path the subprocess proxy can't see, or builds commands internally
from a high-level API that name-level filtering can't inspect. A guard
contains library-specific knowledge (which internal chokepoint to hook, how
to parse its arguments) and therefore must be deliberately authored, tested,
and pinned to the library version it understands.

This registry keeps that wiring uniform: `apply_all` iterates the guards
instead of hardcoding each one, every guard self-gates on its import being
allowlisted, and adding a future guard is a single registry entry rather than
another special-case line in the enforcement pipeline.

Design intent: this list should stay SHORT. A library that needs a guard is a
maintenance liability (it depends on a third party's internals). If many
shell-out libraries start needing guards, that's the signal to invest in a
universal process-creation chokepoint (hooking _winapi.CreateProcess /
_posixsubprocess.fork_exec) rather than to grow this list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from pyddock._gitpython_patch import apply_gitpython_patch

# Signature of a guard's apply function: (config_dict, deny_messages) -> installed?
ApplyFn = Callable[[dict, "list[tuple[re.Pattern[str], str]]"], bool]


@dataclass(frozen=True)
class LibraryGuard:
    """A named, import-gated enforcement guard for one third-party library.

    Attributes:
        name: Stable identifier for the guard (for logging/diagnostics).
        import_name: The [imports] key that gates this guard. The guard only
            runs when that module is allowlisted.
        apply_fn: Installs the guard. Returns True if it was actually installed
            (e.g. the library is present), False if it was a no-op. Must be
            idempotent — apply_all may run within a single subprocess only once,
            but guards should tolerate re-application defensively.
    """

    name: str
    import_name: str
    apply_fn: ApplyFn

    def applies(self, config: dict) -> bool:
        allowed = config.get("imports", {}).get("allowed", [])
        return self.import_name in allowed

    def apply(
        self, config: dict, deny_messages: "list[tuple[re.Pattern[str], str]]"
    ) -> bool:
        return self.apply_fn(config, deny_messages)


# The registry. Keep this short and well-justified (see module docstring).
LIBRARY_GUARDS: list[LibraryGuard] = [
    LibraryGuard(
        name="gitpython",
        import_name="git",
        apply_fn=apply_gitpython_patch,
    ),
]


def apply_library_guards(
    config: dict,
    deny_messages: "list[tuple[re.Pattern[str], str]] | None" = None,
) -> list[str]:
    """Apply every registered guard whose gating import is allowlisted.

    Returns the names of guards that were actually installed (apply_fn returned
    True), so callers can log/inspect what enforcement is active.

    Guard exceptions are NOT swallowed. A guard that is supposed to constrain a
    library but fails to install would otherwise leave that library running
    *unguarded* (e.g. GitPython's subprocess bypass restored) — a silent
    security hole. Failing loud during bootstrap is the safer outcome: it's a
    visible error the operator will notice, not an invisible bypass.
    """
    deny_messages = deny_messages or []
    installed: list[str] = []
    for guard in LIBRARY_GUARDS:
        if not guard.applies(config):
            continue
        if guard.apply(config, deny_messages):
            installed.append(guard.name)
    return installed
