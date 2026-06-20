"""Shared constants and utilities for pyddock runtime enforcement.

This is the leaf module in the dependency graph — it has NO imports from
other pyddock modules. All other _runtime split modules import from here.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types
from typing import Any

# Filename used for compile() when executing agent snippets.
# Defined here so the sandbox subprocess doesn't need to import the
# pyddock package (which would trigger importlib.metadata).
SNIPPET_FILENAME = "<snippet>"

# Resolved at module load time (before the import hook activates).
# Used by _caller_is_trusted to skip pyddock's own frames.
import os as _os_for_path
_PYDDOCK_DIR = _os_for_path.path.dirname(_os_for_path.path.abspath(__file__))
# Cache path helpers for _caller_is_trusted (avoids repeated attribute lookups).
_normcase = _os_for_path.path.normcase
_realpath = _os_for_path.path.realpath
_abspath = _os_for_path.path.abspath
_os_name = _os_for_path.name
del _os_for_path

# Module-level dict for storing original (unpatched) function references.
# Security-critical wrappers look up originals here instead of capturing them
# in closure cells. This prevents agent code from extracting originals via
# function.__closure__[N].cell_contents or descriptor-protocol introspection.
# Since pyddock._* modules are not importable by agent code, this dict is inaccessible.
_ORIGINALS: dict[str, Any] = {}


def has_ntfs_stream(path: Any) -> bool:
    """Detect an NTFS alternate-data-stream (ADS) reference in a path.

    On Windows, ``name:stream`` (and ``name::$DATA``, ``dir:stream``) addresses a
    *different* securable object than the path's lexical components imply: a write
    to ``.pyddock:pwned`` lands a data stream on the ``.pyddock`` directory, yet
    the lexical leaf ``.pyddock:pwned`` is not a child of ``.pyddock`` — so a
    ``relative_to('.pyddock')`` containment check decides "not protected" and the
    write slips through. ``realpath`` does not reliably normalize the stream away
    (a not-yet-existing stream resolves to the literal), so the only sound defense
    is to reject the syntax up front.

    A ``:`` is only legal as the drive designator at index 1 of the FIRST path
    component (``C:\\...`` or the drive-relative ``C:rest``). A ``:`` anywhere else
    is a stream reference. Returns False on non-Windows, where ``:`` is an
    ordinary, legal filename character.
    """
    if _os_name != "nt":
        return False
    parts = str(path).replace("/", "\\").split("\\")
    for idx, comp in enumerate(parts):
        if not comp:
            continue
        start = 0
        # First component may carry a 'X:' drive designator (X: or X:rest).
        if idx == 0 and len(comp) >= 2 and comp[0].isalpha() and comp[1] == ":":
            start = 2
        if ":" in comp[start:]:
            return True
    return False


def canonical_path(path: Any) -> pathlib.Path:
    """Canonicalize a filesystem path for security containment checks.

    Resolves the path with os.path.realpath, which (on Windows) expands NTFS
    8.3 short names — e.g. ``PYDDOC~1`` -> ``.pyddock`` — AND resolves symlinks,
    directory junctions, and subst drives. It falls back to abspath only if
    realpath raises.

    WHY THIS EXISTS (security-critical):
        Protected-directory checks compare ``candidate.relative_to(protected)``
        using a *lexically* normalized path. os.path.abspath() only canonicalizes
        ``.``/``..`` and is purely textual: it leaves OS-level aliases (8.3 short
        names, symlinks, junctions) intact. The OS, however, DOES resolve those
        aliases when the file is actually opened. That gap let agent code defeat
        the ``.pyddock/`` write protection by writing to ``PYDDOC~1/pwned.txt``:
        ``abspath`` kept ``PYDDOC~1`` so ``relative_to('.pyddock')`` raised
        ValueError (the check decided "not in .pyddock"), but ``open()`` resolved
        ``PYDDOC~1`` -> ``.pyddock`` and the write landed in the protected dir.

        realpath closes the gap by resolving the alias the same way the OS will.

    CONSISTENCY REQUIREMENT:
        realpath rewrites the path (drive-letter case, short-name expansion,
        symlink targets). Callers MUST canonicalize BOTH the protected roots and
        the candidate path with this function so a symlinked / subst-drive
        workspace still compares equal.

    The result is NOT guaranteed to exist; realpath resolves the existing prefix
    and appends any trailing non-existent components verbatim, so it is safe to
    use for write targets that have not been created yet.
    """
    s = str(path)
    try:
        return pathlib.Path(_realpath(s))
    except (OSError, ValueError):
        return pathlib.Path(_abspath(s))


def _find_deny_hint(attempted: str, deny_messages: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Return the first matching deny hint for the attempted action, or None.

    This is the subprocess-side equivalent of config.find_deny_hint().
    It operates on pre-compiled (pattern, message) tuples reconstructed
    from the serialized config dict.
    """
    for pattern, message in deny_messages:
        if pattern.search(attempted):
            return message
    return None


def _is_module_bound_builtin(val: Any) -> bool:
    """Return True if `val` is a C builtin bound to a module via `__self__`.

    Bound C builtins (e.g. os.getcwd, sys.getrecursionlimit, io.open_code) carry
    a reference to their defining module in `func.__self__`. Handing such a
    callable to agent code is a sandbox-escape vector: the real module exposes
    unpatched primitives (nt.open/write) or, for sys, sys.modules / meta_path /
    _getframe. This predicate identifies those callables so they can be wrapped.

    The check is gated on `types.BuiltinFunctionType` first. That matters for
    correctness AND safety: many libraries expose lazy-import shims with custom
    __getattr__ (e.g. polars' optional-dependency proxies), and probing
    `__self__` on those would trigger arbitrary imports / side effects. A real
    builtin's `__self__` is a C-level slot, so reading it is always safe and
    free of side effects.
    """
    if not isinstance(val, types.BuiltinFunctionType):
        return False
    return isinstance(getattr(val, "__self__", None), types.ModuleType)


def _wrap_safe_callable(func: Any) -> Any:
    """Return a thin Python wrapper around a module-bound builtin.

    The wrapper forwards calls but, being an ordinary function, exposes no
    `__self__` — closing the `func.__self__` -> real module leak. The original
    callable is captured in the wrapper's `__closure__`, which is already listed
    in ast.block_attributes (and screened by the getattr/attrgetter guards), so
    it can't be extracted that way either.

    NOTE: we deliberately do NOT use functools.wraps. wraps sets
    `wrapper.__wrapped__ = func`, which would re-open the exact
    `wrapper.__wrapped__.__self__` leak this wrapper exists to close.
    """
    def _safe_callable(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    _safe_callable.__name__ = getattr(func, "__name__", "wrapped")
    _safe_callable.__qualname__ = _safe_callable.__name__
    _safe_callable.__doc__ = getattr(func, "__doc__", None)
    # Mirror the original's module so introspection doesn't reveal the wrapper's
    # defining module (pyddock._base). Cosmetic — pyddock internals aren't
    # importable by agent code — but avoids a confusing __module__ on os.getcwd.
    _safe_callable.__module__ = getattr(func, "__module__", None)
    return _safe_callable
