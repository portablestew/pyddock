"""Pre-warm lazy stdlib imports before the import hook activates.

Some stdlib modules are frozen in Python 3.12+ and lazily import internal
modules on first use (e.g. datetime imports _strptime on first strptime()
call). Once the import hook is active, these lazy imports fail because the
frozen frame chain confuses the trusted-caller check.

This module pre-warms those lazy imports during bootstrap (before the hook
is installed) and returns the set of internal module names that were
successfully cached in sys.modules. The _guarded_import function allows
re-import of these specific modules from cache without re-validating.

To add a new warmup:
    @_register("description", ["internal_module_name"])
    def _warm_something():
        # trigger the lazy import
        ...
"""

from __future__ import annotations

from typing import Callable

# Each entry: (description, warmup callable, internal modules it caches)
_WARMUPS: list[tuple[str, Callable[[], None], list[str]]] = []


def _register(description: str, internals: list[str]) -> Callable:
    """Decorator to register a warmup function."""
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _WARMUPS.append((description, fn, internals))
        return fn
    return decorator


@_register("datetime.strptime → _strptime", ["_strptime"])
def _warm_strptime() -> None:
    import datetime
    datetime.datetime.strptime("2000", "%Y")


@_register("codecs → encodings.*", [])
def _warm_codecs() -> None:
    import codecs
    for name in ("idna", "utf-8", "ascii", "latin-1", "cp1252", "utf-16-le"):
        try:
            codecs.lookup(name)
        except LookupError:
            pass


def run_all() -> frozenset[str]:
    """Execute all warmups. Returns the set of pre-warmed internal module names.

    These module names are safe to allow through the _guarded_import cache
    bypass because:
    1. They were loaded before the import hook was active (trusted)
    2. They are stdlib internals with no I/O capabilities of their own
    3. They cannot be imported directly by agent code (only via the
       pre-warmed_internals bypass when triggered by frozen stdlib)
    """
    prewarmed: set[str] = set()
    for _desc, fn, internals in _WARMUPS:
        try:
            fn()
            prewarmed.update(internals)
        except Exception:
            pass
    return frozenset(prewarmed)
