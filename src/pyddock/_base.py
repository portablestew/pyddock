"""Shared constants and utilities for pyddock runtime enforcement.

This is the leaf module in the dependency graph — it has NO imports from
other pyddock modules. All other _runtime split modules import from here.
"""

from __future__ import annotations

import re
import sys
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
del _os_for_path

# Module-level dict for storing original (unpatched) function references.
# Security-critical wrappers look up originals here instead of capturing them
# in closure cells. This prevents agent code from extracting originals via
# function.__closure__[N].cell_contents or descriptor-protocol introspection.
# Since pyddock._* modules are not importable by agent code, this dict is inaccessible.
_ORIGINALS: dict[str, Any] = {}


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
