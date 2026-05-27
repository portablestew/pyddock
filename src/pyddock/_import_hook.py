"""Import hook enforcement for pyddock.

Provides the import blocking mechanism that runs inside the subprocess.
Blocks non-allowlisted imports unless the import originates from trusted
code (workspace modules, site-packages, stdlib).
"""

from __future__ import annotations

import sys
from typing import Any

from pyddock._base import (
    SNIPPET_FILENAME,
    _PYDDOCK_DIR,
    _find_deny_hint,
    _normcase,
    _realpath,
)


def _is_infra_frame(filename: str) -> bool:
    """Return True if the frame belongs to import infrastructure or pyddock itself.

    These frames are skipped during stack inspection — they're not
    considered "the real caller" of an import.
    """
    return (
        filename.startswith("<frozen")
        or filename.startswith(_PYDDOCK_DIR)
    )


def _caller_is_trusted(trusted_prefixes: tuple[str, ...]) -> bool:
    """Check if the import call originates from trusted code.

    Walks the entire call stack. If any frame between the import guard and
    the agent snippet belongs to trusted code (workspace module or
    site-packages), the import is allowed. If the agent snippet is reached
    with no trusted frame found, the import is blocked.

    Infrastructure frames (frozen importlib, pyddock enforcement) are skipped.
    All other frames are checked against trusted_prefixes or SNIPPET_FILENAME.
    """
    frame = sys._getframe(1)
    found_trusted = False
    while frame is not None:
        filename = frame.f_code.co_filename
        # Skip import machinery and pyddock enforcement frames
        if _is_infra_frame(filename):
            frame = frame.f_back
            continue
        # Agent snippet code — verdict based on whether we found trusted code
        if filename == SNIPPET_FILENAME:
            return found_trusted
        # Resolve symlinks/subst drives and normalize case so that trusted
        # prefix comparison works regardless of path indirection.
        resolved = _normcase(_realpath(filename))
        # Any frame in a trusted path means this import chain was initiated
        # by trusted code (even if it went through stdlib or other libs).
        if resolved.startswith(trusted_prefixes):
            found_trusted = True
        frame = frame.f_back
    return found_trusted


class _ImportBlocker:
    """Custom import hook that blocks modules not in the allowlist.

    Installed via sys.meta_path. Raises ImportError with a helpful message
    listing allowed imports when a disallowed module is imported.

    Uses the modern MetaPathFinder interface (find_spec) which is the
    standard mechanism in Python 3.4+.

    The hook uses a pure allowlist: only top-level modules explicitly listed
    in the allowlist are permitted — unless the import originates from a
    trusted path (workspace module or its transitive dependency in site-packages).
    """

    def __init__(
        self,
        allowed: list[str],
        trusted_prefixes: list[str],
        deny_messages: list[tuple[Any, str]] | None = None,
    ) -> None:
        self._allowed = set(allowed)
        self._trusted_prefixes = tuple(trusted_prefixes)
        self._deny_messages = deny_messages or []

    def find_module(self, fullname: str, path: Any = None) -> "_ImportBlocker | None":
        """Legacy import hook interface for compatibility."""
        if self._should_block(fullname):
            return self
        return None

    def load_module(self, fullname: str) -> None:
        """Raise ImportError for blocked modules (legacy interface)."""
        allowed_list = ", ".join(sorted(self._allowed))
        msg = (
            f"ImportError: '{fullname}' is not an allowed import. "
            f"Please use one of the following allowed imports instead: {allowed_list}"
        )
        hint = _find_deny_hint(fullname, self._deny_messages)
        if hint:
            msg += f"\n[{hint}]"
        raise ImportError(msg)

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> None:
        """Modern import hook interface (Python 3.4+).

        Returns None for allowed imports (letting other finders handle them).
        Raises ImportError directly for blocked imports.
        """
        if self._should_block(fullname):
            allowed_list = ", ".join(sorted(self._allowed))
            msg = (
                f"ImportError: '{fullname}' is not an allowed import. "
                f"Please use one of the following allowed imports instead: "
                f"{allowed_list}"
            )
            hint = _find_deny_hint(fullname, self._deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            raise ImportError(msg)
        return None

    def _should_block(self, fullname: str) -> bool:
        top_level = fullname.split(".")[0]
        if top_level in self._allowed:
            return False
        # Check if the import originates from trusted code (workspace module
        # or its transitive dependency in site-packages).
        if self._trusted_prefixes and _caller_is_trusted(self._trusted_prefixes):
            return False
        return True
