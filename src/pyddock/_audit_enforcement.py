"""Audit-hook backstop for filesystem enforcement.

`sys.addaudithook`-based enforcement that catches filesystem operations which
evade the name-based monkeypatches in `_fs_enforcement` — for example
constructing the genuine `_io.FileIO` obtained via
`type(sys.stdout.buffer.raw)(...)`, or low-level `os.open` / `os.replace`.

Audit events are raised from CPython's C layer, *beneath* the Python name
bindings, so they cannot be bypassed by re-deriving a real class from a live
object. That makes this the authoritative chokepoint; the monkeypatches remain
as a UX layer (early failure with actionable error messages).

Design (validated empirically against the live event stream):

* **Backstop parity.** The hook applies the *same* `_check_read`/`_check_write`
  decision functions used by the patched `open`/`FileIO`, so policy is identical
  whether an operation goes through the patched name or a raw bypass. The
  per-path trusted-library relaxation (`deny-agent` guards) already lives inside
  `_check_*`, so no separate caller tier is needed here.

* **Performance.** The very first action is a frozenset membership test on the
  event name; non-filesystem events return immediately. Hot events
  (`object.__getattr__`, `builtins.id`) fire thousands of times per run and must
  not pay for stack inspection.

* **Import-machinery exemption.** Operations whose nearest non-pyddock caller is
  `<frozen importlib...>` are allowed unconditionally. CPython's bytecode-cache
  writer uses `os.open` + `os.replace` from frozen import machinery to write
  `.pyc` files into the venv (which lives under the protected `.pyddock/`), and
  source reads originate there too; policing them would break imports. Agent
  code cannot forge a frozen-importlib frame.

* **Reentrancy guard.** `_check_*` canonicalizes paths via `os.path.realpath`,
  which can itself raise nested audit events (e.g. a Windows handle `open`); the
  guard prevents infinite recursion and is fail-open for the nested call (the
  outer call already enforces policy).

* **Failure mode.** `PermissionError` (a policy denial) propagates out of the
  hook and aborts the audited operation. Any *unexpected* internal error is
  swallowed so a bug here cannot brick all I/O — the monkeypatches remain the
  first line of defense — but it is surfaced once to stderr (deduped per
  event + exception type) so the failure is discoverable rather than silent.
"""
from __future__ import annotations

import sys
from typing import Any, Callable

# Captured at import time (before enforcement installs the sys proxy). This is
# the real C builtin regardless of the proxy, but we bind it explicitly so the
# hook never performs attribute lookups against a proxied module at call time.
_REAL_GETFRAME = sys._getframe

# --- audit events we police -------------------------------------------------
# See https://docs.python.org/3/library/audit-events.html
_OPEN_EVENT = "open"
# single-path mutations: args[0] is the target path
_WRITE_EVENTS_SINGLE = frozenset({
    "os.remove", "os.unlink", "os.mkdir", "os.rmdir",
    "os.chmod", "os.chown", "os.truncate",
})
# two-path mutations: args[0] (src) and args[1] (dst) are both targets.
# `os.replace` raises the "os.rename" event; hard/sym links raise their own.
_WRITE_EVENTS_PAIR = frozenset({
    "os.rename", "os.link", "os.symlink",
})
_POLICED = frozenset({_OPEN_EVENT}) | _WRITE_EVENTS_SINGLE | _WRITE_EVENTS_PAIR


def install_audit_enforcement(
    *,
    check_read: Callable[[Any], None],
    check_write: Callable[[Any], None],
    is_write_mode: Callable[[str], bool],
    pyddock_dir: str,
    real_os: Any,
) -> None:
    """Install the filesystem audit-hook backstop.

    Args:
        check_read / check_write: the same closures created by
            `apply_filesystem_scoping`; they raise `PermissionError` on denial.
        is_write_mode: predicate mapping a textual file mode to write-intent.
        pyddock_dir: absolute path of the pyddock package (frames here are
            skipped when locating the real caller).
        real_os: the genuine `os` module (for `O_*` flag constants and
            `fspath`), captured before the safe-os proxy is installed.
    """
    _pyddock_norm = pyddock_dir.replace("\\", "/").lower()
    _fsencoding = sys.getfilesystemencoding()
    _fspath = getattr(real_os, "fspath", None)

    # Write-intent mask for os.open (whose audit event carries flags, not a
    # textual mode). O_RDONLY is 0, so any of these bits implies a write.
    _write_mask = 0
    for _flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"):
        _write_mask |= getattr(real_os, _flag, 0)

    _state = {"reentry": False, "errors": 0}
    # Distinct (event, exception-type) pairs already warned about — dedup so a
    # recurring internal bug surfaces once instead of flooding stderr.
    _warned: set[str] = set()

    def _warn_internal_error(event: str, exc: BaseException) -> None:
        """Surface an unexpected hook bug to stderr, reentry-safe and last-resort.

        Writes directly to fd 2 via the real os.write (no audit event, bypasses
        the patched open/FileIO), and never raises — a failure here must not
        abort the audited operation we already decided to allow.
        """
        _state["errors"] += 1
        key = f"{event}:{type(exc).__name__}"
        if key in _warned:
            return
        _warned.add(key)
        try:
            msg = (
                f"pyddock: WARNING: audit backstop swallowed an internal error on "
                f"event {event!r}: {type(exc).__name__}: {exc} — operation was "
                f"ALLOWED (the _fs_enforcement monkeypatch layer is still active). "
                f"This is a bug in the audit hook; please report it.\n"
            )
            real_os.write(2, msg.encode("utf-8", "replace"))
        except Exception:
            pass  # last resort: the warning path must never propagate

    def _extract_path(p: Any) -> str | None:
        """Normalize an audit-event path argument to a str, or None to skip."""
        if p is None or isinstance(p, int):
            return None  # int = already-open fd; nothing to check
        if isinstance(p, bytes):
            try:
                return p.decode(_fsencoding, "surrogateescape")
            except Exception:
                return None
        if isinstance(p, str):
            return p
        if _fspath is not None:
            try:
                fp = _fspath(p)
                return fp.decode(_fsencoding, "surrogateescape") if isinstance(fp, bytes) else fp
            except Exception:
                return None
        return None

    def _caller_is_import_machinery() -> bool:
        """True if the nearest non-pyddock caller is frozen import machinery."""
        frame = _REAL_GETFRAME(1)  # caller = the hook; walk outward
        while frame is not None:
            fn = frame.f_code.co_filename
            if fn.replace("\\", "/").lower().startswith(_pyddock_norm):
                frame = frame.f_back
                continue
            return fn.startswith("<frozen importlib")
        return False

    def _handle_open(args: tuple) -> None:
        path = _extract_path(args[0] if args else None)
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        if isinstance(mode, str):
            write = is_write_mode(mode)
        else:
            # os.open path: determine intent from flags (args[2]).
            flags = args[2] if len(args) > 2 else None
            if isinstance(flags, int):
                write = bool(flags & _write_mask)
            else:
                write = True  # indeterminate -> conservative (treat as write)
        if write:
            check_write(path)
        else:
            check_read(path)

    def _hook(event: str, args: tuple) -> None:
        # Hot path: cheap membership test, return for everything we don't police.
        if event not in _POLICED:
            return
        if _state["reentry"]:
            return
        _state["reentry"] = True
        try:
            # Let CPython's import machinery read source and write .pyc freely.
            if _caller_is_import_machinery():
                return
            if event == _OPEN_EVENT:
                _handle_open(args)
            elif event in _WRITE_EVENTS_SINGLE:
                p = _extract_path(args[0] if args else None)
                if p is not None:
                    check_write(p)
            else:  # _WRITE_EVENTS_PAIR
                for raw in args[:2]:
                    p = _extract_path(raw)
                    if p is not None:
                        check_write(p)
        except PermissionError:
            raise  # policy denial -> abort the audited operation
        except Exception as exc:
            # Unexpected INTERNAL error (not a policy denial). Fail open so a bug
            # here can't abort every audited operation and brick all I/O — but
            # make it discoverable instead of silent.
            _warn_internal_error(event, exc)
        finally:
            _state["reentry"] = False

    sys.addaudithook(_hook)
