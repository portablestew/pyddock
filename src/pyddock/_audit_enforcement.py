"""Audit-event policy engine (`sys.addaudithook`).

A single audit hook dispatches every audited event through a configurable
**disposition table** (`[audit]` in config). Audit events fire from CPython's C
layer, *beneath* the Python name bindings, so they cannot be bypassed by
re-deriving a real primitive from a live object (the `_io.FileIO` /
`type(sys.stdout.buffer.raw)` class of escape). The name-based monkeypatches in
`_fs_enforcement` remain a UX layer (early failure, good messages); this hook is
the authoritative chokepoint for everything that raises an audit event.

Dispositions
------------
* ``fs``            — `open`: route to `_check_read`/`_check_write` by mode/flags.
* ``fs-write``      — single-path mutation (`os.remove`, `os.mkdir`, ...).
* ``fs-write-pair`` — two-path mutation (`os.rename`/`replace`, `os.link`, ...).
* ``agent-deny``    — deny iff the originating caller is AGENT code; allow when a
                      trusted library / stdlib / import-machinery frame is on the
                      stack. For primitives with no sanctioned agent use
                      (`ctypes.*`, `marshal.loads`, `code.__new__`, `sys.settrace`,
                      ...). This is defense-in-depth: the import allowlist already
                      stops agents importing these, but `agent-deny` also closes a
                      future object-graph leak that *reaches* one of them.
* ``network``       — network egress: same caller-scoped deny as ``agent-deny``
                      (agents have no socket path; trusted libraries are allowed).
                      Used for `socket.connect`/`bind`/`sendto`/`sendmsg` and
                      `urllib.Request`. Host/port scoping is a future refinement.
* ``observe``       — log only (when ``--debug``); no enforcement.
* ``allow``         — explicit, silent no-op: no enforcement and no debug log
                      (overrides a default-deny). Distinct from ``observe``,
                      which logs under ``--debug``.

Shared skeleton (all enforcing dispositions)
--------------------------------------------
reentrancy guard → import-machinery exemption → (path check | caller classify) →
optional debug log. Notes:

* **Performance.** First action is a table lookup on the event name; events not
  in the table return immediately. Hot events (`object.__getattr__`,
  `builtins.id`) are simply absent from the table.
* **Import-machinery exemption first.** `.pyc` writes/reads and bytecode
  `marshal.loads` come from `<frozen importlib...>`; they must be exempt or
  imports break. Agent code cannot forge a frozen-importlib frame.
* **Reentrancy guard.** `_check_*` canonicalizes paths (`realpath`), which can
  raise nested audit events; the guard prevents recursion.
* **Failure mode.** `PermissionError` (policy denial) propagates and aborts the
  operation. Unexpected internal errors are swallowed (fail open — the
  monkeypatches remain the first line) but warned once to stderr.

Debug logging (`pyddock serve --debug`)
---------------------------------------
When enabled, every table hit is recorded as JSONL (event, decision, caller
class, detail) — the inventory/observability trail and the design-time tool for
future capabilities. Writes go to a pre-opened fd via `real_os.write` (no audit
event, cannot recurse) and never raise.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

from pyddock._base import SNIPPET_FILENAME

# Real C builtin, captured before the sys proxy is installed.
_REAL_GETFRAME = sys._getframe

# Disposition vocabulary.
_FS_DISPOSITIONS = frozenset({"fs", "fs-write", "fs-write-pair"})
_DENY_DISPOSITIONS = frozenset({"agent-deny", "network"})
_ENFORCING = _FS_DISPOSITIONS | _DENY_DISPOSITIONS
VALID_DISPOSITIONS = _ENFORCING | frozenset({"observe", "allow"})


def install_audit_enforcement(
    *,
    check_read: Callable[[Any], None],
    check_write: Callable[[Any], None],
    is_write_mode: Callable[[str], bool],
    pyddock_dir: str,
    real_os: Any,
    audit_rules: list[tuple[str, str]] | None = None,
    trusted_prefixes: tuple[str, ...] = (),
    debug: bool = False,
    log_path: str | None = None,
) -> None:
    """Install the audit-event policy engine.

    Args:
        check_read / check_write: closures from `apply_filesystem_scoping`;
            raise `PermissionError` on denial.
        is_write_mode: predicate mapping a textual file mode to write-intent.
        pyddock_dir: absolute pyddock package path (frames here are skipped).
        real_os: genuine `os` module (flags, getpid, write, realpath).
        audit_rules: ordered (event-pattern, disposition) list (from `[audit]`
            config — the single source of truth). A pattern ending in ``*`` is a
            prefix match (e.g. ``ctypes.*``). When empty and ``debug`` is off, no
            hook is installed.
        trusted_prefixes: normalized trusted path prefixes (caller classification).
        debug: enable the JSONL observability trail.
        log_path: destination for the debug trail.
    """
    rules = audit_rules or []
    if not rules and not debug:
        return  # nothing to enforce or log — don't install a no-op hook

    # Split rules into exact lookups and ordered wildcard (prefix) matches.
    _exact: dict[str, str] = {}
    _wild: list[tuple[str, str]] = []
    for pattern, disp in rules:
        if pattern.endswith("*"):
            _wild.append((pattern[:-1], disp))  # keep trailing '.', drop '*'
        else:
            _exact[pattern] = disp

    def _resolve(event: str) -> str | None:
        disp = _exact.get(event)
        if disp is not None:
            return disp
        for prefix, d in _wild:
            if event.startswith(prefix):
                return d
        return None

    _pyddock_norm = pyddock_dir.replace("\\", "/").lower()
    _fsencoding = sys.getfilesystemencoding()
    _fspath = getattr(real_os, "fspath", None)
    _trusted_prefixes = tuple(trusted_prefixes)
    _realpath = real_os.path.realpath
    _normcase = real_os.path.normcase

    _write_mask = 0
    for _flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"):
        _write_mask |= getattr(real_os, _flag, 0)

    _state = {"reentry": False, "errors": 0}
    _warned: set[str] = set()

    # --- optional debug log sink (opened before the hook → never self-logs) ---
    _log_fd = -1
    _debug_log = False
    if debug and log_path:
        try:
            real_os.makedirs(real_os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass
        try:
            _flags = real_os.O_CREAT | real_os.O_WRONLY | real_os.O_APPEND
            _log_fd = real_os.open(log_path, _flags, 0o600)
            _debug_log = True
        except Exception:
            _debug_log = False
    _pid = real_os.getpid()

    def _warn_internal_error(event: str, exc: BaseException) -> None:
        _state["errors"] += 1
        key = f"{event}:{type(exc).__name__}"
        if key in _warned:
            return
        _warned.add(key)
        try:
            msg = (
                f"pyddock: WARNING: audit hook swallowed an internal error on "
                f"event {event!r}: {type(exc).__name__}: {exc} — operation was "
                f"ALLOWED (the _fs_enforcement monkeypatch layer is still active). "
                f"This is a bug in the audit hook; please report it.\n"
            )
            real_os.write(2, msg.encode("utf-8", "replace"))
        except Exception:
            pass

    def _extract_path(p: Any) -> str | None:
        if p is None or isinstance(p, int):
            return None
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
        frame = _REAL_GETFRAME(1)
        while frame is not None:
            fn = frame.f_code.co_filename
            if fn.replace("\\", "/").lower().startswith(_pyddock_norm):
                frame = frame.f_back
                continue
            return fn.startswith("<frozen importlib")
        return False

    def _classify_caller() -> str:
        """AGENT | TRUSTED | IMPORT | OTHER, mirroring `_caller_is_trusted`."""
        frame = _REAL_GETFRAME(1)
        found_trusted = False
        saw_import = False
        while frame is not None:
            fn = frame.f_code.co_filename
            nfn = fn.replace("\\", "/").lower()
            if nfn.startswith(_pyddock_norm):
                frame = frame.f_back
                continue
            if fn.startswith("<frozen importlib"):
                saw_import = True
                frame = frame.f_back
                continue
            if fn == SNIPPET_FILENAME:
                return "TRUSTED" if found_trusted else "AGENT"
            if fn.startswith("<frozen"):
                found_trusted = True
                frame = frame.f_back
                continue
            try:
                resolved = _normcase(_realpath(fn))
            except Exception:
                resolved = nfn
            if _trusted_prefixes and resolved.startswith(_trusted_prefixes):
                found_trusted = True
            frame = frame.f_back
        if found_trusted:
            return "TRUSTED"
        if saw_import:
            return "IMPORT"
        return "OTHER"

    def _summarize(event: str, args: tuple) -> str:
        try:
            if event == "open":
                return f"{_extract_path(args[0] if args else None)} mode={args[1] if len(args) > 1 else None!r}"
            if event.startswith("socket."):
                return f"addr={args[1]!r}" if len(args) > 1 else repr(args)
            if event == "subprocess.Popen":
                return f"exe={args[0]!r} args={args[1] if len(args) > 1 else None!r}"
            if event == "os.system":
                return f"cmd={args[0]!r}" if args else repr(args)
            return " | ".join(str(a) for a in args)[:200]
        except Exception:
            return "<unsummarizable>"

    def _log_event(event: str, args: tuple, decision: str, bucket: str | None = None) -> None:
        try:
            rec = {
                "t": round(time.time(), 6),
                "pid": _pid,
                "event": event,
                "decision": decision,
                "caller": bucket if bucket is not None else _classify_caller(),
                "detail": _summarize(event, args)[:300],
            }
            real_os.write(_log_fd, (json.dumps(rec, default=str) + "\n").encode("utf-8", "replace"))
        except Exception:
            pass

    def _enforce_fs(disp: str, event: str, args: tuple) -> None:
        if disp == "fs":  # open
            path = _extract_path(args[0] if args else None)
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            if isinstance(mode, str):
                write = is_write_mode(mode)
            else:
                flags = args[2] if len(args) > 2 else None
                write = bool(flags & _write_mask) if isinstance(flags, int) else True
            (check_write if write else check_read)(path)
        elif disp == "fs-write":
            p = _extract_path(args[0] if args else None)
            if p is not None:
                check_write(p)
        else:  # fs-write-pair
            for raw in args[:2]:
                p = _extract_path(raw)
                if p is not None:
                    check_write(p)

    def _deny_message(event: str, disp: str) -> str:
        if disp == "network":
            return (
                f"PermissionError: network access via '{event}' is not permitted "
                f"from agent code (audit policy: network)."
            )
        return (
            f"PermissionError: '{event}' is not permitted from agent code "
            f"(audit policy: agent-deny). This primitive has no sanctioned use "
            f"inside sandboxed snippets."
        )

    def _hook(event: str, args: tuple) -> None:
        disp = _resolve(event)
        # `allow` is a silent, explicit no-op (overrides a default-deny): no
        # enforcement and — unlike `observe` — no debug logging either.
        if disp is None or disp == "allow":
            return
        enforcing = disp in _ENFORCING
        if not enforcing and not _debug_log:
            return  # observe with debug off → nothing to do
        if _state["reentry"]:
            return
        _state["reentry"] = True
        try:
            decision = "observe"
            if enforcing:
                if _caller_is_import_machinery():
                    decision = "import-exempt"
                elif disp in _FS_DISPOSITIONS:
                    try:
                        _enforce_fs(disp, event, args)
                    except PermissionError:
                        if _debug_log:
                            _log_event(event, args, "deny")
                        raise
                    decision = "allow"
                else:  # agent-deny / network
                    bucket = _classify_caller()
                    if bucket == "AGENT":
                        if _debug_log:
                            _log_event(event, args, "deny", bucket)
                        raise PermissionError(_deny_message(event, disp))
                    if _debug_log:
                        _log_event(event, args, "allow", bucket)
                    return
            if _debug_log:
                _log_event(event, args, decision)
        except PermissionError:
            raise
        except Exception as exc:
            _warn_internal_error(event, exc)
        finally:
            _state["reentry"] = False

    sys.addaudithook(_hook)
