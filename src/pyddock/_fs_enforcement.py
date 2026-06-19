"""Filesystem scoping enforcement.

This module contains the extracted filesystem scoping logic that patches
filesystem operations (builtins.open, pathlib.Path methods, io.open, etc.)
to enforce path restrictions within the sandbox.
"""
from __future__ import annotations

import builtins
import pathlib
import re
import sys
import tempfile as _tempfile_module
import _io as _cio_module
from typing import Any

from pyddock._base import _ORIGINALS, _PYDDOCK_DIR, _find_deny_hint, canonical_path
from pyddock._import_hook import _caller_is_trusted
from pyddock._audit_enforcement import install_audit_enforcement
from pyddock._process_utils import make_child_env


def apply_filesystem_scoping(
    config: dict,
    workspace_root: pathlib.Path,
    real_os: Any,
    trusted_prefixes: tuple[str, ...],
    io_module: Any,
    debug: bool = False,
) -> None:
    """Patch filesystem operations to enforce path restrictions.

    Patches:
    - builtins.open (read and write modes)
    - pathlib.Path.write_text, write_bytes, open (write modes)
    - pathlib.Path.read_text, read_bytes
    """
    fs_config = config.get("filesystem", {})
    writable_paths = fs_config.get("writable_paths", ["."])
    readable_paths = fs_config.get("readable_paths", ["."])
    _real_os = real_os
    _NULL_DEVICE = real_os.devnull

    def _abspath(p: pathlib.Path) -> pathlib.Path:
        """Canonicalize a path for containment checks.

        Uses realpath (via canonical_path) so that OS-level path aliases —
        Windows 8.3 short names (PYDDOC~1 -> .pyddock), symlinks, junctions,
        and subst drives — are resolved the same way the OS resolves them at
        open() time. abspath alone is purely lexical and would leave such
        aliases intact, allowing the protected-directory checks below to be
        bypassed (see canonical_path docstring). Applied consistently to BOTH
        the protected roots and every candidate path.
        """
        return canonical_path(p)

    # Canonicalize allowed paths (realpath: resolves symlinks/junctions/8.3
    # names) so they compare consistently with canonicalized candidate paths.
    resolved_writable = [
        _abspath(workspace_root / p) for p in writable_paths
    ]

    # "*" means unrestricted reads
    unrestricted_reads = "*" in readable_paths
    resolved_readable = [
        _abspath(workspace_root / p) for p in readable_paths if p != "*"
    ]

    # .pyddock/ is always excluded from writes (self-modification protection)
    pyddock_dir = _abspath(workspace_root / ".pyddock")

    # pyddock's own source directory (enforcement code), canonicalized the same
    # way as candidate paths so the relative_to() check below cannot be bypassed
    # via 8.3 short names / symlinks on the install path.
    pyddock_src_dir = canonical_path(_PYDDOCK_DIR)

    # Resolve workspace module directories (write-protected)
    workspace_imports = config.get("imports", {}).get("workspace", {})
    workspace_module_dirs: list[pathlib.Path] = [
        _abspath(workspace_root / rel_path)
        for rel_path in workspace_imports.values()
    ]

    # Compute stdlib Lib directory path for write protection.
    # Uses sys.base_prefix (same logic as _build_trusted_prefixes) to handle
    # virtualenvs correctly — it points to the original Python installation.
    _stdlib_lib_candidate = _real_os.path.join(sys.base_prefix, "Lib")
    if not _real_os.path.isdir(_stdlib_lib_candidate):
        # Unix layout: lib/pythonX.Y/
        _stdlib_lib_candidate = _real_os.path.join(
            sys.base_prefix, "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}"
        )
    _stdlib_lib_path = pathlib.Path(_real_os.path.realpath(_stdlib_lib_candidate))

    # Derive write-protected paths from shell policies
    shell_config = config.get("shell", {})
    shell_protected_dirs: list[pathlib.Path] = []
    for _name, policy in shell_config.items():
        cmd_regex = policy.get("command", "")
        # Heuristic: regex is path-like if it contains path separators or starts with \.
        if "/" in cmd_regex or "\\\\" in cmd_regex or cmd_regex.startswith("\\."):
            path_pattern = cmd_regex.lstrip("^").rstrip("$")
            if "/" in path_pattern:
                dir_part = path_pattern.rsplit("/", 1)[0]
            elif "\\\\" in path_pattern:
                dir_part = path_pattern.rsplit("\\\\", 1)[0]
            else:
                dir_part = path_pattern
            if dir_part:
                # Clean up regex escapes to get a usable filesystem path
                # Common regex escapes: \. → ., \/ → /, \\ → \
                clean_dir = dir_part.replace("\\.", ".").replace("\\/", "/")
                # Remove any remaining regex metacharacters that aren't path chars
                # (keep alphanumeric, /, \, ., -, _)
                shell_protected_dirs.append(
                    _abspath(workspace_root / clean_dir)
                )

    # Compile filesystem guards (regex → disposition, first match wins).
    # Guards apply to BOTH reads and writes.
    guards_config = fs_config.get("guards", [])
    compiled_guards: list[tuple[re.Pattern[str], str]] = []
    for guard in guards_config:
        pattern_str = guard.get("pattern", "") if isinstance(guard, dict) else ""
        disposition = guard.get("disposition", "deny") if isinstance(guard, dict) else "deny"
        try:
            compiled_guards.append((re.compile(pattern_str), disposition))
        except re.error:
            pass  # Skip invalid patterns silently (logged at config load time)

    _workspace_root_abs = _abspath(workspace_root)
    _guard_trusted_prefixes = trusted_prefixes

    def _check_guard(path: pathlib.Path, operation: str) -> bool | None:
        """Check path against filesystem guards. Returns:
        - True if access is explicitly allowed
        - False (raises) if access is denied
        - None if no guard matched (fall through to normal logic)

        Args:
            path: The resolved absolute path to check.
            operation: "read" or "write" (for error messages).
        """
        # Normalize to forward slashes for cross-platform regex matching
        path_str = str(path).replace("\\", "/")

        for pattern, disposition in compiled_guards:
            if pattern.search(path_str):
                if disposition == "deny-agent":
                    # "deny-agent" blocks agent code but allows trusted libraries
                    # (site-packages, workspace modules) to access the path.
                    if _guard_trusted_prefixes and _caller_is_trusted(_guard_trusted_prefixes):
                        return True  # trusted library — allow
                    raise PermissionError(
                        f"PermissionError: Cannot {operation} '{path}' — "
                        f"path matches a filesystem guard (pattern: "
                        f"'{pattern.pattern}', disposition: deny-agent). "
                        f"This path is blocked for security."
                    )
                elif disposition == "deny-all":
                    # "deny-all" blocks everyone unconditionally — no
                    # trusted caller bypass. Use for paths where no
                    # legitimate library reader exists in this sandbox.
                    raise PermissionError(
                        f"PermissionError: Cannot {operation} '{path}' — "
                        f"path matches a filesystem guard (pattern: "
                        f"'{pattern.pattern}', disposition: deny-all). "
                        f"This path is blocked for security."
                    )
                elif disposition == "read-only":
                    # "read-only" allows reads but blocks writes.
                    if operation == "read":
                        return True  # reads permitted
                    raise PermissionError(
                        f"PermissionError: Cannot {operation} '{path}' — "
                        f"path matches a filesystem guard (pattern: "
                        f"'{pattern.pattern}', disposition: read-only). "
                        f"This path is read-only."
                    )
                elif disposition == "workspace":
                    try:
                        path.relative_to(_workspace_root_abs)
                        return True  # inside workspace — allowed
                    except ValueError:
                        raise PermissionError(
                            f"PermissionError: Cannot {operation} '{path}' — "
                            f"path matches a filesystem guard (pattern: "
                            f"'{pattern.pattern}', disposition: workspace). "
                            f"This path is only accessible inside the workspace."
                        )
                elif disposition == "allow":
                    return True  # explicitly allowed
        return None  # no guard matched

    def _is_within(target: pathlib.Path, allowed_roots: list[pathlib.Path]) -> bool:
        """Check if target path is within any of the allowed roots."""
        resolved_target = _abspath(target)
        for root in allowed_roots:
            try:
                resolved_target.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _is_pyddock_path(target: pathlib.Path) -> bool:
        """Check if target path is inside .pyddock/ directory (excluding tmp/)."""
        try:
            rel = _abspath(target).relative_to(pyddock_dir)
            # Allow writes to .pyddock/tmp/ and anything inside it.
            # This is used by tempfile (NamedTemporaryFile passes the dir
            # path to _io.open with a custom opener).
            rel_str = str(rel)
            if rel_str.startswith("tmp"):
                return False
            return True
        except ValueError:
            return False

    def _is_null_device(target: Any) -> bool:
        """True if the path is the OS null device (nul / /dev/null).

        Writing to the null device is a harmless sink. Subprocess/stdio
        redirection (e.g. GitPython spawning git with output to os.devnull)
        opens it via low-level os.open, which the name-based patches never saw
        but the audit hook does — so allow it explicitly here, for everyone.
        """
        s = str(target)
        if s == _NULL_DEVICE:
            return True
        if _real_os.name == "nt":
            # On Windows 'nul' is a reserved device name in any directory and
            # with any extension (nul, dir\\nul, nul.txt all hit the device);
            # real files can't be named that, so a 'nul' component is safe.
            comp = s.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
            return comp == "nul" or comp.startswith("nul.")
        return False

    def _check_read(path: pathlib.Path) -> None:
        """Raise PermissionError if path is outside readable scope."""
        if _is_null_device(path):
            return
        resolved = _abspath(path)
        # Check guards first (first match wins)
        guard_result = _check_guard(resolved, "read")
        if guard_result is True:
            return  # explicitly allowed by guard
        # guard_result is None — no guard matched, fall through
        if unrestricted_reads:
            return
        if not _is_within(resolved, resolved_readable):
            paths_str = ", ".join(str(p) for p in resolved_readable)
            raise PermissionError(
                f"PermissionError: Cannot read '{path}' — reads are restricted "
                f"to the workspace directory. Please use a path within the "
                f"workspace instead."
            )

    def _check_write(path: pathlib.Path) -> None:
        """Raise PermissionError if path is outside writable scope."""
        if _is_null_device(path):
            return
        resolved = _abspath(path)
        # Check guards first (first match wins)
        guard_result = _check_guard(resolved, "write")
        # Even if guard says "allow", still enforce structural protections
        # (.pyddock/, pyddock source, stdlib, workspace modules, shell dirs).
        # Guards can only DENY or relax the writable_paths boundary — they
        # cannot override structural write protections.

        if _is_pyddock_path(path):
            raise PermissionError(
                f"PermissionError: Cannot write to '{path}' — the .pyddock/ "
                f"directory is protected. Snippets cannot modify their own "
                f"configuration or environment."
            )
        # Protect pyddock's own source directory (enforcement code)
        try:
            resolved.relative_to(pyddock_src_dir)
            raise PermissionError(
                f"PermissionError: Cannot write to '{path}' — the pyddock "
                f"source directory is protected."
            )
        except ValueError:
            pass
        # Protect the Python stdlib Lib directory
        try:
            resolved.relative_to(_stdlib_lib_path)
            raise PermissionError(
                f"PermissionError: Cannot write to '{path}' — the Python "
                f"standard library directory is protected."
            )
        except ValueError:
            pass
        # Check workspace module directory protection
        for ws_dir in workspace_module_dirs:
            try:
                resolved.relative_to(ws_dir)
                raise PermissionError(
                    f"Cannot write to '{path}' — workspace module "
                    f"directories are protected."
                )
            except ValueError:
                continue
        # Check shell write protection (prevent write-then-execute escalation)
        for protected_dir in shell_protected_dirs:
            try:
                resolved.relative_to(protected_dir)
                raise PermissionError(
                    f"PermissionError: Cannot write to '{path}' — this path "
                    f"is write-protected because it contains shell-executable "
                    f"scripts. This prevents write-then-execute escalation."
                )
            except ValueError:
                continue

        # If guard explicitly allowed, skip writable_paths boundary check
        if guard_result is True:
            return

        if not _is_within(resolved, resolved_writable):
            paths_str = ", ".join(str(p) for p in resolved_writable)
            raise PermissionError(
                f"PermissionError: Cannot write '{path}' — writes are restricted "
                f"to the workspace directory. Please use a path within the "
                f"workspace instead."
            )

    _WRITE_MODES = {"w", "a", "x", "r+", "w+", "a+", "x+",
                    "wb", "ab", "xb", "r+b", "w+b", "a+b", "x+b",
                    "rb+", "wb+", "ab+", "xb+",
                    "wt", "at", "xt", "r+t", "w+t", "a+t", "x+t",
                    "rt+", "wt+", "at+", "xt+"}

    def _is_write_mode(mode: str) -> bool:
        """Determine if a file mode implies writing."""
        # Any mode containing w, a, x, or + (except bare r) is a write mode
        return any(c in mode for c in "wax+")

    # Patch builtins.open
    _ORIGINALS["builtins.open"] = builtins.open

    def _patched_open(
        file: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        path = pathlib.Path(file) if not isinstance(file, pathlib.Path) else file
        if _is_write_mode(mode):
            _check_write(path)
        else:
            _check_read(path)
        return _ORIGINALS["builtins.open"](file, mode, *args, **kwargs)

    builtins.open = _patched_open

    # Set tempfile to use the workspace-local temp directory.
    # The directory is created by the server at startup (server.py).
    _tempfile_module.tempdir = str(workspace_root / ".pyddock" / "tmp")

    # Patch pathlib.Path methods
    _ORIGINALS["Path.write_text"] = pathlib.Path.write_text
    _ORIGINALS["Path.write_bytes"] = pathlib.Path.write_bytes
    _ORIGINALS["Path.read_text"] = pathlib.Path.read_text
    _ORIGINALS["Path.read_bytes"] = pathlib.Path.read_bytes
    _ORIGINALS["Path.open"] = pathlib.Path.open

    def _patched_write_text(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.write_text"](self_path, *args, **kwargs)

    def _patched_write_bytes(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.write_bytes"](self_path, *args, **kwargs)

    def _patched_read_text(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_read(self_path)
        return _ORIGINALS["Path.read_text"](self_path, *args, **kwargs)

    def _patched_read_bytes(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_read(self_path)
        return _ORIGINALS["Path.read_bytes"](self_path, *args, **kwargs)

    def _patched_path_open(self_path: pathlib.Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if _is_write_mode(mode):
            _check_write(self_path)
        else:
            _check_read(self_path)
        return _ORIGINALS["Path.open"](self_path, mode, *args, **kwargs)

    pathlib.Path.write_text = _patched_write_text
    pathlib.Path.write_bytes = _patched_write_bytes
    pathlib.Path.read_text = _patched_read_text
    pathlib.Path.read_bytes = _patched_read_bytes
    pathlib.Path.open = _patched_path_open

    # Patch Path.rename and Path.replace — both move files to a target path.
    # replace() atomically overwrites the target; rename() fails if target
    # exists on Windows. Both must check the destination against write guards.
    _ORIGINALS["Path.rename"] = pathlib.Path.rename
    _ORIGINALS["Path.replace"] = pathlib.Path.replace

    def _patched_rename(self_path: pathlib.Path, target: Any, *args: Any, **kwargs: Any) -> Any:
        target_path = pathlib.Path(target) if not isinstance(target, pathlib.Path) else target
        _check_write(target_path)
        _check_write(self_path)  # source is also "modified" (removed)
        return _ORIGINALS["Path.rename"](self_path, target, *args, **kwargs)

    def _patched_replace(self_path: pathlib.Path, target: Any, *args: Any, **kwargs: Any) -> Any:
        target_path = pathlib.Path(target) if not isinstance(target, pathlib.Path) else target
        _check_write(target_path)
        _check_write(self_path)  # source is also "modified" (removed)
        return _ORIGINALS["Path.replace"](self_path, target, *args, **kwargs)

    pathlib.Path.rename = _patched_rename
    pathlib.Path.replace = _patched_replace

    # Patch Path.unlink, rmdir, touch, mkdir — all modify the filesystem.
    _ORIGINALS["Path.unlink"] = pathlib.Path.unlink
    _ORIGINALS["Path.rmdir"] = pathlib.Path.rmdir
    _ORIGINALS["Path.touch"] = pathlib.Path.touch
    _ORIGINALS["Path.mkdir"] = pathlib.Path.mkdir

    def _patched_unlink(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.unlink"](self_path, *args, **kwargs)

    def _patched_rmdir(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.rmdir"](self_path, *args, **kwargs)

    def _patched_touch(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.touch"](self_path, *args, **kwargs)

    def _patched_mkdir(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        _check_write(self_path)
        return _ORIGINALS["Path.mkdir"](self_path, *args, **kwargs)

    pathlib.Path.unlink = _patched_unlink
    pathlib.Path.rmdir = _patched_rmdir
    pathlib.Path.touch = _patched_touch
    pathlib.Path.mkdir = _patched_mkdir

    # Disable dangerous Path methods entirely — these enable bypass via
    # indirection (symlinks/hardlinks) or permission escalation.
    def _blocked_symlink_to(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(
            f"PermissionError: Path.symlink_to() is not permitted. "
            f"Symlink creation is disabled to prevent filesystem bypass."
        )

    def _blocked_hardlink_to(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(
            f"PermissionError: Path.hardlink_to() is not permitted. "
            f"Hard link creation is disabled to prevent filesystem bypass."
        )

    def _blocked_link_to(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(
            f"PermissionError: Path.link_to() is not permitted. "
            f"Hard link creation is disabled to prevent filesystem bypass."
        )

    # chmod validation: block special bits (setuid/setgid/sticky) which
    # change how the OS treats the file. Standard permission bits (owner,
    # group, other rwx) are allowed — _check_write() already controls
    # which files can be modified at all.
    _CHMOD_SPECIAL_BITS = 0o7000

    def _validate_chmod(mode: int, caller: str) -> None:
        """Reject non-integer or special-bit modes."""
        if not isinstance(mode, int):
            raise PermissionError(
                f"PermissionError: {caller} mode must be an integer."
            )
        if mode & _CHMOD_SPECIAL_BITS:
            raise PermissionError(
                f"PermissionError: {caller} mode {oct(mode)} requests "
                f"special bits ({oct(mode & _CHMOD_SPECIAL_BITS)}). "
                f"setuid/setgid/sticky are not permitted."
            )

    _ORIGINALS["Path.chmod"] = pathlib.Path.chmod
    if hasattr(pathlib.Path, "lchmod"):
        _ORIGINALS["Path.lchmod"] = pathlib.Path.lchmod

    def _guarded_chmod(self_path: pathlib.Path, mode: int, *args: Any, **kwargs: Any) -> None:
        _validate_chmod(mode, "Path.chmod()")
        _check_write(self_path)
        return _ORIGINALS["Path.chmod"](self_path, mode, *args, **kwargs)

    def _guarded_lchmod(self_path: pathlib.Path, mode: int, *args: Any, **kwargs: Any) -> None:
        _validate_chmod(mode, "Path.lchmod()")
        _check_write(self_path)
        return _ORIGINALS["Path.lchmod"](self_path, mode, *args, **kwargs)

    pathlib.Path.symlink_to = _blocked_symlink_to
    pathlib.Path.hardlink_to = _blocked_hardlink_to
    if hasattr(pathlib.Path, "link_to"):
        pathlib.Path.link_to = _blocked_link_to
    pathlib.Path.chmod = _guarded_chmod
    if hasattr(pathlib.Path, "lchmod"):
        pathlib.Path.lchmod = _guarded_lchmod

    # Patch io.open — it's a separate function from builtins.open
    # and can bypass filesystem scoping if not patched.
    _io_module = io_module
    _ORIGINALS["io.open"] = _io_module.open

    def _patched_io_open(
        file: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        # Skip path checking for integer file descriptors (used internally by subprocess)
        if isinstance(file, int):
            return _ORIGINALS["io.open"](file, mode, *args, **kwargs)
        path = pathlib.Path(file) if not isinstance(file, pathlib.Path) else file
        if _is_write_mode(mode):
            _check_write(path)
        else:
            _check_read(path)
        return _ORIGINALS["io.open"](file, mode, *args, **kwargs)

    _io_module.open = _patched_io_open

    # Also patch _io.open (C implementation) to prevent bypass of filesystem scoping
    _ORIGINALS["_io.open"] = _cio_module.open

    def _patched_cio_open(
        file: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        # File descriptors (int) bypass path checks — they're already-open handles
        if isinstance(file, int):
            return _ORIGINALS["_io.open"](file, mode, *args, **kwargs)
        # Skip path checks when called from frozen import machinery
        # (loading .py/.pyc files). Only frozen code gets this bypass.
        caller = sys._getframe(1)
        if caller and caller.f_code.co_filename.startswith("<frozen"):
            return _ORIGINALS["_io.open"](file, mode, *args, **kwargs)
        path = pathlib.Path(file) if not isinstance(file, pathlib.Path) else file
        if _is_write_mode(mode):
            _check_write(path)
        else:
            _check_read(path)
        return _ORIGINALS["_io.open"](file, mode, *args, **kwargs)

    _cio_module.open = _patched_cio_open

    # Patch io.FileIO — its constructor opens files directly without going
    # through builtins.open or io.open. Same for _io.FileIO (C implementation).
    _ORIGINALS["io.FileIO"] = _io_module.FileIO
    _ORIGINALS["_io.FileIO"] = _cio_module.FileIO

    class _PatchedFileIO(_ORIGINALS["_io.FileIO"]):
        """FileIO subclass that enforces filesystem scoping on construction."""

        def __init__(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> None:
            if not isinstance(file, int):
                path = pathlib.Path(file) if not isinstance(file, pathlib.Path) else file
                if _is_write_mode(mode):
                    _check_write(path)
                else:
                    _check_read(path)
            super().__init__(file, mode, *args, **kwargs)

    _io_module.FileIO = _PatchedFileIO
    _cio_module.FileIO = _PatchedFileIO

    # Lock down tempfile.tempdir — prevent agent from redirecting temp file
    # creation outside the workspace by reassigning tempfile.tempdir.
    # We freeze it by replacing the tempfile module's __dict__ entry with a
    # property-like mechanism (module-level setattr override isn't possible,
    # so we delete the attribute and rely on the patched open() to enforce
    # path checks on any tempfile operations regardless of tempdir value).
    # Simpler approach: just remove tempdir setter by making tempfile use
    # a wrapper that ignores reassignment.
    _frozen_tempdir = _tempfile_module.tempdir

    class _TempfileProxy:
        """Proxy that freezes tempdir and blocks mkstemp while forwarding everything else."""

        # mkstemp returns a raw (fd, path) tuple that requires os.close/os.write
        # which aren't on the safe os proxy. Block it to avoid leaked fds.
        _BLOCKED = frozenset({"mkstemp", "mkdtemp"})

        def __getattr__(self, name: str) -> Any:
            if name == "tempdir":
                return _frozen_tempdir
            if name in self._BLOCKED:
                def _blocked(*args: Any, **kwargs: Any) -> None:
                    raise PermissionError(
                        f"PermissionError: tempfile.{name}() is not permitted. "
                        f"Use tempfile.NamedTemporaryFile() or tempfile.TemporaryFile() instead."
                    )
                return _blocked
            return getattr(_tempfile_module, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "tempdir":
                raise PermissionError(
                    "PermissionError: Cannot reassign tempfile.tempdir. "
                    "Temp files are restricted to the workspace .pyddock/tmp/ directory."
                )
            setattr(_tempfile_module, name, value)

    sys.modules["tempfile"] = _TempfileProxy()

    # Patch os.makedirs and os.mkdir on the safe os proxy to use _check_write.
    # This ensures .pyddock/, workspace module dirs, and other protected paths
    # are enforced (not just the workspace boundary check).

    def _safe_makedirs(name: str, mode: int = 0o777, exist_ok: bool = False) -> None:
        _check_write(pathlib.Path(name))
        _real_os.makedirs(name, mode, exist_ok=exist_ok)

    def _safe_mkdir(name: str, mode: int = 0o777) -> None:
        _check_write(pathlib.Path(name))
        _real_os.mkdir(name, mode)

    if "os" in sys.modules:
        safe_os = sys.modules["os"]
        safe_os.makedirs = _safe_makedirs
        safe_os.mkdir = _safe_mkdir

    # Expose os.chmod with the same validation as Path.chmod.
    def _safe_chmod(path: str, mode: int) -> None:
        _validate_chmod(mode, "os.chmod()")
        _check_write(pathlib.Path(path))
        _real_os.chmod(path, mode)

    if "os" in sys.modules:
        sys.modules["os"].chmod = _safe_chmod

    # Install the audit-hook backstop LAST, sharing the same _check_* closures.
    # Audit events fire beneath the Python name layer, so this catches bypasses
    # that re-derive the real _io.FileIO from a live object or use low-level
    # os.open/os.replace — operations the monkeypatches above cannot see.
    # Install the audit-hook policy engine LAST, sharing the same _check_*
    # closures. Audit events fire beneath the Python name layer, so this catches
    # bypasses that re-derive the real _io.FileIO from a live object or use
    # low-level os.open/os.replace — and enforces the [audit] disposition table
    # (fs scoping + agent-deny for primitives with no sanctioned agent use).
    audit_rules = [
        (entry["pattern"], entry["disposition"])
        for entry in config.get("audit", [])
        if isinstance(entry, dict) and "pattern" in entry and "disposition" in entry
    ]
    install_audit_enforcement(
        check_read=_check_read,
        check_write=_check_write,
        is_write_mode=_is_write_mode,
        pyddock_dir=_PYDDOCK_DIR,
        real_os=_real_os,
        audit_rules=audit_rules,
        trusted_prefixes=trusted_prefixes,
        debug=debug,
        log_path=str(workspace_root / ".pyddock" / "tmp" / "audit.jsonl") if debug else None,
        shell_policies=config.get("shell", {}),
        workspace_root=str(workspace_root),
        workspace_module_dirs=config.get("imports", {}).get("workspace", {}),
        env_base=config.get("env", {}),
        env_snapshot=make_child_env(),
    )
