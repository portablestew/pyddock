"""Runtime enforcement module for pyddock.

This module runs INSIDE the subprocess before user code executes.
It applies runtime restrictions that complement the static AST analysis:
- Import hook (blocks non-allowlisted imports)
- Filesystem scoping (restricts read/write paths)
- Module and class restrictions (deny/allow patterns for module attrs and class methods)

The RuntimeEnforcement class accepts a serialized config dict and workspace_root string.
"""

from __future__ import annotations

import _io as _cio_module
import builtins
import importlib
import pathlib
import re
import sys
import tempfile as _tempfile_module
import types
from typing import Any

from pyddock import SNIPPET_FILENAME

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
# Since pyddock._runtime is not importable by agent code, this dict is inaccessible.
_ORIGINALS: dict[str, Any] = {}


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

    def __init__(self, allowed: list[str], trusted_prefixes: list[str]) -> None:
        self._allowed = set(allowed)
        self._trusted_prefixes = tuple(trusted_prefixes)

    def find_module(self, fullname: str, path: Any = None) -> _ImportBlocker | None:
        """Legacy import hook interface for compatibility."""
        if self._should_block(fullname):
            return self
        return None

    def load_module(self, fullname: str) -> None:
        """Raise ImportError for blocked modules (legacy interface)."""
        allowed_list = ", ".join(sorted(self._allowed))
        raise ImportError(
            f"ImportError: '{fullname}' is not an allowed import. "
            f"Please use one of the following allowed imports instead: {allowed_list}"
        )

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> None:
        """Modern import hook interface (Python 3.4+).

        Returns None for allowed imports (letting other finders handle them).
        Raises ImportError directly for blocked imports.
        """
        if self._should_block(fullname):
            allowed_list = ", ".join(sorted(self._allowed))
            raise ImportError(
                f"ImportError: '{fullname}' is not an allowed import. "
                f"Please use one of the following allowed imports instead: "
                f"{allowed_list}"
            )
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


class MethodFilterProxy:
    """Proxy that intercepts attribute access and blocks disallowed methods.

    Used by FactoryProxy to wrap objects returned by factory functions.
    Only methods matching at least one allow pattern are permitted.
    """

    def __init__(self, wrapped: Any, allow_patterns: list[re.Pattern[str]]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_allow_patterns", allow_patterns)

    def __getattr__(self, name: str) -> Any:
        allow_patterns = object.__getattribute__(self, "_allow_patterns")
        wrapped = object.__getattribute__(self, "_wrapped")

        if name.startswith("__") and name.endswith("__"):
            # Allow dunder access for internal Python machinery (repr, str, etc.)
            return getattr(wrapped, name)

        if not any(p.match(name) for p in allow_patterns):
            patterns_str = ", ".join(p.pattern for p in allow_patterns)
            raise PermissionError(
                f"PermissionError: '{name}' is not permitted. "
                f"Allowed method patterns: {patterns_str}. "
                f"Please use one of the allowed methods instead."
            )
        return getattr(wrapped, name)


class FactoryProxy:
    """Wraps a factory function to return proxied objects.

    Objects returned by the factory are wrapped in MethodFilterProxy,
    which enforces the allow-pattern list on method access.
    """

    def __init__(
        self, original_factory: Any, allow_patterns: list[re.Pattern[str]]
    ) -> None:
        self._original = original_factory
        self._allow_patterns = allow_patterns

    def __call__(self, *args: Any, **kwargs: Any) -> MethodFilterProxy:
        obj = self._original(*args, **kwargs)
        return MethodFilterProxy(obj, self._allow_patterns)


def _expand_patterns(
    patterns: list[str], module: types.ModuleType
) -> frozenset[str]:
    """Pre-compute the set of attribute names matching any regex pattern.

    Evaluates each pattern against dir(module) at proxy creation time.
    This avoids regex evaluation on every attribute access.

    Args:
        patterns: List of regex pattern strings (from module_allow config).
        module: The real module to scan.

    Returns:
        Frozenset of concrete attribute names that match at least one pattern.
    """
    matched: set[str] = set()
    for name in dir(module):
        for pattern in patterns:
            if re.match(pattern, name):
                matched.add(name)
                break
    return frozenset(matched)


def _compute_exported_api(
    module: types.ModuleType,
    *,
    exclude_foreign_classes: bool = False,
) -> frozenset[str]:
    """Determine which attributes constitute a module's public API.

    Algorithm:
    1. If module defines __all__, use exactly those names.
    2. Otherwise, include all attributes that are NOT instances of
       types.ModuleType (excludes re-exported imports like `os`, `sys`).
       Also exclude private names (starting with _).
    3. If exclude_foreign_classes is True, also exclude classes whose
       __module__ belongs to a different top-level package. This prevents
       workspace modules from leaking network-capable factory classes
       (e.g. Jira, SSHClient) imported from third-party dependencies.

    Args:
        module: The module to compute the exported API for.
        exclude_foreign_classes: If True, classes defined in a different
            top-level package are excluded from the exported API. Workspace
            modules that intentionally expose such classes should define
            __all__ to opt in.

    Returns:
        Frozenset of attribute names considered part of the exported API.
    """
    if hasattr(module, "__all__"):
        return frozenset(module.__all__)

    module_name = getattr(module, "__name__", "") or ""
    top_level_pkg = module_name.split(".")[0]

    exported: set[str] = set()
    for name in dir(module):
        if name.startswith("_"):
            continue
        val = getattr(module, name, None)
        if isinstance(val, types.ModuleType):
            continue
        # Exclude foreign classes on workspace modules to prevent leakage
        # of network-capable factories (e.g. atlassian.Jira, paramiko.SSHClient).
        if exclude_foreign_classes and isinstance(val, type):
            cls_module = getattr(val, "__module__", "") or ""
            cls_top_level = cls_module.split(".")[0]
            if cls_module == "builtins":
                # Built-in types (str, int, Exception, etc.) are always safe
                exported.add(name)
            elif cls_top_level == top_level_pkg:
                # Class defined within the same package — safe
                exported.add(name)
            # else: foreign class — exclude from exported API
            continue
        exported.add(name)
    return frozenset(exported)


# Internal state for _CallerScopedModuleProxy instances. Stored here (not on
# proxy instances) so agent code cannot extract it via object.__getattribute__.
_PROXY_STATE: dict[int, tuple] = {}


class _CallerScopedModuleProxy(types.ModuleType):
    """Caller-scoped module proxy that enforces attribute access control.

    Operates in two modes:

    Simple mode (trusted_prefixes=None):
        - always_allowed: Pre-cached on instance, returned without stack walk.
        - always_blocked: Always raise AttributeError.
        - Everything else: Raise AttributeError (deny by default).
        - Used by: deny-mode modules (boto3), workspace modules.
        - No _caller_is_trusted() invocation — O(1) set check only.

    Caller-scoped mode (trusted_prefixes provided):
        - always_allowed: Pre-cached on instance, returned without stack walk.
        - always_blocked: Always raise AttributeError.
        - Everything else: Returned only if _caller_is_trusted() is True.
        - Used by: sys proxy only.
    """

    def __init__(
        self,
        module_name: str,
        real_module: types.ModuleType,
        always_allowed: frozenset[str],
        always_blocked: frozenset[str],
        trusted_prefixes: tuple[str, ...] | None = None,  # None = simple mode
        *,
        custom_attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            module_name: Name for the proxy module (e.g. "sys", "boto3").
            real_module: The real module being proxied.
            always_allowed: Attribute names always accessible to any caller.
            always_blocked: Attribute names that always raise AttributeError.
            trusted_prefixes: Tuple passed to _caller_is_trusted(). If None,
                              the proxy operates in simple mode (allowlist-only,
                              no stack walk). If provided, operates in caller-
                              scoped mode (sys proxy).
            custom_attrs: Optional dict of custom attribute implementations
                          (e.g. sys.exit replacement, FactoryProxy-wrapped
                          callables). These are pre-cached and override real
                          module values. The real module is NOT mutated.
        """
        super().__init__(module_name)
        # Store internal state in module-level dict, NOT on the instance.
        # This prevents agent code from using object.__getattribute__(proxy, '_real_module').
        _PROXY_STATE[id(self)] = (
            real_module, always_blocked, always_allowed, trusted_prefixes, module_name
        )

        _custom = custom_attrs or {}
        _MISSING = object()

        # Pre-cache all always_allowed attributes on the instance.
        # Check custom_attrs first, then fall back to real module.
        # Note: getattr may trigger lazy __getattr__ on modules, which can
        # import submodules that use blocked patterns (e.g. six using
        # attrgetter('__closure__')). We skip those gracefully.
        for name in always_allowed:
            if name in _custom:
                object.__setattr__(self, name, _custom[name])
            else:
                try:
                    val = getattr(real_module, name, _MISSING)
                except (PermissionError, ImportError):
                    continue
                if val is not _MISSING:
                    object.__setattr__(self, name, val)

        # Pre-cache ALL custom_attrs entries (some may not be in always_allowed).
        for name, val in _custom.items():
            object.__setattr__(self, name, val)

        # Forward package metadata so Python's import machinery treats this
        # proxy as a proper package (needed for `import boto3.s3` etc.).
        for meta_attr in ("__path__", "__package__", "__file__", "__spec__"):
            val = getattr(real_module, meta_attr, _MISSING)
            if val is not _MISSING:
                object.__setattr__(self, meta_attr, val)

    def __dir__(self) -> list[str]:
        _, _, always_allowed, _, _ = _PROXY_STATE[id(self)]
        return list(always_allowed)

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access for non-cached attributes.

        Only invoked for EXTERNAL access. Internal module code (code within
        the proxied module) accesses attributes through __globals__ which is
        the module's __dict__ — this method is never called for internal access.
        """
        real_module, always_blocked, always_allowed, trusted_prefixes, module_name = (
            _PROXY_STATE[id(self)]
        )

        if name in always_blocked:
            raise AttributeError(
                f"module '{module_name}' has no attribute '{name}'"
            )

        # Fallback for always-allowed (shouldn't fire normally due to pre-caching)
        if name in always_allowed:
            if hasattr(real_module, name):
                return getattr(real_module, name)
            raise AttributeError(
                f"module '{module_name}' has no attribute '{name}'"
            )

        # Caller-scoped mode (all universal proxies + sys proxy)
        if trusted_prefixes is not None:
            if _caller_is_trusted(trusted_prefixes):
                if hasattr(real_module, name):
                    return getattr(real_module, name)

        # Simple mode or untrusted caller in caller-scoped mode
        raise AttributeError(
            f"module '{module_name}' has no attribute '{name}'"
        )


def _build_trusted_prefixes(
    workspace_root: pathlib.Path,
    workspace_imports: dict[str, str],
    venv_path: pathlib.Path | None,
    real_os: types.ModuleType | None = None,
) -> tuple[str, ...]:
    """Build the complete set of trusted path prefixes.

    Returns normalized, resolved path prefixes for:
    1. Workspace module directories (editable installs)
    2. The venv's site-packages (transitive deps)
    3. The Python stdlib Lib directory

    This extracts and extends the existing inline prefix-building logic
    from install_import_hook(). The current implementation builds prefixes
    inline; this refactor moves it to a standalone function and adds the
    stdlib Lib directory as a trusted prefix. Called from install_import_hook()
    in place of the inline prefix construction.

    Args:
        workspace_root: Absolute path to workspace directory.
        workspace_imports: Dict of {module_name: relative_path} from config.
        venv_path: Path to .pyddock/venv if it exists, else None.
        real_os: The real os module (saved in apply_all()). By the time this
            runs, os in sys.modules may be the safe proxy. If None, falls back
            to importing os directly (only safe during early init).

    Returns:
        Tuple of normalized, lowercased path prefixes with no duplicates.
    """
    _os = real_os if real_os is not None else __import__("os")
    prefixes: list[str] = []

    # 1. Workspace module directories (editable installs)
    for _name, rel_path in workspace_imports.items():
        abs_path = _os.path.normcase(
            _os.path.realpath(str(workspace_root / rel_path))
        )
        prefixes.append(abs_path)

    # 2. Venv site-packages (transitive deps of allowed packages)
    if venv_path is not None and venv_path.is_dir():
        # Find site-packages inside the venv (Windows: Lib/site-packages)
        for root, dirs, _files in _os.walk(str(venv_path / "Lib")):
            if "site-packages" in dirs:
                sp = _os.path.join(root, "site-packages")
                prefixes.append(_os.path.normcase(_os.path.realpath(sp)))
                break
        else:
            # Unix-style layout: lib/pythonX.Y/site-packages
            for root, dirs, _files in _os.walk(str(venv_path / "lib")):
                if "site-packages" in dirs:
                    sp = _os.path.join(root, "site-packages")
                    prefixes.append(_os.path.normcase(_os.path.realpath(sp)))
                    break

    # 3. Python stdlib Lib directory
    # Use sys.base_prefix to handle virtualenvs correctly — it points to
    # the original Python installation, not the venv.
    stdlib_lib = _os.path.join(sys.base_prefix, "Lib")
    if not _os.path.isdir(stdlib_lib):
        # Unix layout: lib/pythonX.Y/
        stdlib_lib = _os.path.join(
            sys.base_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}"
        )
    prefixes.append(_os.path.normcase(_os.path.realpath(stdlib_lib)))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_prefixes: list[str] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            unique_prefixes.append(p)

    return tuple(unique_prefixes)


def _proxy_module_universal(
    name: str,
    trusted_prefixes: tuple[str, ...],
    skip_modules: frozenset[str],
    workspace_module_names: frozenset[str] = frozenset(),
) -> None:
    """Wrap a module in a caller-scoped proxy (universal mode).

    Every allowed module gets caller-scoped mode. The always_allowed set
    is the module's exported API (non-ModuleType, non-private attrs).
    Trusted code (stdlib, site-packages, workspace) can access anything.
    Agent code can only access the exported API.

    Args:
        name: Fully-qualified module name (e.g. "pathlib", "json.decoder").
        trusted_prefixes: Tuple of trusted path prefixes for _caller_is_trusted().
        skip_modules: Set of module names that must not be proxied — only
                      {"os", "sys"} plus restriction module names.
        workspace_module_names: Set of top-level workspace module names. Modules
                      in this set get stricter filtering that excludes foreign
                      classes from the exported API.
    """
    module = sys.modules.get(name)
    if module is None or isinstance(module, _CallerScopedModuleProxy):
        return
    # Skip if this module OR its top-level package is in the skip set.
    # This ensures submodules of os (os.path), sys, and threading are also skipped.
    top_level = name.split(".")[0]
    if name in skip_modules or top_level in skip_modules:
        return

    # Compute the exported API — these attrs are always accessible to agent code.
    # Workspace modules get stricter filtering: foreign classes are excluded
    # to prevent leakage of network-capable factories (e.g. Jira, SSHClient).
    is_workspace = top_level in workspace_module_names
    always_allowed = _compute_exported_api(module, exclude_foreign_classes=is_workspace)

    # Include already-loaded submodule names so `import json; json.decoder` works
    prefix = name + "."
    submod_names = frozenset(
        key[len(prefix):].split(".")[0]
        for key in sys.modules
        if key.startswith(prefix)
    )
    always_allowed = always_allowed | submod_names

    # Always include module metadata attrs
    always_allowed = always_allowed | frozenset({
        "__name__", "__doc__", "__loader__", "__spec__",
        "__package__", "__path__", "__file__",
    })

    proxy = _CallerScopedModuleProxy(
        module_name=name,
        real_module=module,
        always_allowed=always_allowed,
        always_blocked=frozenset(),
        trusted_prefixes=trusted_prefixes,  # caller-scoped mode for ALL
    )
    sys.modules[name] = proxy

    # Update parent module's reference to point to the proxy.
    # Skip the update if the parent's existing attribute is callable — this
    # handles the case where a package re-exports a function from a submodule
    # (e.g. polars.functions.lit is both a module AND a function attribute).
    # Replacing the callable with a non-callable proxy would break internal calls.
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent_name, child_name = parts
        parent = sys.modules.get(parent_name)
        if parent is not None:
            try:
                existing = object.__getattribute__(parent, child_name) if hasattr(parent, child_name) else None
            except (AttributeError, TypeError):
                existing = None
            # Only update if the existing attr isn't a callable function/class
            # that would be broken by replacing it with a non-callable proxy.
            if existing is None or isinstance(existing, types.ModuleType) or isinstance(existing, _CallerScopedModuleProxy):
                try:
                    object.__setattr__(parent, child_name, proxy)
                except (AttributeError, TypeError):
                    pass


class RuntimeEnforcement:
    """Applies runtime enforcement inside the subprocess.

    Constructor accepts:
        config: dict — serialized PyddockConfig (plain dict from executor)
        workspace_root: str — absolute path to the workspace directory

    Call apply_all() to install all enforcement hooks in order.
    """

    def __init__(self, config: dict, workspace_root: str) -> None:
        self._config = config
        # Use abspath (not resolve) to preserve symlinks/subst drives
        import os as _os_init
        self._workspace_root = pathlib.Path(_os_init.path.abspath(workspace_root))

    def apply_all(self) -> None:
        """Apply all enforcement in order: import hook, filesystem, proxies, patches, subprocess."""
        # Save references to modules before safe proxies replace them in sys.modules
        import os as _real_os
        self._real_os = _real_os
        import sys as _real_sys
        self._real_sys = _real_sys
        try:
            import subprocess as _subprocess_module
            self._subprocess_module = _subprocess_module
        except ImportError:
            self._subprocess_module = None

        # Import resolve_command before the hook blocks pyddock
        from pyddock.shell_executor import resolve_command, _looks_like_path, _extract_path_candidates
        self._resolve_command = resolve_command
        self._looks_like_path = _looks_like_path
        self._extract_path_candidates = _extract_path_candidates

        # Pre-import modules needed by enforcement methods before the hook activates.
        # These are runtime-internal dependencies, not user-facing.
        import operator as _operator_module
        self._operator_module = _operator_module
        import types as _types_module
        self._types_module = _types_module
        import io as _io_module
        self._io_module = _io_module
        import traceback as _traceback_module
        self._traceback_module = _traceback_module

        self.install_import_hook()
        self.apply_attrgetter_guard()
        self.apply_filesystem_scoping()
        self.apply_restrictions()
        self.apply_subprocess_patch()
        self.install_module_proxies()

    def apply_attrgetter_guard(self) -> None:
        """Patch operator.attrgetter and builtins.getattr to reject blocked attribute names.

        This closes bypasses where attrgetter('__globals__') or
        getattr(obj, '__globals__') is used to access dangerous attributes
        at runtime, bypassing AST-level attribute checks.
        """
        operator = self._operator_module

        blocked_attrs = set(
            self._config.get("ast", {}).get("block_attributes", [])
        )
        _ORIGINALS["attrgetter"] = operator.attrgetter
        _ORIGINALS["getattr"] = builtins.getattr

        def _safe_attrgetter(*names: str) -> Any:
            for name in names:
                # Check each dotted component (attrgetter supports 'a.b.c')
                for part in name.split("."):
                    if part in blocked_attrs:
                        raise PermissionError(
                            f"PermissionError: attrgetter access to '{part}' is "
                            f"not permitted. Please rewrite your snippet to avoid "
                            f"this attribute."
                        )
            return _ORIGINALS["attrgetter"](*names)

        def _safe_getattr(obj: Any, name: str, *default: Any) -> Any:
            if isinstance(name, str) and name in blocked_attrs:
                raise PermissionError(
                    f"PermissionError: getattr access to '{name}' is "
                    f"not permitted. Please rewrite your snippet to avoid "
                    f"this attribute."
                )
            return _ORIGINALS["getattr"](obj, name, *default)

        operator.attrgetter = _safe_attrgetter
        builtins.getattr = _safe_getattr

    def install_import_hook(self) -> None:
        """Install a pure-allowlist import hook with stack-aware bypass.

        Pre-imports all allowed modules first so their transitive dependencies
        are cached in sys.modules before the hook is active. Then installs
        two enforcement points:
        1. _ImportBlocker on sys.meta_path (fires for uncached modules)
        2. _guarded_import on builtins.__import__ (fires for ALL imports)

        Both use the same logic: top-level module in allowlist → allow,
        caller is inside a trusted path (workspace module or site-packages) → allow,
        else → block.

        Trusted paths include:
        - Workspace module directories (editable installs)
        - The venv's site-packages (transitive dependencies)
        """
        allowed = self._config.get("imports", {}).get("allowed", [])
        allowed_set = set(allowed)
        workspace_imports = self._config.get("imports", {}).get("workspace", {})

        # Pre-import allowed modules so their transitive dependencies
        # are in sys.modules before the hook activates
        for module_name in allowed:
            try:
                importlib.import_module(module_name)
            except ImportError:
                # Module not installed yet — that's fine, skip it
                pass

        # Build trusted path prefixes for stack-aware bypass.
        # These are directories where code is allowed to import freely.
        # Includes workspace module dirs, venv site-packages, and stdlib Lib.
        venv_path = self._workspace_root / ".pyddock" / "venv"
        trusted_prefixes_tuple = _build_trusted_prefixes(
            workspace_root=self._workspace_root,
            workspace_imports=workspace_imports,
            venv_path=venv_path if venv_path.is_dir() else None,
            real_os=self._real_os,
        )
        self._trusted_prefixes = trusted_prefixes_tuple

        # Warm up codecs that may be lazily loaded after the hook activates.
        # Codec lookups trigger imports (e.g. encodings.idna → stringprep)
        # that would be blocked by the pure allowlist. Pre-loading them here
        # caches everything before the hook is installed.
        try:
            import codecs
            for codec_name in ("idna", "utf-8", "ascii", "latin-1", "cp1252", "utf-16-le"):
                try:
                    codecs.lookup(codec_name)
                except LookupError:
                    pass
        except ImportError:
            pass

        blocker = _ImportBlocker(allowed, trusted_prefixes_tuple)
        # Insert at the beginning so it's checked first
        sys.meta_path.insert(0, blocker)

        # Patch builtins.__import__ to enforce the allowlist directly.
        # This closes the bypass where attacker accesses __import__ from
        # a function's __globals__['__builtins__'] dict.
        _ORIGINALS["import"] = builtins.__import__
        _trusted = trusted_prefixes_tuple
        _loading_depth = [0]  # reentrant counter: >0 while loading an allowed module
        # Modules that must NOT be wrapped by _proxy_module_universal:
        # os: plain types.ModuleType with safe attrs only (not a _CallerScopedModuleProxy)
        # sys: specialized caller-scoped proxy with custom_attrs
        # threading: _shutdown() called from C/frozen frames during interpreter exit
        # Restriction modules are NOT skipped — mode="deny" modules already have a
        # _CallerScopedModuleProxy (isinstance early-return), and mode="allow" modules
        # need caller-scoped proxying to block ModuleType attr leakage (e.g. polars.os).
        _skip_proxy = frozenset({"os", "sys", "threading"})
        _ws_module_names = frozenset(workspace_imports.keys())

        def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            # For relative imports (level > 0), the name is relative to the
            # calling package which is already allowed. Let them through.
            # args[3] is 'level' in the __import__ signature:
            # __import__(name, globals, locals, fromlist, level)
            level = args[3] if len(args) > 3 else kwargs.get("level", 0)
            if level and level > 0:
                return _ORIGINALS["import"](name, *args, **kwargs)
            top_level = name.split(".")[0]
            if top_level in allowed_set:
                # Track that we're loading an allowed module — imports
                # triggered by frozen import machinery during this load
                # are allowed (e.g. _io for .pyc cache writing).
                _loading_depth[0] += 1
                try:
                    result = _ORIGINALS["import"](name, *args, **kwargs)
                finally:
                    _loading_depth[0] -= 1
                # Proxy ANY submodule import so agent code can't access
                # leaked imports (e.g. pathlib.os, tempfile.os, json.decoder.os).
                if "." in name:
                    _proxy_module_universal(name, _trusted, _skip_proxy, _ws_module_names)
                return result
            # If we're inside the loading of an allowed module AND the
            # immediate caller is frozen import machinery, permit the import.
            # This narrowly handles internal machinery needs (codec loading,
            # .pyc writing) without opening a general bypass.
            if _loading_depth[0] > 0:
                caller = sys._getframe(1)
                if caller and caller.f_code.co_filename.startswith("<frozen"):
                    return _ORIGINALS["import"](name, *args, **kwargs)
            # Stack-aware bypass: allow imports from trusted code paths
            # (workspace modules, their transitive deps in site-packages).
            if _trusted and _caller_is_trusted(_trusted):
                return _ORIGINALS["import"](name, *args, **kwargs)
            allowed_list = ", ".join(sorted(allowed_set))
            raise ImportError(
                f"ImportError: '{name}' is not an allowed import. "
                f"Please use one of the following allowed imports "
                f"instead: {allowed_list}"
            )

        builtins.__import__ = _guarded_import

        # Install a safe sys proxy if 'sys' is in the allowlist.
        # This gives user code access to sys.argv, sys.platform, etc.
        # without exposing sys.modules, sys.meta_path, or other dangerous attrs.
        # Trusted code (workspace modules, site-packages) gets full sys access.
        if "sys" in allowed_set:
            self._install_safe_sys(trusted_prefixes_tuple)

        # Install a safe os proxy if 'os' is in the allowlist.
        # Exposes os.environ, os.getcwd(), os.name, os.sep, os.path, etc.
        # without exposing os.system, os.popen, os.exec*, os.spawn*, etc.
        if "os" in allowed_set:
            self._install_safe_os()

        # Install module proxies on ALL allowed modules to prevent attribute
        # leakage. This blocks agent code from accessing imported modules via
        # e.g. pathlib.os, tempfile.os, workspace_pkg.sys, etc.
        # Skip: os/sys (have specialized proxies), restriction modules (handled
        # by apply_restrictions()), and anything already proxied.
        # NOTE: This is called later by install_module_proxies() after all
        # patches are applied, so proxies capture the patched versions.
        self._allowed_set = allowed_set

    def install_module_proxies(self) -> None:
        """Wrap ALL allowed modules in caller-scoped proxies.

        Replaces the old implementation that only proxied workspace modules
        and patched os references on stdlib modules.

        Called LAST in apply_all() so that all runtime patches are applied first.
        The proxies then pre-cache the patched versions of attributes.

        This blocks agent code from accessing imported modules via e.g.
        pathlib.os, tempfile.os, workspace_pkg.sys, etc.
        """
        # Build the skip set: only os/sys/threading need skipping.
        # subprocess and tempfile are NOT skipped — they are patched on the real module
        # before proxying, and the proxy pre-caches the patched versions (since
        # install_module_proxies runs last). threading IS skipped because _shutdown()
        # is called during interpreter exit from C code / frozen frames, not from
        # stdlib code with a file path — so _caller_is_trusted() would return False.
        # Restriction modules are NOT skipped:
        #   - mode="deny" (boto3): already a _CallerScopedModuleProxy → isinstance early-return
        #   - mode="allow" (polars): real module with _blocked stubs → gets caller-scoped
        #     proxy to block ModuleType attr leakage (e.g. polars.os)
        skip_modules = frozenset({"os", "sys", "threading"})

        # Identify workspace module names for stricter class filtering.
        workspace_imports = self._config.get("imports", {}).get("workspace", {})
        workspace_module_names = frozenset(workspace_imports.keys())

        for mod_name in list(sys.modules.keys()):
            top = mod_name.split(".")[0]
            if top not in self._allowed_set:
                continue
            _proxy_module_universal(
                mod_name, self._trusted_prefixes, skip_modules, workspace_module_names
            )

    def _install_safe_sys(self, trusted_prefixes: tuple[str, ...]) -> None:
        """Replace sys in sys.modules with a caller-scoped proxy.

        The proxy has three access tiers:
        1. SAFE_ATTRS: Always accessible by anyone (argv, platform, stdout, etc.)
        2. BLOCKED_ATTRS: Always blocked for everyone (meta_path, _getframe —
           these would allow sandbox escape even from trusted code)
        3. Everything else: Accessible only to trusted callers (workspace modules,
           site-packages). Agent snippet code gets AttributeError.

        This allows libraries like pywin32 (running from site-packages) to access
        sys.modules, sys.path, etc. during their bootstrap, while preventing agent
        code from using those attributes to escape the sandbox.
        """
        _real_sys = sys
        _traceback_mod = self._traceback_module

        # Attributes always safe for anyone (read-only values, streams, etc.)
        _SAFE_ATTRS = frozenset({
            "argv", "platform", "version", "version_info", "maxsize",
            "byteorder", "executable", "prefix", "exec_prefix",
            "stdout", "stderr", "stdin",
            "float_info", "int_info", "hash_info",
            "getdefaultencoding", "getfilesystemencoding",
            "getrecursionlimit",
            "path",  # frozen tuple copy for agent code (read-only)
            "exit",
            "exc_info",
            "_pyddock_format_tb",
            # Module metadata attrs Python expects on all modules
            "__name__", "__doc__", "__loader__", "__spec__",
            "__package__", "__path__",
        })

        # Attributes NEVER exposed — even trusted code shouldn't remove
        # the import hook or inspect frames through the proxy.
        _BLOCKED_ATTRS = frozenset({
            "meta_path",       # removing import blocker = sandbox escape
            "path_hooks",      # manipulating import machinery
            "path_importer_cache",
            "_getframe",       # frame access = globals access = sandbox escape
            "_current_frames",
        })

        # Build custom_attrs dict with sys-specific custom implementations.

        # sys.exit — harmless in subprocess (raises SystemExit)
        def _safe_exit(code: int = 0) -> None:
            raise SystemExit(code)

        # sys.exc_info() — strip traceback to prevent frame access
        def _safe_exc_info() -> tuple:
            typ, val, _tb = _real_sys.exc_info()
            return (typ, val, None)

        # sys._pyddock_format_tb() — safe traceback alternative
        def _safe_format_tb() -> list[tuple[str, int, str, str | None]]:
            _typ, _val, tb = _real_sys.exc_info()
            if tb is None:
                return []
            entries = _traceback_mod.extract_tb(tb)
            return [(e.filename, e.lineno, e.name, e.line) for e in entries]

        custom_attrs = {
            "exit": _safe_exit,
            "exc_info": _safe_exc_info,
            "path": tuple(_real_sys.path),
            "_pyddock_format_tb": _safe_format_tb,
        }

        # Instantiate the unified caller-scoped proxy for sys
        proxy = _CallerScopedModuleProxy(
            module_name="sys",
            real_module=_real_sys,
            always_allowed=_SAFE_ATTRS,
            always_blocked=_BLOCKED_ATTRS,
            trusted_prefixes=tuple(trusted_prefixes),
            custom_attrs=custom_attrs,
        )

        # Put the caller-scoped proxy where 'import sys' will find it
        _real_sys.modules["sys"] = proxy

    def _install_safe_os(self) -> None:
        """Replace os in sys.modules with a safe proxy exposing only benign attrs."""
        _real_os = self._real_os
        types = self._types_module

        safe_os = types.ModuleType("os")
        safe_os.__doc__ = "Safe os proxy provided by pyddock."
        safe_os._pyddock_safe = True

        # Safe constants and attributes
        _safe_attrs = [
            "name", "sep", "altsep", "extsep", "pathsep", "linesep",
            "curdir", "pardir", "devnull",
            "cpu_count", "PathLike", "fspath",
        ]
        for attr in _safe_attrs:
            if hasattr(_real_os, attr):
                setattr(safe_os, attr, getattr(_real_os, attr))

        # Safe functions (read-only operations)
        _safe_funcs = [
            "getcwd", "getpid", "getlogin",
            "getenv",
            "listdir", "scandir", "walk",
            "stat", "lstat", "fstat",
            "path",  # os.path submodule (pure path manipulation)
        ]
        for attr in _safe_funcs:
            if hasattr(_real_os, attr):
                setattr(safe_os, attr, getattr(_real_os, attr))

        # os.environ — expose as read-only (MappingProxyType)
        MappingProxyType = self._types_module.MappingProxyType
        safe_os.environ = MappingProxyType(dict(_real_os.environ))

        # Workspace-scoped directory operations are patched in apply_filesystem_scoping()
        # where the full _check_write logic (including .pyddock/ and workspace module
        # protection) is available. Here we just expose stubs that will be replaced.

        # Put the safe proxy where 'import os' will find it
        sys.modules["os"] = safe_os
        # Also update os.path reference to use the real os.path
        sys.modules["os.path"] = _real_os.path

    def apply_filesystem_scoping(self) -> None:
        """Patch filesystem operations to enforce path restrictions.

        Patches:
        - builtins.open (read and write modes)
        - pathlib.Path.write_text, write_bytes, open (write modes)
        - pathlib.Path.read_text, read_bytes
        """
        fs_config = self._config.get("filesystem", {})
        writable_paths = fs_config.get("writable_paths", ["."])
        readable_paths = fs_config.get("readable_paths", ["."])
        workspace_root = self._workspace_root
        _real_os = self._real_os

        def _abspath(p: pathlib.Path) -> pathlib.Path:
            """Normalize path without resolving symlinks/subst drives."""
            return pathlib.Path(_real_os.path.abspath(str(p)))

        # Resolve allowed paths to absolute (preserving symlinks)
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

        # Resolve workspace module directories (write-protected)
        workspace_imports = self._config.get("imports", {}).get("workspace", {})
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
        shell_config = self._config.get("shell", {})
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
        _guard_trusted_prefixes = self._trusted_prefixes

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

        def _check_read(path: pathlib.Path) -> None:
            """Raise PermissionError if path is outside readable scope."""
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
                resolved.relative_to(pathlib.Path(_PYDDOCK_DIR))
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

        def _blocked_chmod(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                f"PermissionError: Path.chmod() is not permitted. "
                f"Permission changes are disabled."
            )

        def _blocked_lchmod(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                f"PermissionError: Path.lchmod() is not permitted. "
                f"Permission changes are disabled."
            )

        pathlib.Path.symlink_to = _blocked_symlink_to
        pathlib.Path.hardlink_to = _blocked_hardlink_to
        if hasattr(pathlib.Path, "link_to"):
            pathlib.Path.link_to = _blocked_link_to
        pathlib.Path.chmod = _blocked_chmod
        if hasattr(pathlib.Path, "lchmod"):
            pathlib.Path.lchmod = _blocked_lchmod

        # Patch io.open — it's a separate function from builtins.open
        # and can bypass filesystem scoping if not patched.
        _io_module = self._io_module
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
        _real_os = self._real_os

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

    def apply_restrictions(self) -> None:
        """Apply module-level and class-level restrictions.

        For mode="deny":
          - module_allow: install a _CallerScopedModuleProxy (simple mode) that
            exposes only attributes matching allow patterns. The real module is
            NOT mutated.
          - class_allow: wrap allowed callable attrs with FactoryProxy and store
            them in custom_attrs on the proxy, so objects they return are
            restricted to class_allow patterns.

        For mode="allow":
          - module_deny: block module-level attrs matching deny patterns
          - class_deny: patch methods matching deny patterns on all public classes
        """
        restrictions = self._config.get("restrictions", {})
        allowed_imports = set(self._config.get("imports", {}).get("allowed", []))

        for module_name, restriction in restrictions.items():
            # Skip restrictions for modules that aren't in [imports] —
            # the agent can't import them anyway, and importing here can
            # trigger side effects (e.g. attrgetter guard conflicts).
            if module_name not in allowed_imports:
                continue

            mode = restriction.get("mode", "allow")
            module_allow = restriction.get("module_allow", [])
            module_deny = restriction.get("module_deny", [])
            class_allow = restriction.get("class_allow", [])
            class_deny = restriction.get("class_deny", [])

            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            # --- Module-level enforcement ---
            if mode == "allow" and module_deny:
                # Block matching patterns on the module
                deny_compiled = [re.compile(p) for p in module_deny]
                for attr_name in list(dir(module)):
                    if attr_name.startswith("_"):
                        continue
                    if any(p.match(attr_name) for p in deny_compiled):
                        self._patch_module_function(module, attr_name, module_name)
            elif mode == "deny" and module_allow:
                # Install a simple-mode _CallerScopedModuleProxy.
                # The proxy exposes only attributes matching module_allow patterns.
                # The real module is NOT mutated — FactoryProxy instances (if any)
                # live in custom_attrs on the proxy.
                always_allowed = _expand_patterns(module_allow, module)
                custom_attrs: dict[str, Any] = {}

                if class_allow:
                    # For each allowed callable attr, wrap it in FactoryProxy
                    # so objects it returns are restricted to class_allow patterns.
                    compiled_class_patterns = [re.compile(p) for p in class_allow]
                    for attr_name in always_allowed:
                        attr = getattr(module, attr_name, None)
                        if attr is not None and callable(attr) and not isinstance(attr, type):
                            custom_attrs[attr_name] = FactoryProxy(
                                attr, compiled_class_patterns
                            )

                proxy = _CallerScopedModuleProxy(
                    module_name=module_name,
                    real_module=module,
                    always_allowed=always_allowed,
                    always_blocked=frozenset(),
                    trusted_prefixes=None,  # simple mode
                    custom_attrs=custom_attrs if custom_attrs else None,
                )
                sys.modules[module_name] = proxy

            # --- Class-level enforcement ---
            if mode == "allow" and class_deny:
                # Patch methods on all public classes matching deny patterns
                deny_compiled = [re.compile(p) for p in class_deny]
                for cls_name in dir(module):
                    if cls_name.startswith("_"):
                        continue
                    cls = getattr(module, cls_name, None)
                    if cls is None or not isinstance(cls, type):
                        continue
                    for method_name in list(dir(cls)):
                        if method_name.startswith("_"):
                            continue
                        if any(p.match(method_name) for p in deny_compiled):
                            self._patch_class_method(cls, method_name, module_name)

    @staticmethod
    def _patch_module_function(
        module: Any, func_name: str, module_name: str
    ) -> None:
        """Replace a module-level function with one that raises PermissionError."""

        def _blocked(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                f"PermissionError: '{func_name}' is not permitted on {module_name}. "
                f"Please rewrite your snippet to avoid this function."
            )

        setattr(module, func_name, _blocked)

    @staticmethod
    def _patch_class_method(
        cls: Any, method_name: str, module_name: str
    ) -> None:
        """Replace a class method with one that raises PermissionError."""

        def _blocked(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                f"PermissionError: '{method_name}' is not permitted on {module_name}. "
                f"Please rewrite your snippet to avoid this function."
            )

        setattr(cls, method_name, _blocked)

    def apply_subprocess_patch(self) -> None:
        """Replace subprocess with a safe proxy module and block os.system.

        Instead of patching individual functions (whack-a-mole), we replace
        the entire subprocess module in sys.modules with a proxy that only
        exposes subprocess.run() and subprocess.Popen() — both validated
        against shell policies.

        Exposed on the proxy:
        - subprocess.run() — validated against shell policies
        - subprocess.Popen() — validated proxy class (same policy checks at construction)
        - subprocess.PIPE, DEVNULL, STDOUT — constants for run()/Popen() calls
        - subprocess.CompletedProcess — return type
        - subprocess.CalledProcessError, TimeoutExpired, SubprocessError — exceptions

        NOT exposed (no bypass surface):
        - call, check_call, check_output, getoutput, getstatusoutput
        """
        types = self._types_module

        _real_os = self._real_os
        _resolve_cmd = self._resolve_command
        shell_policies = self._config.get("shell", {})

        # Build example command for error messages
        if shell_policies:
            first_name = next(iter(shell_policies))
            first_policy = shell_policies[first_name]
            example_cmd = first_policy.get("command", first_name).lstrip("^").rstrip("$")
            allowed_commands_str = ", ".join(
                p.get("command", name) for name, p in shell_policies.items()
            )
        else:
            example_cmd = "command"
            allowed_commands_str = "(none configured)"

        def _find_matching_policy(command: str) -> dict | None:
            """Find first matching shell policy for a command."""
            for _name, policy in shell_policies.items():
                if re.match(policy["command"], command):
                    return policy
            return None

        def _check_args_policy(policy: dict, cmd_args: list[str]) -> str | None:
            """Validate args against policy. Returns error message or None."""
            args_str = " ".join(cmd_args)
            mode = policy.get("mode", "deny")

            if mode == "deny":
                allow_patterns = policy.get("allow", [])
                if not allow_patterns:
                    return "No argument patterns are allowed for this command."
                if not any(re.match(p, args_str) for p in allow_patterns):
                    allowed = ", ".join(allow_patterns)
                    return (
                        f"Arguments '{args_str}' not permitted. "
                        f"Allowed patterns: {allowed}"
                    )
                return None
            elif mode == "allow":
                deny_patterns = policy.get("deny", [])
                for pattern in deny_patterns:
                    if re.match(pattern, args_str):
                        return (
                            f"Arguments '{args_str}' matched deny pattern '{pattern}'."
                        )
                return None
            return None

        # Pre-compute protected paths for arg scanning
        _ws_root = self._workspace_root
        _pyddock_dir = pathlib.Path(_real_os.path.abspath(str(_ws_root / ".pyddock")))
        _workspace_imports = self._config.get("imports", {}).get("workspace", {})
        _ws_module_dirs: list[tuple[str, pathlib.Path]] = [
            (mod_name, pathlib.Path(_real_os.path.abspath(str(_ws_root / rel_path))))
            for mod_name, rel_path in _workspace_imports.items()
        ]
        _shell_protected_dirs: list[tuple[str, pathlib.Path]] = []
        for _sp_name, _sp_policy in shell_policies.items():
            _sp_cmd = _sp_policy.get("command", "")
            if "/" in _sp_cmd or "\\\\" in _sp_cmd or _sp_cmd.startswith("\\."):
                _sp_pattern = _sp_cmd.lstrip("^").rstrip("$")
                if "/" in _sp_pattern:
                    _sp_dir = _sp_pattern.rsplit("/", 1)[0]
                elif "\\\\" in _sp_pattern:
                    _sp_dir = _sp_pattern.rsplit("\\\\", 1)[0]
                else:
                    _sp_dir = _sp_pattern
                if _sp_dir:
                    _sp_clean = _sp_dir.replace("\\.", ".").replace("\\/", "/")
                    _shell_protected_dirs.append(
                        (_sp_clean, pathlib.Path(_real_os.path.abspath(str(_ws_root / _sp_clean))))
                    )
        _ws_root_abs = pathlib.Path(_real_os.path.abspath(str(_ws_root)))

        _looks_like_path_rt = self._looks_like_path
        _extract_path_candidates_rt = self._extract_path_candidates

        def _check_arg_paths(policy: dict, cmd_args: list[str]) -> str | None:
            """Scan args for path-like values and validate against arg_paths policy."""
            arg_paths_mode = policy.get("arg_paths", "workspace")
            if arg_paths_mode == "none":
                return None

            for arg in cmd_args:
                # Extract all path candidates (raw arg + embedded --flag=value)
                candidates = _extract_path_candidates_rt(arg)
                if not candidates:
                    continue

                for candidate in candidates:
                    resolved = pathlib.Path(_real_os.path.abspath(
                        str(_ws_root / candidate)
                    ))

                    # Check .pyddock/ (excluding .pyddock/tmp/)
                    try:
                        rel = resolved.relative_to(_pyddock_dir)
                        if not str(rel).startswith("tmp"):
                            return (
                                f"Argument '{arg}' targets the protected .pyddock/ "
                                f"directory. Shell commands cannot write to .pyddock/ "
                                f"(self-modification protection)."
                            )
                    except ValueError:
                        pass

                    # Check workspace module directories
                    for mod_name, ws_dir in _ws_module_dirs:
                        try:
                            resolved.relative_to(ws_dir)
                            return (
                                f"Argument '{arg}' targets workspace module directory "
                                f"'{mod_name}'. Shell commands cannot write to workspace "
                                f"module directories."
                            )
                        except ValueError:
                            continue

                    # Check shell script directories
                    for dir_label, script_dir in _shell_protected_dirs:
                        try:
                            resolved.relative_to(script_dir)
                            return (
                                f"Argument '{arg}' targets a shell-executable script "
                                f"directory ({dir_label}). Shell commands cannot write "
                                f"to script directories (write-then-execute prevention)."
                            )
                        except ValueError:
                            continue

                    # "workspace" mode: block paths outside the workspace
                    if arg_paths_mode == "workspace":
                        try:
                            resolved.relative_to(_ws_root_abs)
                        except ValueError:
                            return (
                                f"Argument '{arg}' resolves outside the workspace. "
                                f"Shell commands are restricted to workspace-relative "
                                f"paths (arg_paths = \"workspace\")."
                            )

            return None

        def _validated_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            """subprocess.run replacement that validates against shell policies."""
            # Reject shell=True
            if kwargs.get("shell", False):
                raise PermissionError(
                    "PermissionError: shell=True is not permitted in subprocess.run(). "
                    "Pass command as a list instead: "
                    f"subprocess.run(['{example_cmd}', 'arg1', 'arg2'])"
                )
            # Reject string commands
            if isinstance(cmd, str):
                raise PermissionError(
                    "PermissionError: String commands are not permitted in subprocess.run(). "
                    "Pass command as a list instead: "
                    f"subprocess.run(['{example_cmd}', 'arg1', 'arg2'])"
                )
            # If no shell policies configured, block entirely
            if not shell_policies:
                raise PermissionError(
                    "PermissionError: No shell policies configured. "
                    "Add [shell.*] sections to pyddock.toml to enable command execution, "
                    "or use run_shell directly."
                )
            # Validate command against shell policy
            if not cmd:
                raise PermissionError(
                    "PermissionError: Empty command list is not permitted."
                )
            command = str(cmd[0])
            cmd_args = [str(a) for a in cmd[1:]]
            policy = _find_matching_policy(command)
            if policy is None:
                raise PermissionError(
                    f"PermissionError: Command '{command}' is not permitted. "
                    f"No matching shell policy found. "
                    f"Allowed commands: {allowed_commands_str}"
                )
            rejection = _check_args_policy(policy, cmd_args)
            if rejection is not None:
                raise PermissionError(f"PermissionError: {rejection}")
            # Check arg paths against protected directories
            path_rejection = _check_arg_paths(policy, cmd_args)
            if path_rejection is not None:
                raise PermissionError(f"PermissionError: {path_rejection}")
            # Apply interpreter mapping (same as run_shell) and execute
            resolved = _resolve_cmd(command)
            full_cmd = resolved + cmd_args
            kwargs["shell"] = False
            return _ORIGINALS["subprocess.run"](full_cmd, *args, **kwargs)

        def _validate_command(cmd: Any, caller: str) -> tuple[list[str], list[str]]:
            """Shared validation for run() and Popen(). Returns (resolved_cmd, cmd_args).

            Raises PermissionError if the command is not permitted.
            """
            # Reject shell=True handled by caller (kwargs not passed here)
            # Reject string commands
            if isinstance(cmd, str):
                raise PermissionError(
                    f"PermissionError: String commands are not permitted in subprocess.{caller}(). "
                    "Pass command as a list instead: "
                    f"subprocess.{caller}(['{example_cmd}', 'arg1', 'arg2'])"
                )
            # If no shell policies configured, block entirely
            if not shell_policies:
                raise PermissionError(
                    "PermissionError: No shell policies configured. "
                    "Add [shell.*] sections to pyddock.toml to enable command execution, "
                    "or use run_shell directly."
                )
            if not cmd:
                raise PermissionError(
                    "PermissionError: Empty command list is not permitted."
                )
            command = str(cmd[0])
            cmd_args = [str(a) for a in cmd[1:]]
            policy = _find_matching_policy(command)
            if policy is None:
                raise PermissionError(
                    f"PermissionError: Command '{command}' is not permitted. "
                    f"No matching shell policy found. "
                    f"Allowed commands: {allowed_commands_str}"
                )
            rejection = _check_args_policy(policy, cmd_args)
            if rejection is not None:
                raise PermissionError(f"PermissionError: {rejection}")
            path_rejection = _check_arg_paths(policy, cmd_args)
            if path_rejection is not None:
                raise PermissionError(f"PermissionError: {path_rejection}")
            resolved = _resolve_cmd(command)
            return resolved + cmd_args, cmd_args

        class _SafePopen:
            """Proxy around subprocess.Popen that validates commands against shell policies.

            Validates the command at construction time (same checks as subprocess.run),
            then delegates all safe operations to the real Popen instance.
            """

            def __init__(self, cmd: Any, *args: Any, **kwargs: Any) -> None:
                if kwargs.get("shell", False):
                    raise PermissionError(
                        "PermissionError: shell=True is not permitted in subprocess.Popen(). "
                        "Pass command as a list instead: "
                        f"subprocess.Popen(['{example_cmd}', 'arg1', 'arg2'])"
                    )
                full_cmd, _ = _validate_command(cmd, "Popen")
                kwargs["shell"] = False
                self._proc = _ORIGINALS["subprocess.Popen"](full_cmd, *args, **kwargs)

            # --- Process control ---
            def communicate(self, *args: Any, **kwargs: Any) -> tuple:
                return self._proc.communicate(*args, **kwargs)

            def wait(self, *args: Any, **kwargs: Any) -> int:
                return self._proc.wait(*args, **kwargs)

            def poll(self) -> int | None:
                return self._proc.poll()

            def terminate(self) -> None:
                return self._proc.terminate()

            def kill(self) -> None:
                return self._proc.kill()

            def send_signal(self, signal: int) -> None:
                return self._proc.send_signal(signal)

            # --- Properties ---
            @property
            def stdout(self) -> Any:
                return self._proc.stdout

            @property
            def stderr(self) -> Any:
                return self._proc.stderr

            @property
            def stdin(self) -> Any:
                return self._proc.stdin

            @property
            def pid(self) -> int:
                return self._proc.pid

            @property
            def returncode(self) -> int | None:
                return self._proc.returncode

            @property
            def args(self) -> Any:
                return self._proc.args

            # --- Context manager ---
            def __enter__(self) -> "_SafePopen":
                return self

            def __exit__(self, *args: Any) -> None:
                self._proc.__exit__(*args)

            def __repr__(self) -> str:
                return f"<SafePopen pid={self.pid} returncode={self.returncode}>"

        # Build the safe subprocess proxy
        _subprocess_module = self._subprocess_module
        if _subprocess_module is not None:
            _ORIGINALS["subprocess.run"] = _subprocess_module.run
            _ORIGINALS["subprocess.Popen"] = _subprocess_module.Popen

            safe_subprocess = types.ModuleType("subprocess")
            safe_subprocess.__doc__ = "Safe subprocess proxy provided by pyddock. Only subprocess.run() and subprocess.Popen() are available, validated against shell policies."

            # Allowed entry points
            safe_subprocess.run = _validated_run
            safe_subprocess.Popen = _SafePopen

            # Constants needed for run()/Popen() calls
            safe_subprocess.PIPE = _subprocess_module.PIPE
            safe_subprocess.DEVNULL = _subprocess_module.DEVNULL
            safe_subprocess.STDOUT = _subprocess_module.STDOUT

            # Types needed for return values and error handling
            safe_subprocess.CompletedProcess = _subprocess_module.CompletedProcess
            safe_subprocess.CalledProcessError = _subprocess_module.CalledProcessError
            safe_subprocess.TimeoutExpired = _subprocess_module.TimeoutExpired
            safe_subprocess.SubprocessError = _subprocess_module.SubprocessError

            # Replace in sys.modules so 'import subprocess' finds the proxy
            sys.modules["subprocess"] = safe_subprocess

        # Always patch os.system
        def _blocked_os_system(cmd: Any) -> None:
            raise PermissionError(
                "PermissionError: os.system() is not available. "
                f"Use subprocess.run(['{example_cmd}', 'arg1', 'arg2']) instead, "
                "which validates commands against the shell policy."
            )

        # Patch os.system on the real os module
        _real_os.system = _blocked_os_system

        # Also patch it on the safe os proxy if it exists in sys.modules
        if "os" in sys.modules:
            safe_os = sys.modules["os"]
            safe_os.system = _blocked_os_system
