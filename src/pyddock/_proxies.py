"""Proxy classes for pyddock runtime enforcement.

Provides the proxy classes that enforce attribute-level access control on
restricted modules (e.g. P4, boto3, git). Also provides the caller-scoped
module proxy used for universal module protection.

All proxy state is stored in module-level dicts (_MFP_STATE, _FP_STATE,
_PROXY_STATE) rather than on instances, preventing agent code from extracting
internal state via object.__getattribute__.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types
from typing import Any

from pyddock._base import _find_deny_hint
from pyddock._import_hook import _caller_is_trusted

# ---------------------------------------------------------------------------
# MethodFilterProxy
# ---------------------------------------------------------------------------

# Internal state for MethodFilterProxy instances. Stored here (not on
# proxy instances) so agent code cannot extract it via object.__getattribute__.
# The pyddock._* modules are not importable by agent code.
_MFP_STATE: dict[int, tuple] = {}

# Dunders that are BLOCKED on MethodFilterProxy and FactoryProxy.
# These enable bypass of the allow-pattern filtering if passed through.
# All other dunders are allowed (needed for repr, str, iter, context managers, etc.).
_BLOCKED_DUNDERS = frozenset({
    "__getattribute__",  # wrapped.__getattribute__("run_submit") bypasses proxy
    "__getattr__",       # wrapped.__getattr__("run_submit") bypasses proxy
    "__dict__",          # exposes raw instance attributes
    "__reduce__",        # pickle protocol could serialize/deserialize unwrapped
    "__reduce_ex__",     # pickle protocol could serialize/deserialize unwrapped
})


class MethodFilterProxy:
    """Proxy that intercepts attribute access and blocks disallowed methods.

    Used by FactoryProxy to wrap objects returned by factory functions.
    Only methods matching at least one allow pattern are permitted.

    Internal state (wrapped object, patterns) is stored in the module-level
    _MFP_STATE dict, NOT on the instance. This prevents agent code from
    extracting the raw wrapped object via object.__getattribute__.
    """

    __slots__ = ()

    def __init__(
        self,
        wrapped: Any,
        allow_patterns: list[re.Pattern[str]],
        deny_messages: list[tuple[re.Pattern[str], str]] | None = None,
        module_name: str = "",
    ) -> None:
        _MFP_STATE[id(self)] = (wrapped, allow_patterns, deny_messages or [], module_name)

    def __del__(self) -> None:
        _MFP_STATE.pop(id(self), None)

    def __getattribute__(self, name: str) -> Any:
        state = _MFP_STATE.get(id(self))
        if state is None:
            raise RuntimeError("MethodFilterProxy: internal state missing")
        wrapped, allow_patterns, deny_messages, module_name = state

        if name.startswith("__") and name.endswith("__"):
            # Block dangerous dunders that enable proxy bypass
            if name in _BLOCKED_DUNDERS:
                raise PermissionError(
                    f"PermissionError: access to '{name}' is not permitted "
                    f"on restricted objects."
                )
            # Allow safe dunders for internal Python machinery (repr, str, etc.)
            return getattr(wrapped, name)

        if not any(p.match(name) for p in allow_patterns):
            patterns_str = ", ".join(p.pattern for p in allow_patterns)
            msg = (
                f"PermissionError: '{name}' is not permitted. "
                f"Allowed method patterns: {patterns_str}. "
                f"Please use one of the allowed methods instead."
            )
            attempted = f"{module_name}.{name}" if module_name else name
            hint = _find_deny_hint(attempted, deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)
        return getattr(wrapped, name)


# ---------------------------------------------------------------------------
# FactoryProxy
# ---------------------------------------------------------------------------

# Internal state for FactoryProxy instances. Stored here (not on proxy
# instances) so agent code cannot extract the original factory/class via
# proxy.__dict__["_original"] or object.__getattribute__(proxy, "_original").
_FP_STATE: dict[int, tuple] = {}


class FactoryProxy:
    """Wraps a factory function or class to return proxied objects.

    Objects returned by the factory (or class constructor) are wrapped in
    MethodFilterProxy, which enforces the allow-pattern list on method access.

    When wrapping a class, also supports:
    - __getattr__: Proxies class-level attribute access (e.g. classmethods)
      with the same allow-pattern filtering applied.
    - __instancecheck__: Supports isinstance() checks against the wrapped class.

    Internal state is stored in the module-level _FP_STATE dict to prevent
    agent code from accessing the original factory via __dict__ or
    object.__getattribute__.
    """

    __slots__ = ()

    def __init__(
        self,
        original_factory: Any,
        allow_patterns: list[re.Pattern[str]],
        deny_messages: list[tuple[re.Pattern[str], str]] | None = None,
        module_name: str = "",
    ) -> None:
        is_class = isinstance(original_factory, type)
        _FP_STATE[id(self)] = (original_factory, allow_patterns, deny_messages or [], module_name, is_class)

    def __del__(self) -> None:
        _FP_STATE.pop(id(self), None)

    def __call__(self, *args: Any, **kwargs: Any) -> MethodFilterProxy:
        state = _FP_STATE.get(id(self))
        if state is None:
            raise RuntimeError("FactoryProxy: internal state missing")
        original, allow_patterns, deny_messages, module_name, _is_class = state
        obj = original(*args, **kwargs)
        return MethodFilterProxy(obj, allow_patterns, deny_messages, module_name)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped class with filtering.

        Only applies when wrapping a class (for classmethods, constants, etc.).
        Applies the same class_allow patterns to prevent access to restricted
        methods via the class object itself.
        """
        state = _FP_STATE.get(id(self))
        if state is None:
            raise RuntimeError("FactoryProxy: internal state missing")
        original, allow_patterns, deny_messages, module_name, is_class = state

        if not is_class:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Block dangerous dunders that enable bypass
        if name.startswith("__") and name.endswith("__"):
            if name in _BLOCKED_DUNDERS:
                raise PermissionError(
                    f"PermissionError: access to '{name}' is not permitted "
                    f"on restricted objects."
                )
            # Allow safe dunders for Python machinery
            return getattr(original, name)

        # Apply the same allow-pattern filtering as MethodFilterProxy
        if not any(p.match(name) for p in allow_patterns):
            patterns_str = ", ".join(p.pattern for p in allow_patterns)
            msg = (
                f"PermissionError: '{name}' is not permitted. "
                f"Allowed method patterns: {patterns_str}. "
                f"Please use one of the allowed methods instead."
            )
            attempted = f"{module_name}.{name}" if module_name else name
            hint = _find_deny_hint(attempted, deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)

        return getattr(original, name)

    def __instancecheck__(self, instance: Any) -> bool:
        """Support isinstance() checks against the wrapped class."""
        state = _FP_STATE.get(id(self))
        if state is None:
            return NotImplemented
        original, _allow_patterns, _deny_messages, _module_name, is_class = state
        if is_class:
            # Unwrap MethodFilterProxy instances for isinstance checks
            if isinstance(instance, MethodFilterProxy):
                mfp_state = _MFP_STATE.get(id(instance))
                if mfp_state is not None:
                    wrapped = mfp_state[0]
                    return isinstance(wrapped, original)
            return isinstance(instance, original)
        return NotImplemented


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _expand_patterns(
    patterns: list[str], module: types.ModuleType
) -> frozenset[str]:
    """Pre-compute the set of attribute names matching any regex pattern.

    Evaluates each pattern against dir(module) at proxy creation time.
    This avoids regex evaluation on every attribute access.
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
    exclude_foreign_classes: bool,
    include_private: bool,
) -> frozenset[str]:
    """Determine which attributes constitute a module's public API.

    Algorithm:
    1. If module defines __all__, seed the result with those names.
    2. Otherwise (or additionally when include_private is True), include all
       attributes that are NOT instances of types.ModuleType (excludes
       re-exported imports like `os`, `sys`). Dunder names are always skipped
       here (module metadata dunders are attached separately by the proxy).
    3. If exclude_foreign_classes is True, also exclude classes whose
       __module__ belongs to a different top-level package.

    include_private:
        When False (default), single-underscore private names are excluded —
        agent code only sees the public API. Used for pre-enforcement modules
        (imported at startup before the os/sys proxies were installed) which
        may hold real os/sys references.
        When True, single-underscore private non-module attributes are also
        included. Used ONLY for post-enforcement modules (imported lazily by
        agent code after enforcement was active), whose internal references
        already resolve to the safe os/sys proxies and therefore cannot leak
        dangerous capabilities. This is required so native extensions (e.g.
        cryptography's Rust bindings) can read their module-private constants
        through the proxy via sys.modules.
    """
    module_name = getattr(module, "__name__", "") or ""
    top_level_pkg = module_name.split(".")[0]

    exported: set[str] = set()

    if hasattr(module, "__all__"):
        exported |= set(module.__all__)
        if not include_private:
            return frozenset(exported)

    for name in dir(module):
        # Dunder names are handled separately (proxy attaches metadata dunders).
        if name.startswith("__") and name.endswith("__"):
            continue
        if name.startswith("_") and not include_private:
            continue
        val = getattr(module, name, None)
        if isinstance(val, types.ModuleType):
            continue
        if exclude_foreign_classes and isinstance(val, type):
            cls_module = getattr(val, "__module__", "") or ""
            cls_top_level = cls_module.split(".")[0]
            if cls_module == "builtins":
                exported.add(name)
            elif cls_top_level == top_level_pkg:
                exported.add(name)
            continue
        exported.add(name)
    return frozenset(exported)


# ---------------------------------------------------------------------------
# _CallerScopedModuleProxy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trusted prefix computation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Universal module proxying
# ---------------------------------------------------------------------------

def _proxy_module_universal(
    name: str,
    trusted_prefixes: tuple[str, ...],
    skip_modules: frozenset[str],
    workspace_module_names: frozenset[str] = frozenset(),
    *,
    include_private: bool,
) -> None:
    """Wrap a module in a caller-scoped proxy (universal mode).

    Every allowed module gets caller-scoped mode. The always_allowed set
    is the module's exported API (non-ModuleType, non-private attrs).
    Trusted code (stdlib, site-packages, workspace) can access anything.
    Agent code can only access the exported API.

    include_private:
        When True, single-underscore private non-module attributes are also
        exposed to agent code. This is safe ONLY for post-enforcement modules
        (lazily imported after the os/sys proxies were installed) and is
        required so native extensions can read their module-private constants
        through the proxy. It is force-disabled for workspace modules, whose
        private attributes could hold foreign network-capable objects.
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
    # to prevent leakage of network-capable factories (e.g. Jira, SSHClient),
    # and private attributes are never exposed.
    is_workspace = top_level in workspace_module_names
    effective_private = include_private and not is_workspace
    always_allowed = _compute_exported_api(
        module,
        exclude_foreign_classes=is_workspace,
        include_private=effective_private,
    )

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
