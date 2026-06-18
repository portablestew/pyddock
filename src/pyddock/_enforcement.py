"""RuntimeEnforcement orchestrator.

This module contains the RuntimeEnforcement class which coordinates all
sandbox enforcement mechanisms. It delegates filesystem scoping and
subprocess patching to their respective standalone modules while keeping
the remaining enforcement logic (import hooks, module proxies, attribute
guards, and restriction application) in-class.
"""
from __future__ import annotations

import builtins
import importlib
import pathlib
import re
import sys
import types
from typing import Any

from pyddock._base import SNIPPET_FILENAME, _ORIGINALS, _find_deny_hint
from pyddock._import_hook import _ImportBlocker, _caller_is_trusted
from pyddock._proxies import (
    MethodFilterProxy, FactoryProxy, _CallerScopedModuleProxy,
    _expand_patterns, _build_trusted_prefixes, _proxy_module_universal,
    _compute_exported_api,
)
from pyddock._fs_enforcement import apply_filesystem_scoping
from pyddock._subprocess_patch import apply_subprocess_patch
from pyddock._library_guards import apply_library_guards
from pyddock._prewarm import run_all as _run_prewarms


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
        # Pre-compile deny_messages patterns from serialized config
        self._deny_messages: list[tuple[re.Pattern[str], str]] = []
        for entry in config.get("deny_messages", []):
            try:
                pattern = re.compile(entry["pattern"])
                self._deny_messages.append((pattern, entry["message"]))
            except (re.error, KeyError, TypeError):
                pass  # Skip malformed entries silently in subprocess

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
        apply_filesystem_scoping(
            config=self._config,
            workspace_root=self._workspace_root,
            real_os=self._real_os,
            trusted_prefixes=self._trusted_prefixes,
            io_module=self._io_module,
        )
        self.apply_restrictions()
        # Per-library enforcement guards (the rare escape hatch for libraries the
        # declarative imports/restrictions/shell tiers can't fully constrain).
        # Each guard self-gates on its import being allowlisted. Run before
        # install_module_proxies so target modules (e.g. git.cmd) are still real.
        # GitPython is the first such guard: it captured `from subprocess import
        # Popen` at import, so the subprocess proxy below can't see it — its guard
        # hooks git.cmd.Git.execute and validates against [shell.git] instead.
        apply_library_guards(
            config=self._config,
            deny_messages=self._deny_messages,
        )
        apply_subprocess_patch(
            config=self._config,
            workspace_root=self._workspace_root,
            real_os=self._real_os,
            subprocess_module=self._subprocess_module,
            types_module=self._types_module,
            resolve_command=self._resolve_command,
            looks_like_path=self._looks_like_path,
            extract_path_candidates=self._extract_path_candidates,
            deny_messages=self._deny_messages,
        )
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

        # Pre-warmed stdlib internals that are allowed to be re-imported from
        # sys.modules cache. These are lazily imported by frozen stdlib modules
        # (e.g. _strptime by datetime.strptime) and would be blocked by the
        # trusted-caller check due to frozen frame chain issues in Python 3.12+.
        _prewarmed_internals = _run_prewarms()

        blocker = _ImportBlocker(allowed, trusted_prefixes_tuple, self._deny_messages)
        # Insert at the beginning so it's checked first
        sys.meta_path.insert(0, blocker)

        # Patch builtins.__import__ to enforce the allowlist directly.
        # This closes the bypass where attacker accesses __import__ from
        # a function's __globals__['__builtins__'] dict.
        _ORIGINALS["import"] = builtins.__import__
        _trusted = trusted_prefixes_tuple
        _loading_depth = [0]  # reentrant counter: >0 while loading an allowed module
        # Names whose import is currently in-flight. Used to detect re-entrant
        # imports of the SAME module triggered by that module's own __init__.py
        # (e.g. `from cryptography.x509 import oid` executed while x509/__init__.py
        # is still running). Proxying a partially-initialized module would capture
        # an incomplete public API, so we only proxy on the OUTERMOST import.
        _loading_names: set[str] = set()
        # Modules that must NOT be wrapped by _proxy_module_universal:
        # os: plain types.ModuleType with safe attrs only (not a _CallerScopedModuleProxy)
        # sys: specialized caller-scoped proxy with custom_attrs
        # threading: _shutdown() called from C/frozen frames during interpreter exit
        # Restriction modules are NOT skipped — mode="deny" modules already have a
        # _CallerScopedModuleProxy (isinstance early-return), and mode="allow" modules
        # need caller-scoped proxying to block ModuleType attr leakage (e.g. polars.os).
        _skip_proxy = frozenset({"os", "sys", "threading"})
        _ws_module_names = frozenset(workspace_imports.keys())
        _deny_msgs = self._deny_messages

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
                # Is THIS call the outermost import of `name`? A nested
                # re-entrant import of the same name (from the module's own
                # __init__.py) must not proxy the partially-initialized module.
                is_outermost = name not in _loading_names
                if is_outermost:
                    _loading_names.add(name)
                try:
                    result = _ORIGINALS["import"](name, *args, **kwargs)
                finally:
                    _loading_depth[0] -= 1
                    if is_outermost:
                        _loading_names.discard(name)
                # Proxy ANY submodule import so agent code can't access
                # leaked imports (e.g. pathlib.os, tempfile.os, json.decoder.os).
                # Only proxy on the outermost import so the module's __init__.py
                # has finished and its full public API (__all__) is populated.
                # include_private=True: these modules were imported lazily AFTER
                # the os/sys proxies were installed, so their internal references
                # already resolve to safe proxies. Exposing their private
                # non-module attrs is safe and lets native extensions read their
                # module-private constants through the proxy via sys.modules.
                if "." in name and is_outermost:
                    _proxy_module_universal(
                        name, _trusted, _skip_proxy, _ws_module_names,
                        include_private=True,
                    )
                return result
            # If the module was explicitly pre-warmed as a stdlib internal
            # (e.g. _strptime), allow re-import from cache. These are lazily
            # imported by frozen stdlib modules and would be blocked by the
            # trusted-caller check due to frozen frame chain issues.
            if name in _prewarmed_internals and name in sys.modules:
                return _ORIGINALS["import"](name, *args, **kwargs)
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
            msg = (
                f"ImportError: '{name}' is not an allowed import. "
                f"Please use one of the following allowed imports "
                f"instead: {allowed_list}"
            )
            hint = _find_deny_hint(name, _deny_msgs)
            if hint:
                msg += f"\n[{hint}]"
            raise ImportError(msg)

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
                mod_name, self._trusted_prefixes, skip_modules, workspace_module_names,
                include_private=False,
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
            "getenv", "urandom",
            "listdir", "scandir", "walk",
            "stat", "lstat", "fstat",
        ]
        for attr in _safe_funcs:
            if hasattr(_real_os, attr):
                setattr(safe_os, attr, getattr(_real_os, attr))

        # os.environ — expose as read-only (MappingProxyType)
        MappingProxyType = self._types_module.MappingProxyType
        safe_os.environ = MappingProxyType(dict(_real_os.environ))

        # os.path — wrap in a caller-scoped proxy instead of handing out the raw
        # ntpath/posixpath module.
        #
        # SECURITY: the raw path module re-exports the modules it imports at
        # module scope (os.path.os, os.path.sys, os.path.genericpath, ...).
        # `os.path.os` therefore handed agent code the *real* os module — a
        # sandbox-escape vector to the unpatched low-level file primitives
        # (os.open/os.write). The caller-scoped proxy exposes only the path
        # manipulation API (join, exists, dirname, abspath, ...) to agent code;
        # the leaked sub-module references are non-ModuleType-filtered out of the
        # exported API and are only reachable from trusted (stdlib/site-packages)
        # callers, which legitimately need them.
        real_path = getattr(_real_os, "path", None)
        if real_path is not None:
            path_api = _compute_exported_api(
                real_path,
                exclude_foreign_classes=False,
                include_private=False,
            )
            path_proxy = _CallerScopedModuleProxy(
                module_name=getattr(real_path, "__name__", "os.path"),
                real_module=real_path,
                always_allowed=path_api,
                always_blocked=frozenset(),
                trusted_prefixes=tuple(self._trusted_prefixes),
            )
            setattr(safe_os, "path", path_proxy)
        else:
            path_proxy = None

        # Workspace-scoped directory operations are patched in apply_filesystem_scoping()
        # where the full _check_write logic (including .pyddock/ and workspace module
        # protection) is available. Here we just expose stubs that will be replaced.

        # Put the safe proxy where 'import os' will find it
        sys.modules["os"] = safe_os
        # Route os.path (and the underlying ntpath/posixpath name) through the
        # wrapped proxy so neither `import os; os.path.os` nor
        # `import os.path; os.path.os` leaks the real os module.
        if path_proxy is not None:
            sys.modules["os.path"] = path_proxy
            _real_path_name = getattr(real_path, "__name__", None)
            if _real_path_name:
                sys.modules[_real_path_name] = path_proxy
        else:
            sys.modules["os.path"] = _real_os.path

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
                    # Classes are wrapped too (they're callable constructors),
                    # except BaseException subclasses which are never factories.
                    compiled_class_patterns = [re.compile(p) for p in class_allow]
                    for attr_name in always_allowed:
                        attr = getattr(module, attr_name, None)
                        if attr is None or not callable(attr):
                            continue
                        if isinstance(attr, type) and issubclass(attr, BaseException):
                            continue
                        custom_attrs[attr_name] = FactoryProxy(
                            attr, compiled_class_patterns,
                            self._deny_messages, module_name,
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

    def _patch_module_function(
        self, module: Any, func_name: str, module_name: str
    ) -> None:
        """Replace a module-level function with one that raises PermissionError."""
        deny_messages = self._deny_messages

        def _blocked(*args: Any, **kwargs: Any) -> None:
            msg = (
                f"PermissionError: '{func_name}' is not permitted on {module_name}. "
                f"Please rewrite your snippet to avoid this function."
            )
            hint = _find_deny_hint(f"{module_name}.{func_name}", deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)

        setattr(module, func_name, _blocked)

    def _patch_class_method(
        self, cls: Any, method_name: str, module_name: str
    ) -> None:
        """Replace a class method with one that raises PermissionError."""
        deny_messages = self._deny_messages

        def _blocked(*args: Any, **kwargs: Any) -> None:
            msg = (
                f"PermissionError: '{method_name}' is not permitted on {module_name}. "
                f"Please rewrite your snippet to avoid this function."
            )
            hint = _find_deny_hint(f"{module_name}.{method_name}", deny_messages)
            if hint:
                msg += f"\n[{hint}]"
            raise PermissionError(msg)

        setattr(cls, method_name, _blocked)
