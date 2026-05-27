"""Backward-compatible re-export surface for pyddock._runtime.

External consumers (executor.py, tests) import from here.
All logic lives in the submodules:
  _base.py             — constants, _ORIGINALS, _find_deny_hint
  _import_hook.py      — _ImportBlocker, _caller_is_trusted
  _proxies.py          — MethodFilterProxy, FactoryProxy, _CallerScopedModuleProxy
  _fs_enforcement.py   — apply_filesystem_scoping
  _subprocess_patch.py — apply_subprocess_patch
  _enforcement.py      — RuntimeEnforcement
"""

from pyddock._base import (  # noqa: F401
    SNIPPET_FILENAME,
    _find_deny_hint,
    _ORIGINALS,
    _PYDDOCK_DIR,
    _normcase,
    _realpath,
)
from pyddock._import_hook import (  # noqa: F401
    _ImportBlocker,
    _caller_is_trusted,
    _is_infra_frame,
)
from pyddock._proxies import (  # noqa: F401
    MethodFilterProxy,
    FactoryProxy,
    _CallerScopedModuleProxy,
    _MFP_STATE,
    _FP_STATE,
    _PROXY_STATE,
    _BLOCKED_DUNDERS,
    _expand_patterns,
    _compute_exported_api,
    _build_trusted_prefixes,
    _proxy_module_universal,
)
from pyddock._enforcement import RuntimeEnforcement  # noqa: F401
