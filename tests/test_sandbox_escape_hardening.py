"""Regression tests for sandbox-escape hardening.

Covers three fixes landed together:

1. Namespace isolation — the agent snippet runs in a dedicated module
   namespace, NOT the bootstrap module globals. This prevents the snippet from
   reaching enforcement internals (`_enforcement`, `_RE`) and the real
   (unproxied) `sys` module that the bootstrap imports.

2. Sanitized traceback / frame-attribute blocking — frame, traceback, and
   generator/coroutine frame attributes are rejected by AST validation and the
   runtime getattr guard, closing the
   `e.__traceback__.tb_frame.f_back.f_globals` stack-walk escape.

3. os.path wrapping — `os.path` is handed out as a caller-scoped proxy so
   `os.path.os` no longer leaks the real `os` module (and its unpatched
   low-level `os.open`/`os.write` primitives) to agent code.

The functional-preservation tests assert that namespace isolation did not
regress ordinary module-level execution semantics (comprehensions, class
bodies, nested/mutually-recursive functions, dataclasses, and
typing.get_type_hints forward-reference resolution).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock.ast_validator import ASTValidator
from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager

# Mirror the frame/traceback/generator attributes blocked in default_config.toml.
FRAME_ATTRS = [
    "f_globals",
    "f_locals",
    "f_back",
    "f_builtins",
    "tb_frame",
    "gi_frame",
    "cr_frame",
    "ag_frame",
]

DEFAULT_BLOCK_ATTRS = [
    "__subclasses__",
    "__globals__",
    "__code__",
    "__bases__",
    "__mro__",
    "__closure__",
    *FRAME_ATTRS,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PyddockConfig:
    """Config mirroring production defaults relevant to these tests."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(
            allowed=[
                "os",
                "sys",
                "threading",
                "io",
                "json",
                "math",
                "pathlib",
                "types",
                "typing",
                "dataclasses",
                "collections",
            ]
        ),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(
            block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
            block_attributes=list(DEFAULT_BLOCK_ATTRS),
        ),
        restrictions={},
    )


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    """VenvManager that uses the current interpreter (no real venv needed)."""
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


@pytest.fixture
def executor(config: PyddockConfig, venv_manager: VenvManager) -> SubprocessExecutor:
    return SubprocessExecutor(config, venv_manager)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _run(executor: SubprocessExecutor, workspace: Path, source: str):
    return executor.execute(source, args=[], timeout=30, workspace_root=workspace)


# ---------------------------------------------------------------------------
# 1. Namespace isolation
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    """Enforcement internals must not be visible in the snippet namespace."""

    def test_enforcement_object_not_visible(self, executor, workspace) -> None:
        result = _run(executor, workspace, "_enforcement")
        assert result.exit_code != 0
        assert "NameError" in result.stderr

    def test_runtime_enforcement_class_not_visible(self, executor, workspace) -> None:
        result = _run(executor, workspace, "_RE")
        assert result.exit_code != 0
        assert "NameError" in result.stderr

    def test_bare_sys_not_preloaded(self, executor, workspace) -> None:
        # The bootstrap imports the real `sys`; that name must not leak into the
        # snippet namespace. A snippet must `import sys` itself (and get the proxy).
        result = _run(executor, workspace, "sys")
        assert result.exit_code != 0
        assert "NameError" in result.stderr

    def test_imported_sys_is_proxied(self, executor, workspace) -> None:
        # After importing, the proxy must still block sys.modules (the gateway to
        # real module references / _ORIGINALS).
        result = _run(executor, workspace, "import sys\nsys.modules")
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr

    def test_dunder_name_is_main(self, executor, workspace) -> None:
        result = _run(executor, workspace, "__name__")
        assert result.exit_code == 0
        assert result.result == "'__main__'"

    def test_no_real_os_via_enforcement(self, executor, workspace) -> None:
        # The historical escape: _enforcement._real_os.open(...). Must NameError
        # on _enforcement before anything else.
        src = "fd = _enforcement._real_os.open('pwned.txt', 1)"
        result = _run(executor, workspace, src)
        assert result.exit_code != 0
        assert "NameError" in result.stderr


# ---------------------------------------------------------------------------
# 2. Frame / traceback attribute blocking (AST + runtime)
# ---------------------------------------------------------------------------


class TestFrameAttributeBlocking:
    """Frame/traceback attributes are rejected statically and dynamically."""

    @pytest.fixture
    def validator(self, config: PyddockConfig) -> ASTValidator:
        return ASTValidator(config)

    @pytest.mark.parametrize("attr", FRAME_ATTRS)
    def test_ast_blocks_frame_attribute(self, validator: ASTValidator, attr: str) -> None:
        violations = validator.validate(f"x.{attr}")
        assert any(v.kind == "blocked_attribute" and v.name == attr for v in violations)

    def test_ast_blocks_traceback_stack_walk(self, validator: ASTValidator) -> None:
        source = (
            "try:\n"
            "    raise ValueError()\n"
            "except ValueError as e:\n"
            "    g = e.__traceback__.tb_frame.f_back.f_globals\n"
        )
        violations = validator.validate(source)
        blocked = {v.name for v in violations if v.kind == "blocked_attribute"}
        # The dangerous links in the chain must be flagged.
        assert "tb_frame" in blocked
        assert "f_back" in blocked
        assert "f_globals" in blocked

    def test_runtime_getattr_guard_blocks_frame_attr(self, executor, workspace) -> None:
        # AST validation runs upstream in the server; the runtime getattr guard
        # is the backstop for dynamic access. It must reject the name regardless
        # of whether the object actually has it.
        result = _run(executor, workspace, "getattr(object(), 'f_globals')")
        assert result.exit_code != 0
        assert "PermissionError" in result.stderr


# ---------------------------------------------------------------------------
# 3. os.path wrapping
# ---------------------------------------------------------------------------


class TestOsPathWrapping:
    """os.path must expose the path API but not leak the real os module."""

    def test_path_api_still_works(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nos.path.join('a', 'b')")
        assert result.exit_code == 0
        assert result.result in ("'a\\\\b'", "'a/b'")

    def test_path_basename_dirname(self, executor, workspace) -> None:
        result = _run(
            executor, workspace, "import os\nos.path.basename(os.path.dirname('a/b/c'))"
        )
        assert result.exit_code == 0
        assert result.result == "'b'"

    def test_os_path_os_attribute_blocked(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nos.path.os")
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr

    def test_os_path_genericpath_blocked(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nos.path.genericpath")
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr

    def test_write_escape_via_os_path_is_closed(self, executor, workspace) -> None:
        # The original proof-of-exploit path: os.path.os -> real os -> os.open.
        src = (
            "import os\n"
            "real = os.path.os\n"
            "fd = real.open('pwned.txt', real.O_WRONLY | real.O_CREAT)\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr
        assert not (workspace / "pwned.txt").exists()


# ---------------------------------------------------------------------------
# Functional preservation — namespace isolation must not regress semantics
# ---------------------------------------------------------------------------


class TestFunctionalPreservation:
    """Module-level execution semantics must be identical to before isolation."""

    def test_comprehension_reads_toplevel_var(self, executor, workspace) -> None:
        result = _run(executor, workspace, "factor = 10\n[i * factor for i in range(3)]")
        assert result.exit_code == 0, result.stderr
        assert result.result == "[0, 10, 20]"

    def test_class_body_and_method_read_globals(self, executor, workspace) -> None:
        src = (
            "helper = 5\n"
            "class C:\n"
            "    val = helper\n"
            "    def m(self):\n"
            "        return [helper + i for i in range(2)]\n"
            "(C.val, C().m())"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "(5, [5, 6])"

    def test_mutual_recursion(self, executor, workspace) -> None:
        src = (
            "def a(n):\n"
            "    return 1 if n <= 0 else b(n - 1)\n"
            "def b(n):\n"
            "    return a(n - 1)\n"
            "a(4)"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "1"

    def test_dataclass_fields(self, executor, workspace) -> None:
        src = (
            "import dataclasses\n"
            "@dataclasses.dataclass\n"
            "class Pt:\n"
            "    x: int\n"
            "    y: int\n"
            "[f.name for f in dataclasses.fields(Pt)]"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "['x', 'y']"

    def test_get_type_hints_resolves_forward_refs(self, executor, workspace) -> None:
        # The string forward-reference "Node" can only be resolved if the snippet
        # namespace is registered as __main__ in sys.modules — get_type_hints
        # looks up sys.modules[cls.__module__].__dict__ for the globalns. This is
        # the regression guard for the ModuleType-backed namespace fix.
        src = (
            "import dataclasses, typing\n"
            "@dataclasses.dataclass\n"
            "class Node:\n"
            "    value: int\n"
            '    nxt: "Node"\n'
            "hints = typing.get_type_hints(Node)\n"
            "sorted(hints.keys())"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "['nxt', 'value']"

    def test_multi_statement_with_result(self, executor, workspace) -> None:
        result = _run(executor, workspace, "x = 10\ny = 20\nx + y")
        assert result.exit_code == 0, result.stderr
        assert result.result == "30"


# ---------------------------------------------------------------------------
# 4. Module-bound builtin (__self__) leak
# ---------------------------------------------------------------------------


class TestModuleBoundBuiltinLeak:
    """A bound C builtin (os.getcwd, sys.getrecursionlimit, io.open_code) must
    not hand agent code its defining module via ``func.__self__``.

    The safe os/sys proxies copied real builtins verbatim; `func.__self__` then
    pointed back at the real `nt`/`sys` module — a one-step escape to unpatched
    `nt.open`/`nt.write` (file writes anywhere, incl. the protected `.pyddock/`)
    or, for sys, `sys.modules` -> `pyddock._base._ORIGINALS['builtins.open']`.
    The fix wraps such builtins in a plain Python function (no `__self__`).
    """

    # (module, attribute) pairs that previously leaked their real module.
    LEAKY_BUILTINS = [
        ("os", "getcwd"),
        ("os", "urandom"),
        ("os", "listdir"),
        ("sys", "getrecursionlimit"),
        ("sys", "getdefaultencoding"),
        ("sys", "getfilesystemencoding"),
        ("io", "open_code"),
    ]

    @pytest.mark.parametrize("module, attr", LEAKY_BUILTINS)
    def test_self_does_not_expose_module(self, executor, workspace, module, attr) -> None:
        # The wrapped callable is an ordinary function — it has no __self__ at
        # all, so a literal access raises AttributeError and a hasattr is False.
        src = f"import {module}\nhasattr({module}.{attr}, '__self__')"
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "False"

    def test_sys_self_escape_to_sys_modules_is_closed(self, executor, workspace) -> None:
        # Full historical escape chain: sys.getrecursionlimit.__self__ -> real
        # sys -> sys.modules -> pyddock internals / real nt module.
        src = "import sys\nreal = sys.getrecursionlimit.__self__\nreal.modules"
        result = _run(executor, workspace, src)
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr

    def test_write_escape_via_os_self_is_closed(self, executor, workspace) -> None:
        # The proof-of-exploit path: os.getcwd.__self__ -> real nt -> nt.open.
        src = (
            "import os\n"
            "real = os.getcwd.__self__\n"
            "fd = real.open('pwned.txt', real.O_WRONLY | real.O_CREAT)\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code != 0
        assert "AttributeError" in result.stderr
        assert not (workspace / "pwned.txt").exists()

    def test_closure_fallback_is_blocked(self, executor, workspace) -> None:
        # The wrapped builtin is captured in the wrapper's __closure__; that
        # attribute is in block_attributes, so the fallback extraction is also
        # rejected (statically by AST, and dynamically by the getattr guard).
        src = "import os\ngetattr(os.getcwd, '__closure__')"
        result = _run(executor, workspace, src)
        assert result.exit_code != 0
        assert "PermissionError" in result.stderr

    def test_no_module_bound_builtin_leaks_anywhere(self, executor, workspace) -> None:
        # Broad sweep: across every allowlisted module, no agent-reachable
        # callable may expose a module via __self__. This is the catch-all that
        # guards against a future proxy/refactor reopening the class of bug.
        src = (
            "import types as _t\n"
            "import os, sys, io, json, math, pathlib, types, typing, collections\n"
            "mods = {'os': os, 'sys': sys, 'io': io, 'json': json, 'math': math,\n"
            "        'pathlib': pathlib, 'types': types, 'typing': typing,\n"
            "        'collections': collections}\n"
            "leaks = []\n"
            "for _n, _m in mods.items():\n"
            "    for _a in dir(_m):\n"
            "        if _a.startswith('__'):\n"
            "            continue\n"
            "        try:\n"
            "            _v = getattr(_m, _a)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if not callable(_v):\n"
            "            continue\n"
            "        try:\n"
            "            _s = getattr(_v, '__self__', None)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if isinstance(_s, _t.ModuleType):\n"
            "            leaks.append(_n + '.' + _a)\n"
            "leaks\n"
        )
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "[]", f"module-bound builtin leaks: {result.result}"


class TestModuleBoundBuiltinPreserved:
    """Wrapping the builtins must not break their normal behaviour."""

    def test_os_getcwd_works(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nisinstance(os.getcwd(), str)")
        assert result.exit_code == 0, result.stderr
        assert result.result == "True"

    def test_os_listdir_works(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nisinstance(os.listdir('.'), list)")
        assert result.exit_code == 0, result.stderr
        assert result.result == "True"

    def test_os_urandom_works(self, executor, workspace) -> None:
        result = _run(executor, workspace, "import os\nlen(os.urandom(8))")
        assert result.exit_code == 0, result.stderr
        assert result.result == "8"

    def test_sys_getrecursionlimit_works(self, executor, workspace) -> None:
        result = _run(
            executor, workspace, "import sys\nisinstance(sys.getrecursionlimit(), int)"
        )
        assert result.exit_code == 0, result.stderr
        assert result.result == "True"

    def test_wrapped_builtin_keeps_name_and_module(self, executor, workspace) -> None:
        # Introspection still reports the original identity, not the wrapper's.
        src = "import os\n(os.getcwd.__name__, os.getcwd.__module__)"
        result = _run(executor, workspace, src)
        assert result.exit_code == 0, result.stderr
        assert result.result == "('getcwd', 'nt')" or result.result == "('getcwd', 'posix')"
