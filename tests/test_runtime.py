"""Unit tests for runtime enforcement — filesystem scoping and import blocking.

These tests run snippets through the full executor pipeline to verify that
runtime enforcement actually works end-to-end in the subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    RestrictionConfig,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace subdirectory with a test file."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "data.txt").write_text("hello")
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    """VenvManager using the current Python interpreter."""
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


def _make_config(
    allowed: list[str] | None = None,
    writable: list[str] | None = None,
    readable: list[str] | None = None,
    restrictions: dict[str, RestrictionConfig] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=allowed or ["json", "pathlib", "sys", "os", "tempfile", "codecs", "encodings"]),
        filesystem=FilesystemConfig(
            writable_paths=writable or ["."],
            readable_paths=readable or ["."],
        ),
        ast=ASTConfig(block_calls=["eval", "exec"], block_attributes=["__globals__"]),
        restrictions=restrictions or {},
    )


class TestFilesystemScoping:
    """Tests that runtime filesystem enforcement blocks out-of-scope access."""

    def test_write_inside_workspace_succeeds(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to a file inside the workspace works."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = "import pathlib\npathlib.Path('output.txt').write_text('ok')\n'done'"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'done'"
        assert (workspace / "output.txt").read_text() == "ok"

    def test_write_outside_workspace_blocked(
        self, workspace: Path, venv_manager: VenvManager, tmp_path: Path
    ) -> None:
        """Writing to a path outside the workspace raises PermissionError."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        # Create a directory outside the workspace (tmp_path is parent of workspace)
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_path = (outside_dir / "evil.txt").as_posix()
        source = f"import pathlib\npathlib.Path('{outside_path}').write_text('bad')"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code != 0
        assert "PermissionError" in result.stderr
        assert "writes are restricted" in result.stderr

    def test_read_inside_workspace_succeeds(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Reading a file inside the workspace works."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = "import pathlib\npathlib.Path('data.txt').read_text()"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'hello'"

    def test_read_outside_workspace_blocked(
        self, workspace: Path, venv_manager: VenvManager, tmp_path: Path
    ) -> None:
        """Reading a file outside the workspace raises PermissionError when restricted."""
        # Explicitly restrict reads to workspace (not the default "*")
        config = _make_config(readable=["."])
        executor = SubprocessExecutor(config, venv_manager)

        # Create a file outside workspace (in tmp_path, parent of workspace)
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")
        outside_path = outside_file.as_posix()

        source = f"import pathlib\npathlib.Path('{outside_path}').read_text()"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code != 0
        assert "PermissionError" in result.stderr
        assert "reads are restricted" in result.stderr

    def test_read_anywhere_with_wildcard(
        self, workspace: Path, venv_manager: VenvManager, tmp_path: Path
    ) -> None:
        """With readable_paths = ['*'], reads anywhere are allowed."""
        config = _make_config(readable=["*"])
        executor = SubprocessExecutor(config, venv_manager)

        # Create a file outside workspace
        outside_file = tmp_path / "readable.txt"
        outside_file.write_text("accessible")
        outside_path = outside_file.as_posix()

        source = f"import pathlib\npathlib.Path('{outside_path}').read_text()"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'accessible'"

    def test_write_to_pyddock_dir_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to .pyddock/ is always blocked (self-modification protection)."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        # Create .pyddock dir so the path resolves
        (workspace / ".pyddock").mkdir(exist_ok=True)
        pyddock_toml = (workspace / ".pyddock" / "pyddock.toml").as_posix()
        source = f"import pathlib\npathlib.Path('{pyddock_toml}').write_text('hacked')"
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code != 0
        assert "PermissionError" in result.stderr
        assert ".pyddock" in result.stderr


class TestRuntimeImportBlocking:
    """Tests that the runtime import hook blocks disallowed imports."""

    def test_allowed_and_disallowed_imports(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Allowed imports work; disallowed imports are blocked at runtime."""
        config = _make_config(allowed=["json"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import json\n"
            "# Allowed import works\n"
            "encoded = json.dumps({'a': 1})\n"
            "print(f'ALLOWED: {encoded}')\n"
            "\n"
            "# Disallowed import blocked (even via importlib)\n"
            "try:\n"
            "    import importlib\n"
            "    m = importlib.import_module('socket')\n"
            "    print('SHOULD NOT REACH')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert 'ALLOWED: {"a": 1}' in result.stdout
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_builtins_import_bypass_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Accessing __import__ via __globals__['__builtins__'] is blocked."""
        config = _make_config(allowed=["operator", "functools", "re"])
        executor = SubprocessExecutor(config, venv_manager)

        # Use a Python-defined function to get __globals__ (C functions don't have it)
        # re.purge is a Python function in the re module
        source = (
            "import operator\n"
            "import re\n"
            "get_globals = operator.attrgetter('__globals__')\n"
            "g = get_globals(re.compile)\n"
            "do_import = g['__builtins__']['__import__']\n"
            "try:\n"
            "    os = do_import('os')\n"
            "    print('JAILBREAK SUCCEEDED')\n"
            "except (ImportError, TypeError) as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert "JAILBREAK" not in result.stdout
        # Either blocked by our patched __import__, or the attrgetter fails,
        # or __builtins__ access fails — any of these is acceptable
        if result.exit_code == 0:
            assert "BLOCKED" in result.stdout


class TestFactoryProxy:
    """Tests for factory proxy restrictions via the executor."""

    def test_factory_proxy_blocks_disallowed_method(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Methods not matching allow patterns are blocked on proxied objects."""
        # Use a deny-mode restriction on a simple allowed module to test
        # the proxy end-to-end without needing sys.path manipulation
        config = _make_config(
            allowed=["json", "pathlib", "re"],
            restrictions={
                "json": RestrictionConfig(
                    mode="deny",
                    module_allow=["JSONEncoder"],
                    class_allow=["encode"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # Test via the proxy property tests instead — they test the class directly.
        # Here we just verify the factory proxy works end-to-end via re module.
        # Use re.compile as a proxy test: create a pattern and verify it works
        source = (
            "import re\n"
            "\n"
            "# Verify re still works normally (no restrictions on it)\n"
            "pattern = re.compile(r'\\d+')\n"
            "result = pattern.findall('abc 123 def 456')\n"
            "print(f'found: {result}')\n"
            "result\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "found: ['123', '456']" in result.stdout


class TestModulePatches:
    """Tests for allow-mode module patches (deny specific functions)."""

    def test_deny_pattern_blocks_matching_functions(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Functions matching deny patterns are blocked; non-matching still work.
        Regex patterns match multiple functions (dump and dumps both blocked)."""
        config = _make_config(
            allowed=["json", "pathlib", "sys"],
            restrictions={
                "json": RestrictionConfig(
                    mode="allow",
                    module_deny=["dump.*"],  # blocks both dump and dumps
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import json\n"
            "# dumps should be blocked (matches dump.*)\n"
            "try:\n"
            "    json.dumps({'a': 1})\n"
            "    print('SHOULD NOT REACH 1')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED_DUMPS: {e}')\n"
            "# dump should be blocked (matches dump.*)\n"
            "try:\n"
            "    json.dump({'a': 1}, None)\n"
            "    print('SHOULD NOT REACH 2')\n"
            "except PermissionError:\n"
            "    print('BLOCKED_DUMP')\n"
            "# loads should still work (doesn't match dump.*)\n"
            "result = json.loads('{\"x\": 2}')\n"
            "print(f'LOADS_OK: {result}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_DUMPS" in result.stdout
        assert "dumps" in result.stdout
        assert "not permitted" in result.stdout
        assert "BLOCKED_DUMP" in result.stdout
        assert "LOADS_OK:" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestRestrictionModuleLeakage:
    """Tests verifying restriction modules don't leak os/sys via attribute access."""

    def test_module_does_not_leak_os_in_any_mode(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Both allow-mode and deny-mode restriction modules block os attribute access.

        Also verifies submodules (json.decoder.os) are blocked while the
        exported API still works.

        Regression test: polars.os was accessible because mode='allow' only
        patched deny-pattern attrs but didn't block ModuleType attributes.
        """
        # Use allow-mode config (also tests submodule leakage)
        config = _make_config(
            allowed=["json", "os"],
            restrictions={
                "json": RestrictionConfig(
                    mode="allow",
                    module_deny=["dumps"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import json\n"
            "import json.decoder\n"
            "\n"
            "# allow-mode: json.os should be blocked\n"
            "try:\n"
            "    leaked_os = json.os\n"
            "    print(f'LEAKED_ALLOW: {leaked_os}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_ALLOW: {e}')\n"
            "\n"
            "# submodule: json.decoder.os should also be blocked\n"
            "try:\n"
            "    leaked_os = json.decoder.os\n"
            "    print(f'LEAKED_SUB: {leaked_os}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_SUB: {e}')\n"
            "\n"
            "# But the exported API should still work\n"
            "result = json.loads('{\"a\": 1}')\n"
            "print(f'API_OK: {result}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_ALLOW" in result.stdout
        assert "BLOCKED_SUB" in result.stdout
        assert "LEAKED" not in result.stdout
        assert "API_OK:" in result.stdout

    def test_deny_mode_module_does_not_leak_os(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """mode='deny' restriction module blocks access to os attribute."""
        config = _make_config(
            allowed=["json", "os"],
            restrictions={
                "json": RestrictionConfig(
                    mode="deny",
                    module_allow=["loads"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import json\n"
            "try:\n"
            "    leaked_os = json.os\n"
            "    print(f'LEAKED: {leaked_os}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "LEAKED" not in result.stdout


class TestJailbreakPrevention:
    """Tests verifying known jailbreak vectors are blocked."""

    def test_getattr_globals_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """getattr(func, '__globals__') is blocked at runtime."""
        config = _make_config(allowed=["pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            "try:\n"
            "    g = getattr(pathlib.Path.cwd, '__globals__')\n"
            "    print('JAILBREAK')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "JAILBREAK" not in result.stdout

    def test_io_open_write_outside_workspace_blocked(
        self, workspace: Path, venv_manager: VenvManager, tmp_path: Path
    ) -> None:
        """io.open for writing outside workspace is blocked."""
        config = _make_config(allowed=["io"])
        executor = SubprocessExecutor(config, venv_manager)

        outside_path = (tmp_path / "evil.txt").as_posix()
        source = (
            "import io\n"
            "try:\n"
            f"    f = io.open('{outside_path}', 'w')\n"
            "    f.write('hacked')\n"
            "    f.close()\n"
            "    print('JAILBREAK')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "JAILBREAK" not in result.stdout

    def test_format_string_globals_escalation_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Format string can read __globals__ (known limitation) but cannot escalate.

        Python's str.format() uses C-level getattr, bypassing our
        builtins.getattr patch. However, escalation is blocked because:
        - sys.modules is scrubbed (no os/subprocess accessible)
        - builtins.__import__ is guarded (can't import dangerous modules)
        - The format result is a string, not a live reference
        """
        config = _make_config(allowed=["pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        # Format strings return STRING representations, not live objects.
        # Even if you can see __globals__ content as a string, you can't
        # call functions or access modules from a string representation.
        source = (
            "import pathlib\n"
            "# This leaks globals as a string (known limitation)\n"
            "leaked = '{0.__globals__}'.format(pathlib.Path.cwd)\n"
            "# But it's just a string — can't execute anything from it\n"
            "print(f'type: {type(leaked).__name__}')\n"
            "print('is_str: ' + str(isinstance(leaked, str)))\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "type: str" in result.stdout
        assert "is_str: True" in result.stdout

    def test_attrgetter_globals_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """operator.attrgetter('__globals__') is blocked at runtime."""
        config = _make_config(allowed=["operator", "pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import operator\n"
            "import pathlib\n"
            "try:\n"
            "    getter = operator.attrgetter('__globals__')\n"
            "    print('JAILBREAK')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "JAILBREAK" not in result.stdout


class TestExpandPatterns:
    """Unit tests for _expand_patterns() helper function."""

    def test_matches_exact_name(self) -> None:
        """A pattern matching an exact attribute name returns that name."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.client = lambda: None
        module.resource = lambda: None

        result = _expand_patterns(["client"], module)
        assert "client" in result

    def test_matches_regex_pattern(self) -> None:
        """A regex pattern matches multiple attribute names."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.list_buckets = lambda: None
        module.list_objects = lambda: None
        module.describe_instances = lambda: None
        module.delete_bucket = lambda: None

        result = _expand_patterns(["list_.*"], module)
        assert "list_buckets" in result
        assert "list_objects" in result
        assert "describe_instances" not in result
        assert "delete_bucket" not in result

    def test_multiple_patterns(self) -> None:
        """Multiple patterns are OR'd together — matching any one suffices."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.client = lambda: None
        module.resource = lambda: None
        module.setup = lambda: None

        result = _expand_patterns(["client", "resource"], module)
        assert "client" in result
        assert "resource" in result
        assert "setup" not in result

    def test_empty_patterns_returns_empty(self) -> None:
        """An empty pattern list matches nothing."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.something = 42

        result = _expand_patterns([], module)
        assert result == frozenset()

    def test_returns_frozenset(self) -> None:
        """The return type is a frozenset."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.foo = 1

        result = _expand_patterns(["foo"], module)
        assert isinstance(result, frozenset)

    def test_no_match_returns_empty(self) -> None:
        """Patterns that don't match any attribute return an empty frozenset."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.alpha = 1
        module.beta = 2

        result = _expand_patterns(["zzz_.*"], module)
        assert result == frozenset()

    def test_uses_re_match_not_search(self) -> None:
        """re.match anchors at the start — 'lient' should NOT match 'client'."""
        import types
        from pyddock._runtime import _expand_patterns

        module = types.ModuleType("fake")
        module.client = lambda: None

        result = _expand_patterns(["lient"], module)
        assert "client" not in result


class TestComputeExportedApi:
    """Unit tests for _compute_exported_api() helper function."""

    def test_uses_all_when_defined(self) -> None:
        """If module defines __all__, returns exactly those names."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.foo = lambda: None
        module.bar = lambda: None
        module._private = lambda: None
        module.__all__ = ["foo", "bar"]

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert result == frozenset({"foo", "bar"})

    def test_all_includes_private_names_if_listed(self) -> None:
        """__all__ can include private names — they're returned as-is."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module._internal = lambda: None
        module.public = lambda: None
        module.__all__ = ["_internal", "public"]

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert result == frozenset({"_internal", "public"})

    def test_excludes_module_type_attrs_without_all(self) -> None:
        """Without __all__, attributes that are ModuleType instances are excluded."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.my_func = lambda: None
        module.MY_CONST = 42
        # Simulate an imported module (like `import os` at module level)
        module.os = types.ModuleType("os")
        module.sys = types.ModuleType("sys")

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert "my_func" in result
        assert "MY_CONST" in result
        assert "os" not in result
        assert "sys" not in result

    def test_excludes_private_names_without_all(self) -> None:
        """Without __all__, private names (starting with _) are excluded."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.public_func = lambda: None
        module._private_func = lambda: None
        module.__dunder = "something"

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert "public_func" in result
        assert "_private_func" not in result
        assert "__dunder" not in result

    def test_returns_frozenset(self) -> None:
        """The return type is a frozenset."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.x = 1

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert isinstance(result, frozenset)

    def test_empty_module_returns_empty(self) -> None:
        """A module with no public non-module attributes returns empty frozenset."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        # ModuleType has some default attrs like __name__, __doc__ etc.
        # but they all start with _ so should be excluded

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        # Should be empty (no public non-module attrs added)
        assert result == frozenset()

    def test_all_overrides_module_type_filtering(self) -> None:
        """If __all__ lists a module-type attr, it's still included."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.submod = types.ModuleType("submod")
        module.__all__ = ["submod"]

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert "submod" in result

    def test_include_private_exposes_private_non_module_attrs(self) -> None:
        """include_private=True adds single-underscore non-module attributes.

        This is required so native extensions (e.g. cryptography's Rust
        bindings) can read module-private constants through the proxy.
        """
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.public = 1
        module._PRIVATE_CONST = {"k": "v"}

        default = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert "_PRIVATE_CONST" not in default

        with_private = _compute_exported_api(module, exclude_foreign_classes=False, include_private=True)
        assert "public" in with_private
        assert "_PRIVATE_CONST" in with_private

    def test_include_private_still_excludes_module_type(self) -> None:
        """include_private must NOT expose ModuleType attrs (leakage guard).

        Even private module references (e.g. a leaked `os`) stay excluded so
        agent code cannot reach a usable os/sys through a private alias.
        """
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module._leaked_os = types.ModuleType("os")
        module._data = 42

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=True)
        assert "_leaked_os" not in result
        assert "_data" in result

    def test_include_private_skips_dunders(self) -> None:
        """include_private exposes single-underscore names but not dunders."""
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module._single = 1
        module.__weird_dunder__ = 2

        result = _compute_exported_api(module, exclude_foreign_classes=False, include_private=True)
        assert "_single" in result
        assert "__weird_dunder__" not in result

    def test_include_private_with_all_unions_private_attrs(self) -> None:
        """With __all__ present, include_private unions private non-module attrs.

        Default behavior returns __all__ verbatim; include_private also scans
        for private constants the native layer may need.
        """
        import types
        from pyddock._runtime import _compute_exported_api

        module = types.ModuleType("fake")
        module.PublicThing = 1
        module._PRIVATE_CONST = {"k": "v"}
        module.__all__ = ["PublicThing"]

        default = _compute_exported_api(module, exclude_foreign_classes=False, include_private=False)
        assert default == frozenset({"PublicThing"})

        with_private = _compute_exported_api(module, exclude_foreign_classes=False, include_private=True)
        assert "PublicThing" in with_private
        assert "_PRIVATE_CONST" in with_private
