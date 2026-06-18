"""Security-focused tests for pyddock runtime enforcement.

Tests verify that known attack vectors are blocked:
- sys.exc_info traceback stripping (prevents frame access escalation)
- sys.modules/sys.meta_path inaccessible (prevents hook removal)
- _PYDDOCK_DIR write protection (prevents enforcement code tampering)
- _loading_depth bypass is narrow (only frozen callers)
- Agent cannot forge trusted caller frames
"""

from __future__ import annotations

import os
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
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


def _make_config(
    allowed_imports: list[str] | None = None,
) -> PyddockConfig:
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(
            allowed=allowed_imports or ["json", "sys", "os", "pathlib"],
        ),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(
            block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
            block_attributes=["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__"],
        ),
        restrictions={},
    )


class TestSysExcInfoSanitized:
    """Verify sys.exc_info() returns None for traceback (no frame access)."""

    def test_exc_info_traceback_is_none_and_frame_inaccessible(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """sys.exc_info()[2] is always None — no traceback/frame access possible."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import sys\n"
            "try:\n"
            "    1/0\n"
            "except:\n"
            "    typ, val, tb = sys.exc_info()\n"
            "    print(f'type={typ.__name__}')\n"
            "    print(f'value={val}')\n"
            "    print(f'tb={tb}')\n"
            "    # Even if tb were somehow obtained, frame access fails\n"
            "    try:\n"
            "        frame = tb.tb_frame\n"
            "        print('SHOULD NOT REACH')\n"
            "    except (AttributeError, TypeError) as e:\n"
            "        print(f'FRAME_BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "type=ZeroDivisionError" in result.stdout
        assert "tb=None" in result.stdout
        assert "FRAME_BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestSysModulesInaccessible:
    """Verify agent cannot access sys.modules or sys.meta_path."""

    def test_sys_modules_and_meta_path_not_exposed(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """sys.modules and sys.meta_path are not on the safe sys proxy."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import sys\n"
            "# sys.modules should be blocked\n"
            "try:\n"
            "    mods = sys.modules\n"
            "    print('LEAKED_MODULES')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_MODULES: {e}')\n"
            "\n"
            "# sys.meta_path should be blocked\n"
            "try:\n"
            "    hooks = sys.meta_path\n"
            "    print('LEAKED_META_PATH')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_META_PATH: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_MODULES" in result.stdout
        assert "BLOCKED_META_PATH" in result.stdout
        assert "LEAKED" not in result.stdout


class TestPyddockDirWriteProtected:
    """Verify agent cannot write to pyddock's source directory."""

    def test_write_to_pyddock_source_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to pyddock's install directory is blocked."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        # Try to write to pyddock's own _runtime.py
        source = (
            "import pathlib, sys\n"
            "# Find pyddock's install path\n"
            "import json  # allowed, just to get a module with __file__\n"
            "pyddock_dir = str(pathlib.Path(json.__file__).parent.parent / 'pyddock')\n"
            "target = pathlib.Path(pyddock_dir) / 'evil.py'\n"
            "try:\n"
            "    target.write_text('malicious')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_write_to_workspace_pyddock_dir_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to the workspace .pyddock/ directory is blocked (not tmp/)."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        # .pyddock/ is inside the workspace (so it passes the writable-paths
        # boundary) but is structurally protected against self-modification.
        source = (
            "import pathlib\n"
            "try:\n"
            "    pathlib.Path('.pyddock/pwned.txt').write_text('x')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    @pytest.mark.skipif(
        sys.platform != "win32", reason="8.3 short-name aliasing is Windows-only"
    )
    def test_write_to_pyddock_via_short_name_alias_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Regression: a Windows 8.3 short-name alias for .pyddock cannot be used
        to bypass the write protection.

        os.path.abspath() (lexical) leaves an 8.3 alias like PYDDOC~1 intact, so
        the relative_to('.pyddock') containment check would wrongly decide the
        path is outside .pyddock/ — while the OS resolves PYDDOC~1 -> .pyddock at
        open() time. The fix canonicalizes with realpath (expands short names),
        closing the gap. Skips if the volume has 8.3 generation disabled.
        """
        import ctypes
        from ctypes import wintypes

        pyddock_dir = workspace / ".pyddock"
        pyddock_dir.mkdir()

        # Ask Windows for the directory's 8.3 short path.
        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        _GetShortPathNameW.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = _GetShortPathNameW(str(pyddock_dir), buf, 1024)
        if n == 0:
            pytest.skip("GetShortPathNameW failed")
        short_full = buf.value
        short_name = os.path.basename(short_full)

        # If no distinct short name was generated (8.3 disabled on this volume),
        # there is nothing to test.
        if short_name.lower() == ".pyddock":
            pytest.skip("8.3 short-name generation disabled on this volume")

        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        # Write through the short-name alias, relative to the workspace cwd.
        source = (
            "import pathlib\n"
            f"alias = {short_name + '/pwned.txt'!r}\n"
            "try:\n"
            "    pathlib.Path(alias).write_text('x')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout, result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
        # And the file must not have been created in the real .pyddock/.
        assert not (pyddock_dir / "pwned.txt").exists()


class TestImportBypassSecurity:
    """Verify the import bypass cannot be exploited by agent code."""

    def test_agent_cannot_bypass_via_file_with_pyddock_in_name(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """A file named with 'pyddock' in the workspace doesn't get trusted status."""
        # Create a file with "pyddock" in its name in the workspace
        trick_file = workspace / "fake_pyddock_helper.py"
        trick_file.write_text(
            "# This file has 'pyddock' in its name but should NOT be trusted\n"
            "import sqlite3  # should be blocked\n"
        )

        config = _make_config(allowed_imports=["json", "sys", "pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        # Try to import the trick file (it's not in the allowlist anyway)
        source = (
            "try:\n"
            "    import sqlite3\n"
            "    print('SHOULD NOT REACH')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout

    def test_os_getenv_accessible(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """os.getenv is available and returns environment values."""
        config = _make_config()
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import os\n"
            "# getenv should work (read-only operation)\n"
            "result = os.getenv('PATH')\n"
            "print(f'got_path={result is not None}')\n"
            "# Also test with default\n"
            "missing = os.getenv('PYDDOCK_NONEXISTENT_VAR', 'default_val')\n"
            "print(f'default={missing}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "got_path=True" in result.stdout
        assert "default=default_val" in result.stdout


class TestDenyModeModuleProxy:
    """Verify deny-mode module proxy blocks non-allowed attributes for agent code.

    Uses the `json` module with deny-mode restrictions as a test stand-in
    (boto3 may not be installed in the test environment).

    Requirements: 2.2, 2.3, 7.5
    """

    def test_non_allowed_attr_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot access attributes not in module_allow."""
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["json", "sys", "os", "pathlib"],
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__"],
            ),
            restrictions={
                "json": RestrictionConfig(
                    mode="deny",
                    module_allow=["loads"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # json.dumps is NOT in module_allow, so it should be blocked
        source = (
            "import json\n"
            "try:\n"
            "    json.dumps({'a': 1})\n"
            "    print('SHOULD NOT REACH')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_allowed_attr_accessible(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Agent can access attributes matching module_allow patterns."""
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["json", "sys", "os", "pathlib"],
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__"],
            ),
            restrictions={
                "json": RestrictionConfig(
                    mode="deny",
                    module_allow=["loads", "dumps"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # json.loads and json.dumps are in module_allow (no class_allow, so no FactoryProxy)
        source = (
            "import json\n"
            "result = json.loads('{\"a\": 1}')\n"
            "print(f'loads_works={result}')\n"
            "encoded = json.dumps({'b': 2})\n"
            "print(f'dumps_works={encoded}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "loads_works={'a': 1}" in result.stdout
        assert 'dumps_works={"b": 2}' in result.stdout

    def test_class_allow_blocks_disallowed_methods(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Objects returned by allowed functions have method filtering via FactoryProxy.

        When class_allow is configured, allowed non-type callables are wrapped
        in FactoryProxy. Methods not matching class_allow patterns are blocked
        on returned objects.
        """
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["json", "sys", "os", "pathlib"],
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(
                block_calls=["eval", "exec", "compile", "breakpoint", "__import__"],
                block_attributes=["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__"],
            ),
            restrictions={
                "json": RestrictionConfig(
                    mode="deny",
                    module_allow=["loads"],
                    class_allow=["get"],
                ),
            },
        )
        executor = SubprocessExecutor(config, venv_manager)

        # json.loads is a function (not a type), so it gets wrapped in FactoryProxy.
        # Calling it returns a MethodFilterProxy wrapping the result (a dict).
        # Only 'get' matches class_allow, so 'keys' should be blocked.
        source = (
            "import json\n"
            "result = json.loads('{\"a\": 1}')\n"
            "# 'get' matches class_allow pattern\n"
            "val = result.get('a')\n"
            "print(f'get_works={val}')\n"
            "# 'keys' does NOT match class_allow pattern\n"
            "try:\n"
            "    result.keys()\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "get_works=1" in result.stdout
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestWorkspaceModuleLeakageBlocked:
    """Verify workspace module proxy blocks access to imported modules.

    Requirements: 3.2, 3.3
    """

    @pytest.fixture
    def workspace_with_module(self, workspace: Path) -> Path:
        """Create a workspace module that imports os and sys internally."""
        pkg_dir = workspace / "my_workspace_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import os\n"
            "import sys\n"
            "\n"
            "def get_cwd():\n"
            "    \"\"\"Public function that uses os internally.\"\"\"\n"
            "    return os.getcwd()\n"
            "\n"
            "def get_platform():\n"
            "    \"\"\"Public function that uses sys internally.\"\"\"\n"
            "    return sys.platform\n"
            "\n"
            "MY_CONSTANT = 42\n"
        )
        return workspace

    def _make_executor_with_pkg_path(
        self,
        config: PyddockConfig,
        venv_manager: VenvManager,
        pkg_parent_path: Path,
    ) -> SubprocessExecutor:
        """Create an executor that adds the package parent to sys.path."""
        executor = SubprocessExecutor(config, venv_manager)
        original_build = executor._build_bootstrap

        def _patched_build(source: str, args: list[str], workspace_root: Path) -> str:
            bootstrap = original_build(source, args, workspace_root)
            inject_line = f"sys.path.insert(0, {str(pkg_parent_path)!r})\n"
            marker = "sys.argv = "
            idx = bootstrap.find(marker)
            if idx != -1:
                end_of_line = bootstrap.find("\n", idx)
                bootstrap = bootstrap[:end_of_line + 1] + inject_line + bootstrap[end_of_line + 1:]
            return bootstrap

        executor._build_bootstrap = _patched_build  # type: ignore[method-assign]
        return executor

    def test_agent_cannot_access_os_via_workspace_module(
        self, workspace_with_module: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot access workspace_module.os or .sys (imported module leakage blocked)."""
        workspace = workspace_with_module

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_workspace_pkg", "os", "sys"],
                workspace={"my_workspace_pkg": "my_workspace_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_workspace_pkg\n"
            "# os should be blocked\n"
            "try:\n"
            "    leaked_os = my_workspace_pkg.os\n"
            "    print('LEAKED_OS')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_OS: {e}')\n"
            "\n"
            "# sys should be blocked\n"
            "try:\n"
            "    leaked_sys = my_workspace_pkg.sys\n"
            "    print('LEAKED_SYS')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_SYS: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_OS" in result.stdout
        assert "BLOCKED_SYS" in result.stdout
        assert "LEAKED" not in result.stdout

    def test_exported_functions_are_accessible(
        self, workspace_with_module: Path, venv_manager: VenvManager
    ) -> None:
        """Agent CAN access exported functions on the workspace module."""
        workspace = workspace_with_module

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_workspace_pkg", "os", "sys"],
                workspace={"my_workspace_pkg": "my_workspace_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_workspace_pkg\n"
            "# Public functions should be accessible\n"
            "result = my_workspace_pkg.get_platform()\n"
            "print(f'platform={result}')\n"
            "# Constants should be accessible\n"
            "print(f'constant={my_workspace_pkg.MY_CONSTANT}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "platform=" in result.stdout
        assert "constant=42" in result.stdout

    @pytest.fixture
    def workspace_with_submodule(self, workspace: Path) -> Path:
        """Create a workspace package with a submodule that imports os and sys."""
        pkg_dir = workspace / "my_workspace_pkg"
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / "__init__.py").write_text(
            "# Package init\n"
        )
        (pkg_dir / "client.py").write_text(
            "import os\n"
            "import sys\n"
            "\n"
            "def do_work():\n"
            "    \"\"\"Public function that uses os internally.\"\"\"\n"
            "    return os.getcwd()\n"
            "\n"
            "SUBMOD_CONSTANT = 99\n"
        )
        return workspace

    def test_agent_cannot_access_os_via_workspace_submodule(
        self, workspace_with_submodule: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot access submodule.os when importing workspace submodule directly.

        Also verifies Design Property 2: exported API of stdlib modules (pathlib.Path,
        json.loads) remains accessible to agent code.
        """
        workspace = workspace_with_submodule

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_workspace_pkg", "os", "sys", "pathlib", "json"],
                workspace={"my_workspace_pkg": "my_workspace_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_workspace_pkg.client as client\n"
            "# os should be blocked on the submodule\n"
            "try:\n"
            "    leaked_os = client.os\n"
            "    print(f'LEAKED: {leaked_os}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
            "# sys should be blocked on the submodule\n"
            "try:\n"
            "    leaked_sys = client.sys\n"
            "    print(f'LEAKED: {leaked_sys}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_SYS: {e}')\n"
            "# But exported functions should work\n"
            "print(f'constant={client.SUBMOD_CONSTANT}')\n"
            "# Internal use of os by the module's own code still works\n"
            "result = client.do_work()\n"
            "print(f'do_work={result}')\n"
            "\n"
            "# Design Property 2: Exported API of stdlib modules is accessible\n"
            "import pathlib\n"
            "import json\n"
            "# pathlib.Path is part of the exported API and must be accessible\n"
            "p = pathlib.Path('.')\n"
            "print(f'PATHLIB_OK={p.resolve()}')\n"
            "# json.loads is part of the exported API and must be accessible\n"
            "data = json.loads('{\"key\": \"value\"}')\n"
            "print(f'JSON_OK={data}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "BLOCKED_SYS" in result.stdout
        assert "LEAKED" not in result.stdout
        assert "constant=99" in result.stdout
        assert "do_work=" in result.stdout
        # Design Property 2: stdlib exported API accessible to agent code
        assert "PATHLIB_OK=" in result.stdout
        assert "JSON_OK=" in result.stdout

    def test_agent_cannot_access_os_via_pathlib(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot access stdlib module leakage (pathlib.os, tempfile.os, re._compile).

        Validates: Design Properties 1, 3
        """
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["pathlib", "os", "tempfile", "re"],
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            "import tempfile\n"
            "import re\n"
            "\n"
            "# pathlib.os should be blocked\n"
            "try:\n"
            "    leaked = pathlib.os\n"
            "    print(f'LEAKED_PATHLIB: {leaked}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_PATHLIB: {e}')\n"
            "\n"
            "# tempfile.os should be blocked\n"
            "try:\n"
            "    leaked = tempfile.os\n"
            "    print(f'LEAKED_TEMPFILE: {leaked}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_TEMPFILE: {e}')\n"
            "\n"
            "# re._compile (private attr) should be blocked\n"
            "try:\n"
            "    leaked = re._compile\n"
            "    print(f'LEAKED_RE: {leaked}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_RE: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_PATHLIB" in result.stdout
        assert "BLOCKED_TEMPFILE" in result.stdout
        assert "BLOCKED_RE" in result.stdout
        assert "LEAKED" not in result.stdout

    def test_stdlib_write_protection_blocks_writes(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot write to sys.base_prefix/Lib (stdlib write protection).

        Validates: Design Property 5
        """
        # Compute the stdlib Lib path outside the sandbox (test-side)
        stdlib_lib = os.path.join(sys.base_prefix, "Lib")
        if not os.path.isdir(stdlib_lib):
            stdlib_lib = os.path.join(
                sys.base_prefix, "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
            )
        # Use the resolved path as a string literal in agent code
        target_path = os.path.join(stdlib_lib, "test_write.py")

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["pathlib"],
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            f"target = pathlib.Path({target_path!r})\n"
            "try:\n"
            "    target.write_text('hacked')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


class TestReExportingPackageProxy:
    """Regression tests for proxying packages whose __init__.py re-exports.

    A package whose __init__.py does `from <pkg> import <submodule>` triggers
    a re-entrant import of <pkg> while it is still initializing. The proxy must
    be installed only after the outermost import completes, so it captures the
    full public API (__all__) — not a partially-initialized snapshot. This is
    the exact pattern used by cryptography.x509 (re-exports from submodules and
    runs `from cryptography.x509 import oid, ...` during its own init).
    """

    @pytest.fixture
    def workspace_with_reexport_pkg(self, workspace: Path) -> Path:
        """Create a workspace subpackage that re-exports during init.

        reexport_pkg/sub/__init__.py mirrors the cryptography.x509 pattern:
          1. `from reexport_pkg.sub import _impl` — re-entrant import of the
             still-initializing subpackage.
          2. Re-exports PublicThing/make_thing from the submodule.
          3. Imports os (a leaked stdlib module) and defines __all__ AFTER the
             imports — so a prematurely-created proxy would miss the public API.
        """
        pkg = workspace / "reexport_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# top-level package\n")
        sub = pkg / "sub"
        sub.mkdir()
        (sub / "__init__.py").write_text(
            "from __future__ import annotations\n"
            "# Re-entrant import of this very subpackage during its own init.\n"
            "from reexport_pkg.sub import _impl\n"
            "from reexport_pkg.sub._impl import PublicThing, make_thing\n"
            "import os  # leaked stdlib module - must stay inaccessible to agents\n"
            "\n"
            "# __all__ defined AFTER imports: a proxy created mid-init would not\n"
            "# see these names.\n"
            "__all__ = ['PublicThing', 'make_thing']\n"
        )
        (sub / "_impl.py").write_text(
            "import os\n"
            "\n"
            "class PublicThing:\n"
            "    value = 7\n"
            "\n"
            "def make_thing():\n"
            "    return PublicThing()\n"
        )
        return workspace

    def _make_executor_with_pkg_path(
        self,
        config: PyddockConfig,
        venv_manager: VenvManager,
        pkg_parent_path: Path,
    ) -> SubprocessExecutor:
        """Executor that injects the package parent dir onto the subprocess sys.path."""
        executor = SubprocessExecutor(config, venv_manager)
        original_build = executor._build_bootstrap

        def _patched_build(source: str, args: list[str], workspace_root: Path) -> str:
            bootstrap = original_build(source, args, workspace_root)
            inject_line = f"sys.path.insert(0, {str(pkg_parent_path)!r})\n"
            marker = "sys.argv = "
            idx = bootstrap.find(marker)
            if idx != -1:
                end_of_line = bootstrap.find("\n", idx)
                bootstrap = bootstrap[:end_of_line + 1] + inject_line + bootstrap[end_of_line + 1:]
            return bootstrap

        executor._build_bootstrap = _patched_build  # type: ignore[method-assign]
        return executor

    def test_reexported_public_api_is_accessible(
        self, workspace_with_reexport_pkg: Path, venv_manager: VenvManager
    ) -> None:
        """Public API re-exported during init must be reachable on the proxy.

        Without the timing fix, the proxy is created during the re-entrant
        import (before __all__ is populated) and PublicThing/make_thing are
        permanently hidden from agent code.
        """
        workspace = workspace_with_reexport_pkg

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["reexport_pkg", "os"],
                workspace={"reexport_pkg": "reexport_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )
        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import reexport_pkg.sub as sub\n"
            "for attr in ('PublicThing', 'make_thing'):\n"
            "    print(('OK: ' if hasattr(sub, attr) else 'MISSING: ') + attr)\n"
            "thing = sub.make_thing()\n"
            "print(f'VALUE={thing.value}')\n"
        )
        result = executor.execute(source, [], 30, workspace)

        assert result.exit_code == 0, result.stderr
        assert "OK: PublicThing" in result.stdout
        assert "OK: make_thing" in result.stdout
        assert "VALUE=7" in result.stdout
        assert "MISSING" not in result.stdout

    def test_reexport_pkg_module_leakage_still_blocked(
        self, workspace_with_reexport_pkg: Path, venv_manager: VenvManager
    ) -> None:
        """Fixing the proxy timing must NOT expose leaked stdlib modules.

        reexport_pkg.sub imports os at module scope; it must stay inaccessible
        to agent code (ModuleType leakage), even though the public API is now
        correctly exposed.
        """
        workspace = workspace_with_reexport_pkg

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["reexport_pkg", "os"],
                workspace={"reexport_pkg": "reexport_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )
        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import reexport_pkg.sub as sub\n"
            "try:\n"
            "    leaked = sub.os\n"
            "    print('LEAKED_OS')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_OS: {e}')\n"
        )
        result = executor.execute(source, [], 30, workspace)

        assert result.exit_code == 0, result.stderr
        assert "BLOCKED_OS" in result.stdout
        assert "LEAKED" not in result.stdout

