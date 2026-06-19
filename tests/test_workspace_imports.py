"""Tests for workspace imports and simplified import hook.

Tests cover:
- Config parsing of workspace module entries (module = "path")
- Simplified import hook blocks unlisted modules even if cached
- Simplified import hook allows listed modules
- Underscore modules are blocked (no exception)
- Workspace module paths are write-protected
- Transitive deps work via pre-import caching
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
    load_config,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager

from tests._config_helpers import write_workspace_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def venv_manager(tmp_path: Path) -> VenvManager:
    """VenvManager using the current Python interpreter."""
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return manager


def _make_config(
    allowed_imports: list[str] | None = None,
    workspace_modules: dict[str, str] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(
            allowed=allowed_imports or ["json", "math", "pathlib"],
            workspace=workspace_modules or {},
        ),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        restrictions={},
    )


# ---------------------------------------------------------------------------
# Test: workspace config parsing
# ---------------------------------------------------------------------------


class TestWorkspaceConfigParsing:
    """Test that module = 'path' produces correct ImportsConfig."""

    def test_string_value_adds_to_workspace_and_allowed(self, tmp_path: Path) -> None:
        """A string-valued import entry adds the module to both workspace and allowed."""
        write_workspace_config(
            tmp_path,
            imports='[imports]\njson = true\nmy_tool = ".kiro/scripts/my-tool"\n',
            extra="[restrictions]\n",
        )

        config = load_config(tmp_path)

        # Module is in the allowed list
        assert "my_tool" in config.imports.allowed
        # Module is in the workspace dict with its path
        assert "my_tool" in config.imports.workspace
        assert config.imports.workspace["my_tool"] == ".kiro/scripts/my-tool"
        # Bool entries are also in allowed
        assert "json" in config.imports.allowed
        # Bool entries are NOT in workspace
        assert "json" not in config.imports.workspace

    def test_multiple_workspace_modules(self, tmp_path: Path) -> None:
        """Multiple string-valued entries all appear in workspace and allowed."""
        write_workspace_config(
            tmp_path,
            imports=(
                "[imports]\n"
                'invoice_parser = ".kiro/scripts/invoice-parser"\n'
                'metrics_client = ".kiro/scripts/reporting/metrics-client"\n'
                "json = true\n"
            ),
            extra="[restrictions]\n",
        )

        config = load_config(tmp_path)

        assert "invoice_parser" in config.imports.allowed
        assert "metrics_client" in config.imports.allowed
        assert "json" in config.imports.allowed

        assert config.imports.workspace["invoice_parser"] == ".kiro/scripts/invoice-parser"
        assert config.imports.workspace["metrics_client"] == ".kiro/scripts/reporting/metrics-client"

    def test_empty_string_excluded(self, tmp_path: Path) -> None:
        """An empty string value is treated like false — excluded from both."""
        write_workspace_config(
            tmp_path,
            imports='[imports]\njson = true\nexcluded_mod = ""\n',
            extra="[restrictions]\n",
        )

        config = load_config(tmp_path)

        assert "excluded_mod" not in config.imports.allowed
        assert "excluded_mod" not in config.imports.workspace


# ---------------------------------------------------------------------------
# Test: simplified import hook blocks unlisted module even if cached
# ---------------------------------------------------------------------------


class TestImportHookBlocksCachedModules:
    """Test that the import hook blocks modules not in the allowlist even if cached."""

    def test_cached_and_common_modules_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Modules in sys.modules (sqlite3, os) are blocked if not in allowlist."""
        config = _make_config(allowed_imports=["json", "math", "pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "# sqlite3 may be cached from Python startup\n"
            "try:\n"
            "    import sqlite3\n"
            "    print('LEAKED_SQLITE3')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED_SQLITE3: {e}')\n"
            "\n"
            "# os is always in sys.modules but should be blocked\n"
            "try:\n"
            "    import os\n"
            "    print('LEAKED_OS')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED_OS: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_SQLITE3" in result.stdout
        assert "BLOCKED_OS" in result.stdout
        assert "LEAKED" not in result.stdout


# ---------------------------------------------------------------------------
# Test: simplified import hook allows listed module
# ---------------------------------------------------------------------------


class TestImportHookAllowsListedModule:
    """Test that the import hook allows modules in the allowlist."""

    def test_allowed_modules_importable(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """All modules in the allowlist can be imported and used."""
        config = _make_config(allowed_imports=["json", "math", "pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import json\n"
            "import math\n"
            "import pathlib\n"
            "print(f'json={json.__name__} math={math.__name__} pathlib={pathlib.__name__}')\n"
            "json.dumps({'key': 'value'})\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "json=json" in result.stdout
        assert "math=math" in result.stdout
        assert "pathlib=pathlib" in result.stdout
        assert result.result == "'{\"key\": \"value\"}'"


# ---------------------------------------------------------------------------
# Test: underscore module is blocked (no exception)
# ---------------------------------------------------------------------------


class TestUnderscoreModuleBlocked:
    """Test that underscore-prefixed modules are blocked (no exception)."""

    def test_underscore_socket_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """_socket is blocked — prevents raw network access."""
        config = _make_config(allowed_imports=["json", "math", "pathlib"])
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "try:\n"
            "    import _socket\n"
            "    print('SHOULD NOT REACH')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


# ---------------------------------------------------------------------------
# Test: workspace path is write-protected
# ---------------------------------------------------------------------------


class TestWorkspacePathWriteProtected:
    """Test that workspace module directories are write-protected."""

    def test_write_to_workspace_module_dir_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to a workspace module directory raises PermissionError."""
        # Create the workspace module directory
        module_dir = workspace / ".kiro" / "scripts" / "my-tool"
        module_dir.mkdir(parents=True)

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["json", "pathlib", "my_tool"],
                workspace={"my_tool": ".kiro/scripts/my-tool"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )
        executor = SubprocessExecutor(config, venv_manager)

        target = (module_dir / "evil.py").as_posix()
        source = (
            "import pathlib\n"
            "try:\n"
            f"    pathlib.Path('{target}').write_text('malicious code')\n"
            "    print('SHOULD NOT REACH')\n"
            "except PermissionError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout
        assert "workspace module" in result.stdout

    def test_write_outside_workspace_module_dir_allowed(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Writing to workspace root (not a module dir) still works."""
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["json", "pathlib", "my_tool", "tempfile", "codecs", "encodings"],
                workspace={"my_tool": ".kiro/scripts/my-tool"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )
        executor = SubprocessExecutor(config, venv_manager)

        source = (
            "import pathlib\n"
            "pathlib.Path('output.txt').write_text('safe content')\n"
            "'done'\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'done'"
        assert (workspace / "output.txt").read_text() == "safe content"


# ---------------------------------------------------------------------------
# Test: transitive deps work via pre-import caching
# ---------------------------------------------------------------------------


class TestTransitiveDeps:
    """Test that workspace packages can use transitive deps cached during pre-import."""

    @pytest.fixture
    def workspace_with_package(self, workspace: Path) -> Path:
        """Create a minimal workspace package that uses json internally.

        Places the package directly in the workspace root so it's importable
        when the workspace is on sys.path (simulating pip install -e).
        """
        # Create the package directly in workspace root (simulates pip install -e)
        pkg_dir = workspace / "my_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import json as _json\n\n"
            "def to_json(data):\n"
            "    \"\"\"Convert data to JSON string using json module.\"\"\"\n"
            "    return _json.dumps(data)\n"
        )

        return workspace

    def _make_executor_with_pkg_path(
        self,
        config: PyddockConfig,
        venv_manager: VenvManager,
        pkg_parent_path: Path,
    ) -> SubprocessExecutor:
        """Create an executor that adds the package parent to sys.path in bootstrap.

        This simulates what `pip install -e` does in production — making the
        workspace package importable before the import hook activates.
        """
        executor = SubprocessExecutor(config, venv_manager)
        original_build = executor._build_bootstrap

        def _patched_build(source: str, args: list[str], workspace_root: Path) -> str:
            bootstrap = original_build(source, args, workspace_root)
            # Insert the package path addition right after the sys.path.insert for pyddock
            # This must happen before RuntimeEnforcement.apply_all() which does pre-import
            inject_line = f"sys.path.insert(0, {str(pkg_parent_path)!r})\n"
            # Insert after the first sys.path.insert line
            marker = "sys.argv = "
            idx = bootstrap.find(marker)
            if idx != -1:
                # Find end of that line
                end_of_line = bootstrap.find("\n", idx)
                bootstrap = bootstrap[:end_of_line + 1] + inject_line + bootstrap[end_of_line + 1:]
            return bootstrap

        executor._build_bootstrap = _patched_build  # type: ignore[method-assign]
        return executor

    def test_transitive_dep_works_via_preimport(
        self, workspace_with_package: Path, venv_manager: VenvManager
    ) -> None:
        """A workspace package using json internally works even though json is not directly allowed.

        The package imports json at module level during pre-import. When user code
        later calls the package's function, it uses its own module-level reference
        to json — no fresh 'import json' is executed at runtime.
        """
        workspace = workspace_with_package

        # Configure: my_pkg is allowed (workspace module), but json is NOT directly allowed.
        # The package uses json internally via its module-level import.
        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_pkg"],
                workspace={"my_pkg": "my_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        # The snippet imports my_pkg and uses its to_json function.
        # json is NOT in the allowlist, but my_pkg uses it internally
        # via its module-level reference bound during pre-import.
        source = (
            "import my_pkg\n"
            "my_pkg.to_json({'hello': 'world'})\n"
        )

        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result == "'{\"hello\": \"world\"}'"

    def test_direct_import_of_transitive_dep_blocked(
        self, workspace_with_package: Path, venv_manager: VenvManager
    ) -> None:
        """Directly importing json (the transitive dep) is still blocked."""
        workspace = workspace_with_package

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_pkg"],
                workspace={"my_pkg": "my_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "try:\n"
            "    import json\n"
            "    print('SHOULD NOT REACH')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout


# ---------------------------------------------------------------------------
# Test: stack-aware import bypass for workspace module internal imports
# ---------------------------------------------------------------------------


class TestStackAwareImportBypass:
    """Test that workspace modules can perform internal imports of unlisted deps.

    The stack-aware bypass allows code executing inside a workspace module
    (or its transitive dependencies in site-packages) to import modules not
    in the agent's allowlist — while still blocking the agent from importing
    those same modules directly.
    """

    @pytest.fixture
    def workspace_with_lazy_pkg(self, workspace: Path) -> Path:
        """Create a workspace package that does a lazy import inside a function.

        This simulates the lazy-import case: the module imports a dep at
        function call time rather than at module level.
        """
        pkg_dir = workspace / "lazy_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "def get_platform():\n"
            "    \"\"\"Lazily imports platform module and returns info.\"\"\"\n"
            "    import platform\n"
            "    return platform.system()\n"
        )
        return workspace

    @pytest.fixture
    def workspace_with_submodule_pkg(self, workspace: Path) -> Path:
        """Create a workspace package with a submodule that imports an unlisted dep.

        Simulates: from workspace_pkg import server → server.py imports unlisted dep.
        """
        pkg_dir = workspace / "ws_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "server.py").write_text(
            "import platform\n\n"
            "def get_info():\n"
            "    return platform.system()\n"
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

    def test_lazy_import_inside_workspace_module_allowed(
        self, workspace_with_lazy_pkg: Path, venv_manager: VenvManager
    ) -> None:
        """A workspace module can lazily import an unlisted dep inside a function call."""
        workspace = workspace_with_lazy_pkg

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["lazy_pkg"],
                workspace={"lazy_pkg": "lazy_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        # Agent calls the workspace module's function which lazily imports platform
        source = (
            "import lazy_pkg\n"
            "lazy_pkg.get_platform()\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        # Should return the platform name (e.g. 'Windows', 'Linux')
        assert result.result is not None
        assert "SHOULD NOT REACH" not in (result.stdout or "")

    def test_direct_import_of_lazy_dep_still_blocked(
        self, workspace_with_lazy_pkg: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot directly import the same dep that the workspace module uses."""
        workspace = workspace_with_lazy_pkg

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["lazy_pkg"],
                workspace={"lazy_pkg": "lazy_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        # Agent tries to import platform directly — should be blocked
        source = (
            "try:\n"
            "    import platform\n"
            "    print('SHOULD NOT REACH')\n"
            "except ImportError as e:\n"
            "    print(f'BLOCKED: {e}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout
        assert "SHOULD NOT REACH" not in result.stdout

    def test_submodule_import_triggers_lazy_dep(
        self, workspace_with_submodule_pkg: Path, venv_manager: VenvManager
    ) -> None:
        """Importing a workspace package's submodule that imports an unlisted dep works."""
        workspace = workspace_with_submodule_pkg

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["ws_pkg"],
                workspace={"ws_pkg": "ws_pkg"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        # Agent imports the submodule — server.py does `import platform` at module level
        source = (
            "from ws_pkg import server\n"
            "server.get_info()\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert result.result is not None

# ---------------------------------------------------------------------------
# Test: foreign class leakage blocked on workspace modules
# ---------------------------------------------------------------------------


class TestForeignClassLeakageBlocked:
    """Verify workspace module proxy excludes foreign classes from exported API.

    When a workspace module imports a class from a third-party package at
    module level (e.g. `from atlassian import Jira`), that class should NOT
    be accessible to agent code via the module proxy. This prevents agents
    from constructing network-capable clients with arbitrary URLs.

    The fix: _compute_exported_api excludes classes whose __module__ belongs
    to a different top-level package when processing workspace modules.
    Workspace modules can opt in to exposing foreign classes via __all__.
    """

    @pytest.fixture
    def workspace_with_foreign_class(self, workspace: Path) -> Path:
        """Create a workspace module that imports a class from a 'foreign' package.

        Simulates: atlassian_client._client imports Jira from atlassian package.
        We use a fake 'foreign_lib' package to avoid needing real third-party deps.
        """
        # Create the "foreign" third-party package (simulates atlassian, paramiko, etc.)
        foreign_dir = workspace / "foreign_lib"
        foreign_dir.mkdir()
        (foreign_dir / "__init__.py").write_text(
            "class DangerousClient:\n"
            "    \"\"\"A network-capable client class (like Jira, SSHClient).\"\"\"\n"
            "    def __init__(self, url='http://default'):\n"
            "        self.url = url\n"
            "    def connect(self):\n"
            "        return f'connected to {self.url}'\n"
            "\n"
            "class AnotherDangerous:\n"
            "    \"\"\"Another class that shouldn't leak.\"\"\"\n"
            "    pass\n"
            "\n"
            "def safe_helper():\n"
            "    return 'helper_result'\n"
        )

        # Create the workspace module that imports from the foreign package
        ws_mod_dir = workspace / "my_wrapper"
        ws_mod_dir.mkdir()
        (ws_mod_dir / "__init__.py").write_text(
            "from foreign_lib import DangerousClient, AnotherDangerous\n"
            "\n"
            "# This is the safe wrapper function - should be accessible\n"
            "def do_safe_work():\n"
            "    \"\"\"Uses DangerousClient internally but doesn't expose it.\"\"\"\n"
            "    client = DangerousClient(url='http://internal-only')\n"
            "    return client.connect()\n"
            "\n"
            "SAFE_CONSTANT = 'hello'\n"
        )

        return workspace

    @pytest.fixture
    def workspace_with_foreign_class_and_all(self, workspace: Path) -> Path:
        """Create a workspace module with __all__ that explicitly exposes a foreign class."""
        # Create the "foreign" third-party package
        foreign_dir = workspace / "foreign_lib"
        foreign_dir.mkdir()
        (foreign_dir / "__init__.py").write_text(
            "class DangerousClient:\n"
            "    def __init__(self, url='http://default'):\n"
            "        self.url = url\n"
            "    def connect(self):\n"
            "        return f'connected to {self.url}'\n"
        )

        # Create workspace module with __all__ that opts in to exposing the class
        ws_mod_dir = workspace / "my_wrapper"
        ws_mod_dir.mkdir()
        (ws_mod_dir / "__init__.py").write_text(
            "from foreign_lib import DangerousClient\n"
            "\n"
            "__all__ = ['DangerousClient', 'do_safe_work', 'SAFE_CONSTANT']\n"
            "\n"
            "def do_safe_work():\n"
            "    client = DangerousClient(url='http://internal-only')\n"
            "    return client.connect()\n"
            "\n"
            "SAFE_CONSTANT = 'hello'\n"
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

    def test_foreign_class_blocked_on_workspace_module(
        self, workspace_with_foreign_class: Path, venv_manager: VenvManager
    ) -> None:
        """Agent cannot access a foreign class imported by a workspace module."""
        workspace = workspace_with_foreign_class

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_wrapper", "foreign_lib"],
                workspace={"my_wrapper": "my_wrapper", "foreign_lib": "foreign_lib"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_wrapper\n"
            "# Foreign class should be blocked\n"
            "try:\n"
            "    cls = my_wrapper.DangerousClient\n"
            "    print(f'LEAKED: {cls}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_CLASS: {e}')\n"
            "\n"
            "# Another foreign class should also be blocked\n"
            "try:\n"
            "    cls2 = my_wrapper.AnotherDangerous\n"
            "    print(f'LEAKED: {cls2}')\n"
            "except AttributeError as e:\n"
            "    print(f'BLOCKED_CLASS2: {e}')\n"
            "\n"
            "# But safe functions should still work\n"
            "result = my_wrapper.do_safe_work()\n"
            "print(f'SAFE_FUNC={result}')\n"
            "\n"
            "# And constants should be accessible\n"
            "print(f'CONSTANT={my_wrapper.SAFE_CONSTANT}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "BLOCKED_CLASS" in result.stdout
        assert "BLOCKED_CLASS2" in result.stdout
        assert "LEAKED" not in result.stdout
        assert "SAFE_FUNC=connected to http://internal-only" in result.stdout
        assert "CONSTANT=hello" in result.stdout

    def test_foreign_class_allowed_when_all_defined(
        self, workspace_with_foreign_class_and_all: Path, venv_manager: VenvManager
    ) -> None:
        """When __all__ explicitly includes a foreign class, agent can access it."""
        workspace = workspace_with_foreign_class_and_all

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_wrapper", "foreign_lib"],
                workspace={"my_wrapper": "my_wrapper", "foreign_lib": "foreign_lib"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_wrapper\n"
            "# __all__ includes DangerousClient — should be accessible\n"
            "cls = my_wrapper.DangerousClient\n"
            "print(f'CLASS_OK={cls.__name__}')\n"
            "\n"
            "# Safe function should work\n"
            "result = my_wrapper.do_safe_work()\n"
            "print(f'SAFE_FUNC={result}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "CLASS_OK=DangerousClient" in result.stdout
        assert "SAFE_FUNC=connected to http://internal-only" in result.stdout

    def test_own_classes_not_blocked(
        self, workspace: Path, venv_manager: VenvManager
    ) -> None:
        """Classes defined within the workspace module itself are still accessible."""
        # Create workspace module with its own class
        ws_mod_dir = workspace / "my_wrapper"
        ws_mod_dir.mkdir()
        (ws_mod_dir / "__init__.py").write_text(
            "class MyOwnClass:\n"
            "    \"\"\"A class defined in this package - should be accessible.\"\"\"\n"
            "    def greet(self):\n"
            "        return 'hello from own class'\n"
            "\n"
            "def helper():\n"
            "    return 'helper_ok'\n"
        )

        config = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(
                allowed=["my_wrapper"],
                workspace={"my_wrapper": "my_wrapper"},
            ),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
            ast=ASTConfig(block_calls=[], block_attributes=[]),
            restrictions={},
        )

        executor = self._make_executor_with_pkg_path(config, venv_manager, workspace)

        source = (
            "import my_wrapper\n"
            "# Own class should be accessible\n"
            "obj = my_wrapper.MyOwnClass()\n"
            "print(f'OWN_CLASS={obj.greet()}')\n"
            "print(f'HELPER={my_wrapper.helper()}')\n"
        )
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "OWN_CLASS=hello from own class" in result.stdout
        assert "HELPER=helper_ok" in result.stdout
