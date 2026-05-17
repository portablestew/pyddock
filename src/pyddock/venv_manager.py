"""Virtual environment manager for pyddock.

Creates and manages a venv at `.pyddock/venv/`, auto-installs allowed
third-party packages, and exposes the venv Python path for subprocess use.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


class VenvError(Exception):
    """Raised when venv creation or package installation fails."""


class VenvManager:
    """Manages the pyddock virtual environment and package installation.

    Args:
        venv_path: Path to the venv directory. Defaults to `.pyddock/venv/`.
        allowed_imports: List of module names from the config allowlist.
    """

    def __init__(
        self,
        venv_path: Path | None = None,
        allowed_imports: list[str] | None = None,
    ) -> None:
        if venv_path is None:
            venv_path = Path(".pyddock") / "venv"
        self._venv_path = venv_path
        self._allowed_imports = set(allowed_imports or [])
        self._stdlib_modules: set[str] = sys.stdlib_module_names  # type: ignore[assignment]
        self._installed_cache: set[str] = set()  # packages confirmed installed

    def ensure_venv(self) -> None:
        """Create the virtual environment if it doesn't already exist."""
        if self._venv_path.exists() and self.get_python_path().exists():
            return

        try:
            venv.create(str(self._venv_path), with_pip=True)
        except Exception as e:
            raise VenvError(f"Failed to create venv at {self._venv_path}: {e}") from e

    def get_python_path(self) -> Path:
        """Return the path to the venv's Python interpreter.

        On Windows this is `Scripts/python.exe`, on Unix it's `bin/python`.
        """
        if sys.platform == "win32":
            return self._venv_path / "Scripts" / "python.exe"
        return self._venv_path / "bin" / "python"

    def is_stdlib(self, module_name: str) -> bool:
        """Check if a module name is part of the Python standard library.

        Uses `sys.stdlib_module_names` (Python 3.10+) for detection.
        """
        return module_name in self._stdlib_modules

    def _is_installed(self, package_name: str) -> bool:
        """Check if a package is already installed in the venv."""
        if package_name in self._installed_cache:
            return True
        python = self.get_python_path()
        try:
            result = subprocess.run(
                [str(python), "-c", f"import {package_name}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode == 0:
            self._installed_cache.add(package_name)
            return True
        return False

    def install_workspace(
        self, packages: dict[str, str], workspace_root: Path
    ) -> None:
        """Install workspace packages in editable mode via pip install -e.

        Args:
            packages: Mapping of module name → relative path from workspace root.
            workspace_root: The workspace root directory to resolve relative paths.

        Raises:
            VenvError: If pip install -e fails for any package.
        """
        if not packages:
            return

        python = self.get_python_path()
        for module_name, rel_path in packages.items():
            abs_path = workspace_root / rel_path
            try:
                result = subprocess.run(
                    [str(python), "-m", "pip", "install", "-e", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
            except subprocess.TimeoutExpired:
                raise VenvError(
                    f"Timed out installing workspace package '{module_name}' "
                    f"from {abs_path} (exceeded 5 minutes)."
                )

            if result.returncode != 0:
                raise VenvError(
                    f"Failed to install workspace package '{module_name}' "
                    f"from {abs_path}.\n"
                    f"pip output:\n{result.stdout}\n{result.stderr}"
                )

    def install_missing(
        self,
        imports: list[str],
        workspace_skip: set[str] | None = None,
        pip_packages: dict[str, str] | None = None,
    ) -> None:
        """Install any allowed non-stdlib packages not yet in the venv.

        Filters the given import list to only those that are:
        1. In the allowed imports list
        2. Not part of the standard library
        3. Not already installed in the venv
        4. Not a workspace package (handled separately by install_workspace)

        Then pip-installs the remaining packages, using the pip_packages mapping
        to translate import names to PyPI package names where they differ.

        Args:
            imports: List of module names extracted from the snippet's AST.
            workspace_skip: Set of module names that are workspace packages
                and should be skipped (already installed via install_workspace).
            pip_packages: Mapping of import name → pip package name for packages
                where the import name differs from the PyPI name (e.g.
                {"dateutil": "python-dateutil"}).

        Raises:
            VenvError: If pip install fails for any package.
        """
        if workspace_skip is None:
            workspace_skip = set()
        if pip_packages is None:
            pip_packages = {}

        to_install: list[str] = []

        for module_name in imports:
            # Skip if not in the allowed list
            if module_name not in self._allowed_imports:
                continue
            # Skip workspace packages (installed separately)
            if module_name in workspace_skip:
                continue
            # Skip stdlib modules
            if self.is_stdlib(module_name):
                continue
            # Skip already-installed packages
            if self._is_installed(module_name):
                continue
            to_install.append(module_name)

        if not to_install:
            return

        # Translate import names to pip package names where they differ
        pip_names = [pip_packages.get(name, name) for name in to_install]

        python = self.get_python_path()
        try:
            result = subprocess.run(
                [str(python), "-m", "pip", "install", *pip_names],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for pip install
            )
        except subprocess.TimeoutExpired:
            raise VenvError(
                f"Timed out installing packages {to_install} (exceeded 5 minutes). "
                f"Try installing manually: pip install {' '.join(to_install)}"
            )

        if result.returncode != 0:
            raise VenvError(
                f"Failed to install packages {to_install}.\n"
                f"pip output:\n{result.stdout}\n{result.stderr}"
            )
