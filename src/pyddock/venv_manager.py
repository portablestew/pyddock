"""Virtual environment manager for pyddock.

Creates and manages a venv at `.pyddock/venv/`, auto-installs allowed
third-party packages, and exposes the venv Python path for subprocess use.

Uses a stamp file (.pyddock/venv/installed.json) to skip redundant install
checks on subsequent boots. The stamp records:
- venv_created: creation timestamp of pyvenv.cfg (detects venv recreation)
- allowlist_hash: hash of sorted allowlist + pip_packages (detects config changes)
- workspace: {module_name: hash_of_pyproject_toml} (detects metadata changes)

If the stamp is missing, corrupt, or stale, a full install pass runs (same as
the previous always-install behavior) and a fresh stamp is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import venv
from pathlib import Path

logger = logging.getLogger(__name__)

# Name of the stamp file inside the venv directory.
_STAMP_FILENAME = "installed.json"


class VenvError(Exception):
    """Raised when venv creation or package installation fails."""


def _hash_string(s: str) -> str:
    """Return a short hex digest of a string (SHA-256, first 16 hex chars)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    """Return a short hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _get_venv_created(venv_path: Path) -> str | None:
    """Return the creation/mtime timestamp of pyvenv.cfg as an ISO string.

    Uses creation time on Windows (st_ctime_ns) and mtime on Unix.
    Returns None if pyvenv.cfg doesn't exist.
    """
    cfg = venv_path / "pyvenv.cfg"
    if not cfg.exists():
        return None
    stat = cfg.stat()
    # On Windows, st_ctime is the file creation time.
    # On Unix, st_ctime is the last metadata change (not creation), so use mtime.
    if sys.platform == "win32":
        ts = stat.st_ctime_ns
    else:
        ts = stat.st_mtime_ns
    return str(ts)


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
        self._stamp: dict | None = None  # loaded stamp data (None = not loaded yet)

    # ------------------------------------------------------------------
    # Stamp file management
    # ------------------------------------------------------------------

    def _stamp_path(self) -> Path:
        """Path to the stamp file in the .pyddock directory (parent of venv)."""
        return self._venv_path.parent / _STAMP_FILENAME

    def _load_stamp(self) -> dict:
        """Load and validate the stamp file.

        Returns the parsed dict if valid and the venv_created timestamp
        matches the current pyvenv.cfg. Returns an empty dict (triggering
        full install) if the stamp is missing, corrupt, or stale.
        """
        if self._stamp is not None:
            return self._stamp

        stamp_path = self._stamp_path()
        if not stamp_path.exists():
            logger.debug("No stamp file found at %s", stamp_path)
            self._stamp = {}
            return self._stamp

        try:
            data = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Stamp file unreadable, will rebuild: %s", e)
            self._stamp = {}
            return self._stamp

        if not isinstance(data, dict):
            self._stamp = {}
            return self._stamp

        # Validate venv identity — if pyvenv.cfg was recreated, stamp is stale
        expected_created = _get_venv_created(self._venv_path)
        if expected_created is None or data.get("venv_created") != expected_created:
            logger.debug(
                "Venv timestamp mismatch (stamp=%s, actual=%s), will rebuild",
                data.get("venv_created"),
                expected_created,
            )
            self._stamp = {}
            return self._stamp

        self._stamp = data
        return self._stamp

    def _save_stamp(
        self,
        *,
        allowlist_hash: str,
        workspace_hashes: dict[str, str],
    ) -> None:
        """Write the stamp file with current state.

        Overwrites any existing stamp. If writing fails (permissions, disk full),
        logs a warning but does not raise — the next boot will just do a full
        install again.
        """
        venv_created = _get_venv_created(self._venv_path)
        data = {
            "venv_created": venv_created,
            "allowlist_hash": allowlist_hash,
            "workspace": workspace_hashes,
        }
        try:
            self._stamp_path().write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Failed to write stamp file: %s", e)
        self._stamp = data

    def _compute_allowlist_hash(self, pip_packages: dict[str, str] | None = None) -> str:
        """Compute a hash representing the current allowlist + pip_packages mapping."""
        if pip_packages is None:
            pip_packages = {}
        # Deterministic: sorted allowlist + sorted pip_packages items
        parts = sorted(self._allowed_imports)
        parts.append("||")
        parts.extend(f"{k}={v}" for k, v in sorted(pip_packages.items()))
        return _hash_string("\n".join(parts))

    # ------------------------------------------------------------------
    # Venv lifecycle
    # ------------------------------------------------------------------

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

        Skips packages whose pyproject.toml hash matches the stamp file,
        meaning their project metadata hasn't changed since last install.

        Args:
            packages: Mapping of module name → relative path from workspace root.
            workspace_root: The workspace root directory to resolve relative paths.

        Raises:
            VenvError: If pip install -e fails for any package.
        """
        if not packages:
            return

        stamp = self._load_stamp()
        stamp_workspace = stamp.get("workspace", {})
        python = self.get_python_path()
        new_hashes: dict[str, str] = {}

        for module_name, rel_path in packages.items():
            abs_path = workspace_root / rel_path
            # Hash pyproject.toml (or setup.py/setup.cfg) to detect metadata changes
            toml_path = abs_path / "pyproject.toml"
            if not toml_path.exists():
                # Try setup.py as fallback
                toml_path = abs_path / "setup.py"
            if not toml_path.exists():
                # Try setup.cfg as fallback
                toml_path = abs_path / "setup.cfg"

            if toml_path.exists():
                current_hash = _hash_file(toml_path)
                new_hashes[module_name] = current_hash
                # Skip if hash matches stamp
                if stamp_workspace.get(module_name) == current_hash:
                    logger.debug(
                        "Workspace module '%s' unchanged (hash=%s), skipping install",
                        module_name,
                        current_hash,
                    )
                    continue
            else:
                # No project metadata file — nothing to install or track.
                # The module is on sys.path via trusted prefixes already.
                logger.debug(
                    "Workspace module '%s' has no project metadata, skipping install",
                    module_name,
                )
                continue

            logger.debug("Installing workspace module '%s' from %s", module_name, abs_path)
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

        # Store computed workspace hashes for _save_stamp (called after install_missing)
        self._workspace_hashes = new_hashes

    def install_missing(
        self,
        imports: list[str],
        workspace_skip: set[str] | None = None,
        pip_packages: dict[str, str] | None = None,
    ) -> None:
        """Install any allowed non-stdlib packages not yet in the venv.

        Uses the stamp file to skip the entire check when the allowlist hash
        hasn't changed. If the hash matches, all packages are assumed installed.
        If it doesn't match, falls back to per-package checks and installs.

        After completing, writes a fresh stamp file recording the current state.

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

        allowlist_hash = self._compute_allowlist_hash(pip_packages)
        stamp = self._load_stamp()

        # Fast path: if allowlist hash matches, all packages are installed
        if stamp.get("allowlist_hash") == allowlist_hash:
            logger.debug("Allowlist hash unchanged (%s), skipping install checks", allowlist_hash)
            # Still save stamp to update workspace hashes if they changed
            self._save_stamp(
                allowlist_hash=allowlist_hash,
                workspace_hashes=getattr(self, "_workspace_hashes", {}),
            )
            return

        logger.debug("Allowlist hash changed (stamp=%s, current=%s), checking packages",
                     stamp.get("allowlist_hash"), allowlist_hash)

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

        if to_install:
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

        # Write stamp with current state
        self._save_stamp(
            allowlist_hash=allowlist_hash,
            workspace_hashes=getattr(self, "_workspace_hashes", {}),
        )
