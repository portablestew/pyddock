"""Tests for the VenvManager stamp file (installed.json) logic.

Verifies that:
- A valid stamp skips install checks (no subprocess spawns).
- A changed allowlist triggers install checks.
- A changed workspace pyproject.toml triggers reinstall.
- A missing/corrupt stamp falls through to full install.
- A recreated venv (different pyvenv.cfg timestamp) invalidates the stamp.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pyddock.venv_manager import (
    VenvManager,
    _get_venv_created,
    _hash_file,
    _hash_string,
    _STAMP_FILENAME,
)


@pytest.fixture
def venv_dir(tmp_path: Path) -> Path:
    """Create a fake venv directory with pyvenv.cfg and python executable."""
    venv_path = tmp_path / ".pyddock" / "venv"
    venv_path.mkdir(parents=True)
    # Create pyvenv.cfg (its timestamp is used for identity)
    (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    # Create fake python executable
    if sys.platform == "win32":
        scripts = venv_path / "Scripts"
        scripts.mkdir()
        (scripts / "python.exe").write_text("")
    else:
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").write_text("")
    return venv_path


@pytest.fixture
def manager(venv_dir: Path) -> VenvManager:
    """VenvManager pointed at the fake venv."""
    mgr = VenvManager(venv_path=venv_dir, allowed_imports=["polars", "dateutil"])
    # Mock get_python_path so _is_installed doesn't need a real interpreter
    mgr.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
    return mgr


@pytest.fixture
def stamp_path(venv_dir: Path) -> Path:
    """Path where the stamp file should be written."""
    return venv_dir.parent / _STAMP_FILENAME


class TestStampSkipsInstall:
    """When the stamp is valid and hashes match, no installs should run."""

    def test_matching_allowlist_skips_install_checks(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """If allowlist hash matches the stamp, _is_installed is never called."""
        # Compute the expected hash and write a valid stamp
        allowlist_hash = manager._compute_allowlist_hash({"dateutil": "python-dateutil"})
        venv_created = _get_venv_created(venv_dir)
        stamp_data = {
            "venv_created": venv_created,
            "allowlist_hash": allowlist_hash,
            "workspace": {},
        }
        stamp_path.write_text(json.dumps(stamp_data))

        # Patch _is_installed — it should NOT be called
        with patch.object(manager, "_is_installed") as mock_check:
            manager.install_missing(
                imports=["polars", "dateutil"],
                workspace_skip=set(),
                pip_packages={"dateutil": "python-dateutil"},
            )
            mock_check.assert_not_called()

    def test_matching_workspace_hash_skips_pip_install(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path, tmp_path: Path
    ) -> None:
        """If workspace module's pyproject.toml hash matches, pip install -e is skipped."""
        # Create a workspace module with pyproject.toml
        ws_root = tmp_path / "workspace"
        ws_root.mkdir()
        mod_path = ws_root / "my_mod"
        mod_path.mkdir()
        toml = mod_path / "pyproject.toml"
        toml.write_text('[project]\nname = "my_mod"\n')

        # Write stamp with matching hash
        toml_hash = _hash_file(toml)
        venv_created = _get_venv_created(venv_dir)
        stamp_data = {
            "venv_created": venv_created,
            "allowlist_hash": "",
            "workspace": {"my_mod": toml_hash},
        }
        stamp_path.write_text(json.dumps(stamp_data))

        # Patch subprocess.run — it should NOT be called for workspace install
        with patch("pyddock.venv_manager.subprocess.run") as mock_run:
            manager.install_workspace({"my_mod": "my_mod"}, ws_root)
            mock_run.assert_not_called()


class TestStampDetectsChanges:
    """When config or metadata changes, installs should run."""

    def test_changed_allowlist_triggers_install_checks(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """If allowlist hash differs from stamp, _is_installed is called."""
        # Write stamp with a DIFFERENT allowlist hash
        venv_created = _get_venv_created(venv_dir)
        stamp_data = {
            "venv_created": venv_created,
            "allowlist_hash": "stale_hash_value",
            "workspace": {},
        }
        stamp_path.write_text(json.dumps(stamp_data))

        # Patch _is_installed to return True (already installed, no pip needed)
        with patch.object(manager, "_is_installed", return_value=True) as mock_check:
            manager.install_missing(
                imports=["polars"],
                workspace_skip=set(),
                pip_packages={},
            )
            # _is_installed should have been called for polars
            mock_check.assert_called()

    def test_changed_workspace_toml_triggers_reinstall(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path, tmp_path: Path
    ) -> None:
        """If workspace pyproject.toml hash differs, pip install -e runs."""
        # Create workspace module
        ws_root = tmp_path / "workspace"
        ws_root.mkdir()
        mod_path = ws_root / "my_mod"
        mod_path.mkdir()
        toml = mod_path / "pyproject.toml"
        toml.write_text('[project]\nname = "my_mod"\nversion = "1.0"\n')

        # Write stamp with OLD hash
        venv_created = _get_venv_created(venv_dir)
        stamp_data = {
            "venv_created": venv_created,
            "allowlist_hash": "",
            "workspace": {"my_mod": "old_hash_that_wont_match"},
        }
        stamp_path.write_text(json.dumps(stamp_data))

        # Patch subprocess.run to simulate successful install
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("pyddock.venv_manager.subprocess.run", return_value=mock_result) as mock_run:
            manager.install_workspace({"my_mod": "my_mod"}, ws_root)
            # pip install -e should have been called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "-e" in args


class TestStampInvalidation:
    """Corrupt or stale stamps should fall through to full install."""

    def test_missing_stamp_triggers_full_install(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """No stamp file means _is_installed is called (full check)."""
        assert not stamp_path.exists()

        with patch.object(manager, "_is_installed", return_value=True) as mock_check:
            manager.install_missing(
                imports=["polars"],
                workspace_skip=set(),
                pip_packages={},
            )
            mock_check.assert_called()

    def test_corrupt_stamp_triggers_full_install(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """Invalid JSON in stamp file triggers full install pass."""
        stamp_path.write_text("not valid json {{{")

        with patch.object(manager, "_is_installed", return_value=True) as mock_check:
            manager.install_missing(
                imports=["polars"],
                workspace_skip=set(),
                pip_packages={},
            )
            mock_check.assert_called()

    def test_recreated_venv_invalidates_stamp(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """If pyvenv.cfg has a different timestamp, stamp is invalidated."""
        # Write stamp with a venv_created that won't match
        stamp_data = {
            "venv_created": "99999999999999",
            "allowlist_hash": manager._compute_allowlist_hash({}),
            "workspace": {},
        }
        stamp_path.write_text(json.dumps(stamp_data))

        with patch.object(manager, "_is_installed", return_value=True) as mock_check:
            manager.install_missing(
                imports=["polars"],
                workspace_skip=set(),
                pip_packages={},
            )
            # Should have fallen through to full check
            mock_check.assert_called()

    def test_stamp_written_after_install(
        self, manager: VenvManager, venv_dir: Path, stamp_path: Path
    ) -> None:
        """After install_missing completes, a valid stamp file exists."""
        assert not stamp_path.exists()

        with patch.object(manager, "_is_installed", return_value=True):
            manager.install_missing(
                imports=["polars"],
                workspace_skip=set(),
                pip_packages={"dateutil": "python-dateutil"},
            )

        assert stamp_path.exists()
        data = json.loads(stamp_path.read_text())
        assert "allowlist_hash" in data
        assert "venv_created" in data
        assert data["venv_created"] == _get_venv_created(venv_dir)
