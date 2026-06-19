"""Trust-set tests for _build_trusted_prefixes (venv-authoritative model).

Background
----------
`_caller_is_trusted()` decides whether an import originates from library code
(allowed to import freely) or from agent snippet code (gated by the allowlist).
It compares each caller frame's ``normcase(realpath(co_filename))`` against the
trusted prefixes produced by ``_build_trusted_prefixes()``.

The original cryptography/``__future__`` bug was a resolution-order problem: when
pyddock is provisioned by uv, the bootstrap used to prepend pyddock's install dir
(a shared uv cache that also holds pyddock's deps) to ``sys.path``, so allowed
packages loaded from the untrusted cache instead of ``.pyddock/venv`` and their
internal imports were rejected.

The chosen fix makes ``.pyddock/venv`` authoritative:
  * executor.py APPENDS pyddock's install dir to ``sys.path`` (so the venv wins
    import resolution), and
  * ``_build_trusted_prefixes`` trusts ONLY the venv site-packages (plus the
    realpath targets of the venv's own symlinked entries, for uv symlink-mode),
    workspace module dirs, and stdlib — NOT arbitrary site-packages dirs found
    on ``sys.path``.

Trusting only the venv keeps the trusted set a closed, config-pinned unit, so
sandbox behavior is repeatable regardless of how pyddock itself was installed or
what a shared cache contains. A package missing from the venv fails loudly
rather than being silently trusted from elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from pyddock._proxies import _build_trusted_prefixes


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


def test_venv_site_packages_is_trusted(tmp_path: Path) -> None:
    """The .pyddock/venv site-packages is the authoritative trusted location."""
    workspace = tmp_path / "workspace"
    venv_site = workspace / ".pyddock" / "venv" / "Lib" / "site-packages"
    venv_site.mkdir(parents=True)

    prefixes = _build_trusted_prefixes(
        workspace_root=workspace,
        workspace_imports={},
        venv_path=workspace / ".pyddock" / "venv",
    )

    assert _norm(str(venv_site)) in prefixes


def test_sys_path_site_packages_outside_venv_is_NOT_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site-packages dir on sys.path but outside the venv is NOT trusted.

    This is the deliberate venv-authoritative behavior: a uv cache (or any other
    site-packages dir) injected onto sys.path must not be trusted. Allowed
    packages are expected to resolve from the venv (executor.py appends pyddock's
    install dir so the venv wins resolution); anything that still loads from the
    cache fails loudly instead of being silently trusted.
    """
    # Simulate uv's cache archive being on sys.path.
    cache_site = tmp_path / "uv_cache" / "archive-v0" / "deadbeef" / "Lib" / "site-packages"
    cache_site.mkdir(parents=True)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.syspath_prepend(str(cache_site))

    prefixes = _build_trusted_prefixes(
        workspace_root=workspace,
        workspace_imports={},
        venv_path=None,
    )

    assert _norm(str(cache_site)) not in prefixes, (
        "a site-packages dir on sys.path (outside the venv) must NOT be trusted"
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs elevation on Windows CI"
)
def test_venv_symlinked_package_into_cache_is_trusted(
    tmp_path: Path,
) -> None:
    """uv symlink-mode: the venv's own packages stored in a shared cache.

    The venv site-packages dir is real, but each package inside is a symlink into
    the global cache. Those are still the venv's declared contents, so the
    symlink target's directory must be trusted.
    """
    cache_site = tmp_path / "cache" / "Lib" / "site-packages"
    cache_site.mkdir(parents=True)
    real_pkg = cache_site / "somepkg"
    real_pkg.mkdir()
    (real_pkg / "__init__.py").write_text("x = 1\n")

    workspace = tmp_path / "workspace"
    venv_site = workspace / ".pyddock" / "venv" / "Lib" / "site-packages"
    venv_site.mkdir(parents=True)
    os.symlink(real_pkg, venv_site / "somepkg", target_is_directory=True)

    prefixes = _build_trusted_prefixes(
        workspace_root=workspace,
        workspace_imports={},
        venv_path=workspace / ".pyddock" / "venv",
    )

    assert _norm(str(cache_site)) in prefixes, (
        "the venv's symlink target (the shared cache) must be trusted"
    )


def test_workspace_internal_site_packages_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site-packages dir INSIDE the workspace is skipped (agent-writable).

    Even though sys.path scanning is gone, a venv symlink could in principle
    resolve back into the workspace; the agent-writable guard still applies.
    """
    workspace = tmp_path / "workspace"
    rogue_site = workspace / "vendor" / "site-packages"
    rogue_site.mkdir(parents=True)

    monkeypatch.syspath_prepend(str(rogue_site))

    prefixes = _build_trusted_prefixes(
        workspace_root=workspace,
        workspace_imports={},
        venv_path=None,
    )

    assert _norm(str(rogue_site)) not in prefixes


def test_pyddock_venv_inside_workspace_is_still_trusted(
    tmp_path: Path,
) -> None:
    """The managed .pyddock/venv lives in the workspace but is write-protected.

    It is exempt from the workspace-internal skip, so it stays trusted even
    though it is technically inside the workspace root.
    """
    workspace = tmp_path / "workspace"
    venv_site = workspace / ".pyddock" / "venv" / "Lib" / "site-packages"
    venv_site.mkdir(parents=True)

    prefixes = _build_trusted_prefixes(
        workspace_root=workspace,
        workspace_imports={},
        venv_path=workspace / ".pyddock" / "venv",
    )

    assert _norm(str(venv_site)) in prefixes
