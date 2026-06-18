"""Tests for GitPython command enforcement (_gitpython_patch).

GitPython bypasses the subprocess proxy (it captures `from subprocess import
Popen` at import). We instead guard its single execution chokepoint,
git.cmd.Git.execute, validating the command vector against [shell.git].

The validator is tested directly against the *shipped* default policy so the
allow-list and the tests can't silently drift. Command vectors below were
captured by instrumenting GitPython against a real repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.config import load_config
from pyddock._gitpython_patch import (
    build_git_command_validator,
    apply_gitpython_patch,
)


@pytest.fixture(scope="module")
def git_allow() -> list[str]:
    """The shipped [shell.git] allow-list from the bundled default config."""
    # tmp workspace with no .pyddock/pyddock.toml -> bundled default_config.toml
    cfg = load_config(Path(__file__).parent)
    assert "git" in cfg.shell, "default config must define [shell.git]"
    return cfg.shell["git"].allow


@pytest.fixture
def validate(git_allow):
    config = {
        "imports": {"allowed": ["git"]},
        "shell": {"git": {"mode": "deny", "allow": git_allow}},
    }
    return build_git_command_validator(config, deny_messages=[])


_SHA = "dad31159f74005374b12210c6c3d555328a6798f"

# Command vectors GitPython actually generates for common READ operations
# (captured via Git.execute instrumentation). All must be permitted.
ALLOWED_COMMANDS = [
    pytest.param(["git", "cat-file", "--batch-check"], id="head_commit"),
    pytest.param(["git", "rev-list", "--max-count=3", _SHA, "--"], id="iter_commits"),
    pytest.param(["git", "status", "--short"], id="status_short"),
    pytest.param(["git", "log", "-1", "--oneline"], id="log"),
    pytest.param(["git", "status", "--porcelain", "--untracked-files"], id="untracked"),
    pytest.param(
        ["git", "diff", "--cached", "--abbrev=40", "--full-index", "--raw"], id="is_dirty"
    ),
    pytest.param(["git", "remote", "get-url", "--all", "origin"], id="remote_urls"),
    pytest.param(["git", "diff", "--stat"], id="diff_stat"),
    pytest.param(["git", "config", "--get", "user.email"], id="config_read"),
    pytest.param(["git", "version"], id="version"),
    # Local write operations are intentionally permitted (no push/remote).
    pytest.param(["git", "add", "."], id="add"),
    pytest.param(["git", "commit", "-m", "msg"], id="commit"),
    pytest.param(["git", "commit", "-m", "subject\n\nbody line\nmore"], id="commit_multiline"),
    pytest.param(["git", "checkout", "-b", "feature"], id="checkout_branch"),
    pytest.param(["git", "merge", "origin/main"], id="merge"),
    pytest.param(["git", "fetch", "origin"], id="fetch"),
    # Benign global options are tolerated.
    pytest.param(["git", "-C", "subdir", "status"], id="dash_C_value"),
    pytest.param(["git", "--git-dir=.git", "status"], id="git_dir_eq"),
    pytest.param(["git", "--no-pager", "log"], id="no_pager_flag"),
    # Absolute / .exe executable path is accepted.
    pytest.param(["C:\\Program Files\\Git\\cmd\\git.exe", "status"], id="abs_exe"),
    pytest.param(["/usr/bin/git", "status"], id="posix_abs_exe"),
]

# Vectors that must be rejected.
DENIED_COMMANDS = [
    pytest.param(["git", "push"], id="push"),
    pytest.param(["git", "push", "origin", "main"], id="push_origin"),
    pytest.param(["git", "remote", "add", "evil", "http://x"], id="remote_add"),
    pytest.param(["git", "remote", "set-url", "origin", "http://x"], id="remote_set_url"),
    pytest.param(["git", "config", "user.email", "x@y.z"], id="config_write"),
    pytest.param(["git", "config", "--global", "core.pager", "cat"], id="config_global_write"),
    pytest.param(["git", "gc"], id="gc_not_allowed"),
    # Option-injection / RCE surface.
    pytest.param(["git", "-c", "core.sshCommand=touch pwned", "fetch"], id="dash_c_inject"),
    pytest.param(["git", "-c", "x=y", "status"], id="dash_c_any"),
    pytest.param(["git", "--exec-path=/tmp/evil", "status"], id="exec_path"),
    pytest.param(["git", "--upload-pack=evil", "fetch"], id="upload_pack"),
    pytest.param(["git", "--config-env=core.sshCommand=X", "fetch"], id="config_env"),
    pytest.param(["git", "--unknown-global", "status"], id="unknown_global"),
    # Non-git executable via repo.git.execute([...]).
    pytest.param(["/bin/sh", "-c", "evil"], id="non_git_exe"),
    pytest.param(["python", "-c", "evil"], id="python_exe"),
    # Malformed.
    pytest.param(["git"], id="no_subcommand"),
    pytest.param([], id="empty"),
    # Verb-anchoring: a short allowed verb must NOT prefix-match a longer,
    # unlisted hyphenated subcommand (the checkout-index escape and friends).
    pytest.param(
        ["git", "checkout-index", "--prefix=.pyddock/", "-f", "--", "x"],
        id="checkout_index",
    ),
    pytest.param(["git", "merge-file", "a", "b", "c"], id="merge_file"),
    pytest.param(["git", "fetch-pack", "--all", "origin"], id="fetch_pack"),
    pytest.param(["git", "commit-tree", "HEAD^{tree}"], id="commit_tree"),
]


class TestValidatorAllow:
    @pytest.mark.parametrize("command", ALLOWED_COMMANDS)
    def test_allowed(self, validate, command) -> None:
        # Should not raise.
        validate(command)


class TestValidatorDeny:
    @pytest.mark.parametrize("command", DENIED_COMMANDS)
    def test_denied(self, validate, command) -> None:
        with pytest.raises(PermissionError):
            validate(command)


class TestValidatorMisc:
    def test_string_command_rejected(self, validate) -> None:
        with pytest.raises(PermissionError):
            validate("git status")

    def test_no_git_policy_denies_everything(self) -> None:
        # git allowed as import but no [shell.git] policy -> nothing permitted.
        v = build_git_command_validator(
            {"imports": {"allowed": ["git"]}, "shell": {}}, deny_messages=[]
        )
        with pytest.raises(PermissionError):
            v(["git", "status"])

    def test_dash_c_not_mistaken_for_subcommand(self, validate) -> None:
        # `-c x=y` must be rejected at the global-option stage, not skipped with
        # `x=y` treated as the subcommand.
        with pytest.raises(PermissionError) as exc:
            validate(["git", "-c", "x=y", "status"])
        assert "-c" in str(exc.value)


class TestPatchApplication:
    def test_noop_when_git_not_allowed(self) -> None:
        installed = apply_gitpython_patch(
            {"imports": {"allowed": ["json"]}, "shell": {}}, deny_messages=[]
        )
        assert installed is False

    def test_guard_installed_and_blocks(self, git_allow) -> None:
        git_cmd = pytest.importorskip("git.cmd")
        Git = git_cmd.Git
        original = Git.execute
        try:
            config = {
                "imports": {"allowed": ["git"]},
                "shell": {"git": {"mode": "deny", "allow": git_allow}},
            }
            installed = apply_gitpython_patch(config, deny_messages=[])
            assert installed is True
            assert getattr(Git.execute, "_pyddock_guarded", False) is True

            # A denied command must raise before any process is spawned. We call
            # the unbound method with a dummy self to avoid needing a real repo.
            class _Dummy:
                pass

            with pytest.raises(PermissionError):
                Git.execute(_Dummy(), ["git", "push", "origin", "main"])
        finally:
            Git.execute = original


class TestArgPathScanning:
    """The arg_paths scan (parity with run_shell / subprocess.run) blocks path
    arguments that target protected dirs or, in "workspace" mode, resolve
    outside the workspace — enforced at the policy's configured level."""

    @pytest.fixture
    def validate_paths(self, git_allow, tmp_path):
        config = {
            "imports": {"allowed": ["git"], "workspace": {}},
            "shell": {
                "git": {
                    "mode": "deny",
                    "allow": git_allow,
                    "arg_paths": "workspace",
                }
            },
        }
        return build_git_command_validator(config, str(tmp_path), deny_messages=[])

    def test_blocks_add_into_pyddock(self, validate_paths) -> None:
        # The escape's spirit: a permitted subcommand carrying a .pyddock/ path.
        with pytest.raises(PermissionError, match=r"\.pyddock/"):
            validate_paths(["git", "add", ".pyddock/pwned.txt"])

    def test_blocks_checkout_pathspec_into_pyddock(self, validate_paths) -> None:
        with pytest.raises(PermissionError, match=r"\.pyddock/"):
            validate_paths(["git", "checkout", "--", ".pyddock/pwned.txt"])

    def test_blocks_path_outside_workspace(self, validate_paths) -> None:
        with pytest.raises(PermissionError, match="outside"):
            validate_paths(["git", "add", "../../etc/passwd"])

    def test_allows_workspace_relative_path(self, validate_paths) -> None:
        validate_paths(["git", "add", "output/result.txt"])  # must not raise

    def test_allows_pyddock_tmp(self, validate_paths) -> None:
        validate_paths(["git", "add", ".pyddock/tmp/scratch.txt"])  # must not raise

    def test_allows_dot(self, validate_paths) -> None:
        validate_paths(["git", "add", "."])  # must not raise

    def test_respects_protected_mode_allows_outside(self, git_allow, tmp_path) -> None:
        # "protected" mode permits paths outside the workspace but still blocks
        # protected dirs.
        config = {
            "imports": {"allowed": ["git"], "workspace": {}},
            "shell": {
                "git": {"mode": "deny", "allow": git_allow, "arg_paths": "protected"}
            },
        }
        v = build_git_command_validator(config, str(tmp_path), deny_messages=[])
        v(["git", "add", "C:/somewhere/else/file.txt"])  # outside, but allowed
        with pytest.raises(PermissionError, match=r"\.pyddock/"):
            v(["git", "add", ".pyddock/pwned.txt"])

    def test_scan_skipped_without_workspace_root(self, git_allow) -> None:
        # No workspace_root -> path scan disabled (back-compat for unit tests
        # that only exercise command/allow validation).
        v = build_git_command_validator(
            {
                "imports": {"allowed": ["git"]},
                "shell": {"git": {"mode": "deny", "allow": git_allow}},
            },
            deny_messages=[],
        )
        v(["git", "add", ".pyddock/whatever.txt"])  # no raise (scan skipped)


class TestLibraryGuardRegistry:
    def test_gitpython_is_registered(self) -> None:
        from pyddock._library_guards import LIBRARY_GUARDS
        names = {g.name for g in LIBRARY_GUARDS}
        assert "gitpython" in names
        gp = next(g for g in LIBRARY_GUARDS if g.name == "gitpython")
        assert gp.import_name == "git"

    def test_guard_self_gates_on_import(self) -> None:
        from pyddock._library_guards import LIBRARY_GUARDS
        gp = next(g for g in LIBRARY_GUARDS if g.name == "gitpython")
        assert gp.applies({"imports": {"allowed": ["git", "json"]}}) is True
        assert gp.applies({"imports": {"allowed": ["json"]}}) is False

    def test_registry_skips_unallowed_guards(self) -> None:
        # git not allowlisted -> gitpython guard is skipped, nothing installed,
        # and the gating short-circuits before any GitPython import is attempted.
        from pyddock._library_guards import apply_library_guards
        installed = apply_library_guards(
            {"imports": {"allowed": ["json"]}, "shell": {}}, deny_messages=[]
        )
        assert installed == []

    def test_registry_does_not_swallow_guard_errors(self) -> None:
        # Fail-loud contract: a guard that raises must propagate, not be silently
        # skipped (which would leave its library unguarded).
        import pyddock._library_guards as lg

        def _boom(config, workspace_root, deny_messages):
            raise RuntimeError("guard exploded")

        boom_guard = lg.LibraryGuard(name="boom", import_name="json", apply_fn=_boom)
        original = lg.LIBRARY_GUARDS
        try:
            lg.LIBRARY_GUARDS = [boom_guard]
            with pytest.raises(RuntimeError, match="guard exploded"):
                lg.apply_library_guards(
                    {"imports": {"allowed": ["json"]}}, deny_messages=[]
                )
        finally:
            lg.LIBRARY_GUARDS = original
