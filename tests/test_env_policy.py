"""Unit tests for the subprocess env policy.

Covers the shared primitives in shell_executor:
  * is_unsafe_env_value — the path/URI value detector
  * resolve_env_policy   — global [env] base merged with per-command env
  * filter_child_env     — the proxy's full filter (rewrite + reject)
  * assert_env_locks     — the audit-layer hard-lock backstop (deny-only)

Plus the bundled default_config.toml policy end-to-end (git/p4/docker locks).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.config import EnvConfig, ShellPolicyConfig, load_config
from pyddock.shell_executor import (
    assert_env_locks,
    filter_child_env,
    is_unsafe_env_value,
    resolve_env_policy,
)

SNAPSHOT = {"PATH": "/usr/bin", "LANG": "C", "HOME": "/home/u"}


# ---------------------------------------------------------------------------
# is_unsafe_env_value
# ---------------------------------------------------------------------------

class TestIsUnsafeEnvValue:
    @pytest.mark.parametrize(
        "value",
        [
            "/abs/path",
            "rel/path",
            "win\\path",
            "\\\\server\\share",   # UNC — must NOT be excluded like shell args
            "./relative",
            "../up",
            "~/home",
            "C:\\Windows",
            "tcp://host:2375",
            "http://example.com",
            "ssh://git@host/repo",
        ],
    )
    def test_path_or_uri_like_values_are_unsafe(self, value: str) -> None:
        assert is_unsafe_env_value(value) is True

    @pytest.mark.parametrize(
        "value",
        ["", "C", "en_US.UTF-8", "1", "0", "a@b.com", "John Doe", "UTC", "dumb"],
    )
    def test_inert_scalars_are_safe(self, value: str) -> None:
        assert is_unsafe_env_value(value) is False

    def test_bare_command_is_not_path_like(self) -> None:
        # The known limitation the per-command hard-locks exist for: a bare
        # command name is not path/URI-like, so the value filter alone wouldn't
        # catch GIT_SSH_COMMAND=calc.exe — a deny lock must.
        assert is_unsafe_env_value("calc.exe") is False

    def test_bare_filename_that_exists_is_unsafe(self, tmp_path: Path) -> None:
        # A separator-less value that names a REAL file/dir is unsafe: a tool may
        # resolve it as a path relative to its cwd (e.g. p4 P4ENVIRO=z.txt). This
        # is the bare-filename gap that let P4ENVIRO slip past the inert filter.
        (tmp_path / "z_enviro.txt").write_text("P4EDITOR=evil")
        (tmp_path / "adir").mkdir()
        assert is_unsafe_env_value("z_enviro.txt", (tmp_path,)) is True
        assert is_unsafe_env_value("adir", (tmp_path,)) is True

    def test_bare_filename_that_does_not_exist_stays_inert(self, tmp_path: Path) -> None:
        assert is_unsafe_env_value("nope.txt", (tmp_path,)) is False

    def test_resolve_bases_checks_each_base(self, tmp_path: Path) -> None:
        base_a = tmp_path / "a"; base_a.mkdir()
        base_b = tmp_path / "b"; base_b.mkdir()
        (base_b / "settings").write_text("x")
        # Present only under the second base — still detected.
        assert is_unsafe_env_value("settings", (base_a, base_b)) is True


# ---------------------------------------------------------------------------
# resolve_env_policy
# ---------------------------------------------------------------------------

class TestResolveEnvPolicy:
    def test_merges_global_and_command_deny(self) -> None:
        base = EnvConfig(default="inert", deny=["PATH"])
        cmd = ShellPolicyConfig(command="^git$", mode="deny", env={"deny": ["GIT_.*"]})
        deny, default = resolve_env_policy(base, cmd)
        assert deny == ["PATH", "GIT_.*"]
        assert default == "inert"

    def test_command_overrides_default(self) -> None:
        base = EnvConfig(default="inert", deny=[])
        cmd = ShellPolicyConfig(command="^x$", mode="deny", env={"default": "snapshot"})
        _deny, default = resolve_env_policy(base, cmd)
        assert default == "snapshot"

    def test_accepts_dict_forms(self) -> None:
        # Proxy/audit pass serialized dicts, not dataclasses.
        deny, default = resolve_env_policy(
            {"default": "inert", "deny": ["PATH"]},
            {"env": {"deny": ["GIT_.*"]}},
        )
        assert deny == ["PATH", "GIT_.*"]
        assert default == "inert"

    def test_none_command_policy(self) -> None:
        deny, default = resolve_env_policy({"deny": ["PATH"]}, None)
        assert deny == ["PATH"]
        assert default == "inert"


# ---------------------------------------------------------------------------
# filter_child_env (proxy path)
# ---------------------------------------------------------------------------

class TestFilterChildEnv:
    DENY = ["PATH", "LD_.*", "GIT_SSH_COMMAND", "GIT_CONFIG.*"]

    def _filter(self, agent_env, default="inert"):
        return filter_child_env(
            agent_env, SNAPSHOT, deny_patterns=self.DENY, default=default
        )

    def test_none_returns_snapshot_copy(self) -> None:
        out = self._filter(None)
        assert out == SNAPSHOT
        assert out is not SNAPSHOT  # copy, not alias

    def test_unmentioned_keys_keep_snapshot(self) -> None:
        out = self._filter({"LANG": "C"})  # equals snapshot → no-op
        assert out["PATH"] == "/usr/bin"
        assert out["HOME"] == "/home/u"

    def test_inert_override_allowed(self) -> None:
        out = self._filter({"GIT_AUTHOR_EMAIL": "a@b.com"})
        assert out["GIT_AUTHOR_EMAIL"] == "a@b.com"

    def test_empty_value_removes_var(self) -> None:
        out = self._filter({"LANG": ""})
        assert "LANG" not in out

    def test_snapshot_match_allowed_even_for_locked_key(self) -> None:
        # Passing back the known-good PATH value is a no-op, not a violation.
        out = self._filter({"PATH": "/usr/bin"})
        assert out["PATH"] == "/usr/bin"

    @pytest.mark.parametrize(
        "agent_env",
        [
            {"PATH": "/evil:/usr/bin"},        # locked, path
            {"LD_PRELOAD": "/x.so"},           # locked (regex), path
            {"GIT_SSH_COMMAND": "calc.exe"},   # locked, BARE command (not path)
            {"GIT_CONFIG_COUNT": "1"},         # locked (regex), inert value
        ],
    )
    def test_locked_keys_rejected(self, agent_env: dict) -> None:
        with pytest.raises(PermissionError):
            self._filter(agent_env)

    def test_inert_default_rejects_path_like_unknown_key(self) -> None:
        with pytest.raises(PermissionError):
            self._filter({"SOME_NEW_TOOL_HOME": "/opt/tool"})

    def test_inert_default_rejects_bare_filename_that_exists(self, tmp_path: Path) -> None:
        # End-to-end of the bare-filename gap through the proxy filter: an
        # unknown var set to a separator-less value naming a real file is
        # rejected when resolve_bases (the spawn's cwd/workspace) is supplied.
        (tmp_path / "z_enviro.txt").write_text("P4EDITOR=evil")
        with pytest.raises(PermissionError):
            filter_child_env(
                {"P4ENVIRO_UNKNOWN": "z_enviro.txt"}, SNAPSHOT,
                deny_patterns=self.DENY, default="inert",
                resolve_bases=(tmp_path,),
            )

    def test_inert_default_allows_bare_filename_when_absent(self, tmp_path: Path) -> None:
        out = filter_child_env(
            {"SOME_TOKEN": "not_a_real_file"}, SNAPSHOT,
            deny_patterns=self.DENY, default="inert",
            resolve_bases=(tmp_path,),
        )
        assert out["SOME_TOKEN"] == "not_a_real_file"

    def test_snapshot_default_rejects_any_override(self) -> None:
        with pytest.raises(PermissionError):
            self._filter({"GIT_AUTHOR_EMAIL": "a@b.com"}, default="snapshot")

    def test_snapshot_default_allows_removal(self) -> None:
        out = self._filter({"LANG": ""}, default="snapshot")
        assert "LANG" not in out

    def test_bytes_key_and_value_coerced(self) -> None:
        out = self._filter({b"GIT_AUTHOR_NAME": b"Jane"})
        assert out["GIT_AUTHOR_NAME"] == "Jane"


# ---------------------------------------------------------------------------
# assert_env_locks (audit backstop)
# ---------------------------------------------------------------------------

class TestAssertEnvLocks:
    DENY = ["GIT_SSH_COMMAND", "GIT_CONFIG.*", "LD_.*"]

    def test_none_is_noop(self) -> None:
        assert_env_locks(None, SNAPSHOT, self.DENY)  # no raise

    def test_locked_poisoned_key_raises(self) -> None:
        with pytest.raises(PermissionError):
            assert_env_locks({"GIT_SSH_COMMAND": "calc"}, SNAPSHOT, self.DENY)

    def test_non_locked_key_ignored_even_if_path_like(self) -> None:
        # The backstop is deny-only on hard-locks; it does NOT run the inert
        # filter, so a benign library-set inert var is not a false positive.
        assert_env_locks({"SOME_PATH": "/opt/x"}, SNAPSHOT, self.DENY)  # no raise

    def test_locked_key_matching_snapshot_ok(self) -> None:
        snap = {**SNAPSHOT, "GIT_CONFIG_GLOBAL": "/etc/gitconfig"}
        assert_env_locks({"GIT_CONFIG_GLOBAL": "/etc/gitconfig"}, snap, self.DENY)

    def test_locked_key_removal_ok(self) -> None:
        assert_env_locks({"GIT_SSH_COMMAND": ""}, SNAPSHOT, self.DENY)  # no raise


# ---------------------------------------------------------------------------
# Bundled default_config.toml policy, end-to-end
# ---------------------------------------------------------------------------

class TestBundledDefaultPolicy:
    @pytest.fixture(scope="class")
    def cfg(self):
        # Path with no workspace .pyddock/pyddock.toml → bundled default.
        return load_config(Path(__file__).resolve().parent.parent)

    def test_global_env_present(self, cfg) -> None:
        assert cfg.env.default == "inert"
        # PATH-family / loader / home vars are locked via (case-insensitive)
        # regex rather than literal names. Assert the behavior, not the spelling.
        for key in ("PATH", "Path", "LD_PRELOAD", "HOME", "home", "PYTHONSTARTUP"):
            with pytest.raises(PermissionError):
                filter_child_env(
                    {key: "evilvalue"}, SNAPSHOT,
                    deny_patterns=cfg.env.deny, default=cfg.env.default,
                )

    @pytest.mark.parametrize(
        "command,key",
        [
            ("git", "GIT_SSH_COMMAND"),
            ("git", "GIT_CONFIG_COUNT"),
            ("p4", "P4PORT"),
            ("p4", "P4ENVIRO"),        # settings-file loader (the bare-filename escape)
            ("p4", "P4DIFF"),          # external diff (base name, via P4DIFF.*)
            ("p4", "P4DIFFUNICODE"),   # external diff (unicode variant, via P4DIFF.*)
            ("p4", "P4MERGE"),         # external merge (base name, via P4MERGE.*)
            ("p4", "P4MERGEUNICODE"),  # external merge (unicode variant, via P4MERGE.*)
            ("p4", "P4PAGER"),         # external pager
            ("p4", "P4ALIASES"),       # allow-list bypass via alias rewrite
            ("docker", "DOCKER_HOST"),
        ],
    )
    def test_command_locks_enforced(self, cfg, command: str, key: str) -> None:
        deny, default = resolve_env_policy(cfg.env, cfg.shell[command])
        with pytest.raises(PermissionError):
            filter_child_env(
                {key: "evilvalue"}, SNAPSHOT, deny_patterns=deny, default=default
            )

    def test_benign_override_passes_for_git(self, cfg) -> None:
        deny, default = resolve_env_policy(cfg.env, cfg.shell["git"])
        out = filter_child_env(
            {"GIT_AUTHOR_EMAIL": "a@b.com"}, SNAPSHOT,
            deny_patterns=deny, default=default,
        )
        assert out["GIT_AUTHOR_EMAIL"] == "a@b.com"


# ---------------------------------------------------------------------------
# End-to-end: the proxy actually filters env in the live sandbox subprocess
# ---------------------------------------------------------------------------

import sys

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
)
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager


class TestEnvPolicyEndToEnd:
    """Drive a real run_python subprocess and confirm env overrides are gated.

    The env filter runs BEFORE the spawn, so these assertions are deterministic
    and do not require the target binary (git) to exist on the test machine.
    """

    @pytest.fixture
    def executor(self, tmp_path: Path) -> SubprocessExecutor:
        cfg = PyddockConfig(
            execution=ExecutionConfig(timeout=30.0),
            imports=ImportsConfig(allowed=["subprocess", "sys"]),
            filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["."]),
            ast=ASTConfig(block_calls=["eval", "exec"], block_attributes=["__globals__"]),
            shell={
                "git": ShellPolicyConfig(
                    command="^git$",
                    mode="deny",
                    allow=["status(\\s|$)", "--version(\\s|$)"],
                    env={"deny": ["GIT_SSH_COMMAND", "GIT_CONFIG.*"]},
                )
            },
            env=EnvConfig(default="inert", deny=["PATH", "LD_.*"]),
        )
        manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
        manager.get_python_path = lambda: Path(sys.executable)  # type: ignore[method-assign]
        return SubprocessExecutor(cfg, manager)

    def _run(self, executor: SubprocessExecutor, workspace: Path, snippet: str):
        return executor.execute(snippet, args=[], timeout=20, workspace_root=workspace)

    @pytest.mark.parametrize(
        "poison",
        [
            "{'GIT_SSH_COMMAND': 'calc'}",   # per-command lock (bare command)
            "{'GIT_CONFIG_COUNT': '1'}",      # per-command lock (regex)
            "{'PATH': '/evil'}",              # global lock
            "{'LD_PRELOAD': '/x.so'}",        # global lock (regex)
            "{'NEW_TOOL': '/opt/tool'}",      # inert default: path-like unknown
        ],
    )
    def test_poisoned_env_is_blocked(
        self, executor: SubprocessExecutor, tmp_path: Path, poison: str
    ) -> None:
        snippet = (
            "import subprocess\n"
            "try:\n"
            f"    subprocess.run(['git', 'status'], env={poison})\n"
            "    print('OUTCOME:NOERROR')\n"
            "except PermissionError:\n"
            "    print('OUTCOME:BLOCKED')\n"
        )
        result = self._run(executor, tmp_path, snippet)
        assert "OUTCOME:BLOCKED" in result.stdout, result.stdout + result.stderr

    def test_benign_inert_override_not_blocked_by_policy(
        self, executor: SubprocessExecutor, tmp_path: Path
    ) -> None:
        # GIT_AUTHOR_EMAIL is inert and not locked → the env filter permits it.
        # The spawn may then fail if git is absent (FileNotFoundError), which is
        # fine: we only assert the env policy did NOT reject it.
        snippet = (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['git', '--version'], env={'GIT_AUTHOR_EMAIL': 'a@b.com'})\n"
            "    print('OUTCOME:RAN')\n"
            "except PermissionError as e:\n"
            "    print('OUTCOME:BLOCKED', e)\n"
            "except FileNotFoundError:\n"
            "    print('OUTCOME:RAN')\n"
        )
        result = self._run(executor, tmp_path, snippet)
        assert "OUTCOME:RAN" in result.stdout, result.stdout + result.stderr
