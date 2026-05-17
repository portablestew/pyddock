"""Property-based tests for shell executor correctness.

Uses hypothesis to verify universal properties of policy matching,
argument validation, interpreter mapping, and error messaging.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    ShellPolicyConfig,
)
from pyddock.shell_executor import ShellExecutor


def _make_config(
    shell: dict[str, ShellPolicyConfig] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["json"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
    )


# --- Strategies ---

# Simple command names (alphanumeric, no regex metacharacters)
simple_commands = st.from_regex(r"[a-z][a-z0-9]{0,10}", fullmatch=True)

# Simple arg tokens (no spaces, safe characters)
simple_args = st.from_regex(r"[a-zA-Z0-9_/.\-]{1,20}", fullmatch=True)

# Script extensions
script_extensions = st.sampled_from([".ps1", ".py", ".sh", ".bat"])

# Non-script extensions
non_script_extensions = st.sampled_from([".exe", ".com", ".jar", "", ".rs", ".go"])


# --- Property 1: Policy matching returns first match in insertion order ---

@given(
    cmd=simple_commands,
    n_policies=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=50)
def test_property_first_match_wins(cmd: str, n_policies: int) -> None:
    """For any command matching multiple policies, the first one is returned."""
    # Create n policies that all match the same command
    policies = {}
    for i in range(n_policies):
        name = f"policy{i}"
        policies[name] = ShellPolicyConfig(
            command=f"^{cmd}$",
            mode="deny",
            allow=[f"pattern{i}"],
        )

    config = _make_config(shell=policies)
    executor = ShellExecutor(config, Path("."))
    result = executor._find_matching_policy(cmd)

    assert result is not None
    # First policy's allow pattern should be "pattern0"
    assert result.allow == ["pattern0"]


# --- Property 3: Deny-mode arg validation ---

@given(
    args=st.lists(simple_args, min_size=0, max_size=5),
    allow_patterns=st.lists(
        st.sampled_from(["filelog.*", "files.*", "info", "status.*", "log.*", "diff.*"]),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=100)
def test_property_deny_mode_validation(
    args: list[str], allow_patterns: list[str]
) -> None:
    """Deny-mode permits args iff space-joined string matches at least one allow pattern."""
    import re

    policy = ShellPolicyConfig(
        command="^cmd$", mode="deny", allow=allow_patterns
    )
    config = _make_config(shell={"cmd": policy})
    executor = ShellExecutor(config, Path("."))

    result = executor._check_args_policy(policy, args)
    args_str = " ".join(args)

    # Check if any allow pattern matches
    should_permit = any(re.match(p, args_str) for p in allow_patterns)

    if should_permit:
        assert result is None, f"Expected permit for '{args_str}' with patterns {allow_patterns}"
    else:
        assert result is not None, f"Expected reject for '{args_str}' with patterns {allow_patterns}"


# --- Property 4: Allow-mode arg validation ---

@given(
    args=st.lists(simple_args, min_size=1, max_size=5),
    deny_patterns=st.lists(
        st.sampled_from(["push.*", "force.*", "delete.*", "rm.*", "submit.*"]),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=100)
def test_property_allow_mode_validation(
    args: list[str], deny_patterns: list[str]
) -> None:
    """Allow-mode rejects args iff space-joined string matches any deny pattern."""
    import re

    policy = ShellPolicyConfig(
        command="^cmd$", mode="allow", deny=deny_patterns
    )
    config = _make_config(shell={"cmd": policy})
    executor = ShellExecutor(config, Path("."))

    result = executor._check_args_policy(policy, args)
    args_str = " ".join(args)

    # Check if any deny pattern matches
    should_reject = any(re.match(p, args_str) for p in deny_patterns)

    if should_reject:
        assert result is not None, f"Expected reject for '{args_str}' with deny {deny_patterns}"
    else:
        assert result is None, f"Expected permit for '{args_str}' with deny {deny_patterns}"


# --- Property 5: Interpreter mapping by extension ---

@given(
    base_name=simple_commands,
    ext=script_extensions,
)
@settings(max_examples=50)
def test_property_interpreter_mapping_scripts(base_name: str, ext: str) -> None:
    """Script extensions always get the correct interpreter prefix."""
    config = _make_config()
    executor = ShellExecutor(config, Path("."))
    command = f"{base_name}{ext}"
    result = executor._resolve_command(command)

    if ext == ".ps1":
        assert result[0] in ("pwsh", "powershell")
        assert result[1:] == ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", command]
    elif ext == ".py":
        assert result == ["python", command]
    elif ext == ".sh":
        assert result == ["bash", command]
    elif ext == ".bat":
        assert result == ["cmd", "/c", command]


@given(
    base_name=simple_commands,
    ext=non_script_extensions,
)
@settings(max_examples=50)
def test_property_interpreter_mapping_direct(base_name: str, ext: str) -> None:
    """Non-script extensions always execute directly as [command]."""
    config = _make_config()
    executor = ShellExecutor(config, Path("."))
    command = f"{base_name}{ext}"
    result = executor._resolve_command(command)
    assert result == [command]


# --- Property 12: All error messages suggest run_python ---

@given(cmd=simple_commands)
@settings(max_examples=30)
def test_property_error_messages_suggest_run_python_no_policy(cmd: str) -> None:
    """When command is rejected (no policy), error suggests run_python."""
    config = _make_config(shell={
        "other": ShellPolicyConfig(command="^other$", mode="allow", deny=[]),
    })
    executor = ShellExecutor(config, Path("."))
    assume(cmd != "other")

    result = executor.execute(cmd, [], 10.0)
    assert result.exit_code == 1
    assert "run_python" in result.stderr


@given(args=st.lists(simple_args, min_size=1, max_size=3))
@settings(max_examples=30)
def test_property_error_messages_suggest_run_python_args_rejected(
    args: list[str],
) -> None:
    """When args are rejected, error suggests run_python."""
    # Use a deny-mode policy with a pattern that won't match random args
    config = _make_config(shell={
        "cmd": ShellPolicyConfig(
            command="^cmd$", mode="deny", allow=["^VERY_SPECIFIC_PATTERN_12345$"]
        ),
    })
    executor = ShellExecutor(config, Path("."))

    # These random args won't match the very specific pattern
    args_str = " ".join(args)
    assume(args_str != "VERY_SPECIFIC_PATTERN_12345")

    result = executor.execute("cmd", args, 10.0)
    assert result.exit_code == 1
    assert "run_python" in result.stderr



# --- Property 10: Path-like regex classification and write-protection derivation ---

from pyddock.shell_executor import _derive_write_protected_paths


# Strategies for path-like regexes (contain / or \\ or start with \.)
path_like_regexes = st.sampled_from([
    r"\.kiro/scripts/.*\.ps1",
    r"scripts/build\.sh",
    r"tools/lint\.py",
    r"\.config/hooks/.*",
    r"bin/deploy\.sh",
])

# Strategies for non-path-like regexes (no path separators)
non_path_regexes = st.sampled_from([
    "^p4$",
    "^git$",
    "^npm$",
    "^cargo$",
    "^make$",
    "python",
    "node",
])


@given(regex=path_like_regexes)
@settings(max_examples=30)
def test_property_path_like_produces_protection(regex: str) -> None:
    """Path-like regexes always produce non-empty write-protected path patterns."""
    config = {
        "test": ShellPolicyConfig(command=regex, mode="allow", deny=[]),
    }
    result = _derive_write_protected_paths(config)
    assert len(result) > 0, f"Expected protection for path-like regex: {regex}"


@given(regex=non_path_regexes)
@settings(max_examples=30)
def test_property_non_path_produces_no_protection(regex: str) -> None:
    """Non-path-like regexes never produce write-protected path patterns."""
    config = {
        "test": ShellPolicyConfig(command=regex, mode="allow", deny=[]),
    }
    result = _derive_write_protected_paths(config)
    assert len(result) == 0, f"Expected no protection for non-path regex: {regex}"
