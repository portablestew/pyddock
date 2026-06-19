"""Unit tests for deny_messages feature — config parsing, hint lookup, and integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.config import (
    ASTConfig,
    ConfigError,
    DenyMessageRule,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
    ShellPolicyConfig,
    find_deny_hint,
    load_config,
)
from pyddock.shell_executor import ShellExecutor

from tests._config_helpers import write_workspace_config


def _make_config(
    shell: dict[str, ShellPolicyConfig] | None = None,
    deny_messages: list[DenyMessageRule] | None = None,
) -> PyddockConfig:
    """Create a config with sensible defaults for testing."""
    import re
    return PyddockConfig(
        execution=ExecutionConfig(timeout=30.0),
        imports=ImportsConfig(allowed=["json", "os"]),
        filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["*"]),
        ast=ASTConfig(block_calls=[], block_attributes=[]),
        shell=shell or {},
        deny_messages=deny_messages or [],
    )


class TestFindDenyHint:
    """Tests for find_deny_hint() matching logic."""

    def test_first_match_wins(self) -> None:
        """When multiple patterns match, the first one wins."""
        import re
        rules = [
            DenyMessageRule(pattern=re.compile("git push"), message="specific push hint"),
            DenyMessageRule(pattern=re.compile("^git "), message="generic git hint"),
        ]
        assert find_deny_hint("git push origin", rules) == "specific push hint"

    def test_no_match_returns_none(self) -> None:
        """When no pattern matches, returns None."""
        import re
        rules = [
            DenyMessageRule(pattern=re.compile("aws"), message="aws hint"),
        ]
        assert find_deny_hint("git status", rules) is None

    def test_search_not_match(self) -> None:
        """Pattern uses re.search, not re.match — matches anywhere in string."""
        import re
        rules = [
            DenyMessageRule(pattern=re.compile("push"), message="push hint"),
        ]
        assert find_deny_hint("git push origin", rules) == "push hint"

    def test_empty_rules_returns_none(self) -> None:
        """Empty rules list always returns None."""
        assert find_deny_hint("anything", []) is None


class TestDenyMessagesConfigParsing:
    """Tests for [deny_messages] config section parsing."""

    def test_valid_section_parsed(self, tmp_path: Path) -> None:
        """Valid [deny_messages] section is parsed into DenyMessageRule list."""
        write_workspace_config(
            tmp_path,
            extra=(
                "[deny_messages]\n"
                '"aws" = "Use boto3 instead."\n'
                '"git push" = "Push is not allowed."\n'
            ),
        )

        config = load_config(tmp_path)
        assert len(config.deny_messages) == 2
        assert config.deny_messages[0].message == "Use boto3 instead."
        assert config.deny_messages[1].message == "Push is not allowed."

    def test_invalid_regex_raises(self, tmp_path: Path) -> None:
        """Invalid regex pattern in deny_messages raises ConfigError."""
        write_workspace_config(
            tmp_path, extra='[deny_messages]\n"(unclosed" = "bad regex"\n'
        )

        with pytest.raises(ConfigError, match="not a valid regex"):
            load_config(tmp_path)

    def test_non_string_value_raises(self, tmp_path: Path) -> None:
        """Non-string value in deny_messages raises ConfigError."""
        write_workspace_config(tmp_path, extra='[deny_messages]\n"aws" = 42\n')

        with pytest.raises(ConfigError, match="must be a string"):
            load_config(tmp_path)

    def test_empty_section_produces_empty_list(self, tmp_path: Path) -> None:
        """Empty [deny_messages] section produces empty list."""
        write_workspace_config(tmp_path, extra="[deny_messages]\n")

        config = load_config(tmp_path)
        assert config.deny_messages == []


class TestShellExecutorDenyHints:
    """Tests for deny hints appended to shell rejection messages."""

    def test_command_rejection_includes_hint(self, tmp_path: Path) -> None:
        """Rejected command includes the matching deny hint in stderr."""
        import re
        config = _make_config(
            shell={"git": ShellPolicyConfig(command="^git$", mode="deny", allow=["status.*"])},
            deny_messages=[DenyMessageRule(pattern=re.compile(r"aws"), message="Use boto3.")],
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute("aws", ["s3", "ls"], 30)
        assert result.exit_code == 1
        assert "[Use boto3.]" in result.stderr

    def test_args_rejection_includes_hint(self, tmp_path: Path) -> None:
        """Rejected args include the matching deny hint in stderr."""
        import re
        config = _make_config(
            shell={"git": ShellPolicyConfig(command="^git$", mode="deny", allow=["status.*"])},
            deny_messages=[DenyMessageRule(pattern=re.compile(r"git push"), message="No pushing.")],
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute("git", ["push", "origin"], 30)
        assert result.exit_code == 1
        assert "[No pushing.]" in result.stderr

    def test_no_hint_when_no_match(self, tmp_path: Path) -> None:
        """When no deny_messages pattern matches, no hint is appended."""
        import re
        config = _make_config(
            shell={"git": ShellPolicyConfig(command="^git$", mode="deny", allow=["status.*"])},
            deny_messages=[DenyMessageRule(pattern=re.compile(r"aws"), message="Use boto3.")],
        )
        executor = ShellExecutor(config, tmp_path)
        result = executor.execute("git", ["push", "origin"], 30)
        assert result.exit_code == 1
        assert "[" not in result.stderr


class TestASTValidatorDenyHints:
    """Tests for deny hints in AST validator import rejections."""

    def test_import_rejection_includes_hint(self) -> None:
        """AST validator appends deny hint to blocked import message."""
        import re
        from pyddock.ast_validator import ASTValidator

        config = _make_config(
            deny_messages=[DenyMessageRule(pattern=re.compile(r"requests"), message="Use web_fetch.")],
        )
        validator = ASTValidator(config)
        violations = validator.validate("import requests")
        assert len(violations) == 1
        assert "[Use web_fetch.]" in violations[0].message

    def test_import_rejection_no_hint_when_no_match(self) -> None:
        """AST validator does not append hint when no pattern matches."""
        import re
        from pyddock.ast_validator import ASTValidator

        config = _make_config(
            deny_messages=[DenyMessageRule(pattern=re.compile(r"aws"), message="Use boto3.")],
        )
        validator = ASTValidator(config)
        violations = validator.validate("import requests")
        assert len(violations) == 1
        assert "[" not in violations[0].message
