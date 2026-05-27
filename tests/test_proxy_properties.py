"""Property-based tests for MethodFilterProxy deny-by-default behavior.

Verifies that for any method name and any set of allow patterns:
- If the name matches at least one pattern → access succeeds
- If the name matches no pattern → PermissionError is raised
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from pyddock._runtime import MethodFilterProxy


# --- Strategies ---

# Simple method names (lowercase identifiers, no underscores at start)
method_names = st.from_regex(r"[a-z][a-z0-9_]{1,20}", fullmatch=True)

# Simple regex patterns that are valid and useful
simple_patterns = st.sampled_from([
    "list_.*",
    "describe_.*",
    "get_.*",
    "head_.*",
    "read_.*",
    "fetch_.*",
    "count_.*",
])


class _FakeClient:
    """A fake object with arbitrary attributes for testing the proxy."""

    def __getattr__(self, name: str):
        return f"result_of_{name}"


# --- Property: Deny-by-default ---

# Method names that are guaranteed to match our patterns
matching_method_names = st.sampled_from([
    "list_buckets", "list_objects", "list_users",
    "describe_instances", "describe_clusters",
    "get_object", "get_item", "get_user",
    "head_object", "head_bucket",
    "read_csv", "read_json",
    "fetch_data", "fetch_results",
    "count_items", "count_rows",
])


@given(
    method=matching_method_names,
    patterns=st.lists(simple_patterns, min_size=1, max_size=4),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_proxy_allows_matching_methods(method: str, patterns: list[str]) -> None:
    """If method matches at least one allow pattern, access succeeds."""
    compiled = [re.compile(p) for p in patterns]

    # Only test when the method actually matches
    assume(any(p.match(method) for p in compiled))

    proxy = MethodFilterProxy(_FakeClient(), compiled)
    result = getattr(proxy, method)
    assert result == f"result_of_{method}"


@given(
    method=method_names,
    patterns=st.lists(simple_patterns, min_size=1, max_size=4),
)
@settings(max_examples=100)
def test_proxy_blocks_non_matching_methods(method: str, patterns: list[str]) -> None:
    """If method matches no allow pattern, PermissionError is raised."""
    compiled = [re.compile(p) for p in patterns]

    # Only test when the method does NOT match any pattern
    assume(not any(p.match(method) for p in compiled))

    proxy = MethodFilterProxy(_FakeClient(), compiled)
    with pytest.raises(PermissionError) as exc_info:
        getattr(proxy, method)

    # Error message includes the method name and allowed patterns
    msg = str(exc_info.value)
    assert method in msg
    assert "Allowed method patterns" in msg


@given(method=method_names)
@settings(max_examples=30)
def test_proxy_with_empty_allow_blocks_everything(method: str) -> None:
    """With no allow patterns, all methods are blocked."""
    proxy = MethodFilterProxy(_FakeClient(), [])
    with pytest.raises(PermissionError):
        getattr(proxy, method)


@given(method=method_names)
@settings(max_examples=30)
def test_proxy_with_wildcard_allows_everything(method: str) -> None:
    """With .* pattern, all methods are allowed."""
    proxy = MethodFilterProxy(_FakeClient(), [re.compile(".*")])
    result = getattr(proxy, method)
    assert result == f"result_of_{method}"


def test_proxy_allows_dunder_methods() -> None:
    """Dunder methods (__x__) are always allowed (internal Python machinery).
    Single-underscore methods (_x) are NOT allowed — they're subject to pattern checks.
    This prevents bypasses like client._make_api_call().
    """
    proxy = MethodFilterProxy(_FakeClient(), [re.compile("list_.*")])
    # __repr__ should pass through without pattern check
    result = proxy.__repr__
    # _internal should be blocked (not a dunder)
    with pytest.raises(PermissionError):
        getattr(proxy, "_internal")


def test_proxy_error_message_includes_patterns() -> None:
    """Error message lists all allowed patterns for agent guidance."""
    patterns = [re.compile("list_.*"), re.compile("get_.*")]
    proxy = MethodFilterProxy(_FakeClient(), patterns)

    with pytest.raises(PermissionError) as exc_info:
        getattr(proxy, "delete_bucket")

    msg = str(exc_info.value)
    assert "list_.*" in msg
    assert "get_.*" in msg
    assert "delete_bucket" in msg
