"""Property-based tests for AST validator completeness and soundness.

Uses hypothesis to generate random Python snippets and verify that:
- Completeness: any disallowed construct is always caught
- Soundness: if validate() returns empty, no disallowed constructs are present
"""

from __future__ import annotations

import ast
import keyword

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pyddock.ast_validator import ASTValidator
from pyddock.config import (
    ASTConfig,
    ExecutionConfig,
    FilesystemConfig,
    ImportsConfig,
    PyddockConfig,
)

# Python keywords that can't be used as identifiers
_KEYWORDS = set(keyword.kwlist + keyword.softkwlist)

# --- Test config ---

ALLOWED_IMPORTS = ["json", "math", "re", "pathlib", "collections"]
BLOCKED_CALLS = ["eval", "exec", "compile", "breakpoint", "__import__"]
BLOCKED_ATTRS = ["__subclasses__", "__globals__", "__code__", "__bases__", "__builtins__"]

CONFIG = PyddockConfig(
    execution=ExecutionConfig(timeout=30.0),
    imports=ImportsConfig(allowed=ALLOWED_IMPORTS),
    filesystem=FilesystemConfig(writable_paths=["."], readable_paths=["."]),
    ast=ASTConfig(block_calls=BLOCKED_CALLS, block_attributes=BLOCKED_ATTRS),
    restrictions={},
)

VALIDATOR = ASTValidator(CONFIG)

# --- Strategies ---

# Module names that are NOT in the allowlist
disallowed_modules = st.sampled_from([
    "os", "subprocess", "socket", "ctypes", "shutil", "http",
    "urllib", "requests", "sys", "importlib", "signal",
])

# Module names that ARE in the allowlist
allowed_modules = st.sampled_from(ALLOWED_IMPORTS)

# Blocked call names
blocked_call_names = st.sampled_from(BLOCKED_CALLS)

# Blocked attribute names
blocked_attr_names = st.sampled_from(BLOCKED_ATTRS)

# Valid Python identifiers for variable names
identifiers = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True).filter(
    lambda s: s not in _KEYWORDS
)


# --- Property: AST Completeness ---
# Any source containing a disallowed import always produces at least one violation.

@given(module=disallowed_modules)
@settings(max_examples=50)
def test_completeness_import_statement(module: str) -> None:
    """import <disallowed> is always caught."""
    source = f"import {module}"
    violations = VALIDATOR.validate(source)
    assert len(violations) > 0
    assert any(v.kind == "blocked_import" for v in violations)


@given(module=disallowed_modules, name=identifiers)
@settings(max_examples=50)
def test_completeness_from_import(module: str, name: str) -> None:
    """from <disallowed> import <name> is always caught."""
    source = f"from {module} import {name}"
    violations = VALIDATOR.validate(source)
    assert len(violations) > 0
    assert any(v.kind == "blocked_import" for v in violations)


@given(module=disallowed_modules)
@settings(max_examples=50)
def test_completeness_submodule_import(module: str) -> None:
    """import <disallowed>.submodule is always caught."""
    source = f"import {module}.path"
    violations = VALIDATOR.validate(source)
    assert len(violations) > 0
    assert any(v.kind == "blocked_import" for v in violations)


@given(call_name=blocked_call_names)
@settings(max_examples=30)
def test_completeness_blocked_call(call_name: str) -> None:
    """Direct call to a blocked function is always caught."""
    source = f"{call_name}('x')"
    violations = VALIDATOR.validate(source)
    assert len(violations) > 0
    assert any(v.kind == "blocked_call" for v in violations)


@given(call_name=blocked_call_names, obj=identifiers)
@settings(max_examples=30)
def test_method_calls_not_blocked(call_name: str, obj: str) -> None:
    """Attribute-style calls (obj.eval()) are NOT blocked — only bare calls are.

    This is intentional: re.compile(), json.dumps(), etc. should work.
    The blocked calls list targets builtins (eval, exec, compile), not
    methods with the same name on allowed modules.
    """
    source = f"{obj}.{call_name}('x')"
    violations = VALIDATOR.validate(source)
    # Method calls should NOT produce blocked_call violations
    assert not any(v.kind == "blocked_call" for v in violations)


@given(attr=blocked_attr_names, obj=identifiers)
@settings(max_examples=30)
def test_completeness_blocked_attribute(attr: str, obj: str) -> None:
    """Access to a blocked attribute is always caught."""
    source = f"x = {obj}.{attr}"
    violations = VALIDATOR.validate(source)
    assert len(violations) > 0
    # Either caught as a blocked attribute, or as a syntax error (if obj is a keyword)
    assert any(v.kind in ("blocked_attribute", "syntax_error") for v in violations)


# --- Property: AST Soundness ---
# If validate() returns empty, no disallowed constructs are present.

@given(module=allowed_modules)
@settings(max_examples=30)
def test_soundness_allowed_import(module: str) -> None:
    """Allowed imports produce no violations."""
    source = f"import {module}"
    violations = VALIDATOR.validate(source)
    assert len(violations) == 0


@given(module=allowed_modules, name=identifiers)
@settings(max_examples=30)
def test_soundness_allowed_from_import(module: str, name: str) -> None:
    """from <allowed> import <name> produces no violations."""
    source = f"from {module} import {name}"
    violations = VALIDATOR.validate(source)
    assert len(violations) == 0


@given(var1=identifiers, var2=identifiers)
@settings(max_examples=30)
def test_soundness_safe_code(var1: str, var2: str) -> None:
    """Pure computation with no imports or blocked constructs passes."""
    assume(var1 != var2)
    assume(var1 not in BLOCKED_CALLS and var2 not in BLOCKED_CALLS)
    source = f"{var1} = 42\n{var2} = {var1} + 1\n{var2}"
    violations = VALIDATOR.validate(source)
    assert len(violations) == 0


@given(
    module=allowed_modules,
    var=identifiers,
)
@settings(max_examples=30)
def test_soundness_combined_safe(module: str, var: str) -> None:
    """Allowed import + safe computation passes."""
    assume(var not in BLOCKED_CALLS)
    source = f"import {module}\n{var} = 1 + 2\n{var}"
    violations = VALIDATOR.validate(source)
    assert len(violations) == 0
