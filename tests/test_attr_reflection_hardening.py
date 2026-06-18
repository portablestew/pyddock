"""Lightweight tests for the 2a/2c attribute-access hardening.

2a — ASTValidator also rejects blocked-attribute access expressed as a
     constant-keyed subscript (``type.__dict__["__subclasses__"]``) or as a
     reflective call with a constant name argument
     (``getattr(o, "__globals__")``, ``o.__getattribute__("__globals__")``).

2c — the bundled default config blocks a few additional rarely-used names
     (``__base__``, ``__getattribute__``, ``__getattr__``, ``__reduce__``,
     ``__reduce_ex__``). ``__class__`` / ``__dict__`` are intentionally NOT
     blocked.

Tests are wired to the bundled ``default_config.toml`` (via ``load_config`` on
an empty workspace) so they assert the production policy, not a fixture copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyddock.ast_validator import ASTValidator
from pyddock.config import load_config


@pytest.fixture(scope="module")
def validator(tmp_path_factory: pytest.TempPathFactory) -> ASTValidator:
    # Empty workspace → bundled default_config.toml is loaded.
    workspace = tmp_path_factory.mktemp("ws")
    return ASTValidator(load_config(workspace))


def _kinds(violations) -> set[str]:
    return {v.kind for v in violations}


# --- 2c: the new names are present in the production policy ---


@pytest.mark.parametrize(
    "name",
    ["__base__", "__getattribute__", "__getattr__"],
)
def test_new_blocked_names_in_default_config(validator: ASTValidator, name: str) -> None:
    assert name in validator._block_attr_set


@pytest.mark.parametrize("name", ["__class__", "__dict__", "__reduce__", "__reduce_ex__"])
def test_intentionally_not_blocked(validator: ASTValidator, name: str) -> None:
    # __class__/__dict__: too common in introspection. __reduce__/__reduce_ex__:
    # CPython's enum machinery fetches __reduce_ex__ via getattr for every Enum
    # subclass, so the global getattr guard must not block them.
    assert name not in validator._block_attr_set


# --- 2a: subscript form ---


@pytest.mark.parametrize(
    "src",
    [
        "type.__dict__['__subclasses__'](object)",   # the original exploit
        "vars(object)['__globals__']",
        "d['__bases__']",
        "x['__base__']",
    ],
)
def test_subscript_blocked_name_is_flagged(validator: ASTValidator, src: str) -> None:
    violations = validator.validate(src)
    assert "blocked_subscript" in _kinds(violations)


@pytest.mark.parametrize(
    "src",
    [
        "d['name']",                 # ordinary data key
        "d['__class__']",            # __class__ not blocked
        "d['__dict__']",             # __dict__ not blocked
        "row[0]",                    # non-string key
        "n = '__subclasses__'\nd[n]",  # variable key — not statically resolvable
    ],
)
def test_benign_subscripts_pass(validator: ASTValidator, src: str) -> None:
    assert "blocked_subscript" not in _kinds(validator.validate(src))


# --- 2a: reflective-call form ---


@pytest.mark.parametrize(
    "src",
    [
        "getattr(o, '__globals__')",
        "getattr(o, '__subclasses__')",
        "o.__getattr__('__bases__')",
    ],
)
def test_reflective_call_blocked_name_is_flagged(validator: ASTValidator, src: str) -> None:
    violations = validator.validate(src)
    assert "blocked_reflection" in _kinds(violations)


def test_getattribute_call_is_flagged(validator: ASTValidator) -> None:
    # Flagged by the attribute branch (.__getattribute__ is now blocked) and/or
    # the reflective-call branch — either rejection is sufficient.
    violations = validator.validate("o.__getattribute__('__globals__')")
    assert {"blocked_attribute", "blocked_reflection"} & _kinds(violations)


@pytest.mark.parametrize(
    "src",
    [
        "getattr(o, 'name')",        # benign attribute name
        "getattr(o, attr_var)",      # variable name — runtime guard's job
        "json.dumps(o, default=str)",
    ],
)
def test_benign_reflective_calls_pass(validator: ASTValidator, src: str) -> None:
    kinds = _kinds(validator.validate(src))
    assert "blocked_reflection" not in kinds


# --- soundness: ordinary code is unaffected ---


@pytest.mark.parametrize(
    "src",
    [
        "x = 1 + 2\nx",
        "import json\njson.dumps({'a': 1})",
        "d = {'__class__': 1}\nd['__class__']",
        "[i * 2 for i in range(3)]",
    ],
)
def test_clean_code_has_no_violations(validator: ASTValidator, src: str) -> None:
    assert validator.validate(src) == []
