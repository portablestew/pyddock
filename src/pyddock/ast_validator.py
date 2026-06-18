"""AST validation for pyddock.

Static analysis of Python source code before execution. Provides fast,
deterministic rejection with helpful error messages generated from config.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pyddock.config import PyddockConfig, find_deny_hint


@dataclass
class ASTViolation:
    """A single AST policy violation."""

    kind: str  # "blocked_import", "blocked_call", "blocked_attribute", "syntax_error"
    name: str  # the offending identifier
    line: int
    message: str  # user-facing, generated from config


class ASTValidator:
    """Validates Python source against policy config.

    Walks the AST checking imports against the allowlist, detecting blocked
    function calls, and detecting blocked attribute accesses. All error
    messages are generated from config values (not hardcoded).
    """

    # Call targets that perform string-keyed attribute reflection. A blocked
    # attribute name passed as a constant first argument to one of these is
    # equivalent to a blocked `.attr` access and is flagged (2a).
    _REFLECTIVE_ACCESSORS = frozenset({"getattr", "__getattribute__", "__getattr__"})

    def __init__(self, config: PyddockConfig) -> None:
        self._config = config
        # Precomputed set for O(1) membership in the AST walk.
        self._block_attr_set = frozenset(config.ast.block_attributes)

    def validate(self, source: str) -> list[ASTViolation]:
        """Validate Python source against policy config.

        Preconditions:
            - source is a Python string (may or may not parse)
            - self._config is loaded and valid

        Postconditions:
            - If source is not valid Python syntax, returns a single violation
              with kind="syntax_error"
            - Returns empty list if and only if code passes all checks
            - Each violation contains a user-facing message generated from config
            - No side effects, no execution of the source code
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [
                ASTViolation(
                    kind="syntax_error",
                    name="",
                    line=e.lineno or 0,
                    message=str(e),
                )
            ]

        violations: list[ASTViolation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level not in self._config.imports.allowed:
                        violations.append(
                            self._import_violation(alias.name, node.lineno)
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    top_level = node.module.split(".")[0]
                    if top_level not in self._config.imports.allowed:
                        violations.append(
                            self._import_violation(node.module, node.lineno)
                        )

            elif isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name is not None and name in self._config.ast.block_calls:
                    # Only block bare calls (e.g., eval(...)), not method calls
                    # (e.g., re.compile(...)). Method calls on objects are safe —
                    # the danger is the builtin functions, not methods with the
                    # same name on allowed modules.
                    if isinstance(node.func, ast.Name):
                        violations.append(self._call_violation(name, node.lineno))
                # 2a: reflective accessors — getattr(obj, "__globals__"),
                # obj.__getattribute__("__globals__"), obj.__getattr__("__bases__").
                # A blocked attribute name passed as a constant string argument is
                # equivalent to a blocked `.attr` access. The name is the 1st arg
                # for the method forms and the 2nd for getattr(); rather than
                # special-case each signature, flag any constant string argument
                # that names a blocked attribute. (Variable arguments cannot be
                # resolved statically; the runtime getattr guard is the backstop.)
                if name in self._REFLECTIVE_ACCESSORS:
                    for arg in node.args:
                        if (
                            isinstance(arg, ast.Constant)
                            and isinstance(arg.value, str)
                            and arg.value in self._block_attr_set
                        ):
                            violations.append(
                                self._reflection_violation(arg.value, node.lineno)
                            )

            elif isinstance(node, ast.Attribute):
                if node.attr in self._config.ast.block_attributes:
                    violations.append(
                        self._attr_violation(node.attr, node.lineno)
                    )

            elif isinstance(node, ast.Subscript):
                # 2a: string-keyed access to a blocked attribute via a mapping,
                # e.g. type.__dict__["__subclasses__"], vars(obj)["__globals__"],
                # __builtins__["__globals__"]. The AST attribute check only sees
                # `.attr` syntax; this closes the subscript form. Only constant
                # string keys are resolvable statically; the dunder-name set is
                # narrow enough that legitimate data keys are not affected.
                key = node.slice
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in self._block_attr_set
                ):
                    violations.append(
                        self._subscript_violation(key.value, node.lineno)
                    )

        return violations

    def extract_imports(self, source: str) -> list[str]:
        """Extract deduplicated list of top-level module names from source.

        Returns an empty list if the source has a syntax error.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    modules.add(node.module.split(".")[0])

        return sorted(modules)

    def _import_violation(self, module_name: str, line: int) -> ASTViolation:
        """Create a violation for a disallowed import."""
        allowed_str = ", ".join(self._config.imports.allowed)
        message = (
            f"ImportError: '{module_name}' is not an allowed import. "
            f"Please use one of the following allowed imports instead: {allowed_str}"
        )
        hint = find_deny_hint(module_name, self._config.deny_messages)
        if hint:
            message += f"\n[{hint}]"
        return ASTViolation(
            kind="blocked_import",
            name=module_name,
            line=line,
            message=message,
        )

    def _call_violation(self, call_name: str, line: int) -> ASTViolation:
        """Create a violation for a blocked function call."""
        message = (
            f"SecurityError: '{call_name}()' is not permitted. "
            f"Please rewrite your snippet to avoid this call."
        )
        return ASTViolation(
            kind="blocked_call",
            name=call_name,
            line=line,
            message=message,
        )

    def _attr_violation(self, attr_name: str, line: int) -> ASTViolation:
        """Create a violation for a blocked attribute access."""
        message = (
            f"SecurityError: Access to '{attr_name}' is not permitted. "
            f"Please rewrite your snippet to avoid this attribute."
        )
        return ASTViolation(
            kind="blocked_attribute",
            name=attr_name,
            line=line,
            message=message,
        )

    def _subscript_violation(self, attr_name: str, line: int) -> ASTViolation:
        """Create a violation for blocked-attribute access via subscripting.

        Catches the `__dict__["__subclasses__"]` / `vars(obj)["__globals__"]`
        form that the plain `.attr` attribute check does not see.
        """
        message = (
            f"SecurityError: Subscript access to '{attr_name}' is not permitted "
            f"(e.g. obj.__dict__['{attr_name}'] or vars(obj)['{attr_name}']). "
            f"Please rewrite your snippet to avoid this attribute."
        )
        return ASTViolation(
            kind="blocked_subscript",
            name=attr_name,
            line=line,
            message=message,
        )

    def _reflection_violation(self, attr_name: str, line: int) -> ASTViolation:
        """Create a violation for blocked-attribute access via a reflective call.

        Catches getattr(obj, "__globals__") and
        obj.__getattribute__("__globals__") with a constant name argument.
        """
        message = (
            f"SecurityError: Reflective access to '{attr_name}' is not permitted "
            f"(e.g. getattr(obj, '{attr_name}') or "
            f"obj.__getattribute__('{attr_name}')). "
            f"Please rewrite your snippet to avoid this attribute."
        )
        return ASTViolation(
            kind="blocked_reflection",
            name=attr_name,
            line=line,
            message=message,
        )

    @staticmethod
    def _get_call_name(node: ast.Call) -> str | None:
        """Extract the function name from a Call node.

        Handles both simple calls (e.g., eval(...)) and attribute calls
        (e.g., obj.eval(...)).
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
