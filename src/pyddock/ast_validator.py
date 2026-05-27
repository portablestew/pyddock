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

    def __init__(self, config: PyddockConfig) -> None:
        self._config = config

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

            elif isinstance(node, ast.Attribute):
                if node.attr in self._config.ast.block_attributes:
                    violations.append(
                        self._attr_violation(node.attr, node.lineno)
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
