"""Tests for pyddock file tools (fs_read, fs_stat, fs_append, fs_delete, fs_str_replace).

Uses a session-scoped venv to avoid recreating it per test. Each test gets
its own subdirectory within the shared workspace for isolation.
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import pytest

from pyddock.config import load_config
from pyddock.executor import SubprocessExecutor
from pyddock.script_registry import ScriptToolRegistry
from pyddock.venv_manager import VenvManager


@pytest.fixture(scope="session")
def shared_workspace(tmp_path_factory) -> Path:
    """Session-scoped workspace with venv created once."""
    ws = tmp_path_factory.mktemp("workspace")
    config_dir = ws / ".pyddock"
    config_dir.mkdir()
    (config_dir / "pyddock.toml").write_text(
        """
[execution]
timeout = 30
max_timeout = 60

[imports]
os = true
sys = true
pathlib = true
re = true
difflib = true
datetime = true

[filesystem]
writable_paths = ["."]
readable_paths = ["*"]

[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = []
""",
        encoding="utf-8",
    )
    # Pre-create venv once
    config = load_config(ws)
    vm = VenvManager(venv_path=ws / ".pyddock" / "venv", allowed_imports=config.imports.allowed)
    vm.ensure_venv()
    return ws


@pytest.fixture
def registry(shared_workspace: Path) -> ScriptToolRegistry:
    """Registry using the shared session workspace/venv."""
    config = load_config(shared_workspace)
    vm = VenvManager(
        venv_path=shared_workspace / ".pyddock" / "venv",
        allowed_imports=config.imports.allowed,
    )
    executor = SubprocessExecutor(config, vm)
    reg = ScriptToolRegistry(config, executor, shared_workspace)
    reg.load_scripts()
    return reg


@pytest.fixture
def ws(shared_workspace: Path, request) -> Path:
    """Per-test subdirectory within the shared workspace for file isolation."""
    test_dir = shared_workspace / f"_test_{request.node.name}"
    test_dir.mkdir(exist_ok=True)
    return test_dir


def run(registry: ScriptToolRegistry, script: str, params: dict) -> str:
    """Synchronous helper to run a tool script."""
    return asyncio.run(registry.execute(script, params))


# =============================================================================
# fs_read tests
# =============================================================================


class TestFsRead:
    def test_whole_file(self, ws: Path, registry: ScriptToolRegistry):
        """Whole file → correct line numbers, no hint."""
        f = ws / "test.txt"
        f.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")
        result = run(registry, "read_file", {"path": str(f)})
        assert "1| alpha" in result
        assert "5| epsilon" in result
        assert "Showing" not in result

    def test_range(self, ws: Path, registry: ScriptToolRegistry):
        """start=3, end=7 → lines 3-7, range hint."""
        f = ws / "test.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
        result = run(registry, "read_file", {"path": str(f), "start": 3, "end": 7})
        assert "3| L3" in result
        assert "7| L7" in result
        assert "L2" not in result
        assert "L8" not in result
        assert "Showing lines 3-7 of 10" in result

    def test_tail(self, ws: Path, registry: ScriptToolRegistry):
        """start=-3 → last 3 lines, end ignored."""
        f = ws / "test.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
        result = run(registry, "read_file", {"path": str(f), "start": -3, "end": 2})
        assert "8| L8" in result
        assert "10| L10" in result
        assert "L7" not in result

    def test_tail_larger_than_file(self, ws: Path, registry: ScriptToolRegistry):
        """start=-20 on 5-line file → all lines."""
        f = ws / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        result = run(registry, "read_file", {"path": str(f), "start": -20})
        assert "1| a" in result
        assert "5| e" in result

    def test_truncation_and_continuation(self, ws: Path, registry: ScriptToolRegistry):
        """Large file → truncated with correct hint, continuation works."""
        f = ws / "big.txt"
        lines = [f"{'x' * 90} L{i:04d}" for i in range(1, 601)]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = run(registry, "read_file", {"path": str(f)})
        assert "Truncated at ~50 KB" in result
        assert "of 600" in result
        assert len(result) < 55_000

        # Extract continuation line and verify it works
        m = re.search(r"Use start=(\d+)", result)
        assert m is not None
        next_line = int(m.group(1))
        result2 = run(registry, "read_file", {"path": str(f), "start": next_line})
        assert f"| {'x' * 90} L{next_line:04d}" in result2

    def test_binary_rejection(self, ws: Path, registry: ScriptToolRegistry):
        """Binary file → error, no partial output."""
        f = ws / "bin.dat"
        f.write_bytes(b"\x00\x01\x02\xff\xfe" * 100)
        result = run(registry, "read_file", {"path": str(f)})
        assert "binary or not UTF-8" in result

    def test_empty_file(self, ws: Path, registry: ScriptToolRegistry):
        """Empty file → no crash, empty output."""
        f = ws / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = run(registry, "read_file", {"path": str(f)})
        # Should not crash; output is empty or minimal
        assert "Error" not in result


# =============================================================================
# fs_str_replace tests
# =============================================================================


class TestFsStrReplace:
    def test_single_match(self, ws: Path, registry: ScriptToolRegistry):
        """Unique match → replaced, diff returned."""
        f = ws / "code.py"
        f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        result = run(registry, "str_replace", {
            "path": str(f), "oldStr": "return 'world'", "newStr": "return 'universe'",
        })
        content = f.read_text(encoding="utf-8")
        assert "universe" in content
        assert "world" not in content
        assert "-" in result and "+" in result

    def test_multiple_matches_line_numbers(self, ws: Path, registry: ScriptToolRegistry):
        """3 occurrences → file unchanged, correct line numbers reported."""
        f = ws / "multi.py"
        # "return None" on lines 1, 3, 5
        f.write_text("return None\nx = 1\nreturn None\ny = 2\nreturn None\n", encoding="utf-8")
        result = run(registry, "str_replace", {
            "path": str(f), "oldStr": "return None", "newStr": "return 0",
        })
        # File unchanged
        assert f.read_text(encoding="utf-8").count("return None") == 3
        assert "3 of 3" in result
        # Verify line numbers are reported (lines 1, 3, 5)
        assert "line 1" in result
        assert "line 3" in result
        assert "line 5" in result

    def test_no_match_fuzzy(self, ws: Path, registry: ScriptToolRegistry):
        """Wrong whitespace → fuzzy match found near correct line."""
        f = ws / "fuzzy.py"
        f.write_text("def main():\n    print('hello')\n", encoding="utf-8")
        result = run(registry, "str_replace", {
            "path": str(f), "oldStr": "def  main():\n   print('hello')", "newStr": "x",
        })
        assert "hello" in f.read_text(encoding="utf-8")  # unchanged
        assert "partial match" in result.lower() or "near line" in result.lower()

    def test_window_constraint_replaces_correct_occurrence(self, ws: Path, registry: ScriptToolRegistry):
        """Match on lines 3 and 8; window=7-9 replaces only line 8, line 3 untouched."""
        lines = ["a", "b", "TARGET", "d", "e", "f", "g", "TARGET", "i", "j"]
        f = ws / "window.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = run(registry, "str_replace", {
            "path": str(f), "oldStr": "TARGET", "newStr": "DONE",
            "start_line": 7, "end_line": 9,
        })
        new_lines = f.read_text(encoding="utf-8").splitlines()
        assert new_lines[2] == "TARGET"  # line 3 untouched
        assert new_lines[7] == "DONE"    # line 8 replaced
        assert "-TARGET" in result or "- TARGET" in result or "-TARGET" in result.replace(" ", "")

    def test_window_offset_correctness(self, ws: Path, registry: ScriptToolRegistry):
        """Verify replacement lands at correct char position when window has offset."""
        # "needle" appears at line 2 and line 5. Window 4-6 should replace line 5 only.
        f = ws / "offset.txt"
        f.write_text("aaa\nneedle\nccc\nddd\nneedle\nfff\n", encoding="utf-8")
        run(registry, "str_replace", {
            "path": str(f), "oldStr": "needle", "newStr": "FOUND",
            "start_line": 4, "end_line": 6,
        })
        content = f.read_text(encoding="utf-8")
        lines = content.splitlines()
        assert lines[1] == "needle"  # line 2 untouched
        assert lines[4] == "FOUND"   # line 5 replaced

    def test_empty_new_str(self, ws: Path, registry: ScriptToolRegistry):
        """Replace with "" → content deleted."""
        f = ws / "del.txt"
        f.write_text("keep\nremove_me\nkeep\n", encoding="utf-8")
        run(registry, "str_replace", {
            "path": str(f), "oldStr": "remove_me\n", "newStr": "",
        })
        assert "remove_me" not in f.read_text(encoding="utf-8")

    def test_no_match_in_window_fuzzy_full_file(self, ws: Path, registry: ScriptToolRegistry):
        """String outside window → no match, fuzzy searches full file."""
        f = ws / "outside.txt"
        f.write_text("target\nb\nc\nd\ne\n", encoding="utf-8")
        result = run(registry, "str_replace", {
            "path": str(f), "oldStr": "target", "newStr": "X",
            "start_line": 3, "end_line": 5,
        })
        assert f.read_text(encoding="utf-8").splitlines()[0] == "target"  # unchanged
        assert "partial match" in result.lower() or "no exact match" in result.lower() or "near line" in result.lower()


# =============================================================================
# fs_append tests
# =============================================================================


class TestFsAppend:
    def test_create_new_file_with_parents(self, ws: Path, registry: ScriptToolRegistry):
        """Non-existent path → file created, parents created, diff shows additions."""
        f = ws / "sub" / "dir" / "new.txt"
        result = run(registry, "fs_append", {"path": str(f), "content": "hello\nworld"})
        assert f.exists()
        assert f.read_text(encoding="utf-8") == "hello\nworld"
        assert "+" in result

    def test_append_with_trailing_newline(self, ws: Path, registry: ScriptToolRegistry):
        """File ends with \\n → no extra newline."""
        f = ws / "trail.txt"
        f.write_text("existing\n", encoding="utf-8")
        run(registry, "fs_append", {"path": str(f), "content": "appended"})
        assert f.read_text(encoding="utf-8") == "existing\nappended"

    def test_append_without_trailing_newline(self, ws: Path, registry: ScriptToolRegistry):
        """File ends without \\n → one newline separator."""
        f = ws / "notrail.txt"
        f.write_text("existing", encoding="utf-8")
        run(registry, "fs_append", {"path": str(f), "content": "appended"})
        assert f.read_text(encoding="utf-8") == "existing\nappended"

    def test_append_to_empty_file(self, ws: Path, registry: ScriptToolRegistry):
        """Empty existing file → content written, no spurious newline prefix."""
        f = ws / "empty.txt"
        f.write_text("", encoding="utf-8")
        run(registry, "fs_append", {"path": str(f), "content": "first"})
        assert f.read_text(encoding="utf-8") == "first"

    def test_large_file_append(self, ws: Path, registry: ScriptToolRegistry):
        """Append to >50 KB file → old content untouched, new content at end."""
        f = ws / "big.txt"
        f.write_text("x" * 60_000 + "\n", encoding="utf-8")
        run(registry, "fs_append", {"path": str(f), "content": "TAIL"})
        content = f.read_text(encoding="utf-8")
        assert content.startswith("x" * 1000)
        assert content.endswith("TAIL")


# =============================================================================
# fs_delete tests
# =============================================================================


class TestFsDelete:
    def test_delete_file(self, ws: Path, registry: ScriptToolRegistry):
        """File removed, diff shows removed lines."""
        f = ws / "doomed.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = run(registry, "fs_delete", {"path": str(f)})
        assert not f.exists()
        assert "line1" in result
        assert "-" in result

    def test_delete_empty_directory(self, ws: Path, registry: ScriptToolRegistry):
        """Empty dir removed."""
        d = ws / "empty_dir"
        d.mkdir()
        run(registry, "fs_delete", {"path": str(d)})
        assert not d.exists()

    def test_non_empty_directory(self, ws: Path, registry: ScriptToolRegistry):
        """Non-empty dir → error, dir still exists."""
        d = ws / "full_dir"
        d.mkdir()
        (d / "file.txt").write_text("x", encoding="utf-8")
        result = run(registry, "fs_delete", {"path": str(d)})
        assert d.exists()
        assert "not empty" in result


# =============================================================================
# fs_stat tests
# =============================================================================


class TestFsStat:
    def test_existing_file(self, ws: Path, registry: ScriptToolRegistry):
        """Returns correct metadata."""
        f = ws / "info.txt"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        result = run(registry, "stat_file", {"path": str(f)})
        assert "exists: true" in result
        assert "type: file" in result
        assert "lines: 3" in result
        assert "modified:" in result

    def test_non_existent(self, ws: Path, registry: ScriptToolRegistry):
        """Non-existent → exists: false."""
        result = run(registry, "stat_file", {"path": str(ws / "nope.txt")})
        assert "exists: false" in result


# =============================================================================
# Sandbox enforcement tests
# =============================================================================


class TestSandboxEnforcement:
    def test_write_to_pyddock_dir(self, shared_workspace: Path, registry: ScriptToolRegistry):
        """fs_append to .pyddock/ → blocked."""
        target = shared_workspace / ".pyddock" / "evil.txt"
        result = run(registry, "fs_append", {"path": str(target), "content": "hack"})
        assert not target.exists()
        assert "permission" in result.lower() or ".pyddock" in result.lower()

    def test_read_outside_restricted_readable(self, tmp_path: Path):
        """Read outside workspace with readable_paths=['.'] → blocked."""
        import sys as _sys

        # Create a separate restricted workspace (reuse current interpreter, no venv)
        ws = tmp_path / "restricted_ws"
        ws.mkdir()
        config_dir = ws / ".pyddock"
        config_dir.mkdir()
        (config_dir / "pyddock.toml").write_text(
            """
[execution]
timeout = 10
max_timeout = 30
[imports]
os = true
sys = true
pathlib = true
re = true
difflib = true
datetime = true
[filesystem]
writable_paths = ["."]
readable_paths = ["."]
[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = []
""",
            encoding="utf-8",
        )
        config = load_config(ws)
        vm = VenvManager(venv_path=ws / ".pyddock" / "venv", allowed_imports=config.imports.allowed)
        vm.get_python_path = lambda: Path(_sys.executable)  # type: ignore[method-assign]
        ex = SubprocessExecutor(config, vm)
        reg = ScriptToolRegistry(config, ex, ws)
        reg.load_scripts()

        # Write a file outside this workspace
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        result = run(reg, "read_file", {"path": str(outside)})
        assert "permission" in result.lower() or "restricted" in result.lower()
