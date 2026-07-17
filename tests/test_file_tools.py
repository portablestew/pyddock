"""Tests for pyddock file tools (fs_readfile, fs_stat, fs_append, fs_delete, fs_str_replace).

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
glob = true

[filesystem]
writable_paths = ["."]
readable_paths = ["*"]

[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = []

[audit]
"open" = "fs"
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
# fs_readfile tests
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
# fs_find tests
# =============================================================================


class TestFsFind:
    def test_finds_matching_files(self, ws: Path, registry: ScriptToolRegistry):
        """Glob pattern matches files recursively under path."""
        (ws / "a.py").write_text("x = 1", encoding="utf-8")
        (ws / "sub").mkdir()
        (ws / "sub" / "b.py").write_text("y = 2", encoding="utf-8")
        (ws / "c.txt").write_text("not python", encoding="utf-8")

        result = run(registry, "fs_find", {"file_glob": "*.py", "path": str(ws)})
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_no_matches(self, ws: Path, registry: ScriptToolRegistry):
        """No matching files -> friendly message, not an error."""
        (ws / "a.txt").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {"file_glob": "*.java", "path": str(ws)})
        assert "No files matching" in result

    def test_path_must_be_a_directory(self, ws: Path, registry: ScriptToolRegistry):
        """path pointing at a file (not a directory) -> error."""
        f = ws / "target.py"
        f.write_text("x = 1", encoding="utf-8")
        result = run(registry, "fs_find", {"file_glob": "*.py", "path": str(f)})
        assert "not a directory" in result.lower()

    def test_max_results_truncates(self, ws: Path, registry: ScriptToolRegistry):
        """More matches than max_results -> truncation hint shown."""
        d = ws / "many"
        d.mkdir()
        for i in range(10):
            (d / f"f{i}.py").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_glob": "*.py", "path": str(d), "max_results": 3,
        })
        assert "Showing first 3 results" in result

    def test_nonexistent_path(self, ws: Path, registry: ScriptToolRegistry):
        """Non-existent path -> error."""
        result = run(registry, "fs_find", {
            "file_glob": "*.py", "path": str(ws / "nope"),
        })
        assert "not found" in result.lower()

    def test_hidden_directory_pruned(self, ws: Path, registry: ScriptToolRegistry):
        """Files under a dot-prefixed directory are not returned, and noted."""
        (ws / "visible.py").write_text("x = 1", encoding="utf-8")
        hidden_dir = ws / ".venv"
        hidden_dir.mkdir()
        (hidden_dir / "lib.py").write_text("y = 2", encoding="utf-8")

        result = run(registry, "fs_find", {"file_glob": "*.py", "path": str(ws)})
        assert "visible.py" in result
        assert "lib.py" not in result
        assert "hidden" in result.lower()

    def test_hidden_file_pruned_by_wildcard_glob(self, ws: Path, registry: ScriptToolRegistry):
        """A '*' glob does not match a dot-prefixed file, mirroring shell semantics."""
        (ws / ".hidden.py").write_text("x = 1", encoding="utf-8")
        result = run(registry, "fs_find", {"file_glob": "*.py", "path": str(ws)})
        assert "No files matching" in result

    def test_explicit_dot_glob_matches_hidden_file(self, ws: Path, registry: ScriptToolRegistry):
        """A glob whose final segment starts with '.' (e.g. '.env') explicitly
        matches hidden files, mirroring shell glob semantics."""
        (ws / ".env").write_text("SECRET=1", encoding="utf-8")
        (ws / "visible.txt").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {"file_glob": ".env", "path": str(ws)})
        assert ".env" in result
        assert "visible.txt" not in result

    def test_explicit_dot_glob_does_not_reach_into_hidden_directories(
        self, ws: Path, registry: ScriptToolRegistry,
    ):
        """An explicit dot-glob still doesn't defeat hidden-DIRECTORY pruning —
        only the performance-motivated directory prune is unconditional."""
        hidden_dir = ws / ".venv"
        hidden_dir.mkdir()
        (hidden_dir / ".env").write_text("SECRET=1", encoding="utf-8")
        result = run(registry, "fs_find", {"file_glob": "**/.env", "path": str(ws)})
        assert "No files matching" in result

    def test_exclude_regex_prunes_directory(self, ws: Path, registry: ScriptToolRegistry):
        """exclude_regex matching a subdirectory name prunes it (and is noted)."""
        (ws / "visible.py").write_text("x = 1", encoding="utf-8")
        excluded_dir = ws / "node_modules"
        excluded_dir.mkdir()
        (excluded_dir / "lib.py").write_text("y = 2", encoding="utf-8")

        result = run(registry, "fs_find", {
            "file_glob": "*.py", "path": str(ws), "exclude_regex": "node_modules",
        })
        assert "visible.py" in result
        assert "lib.py" not in result
        assert "excluded" in result.lower()

    def test_hidden_path_named_directly_is_searched(self, ws: Path, registry: ScriptToolRegistry):
        """Pointing path directly at a hidden directory still searches it."""
        hidden_dir = ws / ".config"
        hidden_dir.mkdir()
        (hidden_dir / "settings.py").write_text("x = 1", encoding="utf-8")

        result = run(registry, "fs_find", {"file_glob": "*.py", "path": str(hidden_dir)})
        assert "settings.py" in result


# =============================================================================
# fs_grep tests
# =============================================================================


class TestFsGrep:
    def test_finds_matches_with_line_numbers(self, ws: Path, registry: ScriptToolRegistry):
        """Regex match reports path:line: content."""
        f = ws / "code.py"
        f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "return", "path": str(ws)})
        assert "code.py:2:" in result
        assert "return 'world'" in result

    def test_no_matches(self, ws: Path, registry: ScriptToolRegistry):
        """No matches -> friendly message."""
        f = ws / "code.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "nomatch_xyz", "path": str(ws)})
        assert "No matches" in result

    def test_always_case_insensitive(self, ws: Path, registry: ScriptToolRegistry):
        """Matching is always case-insensitive, no ignore_case param needed."""
        f = ws / "code.py"
        f.write_text("HELLO world\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "hello", "path": str(ws)})
        assert "HELLO world" in result

    def test_file_glob_filters_files(self, ws: Path, registry: ScriptToolRegistry):
        """file_glob restricts which files are searched."""
        (ws / "a.py").write_text("needle\n", encoding="utf-8")
        (ws / "b.txt").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_glob": "*.py", "path": str(ws),
        })
        assert "a.py" in result
        assert "b.txt" not in result

    def test_skips_binary_files_in_directory_scan(self, ws: Path, registry: ScriptToolRegistry):
        """Binary file is skipped and noted when scanning a directory."""
        (ws / "bin.dat").write_bytes(b"\x00\x01needle\x00\x02")
        (ws / "text.txt").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(ws)})
        assert "text.txt" in result
        assert "bin.dat" not in result
        assert "binary file" in result.lower()

    def test_always_searches_directly_named_binary_file(self, ws: Path, registry: ScriptToolRegistry):
        """path pointing directly at a file with null bytes is still searched
        (decode errors replaced), covering a partially corrupted log file."""
        f = ws / "corrupt.log"
        f.write_bytes(b"line one\nneedle in \xff\xfe corrupted line\x00\nline three\n")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(f)})
        assert "needle" in result

    def test_max_results_truncates(self, ws: Path, registry: ScriptToolRegistry):
        """More matches than max_results -> truncation hint shown."""
        f = ws / "many.txt"
        f.write_text("\n".join("needle" for _ in range(10)) + "\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "max_results": 3,
        })
        assert "Showing first 3 matches" in result

    def test_long_line_truncated(self, ws: Path, registry: ScriptToolRegistry):
        """A matched line longer than the per-line cap is truncated with a marker."""
        f = ws / "long.txt"
        long_line = "needle " + ("x" * 500)
        f.write_text(long_line + "\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(ws)})
        assert "[truncated: line too long]" in result
        assert "x" * 500 not in result

    def test_invalid_regex(self, ws: Path, registry: ScriptToolRegistry):
        """Invalid regex -> error, not a crash."""
        f = ws / "code.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "(unclosed", "path": str(ws)})
        assert "Invalid grep_regex" in result

    def test_nonexistent_path(self, ws: Path, registry: ScriptToolRegistry):
        """Non-existent path -> error."""
        result = run(registry, "fs_grep", {
            "grep_regex": "x", "path": str(ws / "nope"),
        })
        assert "not found" in result.lower()

    def test_grep_hidden_directory_pruned(self, ws: Path, registry: ScriptToolRegistry):
        """Matches under a dot-prefixed directory are excluded, and noted."""
        (ws / "visible.py").write_text("needle\n", encoding="utf-8")
        hidden_dir = ws / ".venv"
        hidden_dir.mkdir()
        (hidden_dir / "lib.py").write_text("needle\n", encoding="utf-8")

        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(ws)})
        assert "visible.py" in result
        assert "lib.py" not in result
        assert "hidden" in result.lower()

    def test_grep_exclude_regex_prunes_directory(self, ws: Path, registry: ScriptToolRegistry):
        """exclude_regex matching a subdirectory name prunes it (and is noted)."""
        (ws / "visible.py").write_text("needle\n", encoding="utf-8")
        excluded_dir = ws / "node_modules"
        excluded_dir.mkdir()
        (excluded_dir / "lib.py").write_text("needle\n", encoding="utf-8")

        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "exclude_regex": "node_modules",
        })
        assert "visible.py" in result
        assert "lib.py" not in result
        assert "excluded" in result.lower()

    def test_invalid_exclude_regex(self, ws: Path, registry: ScriptToolRegistry):
        """Invalid exclude_regex -> error, not a crash."""
        f = ws / "code.py"
        f.write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "exclude_regex": "(unclosed",
        })
        assert "Invalid exclude_regex" in result

    def test_hidden_file_pruned_by_wildcard_glob(self, ws: Path, registry: ScriptToolRegistry):
        """A '*' file_glob does not match a dot-prefixed file, mirroring shell semantics."""
        (ws / ".env").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(ws)})
        assert "No matches" in result

    def test_explicit_dot_glob_matches_hidden_file(self, ws: Path, registry: ScriptToolRegistry):
        """A file_glob whose final segment starts with '.' (e.g. '.env')
        explicitly matches hidden files despite the default pruning."""
        (ws / ".env").write_text("needle\n", encoding="utf-8")
        (ws / "visible.txt").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_glob": ".env", "path": str(ws),
        })
        assert ".env:1:" in result
        assert "visible.txt" not in result

    def test_max_results_stops_walk_early(self, ws: Path, registry: ScriptToolRegistry):
        """max_results is enforced during the walk (single pass), not just
        after enumerating every candidate file — verified indirectly via the
        truncation note still firing correctly with many candidate files."""
        d = ws / "many"
        d.mkdir()
        for i in range(20):
            (d / f"f{i}.txt").write_text("needle\n" * 5, encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(d), "max_results": 3,
            "context_lines": 0,
        })
        assert "Showing first 3 matches" in result
        # With context_lines=0, only match lines appear (each contains "needle" once)
        assert result.count("needle") == 3

    def test_max_results_per_file_caps_matches_within_a_file(self, ws: Path, registry: ScriptToolRegistry):
        """max_results_per_file limits matches contributed by a single file
        during a directory scan, preserving breadth across other files."""
        dense = ws / "dense.txt"
        dense.write_text("needle\n" * 10, encoding="utf-8")
        sparse = ws / "sparse.txt"
        sparse.write_text("needle\n", encoding="utf-8")

        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "max_results_per_file": 2,
            "context_lines": 0,
        })
        # Only match lines (context_lines=0), so count filename occurrences directly
        assert result.count("dense.txt") == 2
        assert "sparse.txt" in result

    def test_max_results_per_file_applies_to_directly_named_file(self, ws: Path, registry: ScriptToolRegistry):
        """max_results_per_file also caps matches when path names a single
        file directly, not just during a directory walk."""
        f = ws / "dense.txt"
        f.write_text("needle\n" * 10, encoding="utf-8")

        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(f), "max_results_per_file": 3,
            "context_lines": 0,
        })
        # With context_lines=0, each match line contains the filename once
        assert result.count("dense.txt") == 3

    def test_file_regex_filters_files(self, ws: Path, registry: ScriptToolRegistry):
        """file_regex as alternative to file_glob restricts which files are searched."""
        (ws / "Server.log").write_text("needle\n", encoding="utf-8")
        (ws / "Client.log").write_text("needle\n", encoding="utf-8")
        (ws / "other.txt").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_regex": r"(Server|Client)\.log",
            "path": str(ws), "context_lines": 0,
        })
        assert "Server.log" in result
        assert "Client.log" in result
        assert "other.txt" not in result

    def test_file_glob_and_file_regex_mutually_exclusive(self, ws: Path, registry: ScriptToolRegistry):
        """Passing both file_glob and file_regex is an error."""
        (ws / "a.py").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_glob": "*.py",
            "file_regex": "a", "path": str(ws),
        })
        assert "not both" in result.lower()

    def test_grep_file_regex_with_slash_matches_full_relative_path(
        self, ws: Path, registry: ScriptToolRegistry,
    ):
        """A file_regex containing '/' scopes to the full relative path,
        mirroring file_glob's basename-vs-full-path convention."""
        sub = ws / "sub"
        sub.mkdir()
        (sub / "target.py").write_text("needle\n", encoding="utf-8")
        (ws / "target.py").write_text("needle\n", encoding="utf-8")

        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_regex": r"sub/target\.py",
            "path": str(ws), "context_lines": 0,
        })
        assert "sub/target.py" in result
        assert result.count("target.py") == 1

    def test_grep_file_regex_matches_as_substring(self, ws: Path, registry: ScriptToolRegistry):
        """file_regex matches anywhere in the name (re.search), like
        grep_regex/exclude_regex — a partial pattern missing the extension
        still matches."""
        (ws / "fs_find.py").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_regex": r"fs_find", "path": str(ws),
        })
        assert "fs_find.py" in result

    def test_grep_file_regex_can_be_anchored_for_exact_match(self, ws: Path, registry: ScriptToolRegistry):
        """A caller wanting an exact-name match can anchor file_regex with ^/$."""
        (ws / "fs_find.py").write_text("needle\n", encoding="utf-8")
        (ws / "fs_find.pyc").write_text("needle\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "file_regex": r"^fs_find\.py$", "path": str(ws),
        })
        assert "fs_find.py:" in result
        assert "fs_find.pyc" not in result

    def test_context_lines_default_directory_scan(self, ws: Path, registry: ScriptToolRegistry):
        """Directory scan defaults to 1 line of context around matches."""
        f = ws / "ctx.txt"
        f.write_text("line1\nline2\nneedle\nline4\nline5\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(ws)})
        # Match line uses ':'
        assert "ctx.txt:3:" in result
        # Context lines use '-' separator and show 1 before/after
        assert "ctx.txt-2-" in result
        assert "ctx.txt-4-" in result
        # Lines outside the 1-line context window should NOT appear
        assert "ctx.txt-1-" not in result
        assert "ctx.txt-5-" not in result

    def test_context_lines_default_single_file(self, ws: Path, registry: ScriptToolRegistry):
        """Single-file target defaults to 4 lines of context around matches."""
        f = ws / "ctx_single.txt"
        lines = [f"line{i}" for i in range(1, 12)]
        lines[5] = "needle"  # line 6 (1-indexed)
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = run(registry, "fs_grep", {"grep_regex": "needle", "path": str(f)})
        # Match at line 6 with 4 context -> lines 2..10
        assert "ctx_single.txt:6:" in result
        assert "ctx_single.txt-2-" in result  # 4 lines before
        assert "ctx_single.txt-10-" in result  # 4 lines after

    def test_context_lines_zero_compact(self, ws: Path, registry: ScriptToolRegistry):
        """context_lines=0 produces compact output with no context."""
        f = ws / "compact.txt"
        f.write_text("before\nneedle\nafter\n", encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "context_lines": 0,
        })
        assert "compact.txt:2:" in result
        # No context lines
        assert "compact.txt-1-" not in result
        assert "compact.txt-3-" not in result

    def test_context_lines_overlap_merged(self, ws: Path, registry: ScriptToolRegistry):
        """Adjacent matches have their context windows merged, separated by '--'
        only between non-overlapping groups."""
        f = ws / "overlap.txt"
        # Matches on lines 2 and 4 (context_lines=1 -> windows [1..3] and [3..5] overlap -> merged)
        # Match on line 9 (window [8..10]) is separated from the first group by a gap (lines 6-7)
        content = "line1\nneedle_a\nline3\nneedle_b\nline5\nline6\nline7\nline8\nneedle_c\nline10\n"
        f.write_text(content, encoding="utf-8")
        result = run(registry, "fs_grep", {
            "grep_regex": "needle", "path": str(ws), "context_lines": 1,
        })
        # needle_a (line2) and needle_b (line4) overlap -> merged, no '--' between them
        # needle_c (line9) is separated -> '--' before it
        assert "--" in result
        lines = result.splitlines()
        sep_indices = [i for i, l in enumerate(lines) if l == "--"]
        assert len(sep_indices) == 1


# =============================================================================
# fs_find file_regex tests
# =============================================================================


class TestFsFindFileRegex:
    def test_file_regex_matches(self, ws: Path, registry: ScriptToolRegistry):
        """file_regex matches filenames by regex."""
        (ws / "Server.log").write_text("x", encoding="utf-8")
        (ws / "Client.log").write_text("x", encoding="utf-8")
        (ws / "other.txt").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_regex": r"(Server|Client)\.log", "path": str(ws),
        })
        assert "Server.log" in result
        assert "Client.log" in result
        assert "other.txt" not in result

    def test_file_glob_and_file_regex_mutually_exclusive(self, ws: Path, registry: ScriptToolRegistry):
        """Passing both file_glob and file_regex is an error."""
        (ws / "a.py").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_glob": "*.py", "file_regex": "a", "path": str(ws),
        })
        assert "not both" in result.lower()

    def test_invalid_file_regex(self, ws: Path, registry: ScriptToolRegistry):
        """Invalid file_regex -> error, not a crash."""
        (ws / "a.py").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_regex": "(unclosed", "path": str(ws),
        })
        assert "Invalid file_regex" in result

    def test_find_file_regex_with_slash_matches_full_relative_path(
        self, ws: Path, registry: ScriptToolRegistry,
    ):
        """A file_regex containing '/' matches the full relative path from
        path, mirroring file_glob's basename-vs-full-path convention."""
        sub = ws / "sub"
        sub.mkdir()
        (sub / "target.py").write_text("x", encoding="utf-8")
        (ws / "target.py").write_text("x", encoding="utf-8")

        # Without a slash, matches basenames at any depth -> both files.
        result_basename = run(registry, "fs_find", {
            "file_regex": r"target\.py", "path": str(ws),
        })
        assert result_basename.count("target.py") == 2

        # With a slash, scopes to the full relative path -> only the nested one.
        result_scoped = run(registry, "fs_find", {
            "file_regex": r"sub/target\.py", "path": str(ws),
        })
        assert "sub/target.py" in result_scoped
        assert result_scoped.count("target.py") == 1

    def test_find_file_regex_matches_as_substring(self, ws: Path, registry: ScriptToolRegistry):
        """file_regex matches anywhere in the name (re.search), like
        exclude_regex — a partial pattern missing the extension still matches."""
        (ws / "fs_find.py").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_regex": r"fs_find", "path": str(ws),
        })
        assert "fs_find.py" in result

    def test_find_file_regex_can_be_anchored_for_exact_match(self, ws: Path, registry: ScriptToolRegistry):
        """A caller wanting an exact-name match can anchor file_regex with ^/$."""
        (ws / "fs_find.py").write_text("x", encoding="utf-8")
        (ws / "fs_find.pyc").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_regex": r"^fs_find\.py$", "path": str(ws),
        })
        assert "fs_find.py" in result
        assert "fs_find.pyc" not in result

    def test_find_file_regex_truncation_note_names_file_regex(
        self, ws: Path, registry: ScriptToolRegistry,
    ):
        """The truncation hint names the parameter actually in use — it should
        say 'file_regex', not 'file_glob', when file_regex was the active
        pattern (regression: the note used to hardcode 'file_glob')."""
        d = ws / "many"
        d.mkdir()
        for i in range(10):
            (d / f"f{i}.py").write_text("x", encoding="utf-8")
        result = run(registry, "fs_find", {
            "file_regex": r"\.py$", "path": str(d), "max_results": 3,
        })
        assert "Narrow file_regex or path" in result
        assert "file_glob" not in result


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

    def test_home_tilde_expansion(self, ws: Path, registry: ScriptToolRegistry, monkeypatch: pytest.MonkeyPatch):
        """A leading '~' in path is expanded to the home directory, matching
        the '~' syntax documented for [filesystem] writable_paths/readable_paths."""
        monkeypatch.setenv("USERPROFILE", str(ws))  # Windows
        monkeypatch.setenv("HOME", str(ws))  # POSIX
        f = ws / "tilde_target.txt"
        f.write_text("hello from home\n", encoding="utf-8")
        result = run(registry, "read_file", {"path": "~/tilde_target.txt"})
        assert "hello from home" in result

    def test_fs_grep_reports_permission_denied_not_silent(
        self, tmp_path: Path,
    ):
        """A file blocked by a [filesystem.guards] deny-agent rule must be
        reported as skipped/unreadable, not silently absent from results as
        if it simply didn't match (which would look identical to 'no matches
        found' and mask a security-relevant rejection)."""
        ws = tmp_path / "guarded_ws"
        ws.mkdir()
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
glob = true

[filesystem]
writable_paths = ["."]
readable_paths = ["*"]

[filesystem.guards]
'/secret/' = "deny-agent"

[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = []

[audit]
"open" = "fs"
""",
            encoding="utf-8",
        )
        config = load_config(ws)
        vm = VenvManager(venv_path=ws / ".pyddock" / "venv", allowed_imports=config.imports.allowed)
        vm.ensure_venv()
        executor = SubprocessExecutor(config, vm)
        reg = ScriptToolRegistry(config, executor, ws)
        reg.load_scripts()

        guarded_dir = ws / "secret"
        guarded_dir.mkdir()
        (guarded_dir / "creds.txt").write_text("needle\n", encoding="utf-8")

        result = run(reg, "fs_grep", {"grep_regex": "needle", "path": str(guarded_dir)})
        assert "unreadable" in result.lower() or "permission" in result.lower()

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
glob = true
[filesystem]
writable_paths = ["."]
readable_paths = ["."]
[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = []

[audit]
"open" = "fs"
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
