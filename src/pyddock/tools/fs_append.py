"""fs_append tool script.

Appends content to a file (creates if missing). Uses true append for large
file efficiency — only reads the last ~2 KB to check trailing newline and
capture context lines for the diff response.
"""
import difflib
import os
import sys
from pathlib import Path

# stat.S_IWRITE == 0o200 — avoid importing stat module
_S_IWRITE = 0o200

# --- Parameters ---
path_str = _PARAMS["path"]
content = _PARAMS["content"]
workspace_root = _PARAMS["workspace_root"]


def resolve_path(raw_path: str) -> Path:
    """Resolve a user-provided path to absolute, relative to workspace root."""
    p = Path(raw_path)
    if p.is_absolute():
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(Path(workspace_root) / raw_path)))


def generate_diff(old_lines: list[str], new_lines: list[str], filename: str) -> str:
    """Generate a unified diff between old and new content lines."""
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filename,
        tofile=filename,
        lineterm="",
    )
    return "\n".join(diff)


# --- Main ---
path = resolve_path(path_str)

# Create parent directories (skip if already exists to avoid triggering
# write-protection checks on existing protected-but-writable dirs like .pyddock/tmp/)
if not path.parent.exists():
    path.parent.mkdir(parents=True, exist_ok=True)

if path.exists():
    file_size = path.stat().st_size

    # Read only the tail (last 2 KB) to check trailing newline and capture context
    tail_size = min(file_size, 2048)
    try:
        with open(path, "r", encoding="utf-8") as f:
            if file_size > tail_size:
                f.seek(file_size - tail_size)
                f.readline()  # discard partial line after seek
            tail_text = f.read()
    except UnicodeDecodeError:
        print(
            f"Cannot append to '{path_str}': file is binary or not UTF-8 encoded. "
            f"Use run_python for binary file operations.",
            file=sys.stderr,
        )
        sys.exit(1)

    tail_lines = tail_text.splitlines()
    needs_newline = tail_text and not tail_text.endswith("\n")

    # Capture context (last 5 lines) for the diff
    context_lines = tail_lines[-5:] if tail_lines else []

    # Perform true append
    try:
        with open(path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(content)
    except PermissionError:
        try:
            is_readonly = not (path.stat().st_mode & _S_IWRITE)
        except OSError:
            is_readonly = False
        if is_readonly:
            print(f"Cannot write to '{path_str}': file is read-only. If under version control, ensure the file is checked out and retry.", file=sys.stderr)
        else:
            print(f"Cannot write to '{path_str}': permission denied. The file may be locked by another process or have restrictive permissions.", file=sys.stderr)
        sys.exit(1)

    # Build diff: old fragment (context) vs new fragment (context + appended)
    old_fragment_lines = context_lines
    if needs_newline:
        new_fragment_lines = context_lines + content.splitlines()
    else:
        # Last context line stays as-is, new content appended after it
        new_fragment_lines = context_lines + content.splitlines()

    diff_output = generate_diff(old_fragment_lines, new_fragment_lines, path_str)
else:
    # New file — write content directly
    try:
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        print(f"Cannot create '{path_str}': permission denied.", file=sys.stderr)
        sys.exit(1)

    # Diff: empty → content
    old_fragment_lines = []
    new_fragment_lines = content.splitlines()
    diff_output = generate_diff(old_fragment_lines, new_fragment_lines, path_str)

print(diff_output)
