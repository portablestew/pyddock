"""fs_delete tool script.

Deletes a single file or empty directory. Returns a unified diff of the
removed content (truncated to 16 KB for large files).
"""
import difflib
import os
import sys
from pathlib import Path

# stat.S_IWRITE == 0o200 — avoid importing stat module
_S_IWRITE = 0o200

# --- Parameters ---
path_str = _PARAMS["path"]
workspace_root = _PARAMS["workspace_root"]

# --- Constants ---
_MAX_DELETE_DIFF_BYTES = 16_384  # 16 KB truncation for delete diffs


def resolve_path(raw_path: str) -> Path:
    """Resolve a user-provided path to absolute, relative to workspace root.

    Expands a leading '~' (home directory) before resolving, mirroring how
    [filesystem] scope paths (writable_paths/readable_paths) are resolved.
    """
    expanded = os.path.expanduser(raw_path)
    p = Path(expanded)
    if p.is_absolute():
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(Path(workspace_root) / expanded)))


def generate_diff(old_lines: list[str], new_lines: list[str], filename: str) -> str:
    """Generate a unified diff between old and new content lines."""
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filename,
        tofile="/dev/null",
        lineterm="",
    )
    return "\n".join(diff)


# --- Main ---
path = resolve_path(path_str)

if not path.exists():
    print(f"File not found: '{path_str}'", file=sys.stderr)
    sys.exit(1)

if path.is_dir():
    # Only delete empty directories
    if any(path.iterdir()):
        print(
            f"Cannot delete '{path_str}': directory is not empty. "
            f"Use run_python for recursive deletion.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        path.rmdir()
    except PermissionError:
        print(f"Cannot delete '{path_str}': permission denied. The directory may be locked or have restrictive permissions.", file=sys.stderr)
        sys.exit(1)
    diff_output = generate_diff(["(empty directory)"], [], path_str)
    print(diff_output)
else:
    # Read content for diff (truncate at 16 KB)
    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > _MAX_DELETE_DIFF_BYTES:
            content = content[:_MAX_DELETE_DIFF_BYTES]
            truncated = True
        else:
            truncated = False
    except UnicodeDecodeError:
        content = "(binary file)"
        truncated = False

    # Delete the file
    try:
        path.unlink()
    except PermissionError:
        try:
            is_readonly = not (path.stat().st_mode & _S_IWRITE)
        except OSError:
            is_readonly = False
        if is_readonly:
            print(f"Cannot delete '{path_str}': file is read-only. If under version control, ensure the file is checked out and retry.", file=sys.stderr)
        else:
            print(f"Cannot delete '{path_str}': permission denied. The file may be locked by another process or have restrictive permissions.", file=sys.stderr)
        sys.exit(1)

    # Generate diff showing removal
    old_lines = content.splitlines()
    if truncated:
        old_lines.append("[truncated: file exceeded 16 KB]")
    diff_output = generate_diff(old_lines, [], path_str)
    print(diff_output)
