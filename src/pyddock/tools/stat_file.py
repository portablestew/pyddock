"""stat_file tool script.

Returns file metadata: exists, type, size, line count, modified timestamp.
Non-existent paths return "exists: false" (not an error).
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Parameters ---
path_str = _PARAMS["path"]
workspace_root = _PARAMS["workspace_root"]


def resolve_path(raw_path: str) -> Path:
    """Resolve a user-provided path to absolute, relative to workspace root."""
    p = Path(raw_path)
    if p.is_absolute():
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(Path(workspace_root) / raw_path)))


# --- Main ---
path = resolve_path(path_str)

if not path.exists():
    print("exists: false")
    sys.exit(0)

stat = path.stat()
is_dir = path.is_dir()

parts = [
    "exists: true",
    f"type: {'directory' if is_dir else 'file'}",
    f"size: {stat.st_size} bytes",
]

if not is_dir:
    try:
        content = path.read_text(encoding="utf-8")
        line_count = len(content.splitlines())
        parts.append(f"lines: {line_count}")
    except (UnicodeDecodeError, OSError):
        parts.append("lines: (binary or unreadable)")

modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
parts.append(f"modified: {modified}")

print("\n".join(parts))
