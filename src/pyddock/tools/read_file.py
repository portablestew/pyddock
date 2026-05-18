"""read_file tool script.

Reads a text file with optional line range. Produces line-numbered output.
Enforces a ~50 KB output cap with line-aware truncation and continuation hints.
"""
import os
import sys
from pathlib import Path

# --- Parameters ---
path_str = _PARAMS["path"]
start = _PARAMS.get("start")
end = _PARAMS.get("end")
workspace_root = _PARAMS["workspace_root"]

# --- Constants ---
_MAX_OUTPUT_BYTES = 50_000  # ~50 KB output cap


def resolve_path(raw_path: str) -> Path:
    """Resolve a user-provided path to absolute, relative to workspace root."""
    p = Path(raw_path)
    if p.is_absolute():
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(Path(workspace_root) / raw_path)))


def resolve_line_range(start, end, total):
    """Convert 1-indexed inclusive range to 0-indexed [start, end) slice indices.

    - Negative start = tail (last N lines), end is ignored.
    - Positive start/end are 1-indexed, inclusive.
    - Clamps to valid range.
    """
    if start is None and end is None:
        return (0, total)

    if start is not None and start < 0:
        # Negative start = tail: last N lines, end is ignored
        n = abs(start)
        start_idx = max(0, total - n)
        end_idx = total
    else:
        start_idx = (start - 1) if start is not None else 0
        end_idx = end if end is not None else total

    # Clamp to valid range
    start_idx = max(0, min(start_idx, total))
    end_idx = max(start_idx, min(end_idx, total))
    return (start_idx, end_idx)


# --- Main ---
path = resolve_path(path_str)

if not path.exists():
    print(f"File not found: '{path_str}'", file=sys.stderr)
    sys.exit(1)

if path.is_dir():
    print(f"Cannot read '{path_str}': path is a directory, not a file.", file=sys.stderr)
    sys.exit(1)

# Read with UTF-8 — fails on binary or permission denied
try:
    content = path.read_text(encoding="utf-8")
except PermissionError as e:
    # Extract just the message, not the traceback (avoids leaking internal paths)
    msg = str(e)
    print(msg, file=sys.stderr)
    sys.exit(1)
except UnicodeDecodeError:
    print(
        f"Cannot read '{path_str}': file is binary or not UTF-8 encoded. "
        f"Use run_python with Path.read_bytes() for binary files.",
        file=sys.stderr,
    )
    sys.exit(1)

lines = content.splitlines()
total = len(lines)

# Resolve line range
start_idx, end_idx = resolve_line_range(start, end, total)

# Format with line numbers, enforcing output cap
selected = lines[start_idx:end_idx]
width = len(str(end_idx)) if end_idx > 0 else 1
output_parts = []
output_size = 0
last_emitted_idx = start_idx  # tracks how far we got (0-indexed, exclusive)

for i, line in enumerate(selected):
    numbered_line = f"{start_idx + i + 1:>{width}}| {line}\n"
    if output_size + len(numbered_line) > _MAX_OUTPUT_BYTES:
        # Truncate here — emit continuation hint
        next_line = start_idx + i + 1  # 1-indexed, the line we didn't show
        output_parts.append(
            f"\n[Showing lines {start_idx + 1}-{start_idx + i} of {total}. "
            f"Truncated at ~50 KB. Use start={next_line} to continue.]"
        )
        last_emitted_idx = start_idx + i
        break
    output_parts.append(numbered_line)
    output_size += len(numbered_line)
    last_emitted_idx = start_idx + i + 1
else:
    # No truncation — show range info if partial read
    if start_idx > 0 or end_idx < total:
        output_parts.append(
            f"\n[Showing lines {start_idx + 1}-{end_idx} of {total}.]"
        )

print("".join(output_parts).rstrip("\n"))
