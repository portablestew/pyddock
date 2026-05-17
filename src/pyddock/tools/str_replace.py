"""str_replace tool script.

Find and replace exact text in a file. Requires a unique match.
On failure, provides helpful diagnostics:
- Multiple matches: shows first 5 with line numbers and context
- No match: multi-pass fuzzy search with scored candidates
"""
import difflib
import os
import re
import sys
from pathlib import Path

# stat.S_IWRITE == 0o200 — avoid importing stat module
_S_IWRITE = 0o200

# --- Parameters ---
path_str = _PARAMS["path"]
old_str = _PARAMS["oldStr"]
new_str = _PARAMS["newStr"]
start_line = _PARAMS.get("start_line")
end_line = _PARAMS.get("end_line")
workspace_root = _PARAMS["workspace_root"]


def resolve_path(raw_path: str) -> Path:
    """Resolve a user-provided path to absolute, relative to workspace root."""
    p = Path(raw_path)
    if p.is_absolute():
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(Path(workspace_root) / raw_path)))


def generate_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filename,
        tofile=filename,
    )
    return "".join(diff)


def find_all_occurrences(text: str, pattern: str) -> list[int]:
    """Find all start positions of exact pattern in text."""
    positions = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def char_offset_to_line(content: str, offset: int) -> int:
    """Convert a character offset to a 1-indexed line number."""
    return content[:offset].count("\n") + 1


def format_multiple_matches(content: str, match_positions: list[int], old_str: str) -> str:
    """Format error response showing first 5 of N exact matches with context."""
    lines = content.splitlines()
    total_matches = len(match_positions)
    show_count = min(5, total_matches)

    parts = [f"Multiple matches found. Showing {show_count} of {total_matches} exact matches:"]
    parts.append("")

    for i, pos in enumerate(match_positions[:5]):
        line_num = char_offset_to_line(content, pos)
        # Show 1 line of context above and below
        start_ctx = max(0, line_num - 2)
        end_ctx = min(len(lines), line_num + 1)

        parts.append(f"Match {i + 1} (line {line_num}):")
        for j in range(start_ctx, end_ctx):
            parts.append(f"  {j + 1:>4}| {lines[j]}")
        parts.append("")

    parts.append(
        "Tip: Add surrounding context to oldStr to make it unique, "
        "or use start_line/end_line to constrain the search window."
    )
    return "\n".join(parts)


def fuzzy_search(content: str, old_str: str) -> str:
    """Multi-pass fuzzy matching for str_replace no-match failures."""
    file_lines = content.splitlines()
    old_lines = old_str.splitlines()
    window_size = len(old_lines)

    normalizers = [
        ("exact", lambda s: s),
        ("whitespace-normalized", lambda s: re.sub(r"\s+", "", s).lower()),
        ("alnum-only", lambda s: re.sub(r"[^a-z0-9]", "", s.lower())),
    ]

    for pass_name, normalize in normalizers:
        # Build set of normalized oldStr lines (skip empty after normalization)
        normalized_old = set()
        for line in old_lines:
            n = normalize(line)
            if n:
                normalized_old.add(n)

        if not normalized_old:
            continue

        # Find candidate anchors and score them
        candidates = []  # (line_number, score)

        for i, file_line in enumerate(file_lines):
            normalized_file_line = normalize(file_line)
            if not normalized_file_line:
                continue
            if normalized_file_line in normalized_old:
                # Score: count matching lines in surrounding window
                window_start = max(0, i - window_size + 1)
                window_end = min(len(file_lines), i + window_size)
                score = 0
                for j in range(window_start, window_end):
                    n = normalize(file_lines[j])
                    if n and n in normalized_old:
                        score += 1
                candidates.append((i + 1, score))  # 1-indexed line number

        if candidates:
            # Deduplicate overlapping windows: keep highest score per region
            candidates.sort(key=lambda c: c[1], reverse=True)
            deduped = []
            used_lines = set()
            for line_num, score in candidates:
                # Skip if too close to an already-selected candidate
                if any(abs(line_num - ul) < window_size for ul in used_lines):
                    continue
                deduped.append((line_num, score))
                used_lines.add(line_num)

            total_candidates = len(deduped)
            show_count = min(5, total_candidates)
            top = deduped[:5]

            parts = [
                f"No exact match. Showing {show_count} of {total_candidates} "
                f"partial matches ({pass_name}):"
            ]
            parts.append("")

            for line_num, score in top:
                # Show context around the match
                ctx_start = max(0, line_num - 2)  # 0-indexed
                ctx_end = min(len(file_lines), line_num + window_size)
                parts.append(f"Near line {line_num} (score: {score}/{window_size} lines):")
                for j in range(ctx_start, ctx_end):
                    parts.append(f"  {j + 1:>4}| {file_lines[j]}")
                parts.append("")

            parts.append("Tip: Copy the exact text from the file (whitespace matters).")
            return "\n".join(parts)

    return (
        f"No exact match found in '{path_str}'. No partial matches detected.\n"
        f"File has {len(file_lines)} lines. Use read_file to inspect content."
    )


# --- Main ---
if not old_str:
    print("oldStr must be non-empty.", file=sys.stderr)
    sys.exit(1)

path = resolve_path(path_str)

if not path.exists():
    print(f"File not found: '{path_str}'", file=sys.stderr)
    sys.exit(1)

try:
    content = path.read_text(encoding="utf-8")
except UnicodeDecodeError:
    print(
        f"Cannot read '{path_str}': file is binary or not UTF-8 encoded. "
        f"Use run_python for binary file operations.",
        file=sys.stderr,
    )
    sys.exit(1)

# Optionally constrain search to line window
if start_line or end_line:
    lines = content.splitlines(keepends=True)
    s_idx = (start_line - 1) if start_line else 0
    e_idx = end_line if end_line else len(lines)
    search_region = "".join(lines[s_idx:e_idx])
    # Character offset of the window start within full content
    window_char_offset = sum(len(l) for l in lines[:s_idx])
else:
    search_region = content
    window_char_offset = 0

# Find occurrences in search region
matches = find_all_occurrences(search_region, old_str)

if len(matches) == 1:
    # Single match — replace at exact position in full content
    match_pos = window_char_offset + matches[0]
    new_content = (
        content[:match_pos]
        + new_str
        + content[match_pos + len(old_str):]
    )
    try:
        path.write_text(new_content, encoding="utf-8")
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
    print(generate_diff(content, new_content, path_str))

elif len(matches) > 1:
    # Multiple matches — report locations (offset to full-file positions)
    full_matches = [window_char_offset + m for m in matches]
    print(format_multiple_matches(content, full_matches, old_str))

else:
    # No match — fuzzy search against full content
    print(fuzzy_search(content, old_str))
