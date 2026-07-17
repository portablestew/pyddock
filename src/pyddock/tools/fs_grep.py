"""fs_grep tool script.

Search file contents by regex (always case-insensitive). Approximately:
fs_find() then re.search(grep_regex, line, re.IGNORECASE) per line of each
matched file.

grep_regex is searched within each file matched by file_glob (glob,
default '*') or file_regex (mutually exclusive) under path (default:
workspace root). Returns 'path:line: content' per match, capped at
max_results (default: 100) and 300 chars per line. Optional
max_results_per_file caps matches per file. Like file_glob, file_regex
matches basenames at any depth unless it contains '/' (then it matches the
full relative path); like grep_regex/exclude_regex, file_regex matches
anywhere in the name/path (re.search), not just the whole thing — anchor
with ^/$ for an exact match.

Hidden dot-directories are pruned before descending (see fs_find); an
optional exclude_regex prunes/excludes additional entries the same way. A
file_glob whose final segment starts with '.' (e.g. '.env') explicitly
matches hidden files despite the pruning, mirroring shell glob semantics
where '*' does not match a leading dot but an explicit pattern does.

Binary files are skipped when scanning a directory path (sniffed from the
same read used for matching — no separate open). A single named file is
always searched, binary or not.

context_lines controls how many surrounding lines are shown around each
match. Default: 1 for directory scans, 4 for single-file targets. Set to 0
for compact output with no context.
"""
import glob as _glob
import os
import re
import sys
from pathlib import Path

# --- Parameters ---
grep_regex = _PARAMS["grep_regex"]
file_glob = _PARAMS.get("file_glob")
file_regex = _PARAMS.get("file_regex")
path_str = _PARAMS.get("path") or "."
exclude_regex = _PARAMS.get("exclude_regex")
max_results = _PARAMS.get("max_results") or 100
max_results_per_file = _PARAMS.get("max_results_per_file")
context_lines = _PARAMS.get("context_lines")  # None = adaptive default
workspace_root = _PARAMS["workspace_root"]

# --- Constants ---
_SNIFF_BYTES = 8192  # bytes inspected for the binary heuristic
_MAX_LINE_CHARS = 300  # cap per matched line so one huge line can't dominate output


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


def display_path(p: Path) -> str:
    """Format a path for output: relative to workspace root when possible."""
    try:
        rel = p.relative_to(workspace_root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p)


def is_hidden(name: str) -> bool:
    """True if a single path component is dot-prefixed."""
    return name.startswith(".")


def rel_posix(base: Path, dirpath: str) -> str:
    """Relative path from base to dirpath, forward-slash, '' at base itself."""
    rel = os.path.relpath(dirpath, str(base))
    if rel == ".":
        return ""
    return rel.replace("\\", "/")


def truncate_line(line: str) -> str:
    """Cap a matched line at _MAX_LINE_CHARS, marking truncation."""
    if len(line) <= _MAX_LINE_CHARS:
        return line
    return line[:_MAX_LINE_CHARS] + " [truncated: line too long]"


def read_text_or_skip_reason(p: Path) -> tuple[str | None, str | None]:
    """Read a file once, sniffing for binary content in the same pass.

    Returns (text, None) on success, or (None, reason) where reason is
    "binary" or "unreadable". A single open() call: binary files stop after
    the first _SNIFF_BYTES (no wasted read of large binary blobs); text
    files reuse the already-read prefix instead of reopening.
    """
    try:
        with open(p, "rb") as f:
            chunk = f.read(_SNIFF_BYTES)
            if b"\x00" in chunk:
                return None, "binary"
            rest = f.read()
    except OSError:
        # Covers PermissionError (blocked by a filesystem guard, e.g.
        # ~/.ssh/) and other read failures (deleted mid-walk, broken
        # symlink, etc.).
        return None, "unreadable"
    return (chunk + rest).decode("utf-8", errors="replace"), None


def format_matches_with_context(
    lines: list[str], match_line_nums: list[int], label: str, ctx: int
) -> list[str]:
    """Format matched lines with surrounding context, merging overlapping groups.

    Uses grep-style output:
      path:linenum: matched line content
      path-linenum- context line content
    Groups of matches whose context windows overlap are merged and separated
    from non-overlapping groups by a '--' line.

    Args:
        lines: All lines in the file (0-indexed list).
        match_line_nums: 1-indexed line numbers that matched.
        label: Display path prefix for each output line.
        ctx: Number of context lines before and after each match.

    Returns:
        List of formatted output strings.
    """
    if not match_line_nums:
        return []

    total = len(lines)
    output: list[str] = []

    # Build merged groups of (start_0idx, end_0idx_exclusive, set_of_match_0idxs)
    groups: list[tuple[int, int, set[int]]] = []
    for ln in match_line_nums:
        idx = ln - 1  # convert to 0-indexed
        start = max(0, idx - ctx)
        end = min(total, idx + ctx + 1)
        if groups and start <= groups[-1][1]:
            # Merge with previous group
            prev_start, prev_end, prev_matches = groups[-1]
            groups[-1] = (prev_start, max(prev_end, end), prev_matches | {idx})
        else:
            groups.append((start, end, {idx}))

    for gi, (start, end, match_idxs) in enumerate(groups):
        if gi > 0:
            output.append("--")
        for i in range(start, end):
            line_content = truncate_line(lines[i])
            line_num = i + 1  # 1-indexed for display
            if i in match_idxs:
                output.append(f"{label}:{line_num}: {line_content}")
            else:
                output.append(f"{label}-{line_num}- {line_content}")

    return output


# --- Main ---
if not grep_regex:
    print("grep_regex must be non-empty.", file=sys.stderr)
    sys.exit(1)

try:
    regex = re.compile(grep_regex, re.IGNORECASE)
except re.error as e:
    print(f"Invalid grep_regex '{grep_regex}': {e}", file=sys.stderr)
    sys.exit(1)

# --- File pattern validation (mutually exclusive) ---
if file_glob and file_regex:
    print("Provide file_glob or file_regex, not both.", file=sys.stderr)
    sys.exit(1)

if not file_glob and not file_regex:
    file_glob = "*"

if file_glob:
    file_pattern_rx = re.compile(
        _glob.translate(file_glob, recursive=True, include_hidden=True, seps="/")
    )
    allow_hidden_files = file_glob.rsplit("/", 1)[-1].startswith(".")
    match_basename_only = "/" not in file_glob
else:
    try:
        file_pattern_rx = re.compile(file_regex)
    except re.error as e:
        print(f"Invalid file_regex '{file_regex}': {e}", file=sys.stderr)
        sys.exit(1)
    # See fs_find.py for the rationale on allow_hidden_files and
    # match_basename_only in regex mode.
    allow_hidden_files = False
    match_basename_only = "/" not in file_regex

exclude_re = None
if exclude_regex:
    try:
        exclude_re = re.compile(exclude_regex)
    except re.error as e:
        print(f"Invalid exclude_regex '{exclude_regex}': {e}", file=sys.stderr)
        sys.exit(1)

path = resolve_path(path_str)

if not path.exists():
    print(f"Path not found: '{path_str}'", file=sys.stderr)
    sys.exit(1)

matches: list[str] = []
match_count = 0  # counts actual match lines (not context) for max_results
skipped_hidden = 0
skipped_excluded = 0
skipped_binary = 0
skipped_unreadable = 0
truncated = False

# Resolve adaptive context_lines default: 4 for single-file, 1 for dir scan.
is_single_file = path.is_file()
if context_lines is not None:
    ctx = max(0, int(context_lines))
else:
    ctx = 4 if is_single_file else 1

if is_single_file:
    # A directly named file is always searched — bypasses hidden/exclude/glob
    # filtering and the binary sniff.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        skipped_unreadable += 1
        text = None

    if text is not None:
        label = display_path(path)
        all_lines = text.splitlines()
        match_line_nums: list[int] = []
        for line_num, line in enumerate(all_lines, start=1):
            if regex.search(line):
                match_line_nums.append(line_num)
                match_count += 1
                if match_count >= max_results:
                    truncated = True
                    break
                if max_results_per_file and match_count >= max_results_per_file:
                    break

        if ctx > 0:
            matches = format_matches_with_context(all_lines, match_line_nums, label, ctx)
        else:
            for ln in match_line_nums:
                matches.append(f"{label}:{ln}: {truncate_line(all_lines[ln - 1])}")
else:
    # Single pass: walk, filter, and match together so max_results can stop
    # the walk early instead of enumerating the entire (pruned) tree first.
    for dirpath_str, dirnames, filenames in os.walk(path):
        if truncated:
            break
        rel_dir = rel_posix(path, dirpath_str)

        # Prune hidden / excluded subdirectories BEFORE os.walk descends.
        kept_dirnames = []
        for d in dirnames:
            if is_hidden(d):
                skipped_hidden += 1
                continue
            rel_sub = f"{rel_dir}/{d}" if rel_dir else d
            if exclude_re is not None and exclude_re.search(rel_sub):
                skipped_excluded += 1
                continue
            kept_dirnames.append(d)
        dirnames[:] = kept_dirnames

        for f in filenames:
            if truncated:
                break
            if is_hidden(f) and not allow_hidden_files:
                skipped_hidden += 1
                continue
            rel_file = f"{rel_dir}/{f}" if rel_dir else f
            if exclude_re is not None and exclude_re.search(rel_file):
                skipped_excluded += 1
                continue
            target = f if match_basename_only else rel_file
            # search, not match/fullmatch — see fs_find.py for rationale.
            if not file_pattern_rx.search(target):
                continue

            candidate = Path(dirpath_str) / f
            text, reason = read_text_or_skip_reason(candidate)
            if reason == "binary":
                skipped_binary += 1
                continue
            if reason == "unreadable":
                skipped_unreadable += 1
                continue

            label = display_path(candidate)
            all_lines = text.splitlines()
            file_match_count = 0
            match_line_nums = []
            for line_num, line in enumerate(all_lines, start=1):
                if regex.search(line):
                    match_line_nums.append(line_num)
                    file_match_count += 1
                    match_count += 1
                    if match_count >= max_results:
                        truncated = True
                        break
                    if max_results_per_file and file_match_count >= max_results_per_file:
                        break

            if ctx > 0:
                matches.extend(
                    format_matches_with_context(all_lines, match_line_nums, label, ctx)
                )
            else:
                for ln in match_line_nums:
                    matches.append(f"{label}:{ln}: {truncate_line(all_lines[ln - 1])}")

notes = []
if truncated:
    notes.append(
        f"Showing first {max_results} matches. To see more: narrow path, "
        f"grep_regex, or set max_results_per_file."
    )
if skipped_binary:
    notes.append(f"{skipped_binary} binary file(s) skipped.")
if skipped_unreadable:
    notes.append(f"{skipped_unreadable} unreadable file(s) skipped (permission denied or I/O error).")
if skipped_hidden:
    notes.append(f"{skipped_hidden} hidden file(s)/dir(s) skipped.")
if skipped_excluded:
    notes.append(f"{skipped_excluded} excluded file(s)/dir(s) skipped.")

if not matches:
    msg = f"No matches for '{grep_regex}' found under '{path_str}'."
    if notes:
        msg += " (" + " ".join(notes) + ")"
    print(msg)
else:
    output = "\n".join(matches)
    if notes:
        output += "\n\n[" + " ".join(notes) + "]"
    print(output)
