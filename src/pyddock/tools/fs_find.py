"""fs_find tool script.

Find files by name. Approximately: [p for p in os.walk(path) if file_glob matches]

file_glob is a glob (e.g. '*.py', '**/test_*.py') matched against filenames
under path (default: workspace root, must be a directory). Alternatively,
file_regex provides a raw regex matched the same way (mutually exclusive
with file_glob): a pattern with no '/' matches basenames at any depth, a
pattern containing '/' matches the full relative path. Like grep_regex/
exclude_regex elsewhere in these tools, file_regex matches anywhere in the
name/path (re.search), not just the whole thing — anchor with ^/$ for an
exact match.

Hidden dot-directories (and their contents) are pruned before descending —
they are never walked, not just filtered from results — so this stays fast
on large ignored trees like .venv/ or node_modules/. An optional
exclude_regex prunes/excludes additional entries the same way. A file_glob
whose final segment starts with '.' (e.g. '.env') explicitly matches hidden
files despite the pruning, mirroring shell glob semantics where '*' does not
match a leading dot but an explicit pattern does. Returns matching paths
(relative to the workspace root when possible), one per line, capped at
max_results.
"""
import glob as _glob
import os
import re
import sys
from pathlib import Path

# --- Parameters ---
file_glob = _PARAMS.get("file_glob")
file_regex = _PARAMS.get("file_regex")
path_str = _PARAMS.get("path") or "."
exclude_regex = _PARAMS.get("exclude_regex")
max_results = _PARAMS.get("max_results") or 100
workspace_root = _PARAMS["workspace_root"]


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


# --- Input validation ---
if file_glob and file_regex:
    print("Provide file_glob or file_regex, not both.", file=sys.stderr)
    sys.exit(1)

# Default to '*' glob when neither is specified
if not file_glob and not file_regex:
    file_glob = "*"

# Build the compiled file-matching regex from whichever param was given.
if file_glob:
    # glob.translate() is the stdlib's canonical glob-to-regex translator (Python 3.13+).
    # include_hidden=True because pyddock handles hidden-file pruning separately in the
    # walk loop (via is_hidden()/allow_hidden_files). seps="/" because rel_posix() always
    # produces forward-slash paths regardless of OS.
    file_pattern_rx = re.compile(
        _glob.translate(file_glob, recursive=True, include_hidden=True, seps="/")
    )
    # A glob whose final segment starts with '.' explicitly targets hidden files.
    allow_hidden_files = file_glob.rsplit("/", 1)[-1].startswith(".")
    # A pattern with no '/' matches basenames only; otherwise match the full relative path.
    match_basename_only = "/" not in file_glob
    pattern_display = file_glob
else:
    try:
        file_pattern_rx = re.compile(file_regex)
    except re.error as e:
        print(f"Invalid file_regex '{file_regex}': {e}", file=sys.stderr)
        sys.exit(1)
    # Regex mode: no hidden-file allowance via pattern inspection — unlike a
    # glob's leading '.', there's no reliable syntactic way to tell whether an
    # arbitrary regex "means" to target a hidden file (use exclude_regex or
    # point path at a hidden dir directly instead). Same basename-vs-full-path
    # convention as file_glob: a pattern with no '/' matches basenames at any
    # depth; a pattern containing '/' matches the full relative path.
    allow_hidden_files = False
    match_basename_only = "/" not in file_regex
    pattern_display = file_regex

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

if not path.is_dir():
    print(f"'{path_str}' is not a directory. fs_find searches directories.", file=sys.stderr)
    sys.exit(1)

results: list[Path] = []
truncated = False
skipped_hidden = 0
skipped_excluded = 0

for dirpath, dirnames, filenames in os.walk(path):
    if truncated:
        break
    rel_dir = rel_posix(path, dirpath)

    # Prune hidden / excluded subdirectories BEFORE os.walk descends into
    # them — this is what keeps large ignored trees (.venv/, node_modules/)
    # from ever being scanned at all.
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
        if is_hidden(f) and not allow_hidden_files:
            skipped_hidden += 1
            continue
        rel_file = f"{rel_dir}/{f}" if rel_dir else f
        if exclude_re is not None and exclude_re.search(rel_file):
            skipped_excluded += 1
            continue
        target = f if match_basename_only else rel_file
        # search, not match/fullmatch: glob-derived patterns are anchored on
        # both ends by glob.translate() itself (its regex ends in \z), so
        # search() behaves identically to fullmatch() for the file_glob case.
        # For file_regex, search() keeps it consistent with grep_regex/
        # exclude_regex elsewhere in these tools (substring match by default;
        # the caller anchors with ^/$ if they want a whole-name match).
        if not file_pattern_rx.search(target):
            continue
        results.append(Path(dirpath) / f)
        if len(results) >= max_results:
            truncated = True
            break

notes = []
if truncated:
    pattern_param = "file_glob" if file_glob else "file_regex"
    notes.append(f"Showing first {max_results} results. Narrow {pattern_param} or path to see more.")
if skipped_hidden:
    notes.append(f"{skipped_hidden} hidden file(s)/dir(s) skipped.")
if skipped_excluded:
    notes.append(f"{skipped_excluded} excluded file(s)/dir(s) skipped.")

if not results:
    msg = f"No files matching '{pattern_display}' found under '{path_str}'."
    if notes:
        msg += " (" + " ".join(notes) + ")"
    print(msg)
else:
    output = "\n".join(display_path(p) for p in results)
    if notes:
        output += "\n\n[" + " ".join(notes) + "]"
    print(output)
