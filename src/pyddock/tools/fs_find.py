"""fs_find tool script.

Find files by name under a directory. Approximately:
    [p for p in os.walk(path) if file_glob matches]

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
import os
import re
import sys
from pathlib import Path

# --- Parameters ---
file_glob = _PARAMS["file_glob"]
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


def translate_glob(pattern: str) -> str:
    """Translate a glob pattern to an anchored regex.

    Supports '*' (any run of non-separator chars), '?' (one non-separator
    char), '[seq]'/'[!seq]' character classes, and '**' as a recursive
    any-depth-of-segments wildcard. Matched against a forward-slash path.
    """
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        c = pattern[i]
        i += 1
        if c == "*":
            if i < n and pattern[i] == "*":
                i += 1
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i
            if j < n and pattern[j] == "!":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape("["))
            else:
                stuff = pattern[i:j]
                if stuff.startswith("!"):
                    stuff = "^" + stuff[1:]
                out.append("[" + stuff + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
    return "^" + "".join(out) + "$"


def is_hidden(name: str) -> bool:
    """True if a single path component is dot-prefixed."""
    return name.startswith(".")


def rel_posix(base: Path, dirpath: str) -> str:
    """Relative path from base to dirpath, forward-slash, '' at base itself."""
    rel = os.path.relpath(dirpath, str(base))
    if rel == ".":
        return ""
    return rel.replace("\\", "/")


# --- Main ---
if not file_glob:
    print("file_glob must be non-empty.", file=sys.stderr)
    sys.exit(1)

exclude_re = None
if exclude_regex:
    try:
        exclude_re = re.compile(exclude_regex)
    except re.error as e:
        print(f"Invalid exclude_regex '{exclude_regex}': {e}", file=sys.stderr)
        sys.exit(1)

# Matching strategy mirrors Path.rglob(): a pattern with no '/' matches the
# filename at any depth; a pattern containing '/' matches the full relative
# path from `path` (e.g. '**/test_*.py').
match_basename_only = "/" not in file_glob
file_regex = re.compile(translate_glob(file_glob))

# A glob whose final segment starts with '.' explicitly targets hidden files
# (e.g. '.env', '**/.env') — mirrors shell semantics where '*' does not match
# a leading dot but an explicit pattern does. Hidden DIRECTORIES are still
# pruned regardless (performance); point `path` directly at one to search
# inside it.
allow_hidden_files = file_glob.rsplit("/", 1)[-1].startswith(".")

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
        if not file_regex.match(target):
            continue
        results.append(Path(dirpath) / f)
        if len(results) >= max_results:
            truncated = True
            break

notes = []
if truncated:
    notes.append(f"Showing first {max_results} results. Narrow file_glob or path to see more.")
if skipped_hidden:
    notes.append(f"{skipped_hidden} hidden file(s)/dir(s) skipped.")
if skipped_excluded:
    notes.append(f"{skipped_excluded} excluded file(s)/dir(s) skipped.")

if not results:
    msg = f"No files matching '{file_glob}' found under '{path_str}'."
    if notes:
        msg += " (" + " ".join(notes) + ")"
    print(msg)
else:
    output = "\n".join(display_path(p) for p in results)
    if notes:
        output += "\n\n[" + " ".join(notes) + "]"
    print(output)
