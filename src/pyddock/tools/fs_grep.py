"""fs_grep tool script.

Search file contents by regex (always case-insensitive). Approximately:
fs_find() then re.search(grep_regex, line, re.IGNORECASE) per line of each
matched file.

Hidden dot-directories are pruned before descending (see fs_find); an
optional exclude_regex prunes/excludes additional entries the same way. A
file_glob whose final segment starts with '.' (e.g. '.env') explicitly
matches hidden files despite the pruning, mirroring shell glob semantics
where '*' does not match a leading dot but an explicit pattern does.

Binary files are skipped when scanning a directory path (sniffed from the
same read used for matching — no separate open). A single named file is
always searched, binary or not.
"""
import os
import re
import sys
from pathlib import Path

# --- Parameters ---
grep_regex = _PARAMS["grep_regex"]
file_glob = _PARAMS.get("file_glob") or "*"
path_str = _PARAMS.get("path") or "."
exclude_regex = _PARAMS.get("exclude_regex")
max_results = _PARAMS.get("max_results") or 100
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


# --- Main ---
if not grep_regex:
    print("grep_regex must be non-empty.", file=sys.stderr)
    sys.exit(1)

try:
    regex = re.compile(grep_regex, re.IGNORECASE)
except re.error as e:
    print(f"Invalid grep_regex '{grep_regex}': {e}", file=sys.stderr)
    sys.exit(1)

exclude_re = None
if exclude_regex:
    try:
        exclude_re = re.compile(exclude_regex)
    except re.error as e:
        print(f"Invalid exclude_regex '{exclude_regex}': {e}", file=sys.stderr)
        sys.exit(1)

# Matching strategy mirrors Path.rglob() (see fs_find.py's translate_glob).
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

matches: list[str] = []
skipped_hidden = 0
skipped_excluded = 0
skipped_binary = 0
skipped_unreadable = 0
truncated = False

if path.is_file():
    # A directly named file is always searched — bypasses hidden/exclude/glob
    # filtering and the binary sniff. Covers grepping a partially corrupted
    # log file, or a dotfile named explicitly.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        skipped_unreadable += 1
        text = None

    if text is not None:
        label = display_path(path)
        for line_num, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{label}:{line_num}: {truncate_line(line)}")
                if len(matches) >= max_results:
                    truncated = True
                    break
else:
    # Single pass: walk, filter, and match together so max_results can stop
    # the walk early instead of enumerating the entire (pruned) tree first.
    for dirpath, dirnames, filenames in os.walk(path):
        if truncated:
            break
        rel_dir = rel_posix(path, dirpath)

        # Prune hidden / excluded subdirectories BEFORE os.walk descends into
        # them — never walked, not just filtered from results.
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

            candidate = Path(dirpath) / f
            text, reason = read_text_or_skip_reason(candidate)
            if reason == "binary":
                skipped_binary += 1
                continue
            if reason == "unreadable":
                skipped_unreadable += 1
                continue

            label = display_path(candidate)
            for line_num, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{label}:{line_num}: {truncate_line(line)}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

notes = []
if truncated:
    notes.append(f"Showing first {max_results} matches. Narrow grep_regex or path to see more.")
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
