# pyddock

*py + paddock — an enclosed space where code can run freely within bounds. ([Worked great in Jurassic Park.](https://en.wikipedia.org/wiki/Jurassic_Park))*

MCP server that gives AI agents pre-approved `run_python`, `run_shell`, and file i/o tools. Safely replaces shell scripting for data processing, file analysis, computation, and external command execution — no user confirmation needed.

## Why

Agents write shell snippets for structured tasks. Those snippets require user approval (workflow friction) or blind trust (security risk). Common alternatives each have limitations:

- **Trust specific command strings.** Agents produce variations or chain commands into ad-hoc scripts that break naive string matching, causing repeated approval prompts for equivalent operations.
- **Trust all shell execution.** Unsafe when the environment has access to production resources.
- **Full containerized sandbox (Docker).** Isolation cuts agents off from the dev workspace, host network, and local tooling. Restoring access requires per-project configuration that is brittle to maintain.
- **Larger agent framework with granular permissions.** High adoption cost for what should be a drop-in tool.

pyddock provides Python execution and shell command access with declarative policy controls. Policies are safe to auto-approve because side effects are scoped to the configured workspace.

**Intended use:** Trusted shell replacement for unattended agents in development environments. The full permission policy is reported in the MCP tool description, so agents know what is allowed before writing code. When a snippet does violate a policy, the error includes the specific rule and a recommended workaround. Agents adapt non-interactively rather than blocking on human approval.

**Replaces dedicated MCP servers:** A full Python environment means agents can call service APIs directly (boto3, p4python, etc.) under the same policy enforcement. This eliminates the need for separate MCP servers for AWS, Atlassian, Perforce, and similar services, and allows agents to compose multiple API calls in a single `run_python` invocation.

**Workspace scripts:** Scripts in a configured directory (e.g. `.kiro/scripts/`) are automatically trusted. Scripts authored for workspace agents require no additional pyddock configuration.

**Threat model:** Pyddock's guardrails keep *trustworthy local agents* out of trouble. A determined adversary may discover a jailbreak, or chain with other tools in the workspace for escalation and/or exfiltration. The Python sandbox contains extensive anti-jailbreak protections, but only as a first-line of defense against accidents and casual misuse. Pyddock does not claim to be a security boundary against targeted exploitation or public abuse. In other words, life finds a way.

## Setup

```json
{
  "mcpServers": {
    "pyddock": {
      "command": "uv.exe",
      "args": ["--directory", "<path-to-pyddock-source>", "run", "pyddock", "serve", "--workspace", "<path-to-workspace>"],
      "autoApprove": ["run_python","run_shell","fs_read","fs_stat","fs_append","fs_delete","fs_str_replace"]
    }
  }
}
```

## Tool: `run_python`

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | string | Inline Python snippet |
| `file` | string | Path to a .py file (mutually exclusive with `code`) |
| `args` | string[] | Available as `sys.argv[1:]` |
| `timeout` | number | Seconds (default: 30) |

The last expression in your snippet is captured as RESULT (like a Jupyter cell) when it evaluates to a non-None value. Use `print()` for streaming output or a trailing expression for structured return values.

## Tool: `run_shell`

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | string | Executable name or script path (must match a shell policy) |
| `args` | string[] | Arguments passed directly to the command (no shell interpretation) |
| `timeout` | number | Seconds (default: 30) |

Commands are executed directly — no shell interpretation, pipes, redirects, chaining, or variable expansion. Only commands matching a `[shell.*]` policy in your config are permitted. The tool is only registered when at least one `[shell.*]` section exists in the config.

Script files get automatic interpreter mapping: `.ps1` → powershell, `.py` → python, `.sh` → bash, `.bat` → cmd /c.

## Output format

Both `run_python` and `run_shell` return human-readable text sections:

```
--- RESULT ---
42
--- STDOUT ---
Processing 10 files...
Done.
--- STDERR ---
Warning: deprecated API
--- EXIT CODE: 0 ---
```

- Sections are omitted when empty (except EXIT CODE which is always present)
- RESULT only appears in `run_python` responses, when the last expression is non-None
- `run_shell` responses have STDOUT, STDERR, and EXIT CODE only

## File Tools

Pyddock also exposes file I/O tools that execute through the same sandbox as `run_python` — same filesystem scoping, same protected directories, same policy enforcement. These are convenience shortcuts for common file operations; agents can't do anything they couldn't already do via `run_python` with `pathlib`. On failure, error messages include diagnostic context to guide the agent's next attempt.

- **`fs_read`** — Read a text file. Lines are 1-indexed; negative start = tail. Truncates large output with a continuation hint.
- **`fs_stat`** — File metadata (exists, type, size, line count, modified). Returns `exists: false` for missing paths.
- **`fs_append`** — Append to a file (creates if missing). Returns a unified diff.
- **`fs_delete`** — Delete a file or empty directory. Returns a unified diff of removed content.
- **`fs_str_replace`** — Find-and-replace exact text (unique match required). Returns a unified diff, or match diagnostics on failure.

## Security model

`run_python` enforcement:

- **AST validation** (pre-execution) — rejects disallowed imports, blocked calls (`eval`, `exec`, `compile`), and blocked attribute access (`__globals__`, `__subclasses__`, etc.)
- **Runtime enforcement** (in-subprocess) — import hook, filesystem scoping, `getattr`/`attrgetter` guards, factory proxies for library method restrictions, safe `sys`/`os`/`subprocess` module proxies

`run_shell` and subprocess enforcement:

- **Default deny** — any command not matching a `[shell.*]` config section is rejected outright
- **No shell interpretation** — commands execute via `subprocess.run(shell=False)`, preventing injection, pipes, redirects, and variable expansion
- **Argument validation** — per-command allow/deny regex patterns control which arguments are permitted
- **Argument path scanning** — args that look like filesystem paths are validated against protected directories and workspace boundaries (configurable via `arg_paths`)
- **Write protection** — path-like shell command regexes automatically generate write-deny rules for `run_python`, preventing write-then-execute privilege escalation
- **subprocess proxy** — `run_python` code has access to `subprocess.run()` and `subprocess.Popen()`, both validated against the same shell policies; `shell=True` and string commands are always rejected; `os.system()` is blocked

Writes are restricted to the workspace. The `.pyddock/` directory, pyddock source, and the Python stdlib directory are always write-protected (`.pyddock/tmp/` is the exception, used by tempfile). Reads are unrestricted by default.

## Configuration

Configuration is resolved in two steps:

1. **Base config:** `.pyddock/pyddock.toml` in your workspace (full replacement of the bundled default), or `default_config.toml` if no workspace config exists.
2. **Overlay (optional):** `.pyddock/pyddock.override.toml` is deep-merged on top of the base. Only include sections/keys you want to change — everything else is inherited.

See `default_config.toml` for the full annotated reference. Key sections:

```toml
[execution]
timeout = 30
max_timeout = 3600

[imports]
json = true
csv = true
pathlib = true
# ... (see default_config.toml for full list)
sys = true
os = true
subprocess = true
dateutil = "python-dateutil"
polars = true
boto3 = true

[filesystem]
writable_paths = ["."]   # "." = workspace directory; "*" = unrestricted
readable_paths = ["*"]

[ast]
block_calls = ["eval", "exec", "compile", "breakpoint", "__import__"]
block_attributes = ["__subclasses__", "__globals__", "__code__", "__bases__", "__mro__", "__closure__"]

[restrictions.polars]
mode = "allow"
module_deny = ["write_.*", "sink_.*"]
class_deny = ["write_.*", "sink_.*"]

[restrictions.boto3]
mode = "deny"
module_allow = ["client"]
class_allow = ["list_.*", "describe_.*", "get_.*", "head_.*", "scan", "query", "start_query", "stop_query"]

[shell.p4]
mode = "deny"
allow = ["filelog.*", "files.*", "describe.*", "changes.*", "print.*", "where.*", "info"]

[shell.git]
mode = "deny"
allow = ["status.*", "log.*", "diff.*", "show.*", "branch.*", "rev-parse.*"]

[shell.kiro-scripts]
command = "\\.kiro/scripts/.*"
mode = "allow"
deny = []
arg_paths = "protected"
```

### Import values

Modules are configured in `[imports]` by setting the import name (what you'd write after `import`) to a value:

```toml
[imports]
json = true
pathlib = true
dateutil = "python-dateutil"                          # pip name differs from import name
invoice_parser = ".kiro/scripts/invoice-parser"       # import invoice_parser
metrics_client = ".kiro/scripts/reporting/metrics-client"  # import metrics_client
```

- `true` — allowed module (import name == pip package name)
- `false` — revoked module
- `"<pip-name>"` — allowed module where the PyPI package name differs from the import name (e.g. `dateutil = "python-dateutil"`)
- `"<path>"` — workspace package installed via `pip install -e` at boot (string containing `/` or `\`, or starting with `.`)

Each workspace package must contain a `pyproject.toml`. Dependencies declared in `pyproject.toml` are resolved automatically during installation — transitive deps are available to the package at runtime (including lazy/deferred imports), but not directly importable by agent code unless explicitly listed.

Source code changes in workspace packages are picked up immediately (editable install). Metadata changes (e.g. adding new dependencies in `pyproject.toml`) require a server restart.

Workspace module directories are automatically write-protected to prevent code modification from within `run_python`.

### Configuration override

A minimal `.pyddock/pyddock.override.toml` example — only include what you want to change:

```toml
# .pyddock/pyddock.override.toml — only include what you want to change
[imports]
requests = true    # add a module
boto3 = false      # revoke a module

[execution]
timeout = 120      # override default 30s

[shell.npm]
command = "^npm$"
mode = "deny"
allow = ["run build.*", "test.*"]
```

Merge semantics:

- **Scalars and lists** (timeout, writable_paths, block_calls): overlay value replaces base value.
- **Tables** (imports, restrictions, shell): merge by key — new keys are added, existing keys are overwritten, unmentioned keys are inherited.

## Restrictions

Two modes for per-library restrictions:

- **deny** (with `module_allow` and `class_allow`): blocks all module-level access except functions matching `module_allow` patterns. Wraps allowed callables with a proxy so objects they return are restricted to methods matching `class_allow` patterns. Use for client-producing libraries (boto3, botocore).
- **allow** (with `module_deny` and `class_deny`): blocks specific module-level functions matching `module_deny` patterns and methods on classes matching `class_deny` patterns. Use for libraries where most operations are safe (polars, pandas).

## Shell Policies

Each `[shell.<name>]` section defines a command execution policy:

- **command**: regex pattern matched against the command string (defaults to `^<name>$` if omitted)
- **mode**: `"deny"` (block all args except those matching `allow`) or `"allow"` (permit all args except those matching `deny`)
- **allow**: list of regex patterns for permitted argument strings (used with deny mode)
- **deny**: list of regex patterns for blocked argument strings (used with allow mode)
- **arg_paths**: controls path scanning for arguments that look like filesystem paths:
  - `"workspace"` (default) — blocks args resolving outside the workspace or into protected dirs (`.pyddock/`, workspace modules, script dirs)
  - `"protected"` — only blocks args targeting protected dirs; allows paths outside the workspace
  - `"none"` — no path scanning (fully trusts the command)

Args are space-joined before matching: `["filelog", "//depot/..."]` becomes `"filelog //depot/..."`. All regexes use `re.match()` (implicitly anchored at start).

When `subprocess` is enabled in `[imports]`, `run_python` code can call `subprocess.run([...])` with the same policy enforcement and interpreter mapping. This is useful for composing multiple commands, processing output between calls, or passing environment variables dynamically.

## CLI

```sh
pyddock serve --workspace /path/to/project   # MCP server (stdio)
pyddock run "2 + 2"                          # Direct execution
pyddock run script.py -- arg1 arg2           # File execution with args
```

## Development

```sh
uv sync
uv run pytest
```
