"""pyddock — policy-controlled Python execution via MCP."""

__version__ = "0.1.1"

# Filename used for compile() when executing agent snippets.
# Used by executor (compile source), runtime (stack inspection to identify
# agent code vs trusted library code), and tests.
SNIPPET_FILENAME = "<snippet>"
