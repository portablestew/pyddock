"""pyddock — policy-controlled Python execution via MCP."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyddock")
except PackageNotFoundError:
    # Running inside the sandbox subprocess where pyddock is on sys.path
    # but not installed as a package (no dist-info metadata).
    __version__ = "0.0.0"
