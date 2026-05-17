"""Tool scripts for pyddock.

This package contains Python script assets that are executed via the
ScriptToolRegistry through the sandbox. Each .py file is a standalone
tool script that reads parameters from a `_PARAMS` dict and produces
plaintext output.

These files are NOT imported as modules — they are read as text and
executed via SubprocessExecutor.
"""
