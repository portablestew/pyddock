"""Script tool registry for pyddock.

Generic mechanism for loading Python script assets and executing them
through the sandbox as MCP tools. Scripts are .py files bundled with
pyddock (package data in src/pyddock/tools/). Each script receives
structured parameters via a `_PARAMS` dict preamble and produces
plaintext output.

Execution goes through SubprocessExecutor, so full sandbox enforcement
(filesystem scoping, import hook, protected dirs) applies automatically.
"""

from __future__ import annotations

import asyncio
import logging
from importlib import resources
from pathlib import Path

from pyddock.config import PyddockConfig
from pyddock.executor import RunPythonOutput, SubprocessExecutor

logger = logging.getLogger(__name__)


class ScriptToolRegistry:
    """Loads, caches, and executes Python script assets as MCP tools.

    This pattern works for any tool that:
    1. Accepts structured params from an MCP call
    2. Does computation or I/O that should respect sandbox policy
    3. Returns plaintext results

    The file tools (read_file, stat_file, fs_append, fs_delete, str_replace)
    are the initial set. Future tools can be added by dropping a .py script
    into src/pyddock/tools/ and registering an MCP handler.

    Args:
        config: The loaded pyddock configuration.
        executor: The subprocess executor for running scripts.
        workspace: The workspace root directory.
    """

    def __init__(
        self,
        config: PyddockConfig,
        executor: SubprocessExecutor,
        workspace: Path,
    ) -> None:
        self._config = config
        self._executor = executor
        self._workspace = workspace
        self._scripts: dict[str, str] = {}

    def load_scripts(self, directory: str = "tools") -> dict[str, str]:
        """Load all .py scripts from a package subdirectory.

        Reads script files from the pyddock package data (src/pyddock/tools/)
        and caches their source text in memory. Call once at startup.

        Args:
            directory: Subdirectory name within the pyddock package.

        Returns:
            Mapping of script_name (without .py) → script source text.
        """
        pkg_files = resources.files("pyddock") / directory
        scripts: dict[str, str] = {}

        # Iterate over the package directory contents
        for item in pkg_files.iterdir():
            if item.name.endswith(".py") and item.name != "__init__.py":
                script_name = item.name[:-3]  # strip .py
                scripts[script_name] = item.read_text(encoding="utf-8")
                logger.debug("Loaded tool script: %s", script_name)

        self._scripts = scripts
        logger.info("Loaded %d tool scripts from %s/", len(scripts), directory)
        return scripts

    def build_execution_source(self, script_name: str, params: dict) -> str:
        """Combine a cached script with serialized parameters.

        Produces a single Python source string:
            _PARAMS = { ... serialized params ... }
            <script body>

        The script reads _PARAMS to get its inputs.

        Args:
            script_name: Name of the script (without .py extension).
            params: Dict of parameters to pass to the script.

        Returns:
            Combined source string ready for execution.

        Raises:
            KeyError: If script_name is not in the cache.
        """
        if script_name not in self._scripts:
            available = ", ".join(sorted(self._scripts.keys()))
            raise KeyError(
                f"Unknown tool script '{script_name}'. "
                f"Available scripts: {available}"
            )

        script_source = self._scripts[script_name]
        preamble = f"_PARAMS = {params!r}\n"
        return preamble + script_source

    async def execute(
        self,
        script_name: str,
        params: dict,
        timeout: float | None = None,
    ) -> str:
        """Execute a script tool and return its plaintext output.

        Dispatches to SubprocessExecutor (via asyncio.to_thread) with the
        combined source. Extracts stdout/result from RunPythonOutput.

        Args:
            script_name: Name of the script (without .py extension).
            params: Dict of parameters to pass to the script.
            timeout: Execution timeout in seconds. Defaults to config timeout.

        Returns:
            Plaintext result string. On success: result (if captured) or stdout.
            On non-zero exit: stderr content as error text.
        """
        # Always inject workspace_root so scripts can resolve paths
        params_with_workspace = {
            **params,
            "workspace_root": str(self._workspace),
        }

        source = self.build_execution_source(script_name, params_with_workspace)

        effective_timeout = (
            timeout if timeout is not None else self._config.execution.timeout
        )

        output: RunPythonOutput = await asyncio.to_thread(
            self._executor.execute,
            source,
            [],  # no args
            effective_timeout,
            self._workspace,
        )

        return self._format_output(output)

    @staticmethod
    def _format_output(output: RunPythonOutput) -> str:
        """Extract the meaningful response from a RunPythonOutput.

        Priority:
        1. If exit_code != 0: return stderr (error message)
        2. If result is captured (last expression): return result
        3. Otherwise: return stdout
        """
        if output.exit_code != 0:
            # Error case — return stderr for agent self-correction
            if output.stderr:
                return output.stderr
            # Fallback: if stderr is empty but exit code is non-zero
            return f"Tool script exited with code {output.exit_code}"

        # Success case — prefer result (last expression), fall back to stdout
        if output.result is not None:
            return output.result
        return output.stdout
