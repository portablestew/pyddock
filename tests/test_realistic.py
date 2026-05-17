"""Realistic integration tests — snippets an agent would actually write.

These verify that useful, multi-step scripts work end-to-end with the
default config and runtime enforcement active.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyddock.config import load_config
from pyddock.executor import SubprocessExecutor
from pyddock.venv_manager import VenvManager


@pytest.fixture
def config():
    """Load the real default config (not a minimal test config)."""
    return load_config()


@pytest.fixture
def venv_manager(tmp_path: Path):
    manager = VenvManager(venv_path=tmp_path / "venv", allowed_imports=[])
    manager.get_python_path = lambda: Path(sys.executable)
    return manager


@pytest.fixture
def executor(config, venv_manager):
    return SubprocessExecutor(config, venv_manager)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestRealisticSnippets:
    """Snippets that simulate real agent tasks."""

    def test_parse_log_file_and_extract_errors(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Agent parses a log file, extracts ERROR lines, writes a summary."""
        # Set up a log file
        log_content = "\n".join([
            "2024-01-15 10:00:01 INFO  Server started",
            "2024-01-15 10:00:05 ERROR Connection refused to database",
            "2024-01-15 10:00:06 INFO  Retrying...",
            "2024-01-15 10:00:10 ERROR Timeout waiting for response",
            "2024-01-15 10:00:15 INFO  Connection established",
            "2024-01-15 10:01:00 ERROR Disk space low on /var/log",
        ])
        (workspace / "app.log").write_text(log_content)

        source = """
import pathlib
import json

log = pathlib.Path('app.log').read_text()
errors = [line for line in log.splitlines() if 'ERROR' in line]

summary = {
    'total_lines': len(log.splitlines()),
    'error_count': len(errors),
    'errors': errors,
}

pathlib.Path('error_summary.json').write_text(json.dumps(summary, indent=2))
print(f"Found {len(errors)} errors")
summary
"""
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "Found 3 errors" in result.stdout
        assert "'error_count': 3" in result.result

        # Verify the output file was written
        summary_file = workspace / "error_summary.json"
        assert summary_file.exists()
        import json
        summary = json.loads(summary_file.read_text())
        assert summary["error_count"] == 3

    def test_csv_processing_and_filtering(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Agent reads a CSV, filters rows, computes stats, writes output."""
        csv_content = "name,age,department\nAlice,30,Engineering\nBob,25,Marketing\nCharlie,35,Engineering\nDiana,28,Marketing\nEve,32,Engineering\n"
        (workspace / "employees.csv").write_text(csv_content)

        source = """
import csv
import pathlib
import json

rows = list(csv.DictReader(pathlib.Path('employees.csv').open()))

# Filter to Engineering department
engineers = [r for r in rows if r['department'] == 'Engineering']

# Compute average age
avg_age = sum(int(r['age']) for r in engineers) / len(engineers)

result = {
    'department': 'Engineering',
    'count': len(engineers),
    'average_age': round(avg_age, 1),
    'names': [r['name'] for r in engineers],
}

pathlib.Path('engineering_report.json').write_text(json.dumps(result, indent=2))
result
"""
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "'count': 3" in result.result
        assert "'average_age': 32.3" in result.result
        assert (workspace / "engineering_report.json").exists()

    def test_find_files_by_pattern(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Agent scans a directory tree for files matching a pattern."""
        # Set up a directory structure
        (workspace / "src").mkdir()
        (workspace / "src" / "main.py").write_text("# main")
        (workspace / "src" / "utils.py").write_text("# utils")
        (workspace / "src" / "tests").mkdir()
        (workspace / "src" / "tests" / "test_main.py").write_text("# test")
        (workspace / "docs").mkdir()
        (workspace / "docs" / "readme.md").write_text("# readme")

        source = """
import pathlib

py_files = sorted(str(p.relative_to('.')) for p in pathlib.Path('.').rglob('*.py'))
md_files = sorted(str(p.relative_to('.')) for p in pathlib.Path('.').rglob('*.md'))

print(f"Python files: {len(py_files)}")
print(f"Markdown files: {len(md_files)}")
{'python': py_files, 'markdown': md_files}
"""
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "Python files: 3" in result.stdout
        assert "Markdown files: 1" in result.stdout

    def test_text_transformation_with_regex(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Agent uses regex to transform text content."""
        content = "The API endpoint is http://api.example.com/v1/users and the docs are at http://docs.example.com/api"
        (workspace / "input.txt").write_text(content)

        source = """
import pathlib
import re

text = pathlib.Path('input.txt').read_text()

# Extract all URLs
urls = re.findall(r'https?://[\\w./]+', text)

# Replace URLs with markdown links
def url_to_link(match):
    url = match.group(0)
    return f'[{url}]({url})'

transformed = re.sub(r'https?://[\\w./]+', url_to_link, text)
pathlib.Path('output.txt').write_text(transformed)

{'urls_found': urls, 'transformed_length': len(transformed)}
"""
        result = executor.execute(source, [], 10, workspace)

        assert result.exit_code == 0
        assert "http://api.example.com/v1/users" in result.result
        output = (workspace / "output.txt").read_text()
        assert "[http://api.example.com/v1/users]" in output

    def test_json_data_aggregation_with_args(
        self, executor: SubprocessExecutor, workspace: Path
    ) -> None:
        """Agent processes multiple JSON files specified via args."""
        import json

        (workspace / "data1.json").write_text(json.dumps({"values": [1, 2, 3]}))
        (workspace / "data2.json").write_text(json.dumps({"values": [4, 5, 6]}))
        (workspace / "data3.json").write_text(json.dumps({"values": [7, 8, 9]}))

        source = """
import json
import pathlib
import sys

all_values = []
for filename in sys.argv[1:]:
    data = json.loads(pathlib.Path(filename).read_text())
    all_values.extend(data['values'])

stats = {
    'total': sum(all_values),
    'count': len(all_values),
    'mean': sum(all_values) / len(all_values),
    'min': min(all_values),
    'max': max(all_values),
}
stats
"""
        result = executor.execute(
            source,
            ["data1.json", "data2.json", "data3.json"],
            10,
            workspace,
        )

        assert result.exit_code == 0
        assert "'total': 45" in result.result
        assert "'count': 9" in result.result
        assert "'mean': 5.0" in result.result
