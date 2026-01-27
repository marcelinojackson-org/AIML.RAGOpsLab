from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_list_and_sources(temp_config: Path) -> None:
    list_cmd = [
        sys.executable,
        "-m",
        "ragopslab",
        "list",
        "--config",
        str(temp_config),
        "--format",
        "csv",
        "--limit",
        "0",
    ]
    list_result = _run(list_cmd)
    assert list_result.returncode == 0
    assert "alpha.txt" in list_result.stdout

    sources_cmd = [
        sys.executable,
        "-m",
        "ragopslab",
        "sources",
        "--config",
        str(temp_config),
        "--format",
        "csv",
    ]
    sources_result = _run(sources_cmd)
    assert sources_result.returncode == 0
    assert "beta.pdf" in sources_result.stdout
