from __future__ import annotations

from pathlib import Path

from ragopslab.config import load_config


def test_load_config_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    assert config["paths"]["data_dir"]
    assert config["retrieval"]["search_type"] == "similarity"
    assert "filters" in config["retrieval"]
