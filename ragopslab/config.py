from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data/sample_docs",
        "persist_dir": "storage/chroma",
    },
    "chroma": {
        "collection": "ragopslab",
    },
    "models": {
        "embedding_model": "nomic-embed-text",
    },
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "files": {
        "extensions": ["txt", "md", "pdf"],
    },
    "list": {
        "limit": 5,
        "format": "table",
        "preview_width": 80,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = Path("config.yaml")

    if not path.exists():
        return dict(DEFAULTS)

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping at the top level.")

    return _deep_merge(DEFAULTS, data)

