from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
import pytest

from ragopslab.config import load_config


def ollama_enabled() -> bool:
    return os.getenv("OLLAMA_TESTS", "1").lower() not in {"0", "false", "no"}


def fetch_ollama_models() -> set[str]:
    with urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
        payload = json.load(resp)
    return {m.get("name", "") for m in payload.get("models", [])}


@pytest.fixture(scope="session")
def ollama_models() -> set[str]:
    if not ollama_enabled():
        return set()
    try:
        return fetch_ollama_models()
    except Exception as exc:  # pragma: no cover - network error path
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. "
            "Start Ollama or set OLLAMA_TESTS=0 to run mocked tests."
        ) from exc


@pytest.fixture(scope="session")
def require_ollama(ollama_models: set[str]) -> None:
    if not ollama_enabled():
        pytest.skip("OLLAMA_TESTS=0 (mocked mode).")
    config = load_config(Path("config.yaml"))
    required = {
        config["models"]["embedding_model"],
        config["models"]["chat_model"],
    }
    normalized = {name.split(":")[0] for name in ollama_models if name}
    missing = [m for m in required if m not in ollama_models and m not in normalized]
    if missing:
        raise RuntimeError(
            "Missing Ollama models: "
            f"{', '.join(missing)}. Run 'ollama pull <model>' or set OLLAMA_TESTS=0."
        )


@pytest.fixture()
def temp_collection(tmp_path: Path) -> dict[str, Path | str]:
    persist_dir = tmp_path / "chroma"
    collection_name = "test_collection"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        ids=["doc-1", "doc-2"],
        documents=["alpha content", "beta content"],
        embeddings=[[0.1, 0.2], [0.2, 0.3]],
        metadatas=[
            {
                "source": str(tmp_path / "alpha.txt"),
                "file_name": "alpha.txt",
                "file_ext": "txt",
                "source_type": "txt",
                "page": 0,
            },
            {
                "source": str(tmp_path / "beta.pdf"),
                "file_name": "beta.pdf",
                "file_ext": "pdf",
                "source_type": "pdf",
                "page": 1,
            },
        ],
    )
    return {"persist_dir": persist_dir, "collection_name": collection_name}


@pytest.fixture()
def temp_config(tmp_path: Path, temp_collection: dict[str, Path | str]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  persist_dir: {temp_collection['persist_dir']}",
                "chroma:",
                f"  collection: {temp_collection['collection_name']}",
                "models:",
                "  embedding_model: nomic-embed-text",
                "  chat_model: llama3.1:8b",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
