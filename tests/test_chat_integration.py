from __future__ import annotations

from pathlib import Path

import pytest

from ragopslab.chat import answer_question
from ragopslab.ingest import ingest_directory


@pytest.mark.ollama
def test_ingest_and_chat_integration(require_ollama: None, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = data_dir / "sample.txt"
    sample.write_text(
        "Marcelino Jackson is a DevSecOps Architect with cloud experience.",
        encoding="utf-8",
    )

    persist_dir = tmp_path / "chroma"
    ingest_directory(
        data_dir=data_dir,
        persist_dir=persist_dir,
        collection_name="test_collection",
        embedding_model="nomic-embed-text",
        chunk_size=200,
        chunk_overlap=20,
        extensions=["txt"],
        reset=True,
    )

    result = answer_question(
        query="Who is Marcelino Jackson?",
        persist_dir=persist_dir,
        collection_name="test_collection",
        embedding_model="nomic-embed-text",
        chat_model="llama3.1:8b",
        k=2,
    )
    assert result.answer
    assert "Marcelino" in result.answer
