from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass
class CollectionSummary:
    count: int
    ids: list[str]
    metadatas: list[dict[str, Any]]
    documents: list[str]


def summarize_collection(
    persist_dir: Path, collection_name: str, limit: int
) -> CollectionSummary:
    persist_dir = persist_dir.resolve()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist directory not found: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()

    if count == 0:
        return CollectionSummary(count=0, ids=[], metadatas=[], documents=[])

    if limit == 0:
        sample = collection.get(include=["metadatas", "documents"])
    else:
        sample = collection.peek(limit=limit)
    return CollectionSummary(
        count=count,
        ids=sample.get("ids", []),
        metadatas=sample.get("metadatas", []),
        documents=sample.get("documents", []),
    )
