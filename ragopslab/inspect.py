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
    embeddings: list[list[float]] | None


@dataclass
class SourceSummary:
    source_type: str
    file_name: str
    source: str
    count: int


def summarize_collection(
    persist_dir: Path,
    collection_name: str,
    limit: int,
    include_embeddings: bool = False,
    page: int | None = None,
) -> CollectionSummary:
    persist_dir = persist_dir.resolve()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist directory not found: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()

    if count == 0:
        return CollectionSummary(
            count=0, ids=[], metadatas=[], documents=[], embeddings=None
        )

    include = ["metadatas", "documents"]
    if include_embeddings:
        include.append("embeddings")

    if limit == 0 or include_embeddings or page is not None:
        sample = collection.get(include=include)
    else:
        sample = collection.peek(limit=limit)

    ids = sample.get("ids", []) or []
    metadatas = sample.get("metadatas", []) or []
    documents = sample.get("documents", []) or []
    embeddings = sample.get("embeddings") if include_embeddings else None

    if page is not None:
        filtered_ids: list[str] = []
        filtered_metadatas: list[dict[str, Any]] = []
        filtered_documents: list[str] = []
        filtered_embeddings: list[list[float]] = []

        for idx, metadata in enumerate(metadatas):
            if not metadata:
                continue
            if metadata.get("page") != page:
                continue
            filtered_ids.append(ids[idx])
            filtered_metadatas.append(metadatas[idx])
            filtered_documents.append(documents[idx])
            if include_embeddings and embeddings is not None:
                filtered_embeddings.append(embeddings[idx])

        ids = filtered_ids
        metadatas = filtered_metadatas
        documents = filtered_documents
        embeddings = filtered_embeddings if include_embeddings else None

    if limit != 0 and len(ids) > limit:
        ids = ids[:limit]
        metadatas = metadatas[:limit]
        documents = documents[:limit]
        if include_embeddings and embeddings is not None:
            embeddings = embeddings[:limit]
    return CollectionSummary(
        count=count,
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        embeddings=embeddings,
    )


def list_sources(
    persist_dir: Path,
    collection_name: str,
    source_type: str | None = None,
    file_name: str | None = None,
) -> list[SourceSummary]:
    persist_dir = persist_dir.resolve()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist directory not found: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()
    if count == 0:
        return []

    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", []) or []

    tally: dict[tuple[str, str, str], int] = {}
    for metadata in metadatas:
        if not metadata:
            continue
        stype = str(metadata.get("source_type", "unknown"))
        fname = str(metadata.get("file_name", "unknown"))
        src = str(metadata.get("source", "unknown"))
        if source_type and stype != source_type:
            continue
        if file_name and fname != file_name:
            continue
        key = (stype, fname, src)
        tally[key] = tally.get(key, 0) + 1

    summaries = [
        SourceSummary(source_type=k[0], file_name=k[1], source=k[2], count=v)
        for k, v in tally.items()
    ]
    summaries.sort(key=lambda s: (s.source_type, s.file_name, s.source))
    return summaries
