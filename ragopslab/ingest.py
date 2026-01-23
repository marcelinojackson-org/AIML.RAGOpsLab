from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Iterable, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb


SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
}

# Silence noisy PDF parsing warnings from pypdf (e.g., xref offset issues).
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class IngestStats:
    files_seen: int
    files_loaded: int
    docs_loaded: int
    chunks_created: int
    skipped: int
    duplicates: int


def _gather_files(data_dir: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {ext.lower().lstrip(".") for ext in extensions}
    paths = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") in exts:
            paths.append(path)
    return sorted(paths)


def _load_file(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), autodetect_encoding=True)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_name", path.name)
        doc.metadata.setdefault("file_ext", suffix.lstrip("."))
    return docs


def _existing_sources(persist_dir: Path, collection_name: str) -> Set[str]:
    if not persist_dir.exists():
        return set()

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return set()

    total = collection.count()
    if total == 0:
        return set()

    sources: Set[str] = set()
    batch_size = 1000
    for offset in range(0, total, batch_size):
        result = collection.get(
            include=["metadatas"], limit=batch_size, offset=offset
        )
        for metadata in result.get("metadatas", []) or []:
            if not metadata:
                continue
            source = metadata.get("source")
            if source:
                sources.add(source)
    return sources


def ingest_directory(
    data_dir: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    extensions: Iterable[str],
    reset: bool = False,
) -> IngestStats:
    data_dir = data_dir.resolve()
    persist_dir = persist_dir.resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    paths = _gather_files(data_dir, extensions)
    if not paths:
        raise ValueError("No files found for the given extensions.")

    existing_sources = set() if reset else _existing_sources(persist_dir, collection_name)

    docs = []
    skipped = 0
    duplicates = 0
    for path in paths:
        if str(path) in existing_sources:
            print(f"Duplicate: {path}")
            duplicates += 1
            continue
        loaded = _load_file(path)
        if not loaded:
            skipped += 1
            continue
        docs.extend(loaded)

    if not docs:
        raise ValueError("No new documents loaded. Check duplicates or file types.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)

    return IngestStats(
        files_seen=len(paths),
        files_loaded=len(paths) - skipped,
        docs_loaded=len(docs),
        chunks_created=len(chunks),
        skipped=skipped,
        duplicates=duplicates,
    )
