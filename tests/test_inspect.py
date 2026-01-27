from __future__ import annotations

from pathlib import Path

from ragopslab.inspect import list_sources, summarize_collection


def test_list_sources_filters(temp_collection: dict[str, Path | str]) -> None:
    sources = list_sources(
        persist_dir=Path(temp_collection["persist_dir"]),
        collection_name=str(temp_collection["collection_name"]),
        source_type="pdf",
    )
    assert len(sources) == 1
    assert sources[0].source_type == "pdf"


def test_summarize_collection_page_filter(temp_collection: dict[str, Path | str]) -> None:
    summary = summarize_collection(
        persist_dir=Path(temp_collection["persist_dir"]),
        collection_name=str(temp_collection["collection_name"]),
        limit=0,
        include_embeddings=True,
        page=1,
    )
    assert summary.count == 2
    assert len(summary.ids) == 1
    assert summary.metadatas[0]["page"] == 1
