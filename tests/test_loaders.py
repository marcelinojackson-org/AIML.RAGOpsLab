from __future__ import annotations

import json
from pathlib import Path

from ragopslab.ingest import _load_file


def test_load_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name,role\nAlice,Engineer\nBob,Analyst\n", encoding="utf-8")

    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps([{"name": "Alice"}, {"name": "Bob"}]), encoding="utf-8"
    )

    csv_docs = _load_file(csv_path)
    json_docs = _load_file(json_path)

    assert len(csv_docs) == 2
    assert csv_docs[0].metadata["source_type"] == "csv"
    assert csv_docs[0].metadata["row_id"] == 1

    assert len(json_docs) == 2
    assert json_docs[0].metadata["source_type"] == "json"
    assert json_docs[0].metadata["record_id"] == 1
