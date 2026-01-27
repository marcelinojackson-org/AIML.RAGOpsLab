from __future__ import annotations

import json
from pathlib import Path

from ragopslab.chat import ChatResult
from ragopslab.eval import run_eval


def test_run_eval_with_mock(monkeypatch: object, tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            [
                {"question": "Who is the author?", "expected": "Alice"},
                {"question": "What is the role?", "expected": ["Engineer", "Analyst"]},
            ]
        ),
        encoding="utf-8",
    )

    def fake_answer(*_: object, **__: object) -> ChatResult:
        return ChatResult(answer="Alice is an Engineer.", citations=[])

    monkeypatch.setattr("ragopslab.eval.answer_question", fake_answer)

    result = run_eval(
        eval_file=eval_file,
        persist_dir=tmp_path / "persist",
        collection_name="test",
        embedding_model="nomic-embed-text",
        chat_model="llama3.1:8b",
        k=2,
    )
    assert result["summary"]["passed"] == 2
