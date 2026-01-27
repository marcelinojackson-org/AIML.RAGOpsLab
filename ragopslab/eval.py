from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ragopslab.chat import answer_question


@dataclass
class EvalCase:
    question: str
    expected: str | list[str] | None = None


def _load_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval file must be a JSON array.")
    cases: list[EvalCase] = []
    for item in payload:
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError("Each eval item must include a 'question' field.")
        cases.append(EvalCase(question=str(item["question"]), expected=item.get("expected")))
    return cases


def _expectation_met(answer: str, expected: str | list[str] | None) -> bool:
    if expected is None:
        return False
    answer_lower = answer.lower()
    if isinstance(expected, list):
        return any(str(item).lower() in answer_lower for item in expected)
    return str(expected).lower() in answer_lower


def run_eval(
    eval_file: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chat_model: str,
    k: int,
    filters: dict[str, Any] | None = None,
    search_type: str = "similarity",
    mmr_fetch_k: int | None = None,
) -> dict[str, Any]:
    cases = _load_cases(eval_file)
    results: list[dict[str, Any]] = []
    passed = 0

    for case in cases:
        result = answer_question(
            query=case.question,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            chat_model=chat_model,
            k=k,
            filters=filters,
            search_type=search_type,
            mmr_fetch_k=mmr_fetch_k,
        )
        ok = _expectation_met(result.answer, case.expected)
        if ok:
            passed += 1
        results.append(
            {
                "question": case.question,
                "expected": case.expected,
                "answer": result.answer,
                "citations": result.citations,
                "pass": ok,
            }
        )

    summary = {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
    }
    return {"summary": summary, "results": results}
