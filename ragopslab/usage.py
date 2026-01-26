from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageSummary:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float | None = None


def _heuristic_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_usage_from_metadata(metadata: dict[str, Any] | None) -> dict[str, int]:
    if not metadata:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt = metadata.get("prompt_eval_count")
    completion = metadata.get("eval_count")
    if isinstance(prompt, int) and isinstance(completion, int):
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def estimate_usage(
    prompt_text: str,
    completion_text: str,
) -> dict[str, int]:
    prompt_tokens = _heuristic_tokens(prompt_text)
    completion_tokens = _heuristic_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_cost(
    model: str,
    usage: dict[str, int],
    pricing: dict[str, Any],
    default_prompt_per_1k: float,
    default_completion_per_1k: float,
) -> float:
    model_prices = pricing.get(model, {})
    prompt_per_1k = float(model_prices.get("prompt_per_1k", default_prompt_per_1k))
    completion_per_1k = float(model_prices.get("completion_per_1k", default_completion_per_1k))

    return (
        usage.get("prompt_tokens", 0) / 1000.0 * prompt_per_1k
        + usage.get("completion_tokens", 0) / 1000.0 * completion_per_1k
    )


def build_usage_summary(
    *,
    response_metadata: dict[str, Any] | None,
    estimator: str,
    prompt_text: str,
    completion_text: str,
    model: str,
    pricing: dict[str, Any],
    default_prompt_per_1k: float,
    default_completion_per_1k: float,
    enabled: bool,
) -> UsageSummary | None:
    if not enabled:
        return None

    if estimator == "ollama":
        usage = extract_usage_from_metadata(response_metadata)
        if usage["total_tokens"] == 0:
            usage = estimate_usage(prompt_text, completion_text)
    else:
        usage = estimate_usage(prompt_text, completion_text)

    cost = estimate_cost(
        model=model,
        usage=usage,
        pricing=pricing,
        default_prompt_per_1k=default_prompt_per_1k,
        default_completion_per_1k=default_completion_per_1k,
    )

    return UsageSummary(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        estimated_cost=cost,
    )
