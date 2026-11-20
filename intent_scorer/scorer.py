"""A-method scoring: label-choice prompt -> first-token logprobs -> softmax."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .config import NONE_LABEL, Intent

LABELS = "ABCDEFGHIJKLMNOPQRS"  # up to 18 intents + 해당없음 (top_logprobs=20)
_NONE_DEF = "위 어느 것에도 해당하지 않거나 의도가 불분명함"


def build_prompt(utterance: str, intents: Sequence[Intent]) -> str:
    if len(intents) + 1 > len(LABELS):
        raise ValueError(f"의도는 최대 {len(LABELS) - 1}개까지 지원합니다")
    opts = [f"{LABELS[i]}. {it.name}: {it.definition}" for i, it in enumerate(intents)]
    opts.append(f"{LABELS[len(intents)]}. {NONE_LABEL}: {_NONE_DEF}")
    return (
        "사용자 발화가 어떤 서비스를 원하는지 가장 가까운 것 하나를 고르세요.\n"
        + "\n".join(opts)
        + f'\n\n발화: "{utterance}"\n답은 알파벳 한 글자만.'
    )


def label_probs(top_logprobs: Mapping[str, float], n_options: int) -> list[float]:
    probs = [0.0] * n_options
    for tok, lp in top_logprobs.items():
        i = LABELS.find(tok.strip().upper()) if len(tok.strip()) == 1 else -1
        if 0 <= i < n_options:
            probs[i] += math.exp(lp)
    total = sum(probs)
    if total <= 0:
        return [0.0] * (n_options - 1) + [1.0]
    return [p / total for p in probs]
