"""A-method scoring: label-choice prompt -> first-token logprobs -> softmax."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
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


class IntentScorer:
    """Scores an utterance against intents; averages over cyclic option orders."""

    def __init__(self, client, model: str, intents: Sequence[Intent], rotations: int | None = None,
                 max_concurrency: int = 4):
        self.client, self.model, self.intents = client, model, list(intents)
        names = [it.name for it in self.intents]
        if any(not n.strip() or "\n" in n or n == NONE_LABEL for n in names) or len(set(names)) != len(names):
            raise ValueError(f"의도 이름은 비어 있지 않고, 줄바꿈 없이, 서로 달라야 하며 '{NONE_LABEL}'일 수 없습니다: {names}")
        self.rotations = len(self.intents) if rotations is None else rotations
        if not 1 <= self.rotations <= len(self.intents):
            raise ValueError(f"rotations 는 1..{len(self.intents)} 이어야 합니다 (받은 값: {self.rotations})")
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency)

    def _score_order(self, utterance: str, order: Sequence[Intent]) -> dict[str, float]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_prompt(utterance, order)}],
            max_tokens=1, logprobs=True, top_logprobs=20, temperature=0,
        )
        top = {t.token: t.logprob for t in resp.choices[0].logprobs.content[0].top_logprobs}
        probs = label_probs(top, len(order) + 1)
        return {**{it.name: p for it, p in zip(order, probs)}, NONE_LABEL: probs[-1]}

    def score(self, utterance: str) -> dict[str, float]:
        names = [it.name for it in self.intents] + [NONE_LABEL]
        if not utterance.strip():
            return {n: float(n == NONE_LABEL) for n in names}
        orders = [self.intents[i:] + self.intents[:i] for i in range(self.rotations)]
        results = list(self._pool.map(lambda o: self._score_order(utterance, o), orders))
        return {n: sum(r[n] for r in results) / len(results) for n in names}


def select_intents(scores: Mapping[str, float], threshold: float = 0.2) -> list[str]:
    if scores.get(NONE_LABEL, 0.0) >= max(scores.values()):
        return []
    picked = [(n, s) for n, s in scores.items() if n != NONE_LABEL and s >= threshold]
    return [n for n, _ in sorted(picked, key=lambda x: -x[1])]
