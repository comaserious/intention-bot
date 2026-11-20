"""Accuracy and latency evaluation. Usage: python -m intent_scorer.evaluate --rotations 1 4"""
from __future__ import annotations

import argparse
import math
import statistics
import time
import tomllib
from dataclasses import dataclass
from typing import Mapping

from .config import NONE_LABEL, load_intents, load_settings
from .scorer import IntentScorer, select_intents


@dataclass(frozen=True)
class Case:
    utterance: str
    expected: frozenset[str]


def load_cases(path) -> list[Case]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [Case(c["utterance"], frozenset(c["expected"])) for c in data["case"]]


def run_eval(scorer, cases: list[Case], threshold: float = 0.2) -> dict:
    scorer.score("워밍업")  # warm-up, excluded from timing
    top1, sets, nones, times = [], [], [], []
    for c in cases:
        t0 = time.perf_counter()
        scores = scorer.score(c.utterance)
        times.append(time.perf_counter() - t0)
        picked = select_intents(scores, threshold)
        sets.append(set(picked) == c.expected)
        if c.expected:
            best = max((n for n in scores if n != NONE_LABEL), key=scores.get)
            top1.append(best in c.expected)
        else:
            nones.append(not picked)
    ts = sorted(times)
    rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "n": len(cases),
        "top1_accuracy": rate(top1),
        "set_match": rate(sets),
        "none_accuracy": rate(nones),
        "latency": {
            "mean": statistics.mean(ts),
            "median": statistics.median(ts),
            "p95": ts[math.ceil(0.95 * len(ts)) - 1],
            "max": ts[-1],
        },
    }


def format_report(results: Mapping[str, dict]) -> str:
    head = f"{'method':<10}{'n':>4}{'top1':>8}{'set_match':>11}{'none':>7}{'mean(s)':>9}{'median(s)':>11}{'p95(s)':>8}{'max(s)':>8}"
    rows = [head]
    for name, r in results.items():
        L = r["latency"]
        rows.append(f"{name:<10}{r['n']:>4}{r['top1_accuracy']:>8.2f}{r['set_match']:>11.2f}{r['none_accuracy']:>7.2f}"
                    f"{L['mean']:>9.3f}{L['median']:>11.3f}{L['p95']:>8.3f}{L['max']:>8.3f}")
    return "\n".join(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotations", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--dataset", default="eval/dataset.toml")
    ap.add_argument("--intents", default="config/intents.toml")
    ap.add_argument("--threshold", type=float, default=0.2)
    a = ap.parse_args(argv)
    from openai import OpenAI

    s = load_settings()
    client = OpenAI(base_url=s.base_url, api_key=s.api_key)
    intents, cases = load_intents(a.intents), load_cases(a.dataset)
    results = {f"rot={k}": run_eval(IntentScorer(client, s.model, intents, rotations=k), cases, a.threshold)
               for k in a.rotations}
    print(format_report(results))


if __name__ == "__main__":
    main()
