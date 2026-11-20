import time

import pytest

from intent_scorer.config import NONE_LABEL
from intent_scorer.evaluate import Case, format_report, load_cases, run_eval


class StubScorer:
    def __init__(self, table, delays=None):
        self.table, self.delays, self.calls = table, delays or {}, []

    def score(self, utt):
        self.calls.append(utt)
        time.sleep(self.delays.get(utt, 0))
        return self.table.get(utt, {"정착지원": 0.0, NONE_LABEL: 1.0})


def S(**kw):
    return {"정착지원": 0.0, "생활지원": 0.0, "소상공인지원": 0.0, "공장설립지원": 0.0, NONE_LABEL: 0.0, **kw}


def test_load_cases_reads_dataset():
    cases = load_cases("eval/dataset.toml")
    assert len(cases) == 24
    assert cases[0].expected == frozenset({"정착지원"})
    assert sum(1 for c in cases if not c.expected) == 4


def test_run_eval_metrics_on_known_scores():
    table = {"a": S(정착지원=0.9, 해당없음=0.1), "b": S(정착지원=0.5, 생활지원=0.4, 해당없음=0.1), "c": S(정착지원=0.8, 해당없음=0.2)}
    cases = [Case("a", frozenset({"정착지원"})), Case("b", frozenset({"정착지원", "생활지원"})), Case("c", frozenset({"공장설립지원"}))]
    r = run_eval(StubScorer(table), cases)
    assert r["top1_accuracy"] == pytest.approx(2 / 3)
    assert r["set_match"] == pytest.approx(2 / 3)
    assert r["n"] == 3


def test_run_eval_none_accuracy():
    table = {"x": S(해당없음=0.9, 정착지원=0.1), "y": S(정착지원=0.7, 해당없음=0.3)}
    r = run_eval(StubScorer(table), [Case("x", frozenset()), Case("y", frozenset())])
    assert r["none_accuracy"] == 0.5


def test_latency_stats():
    cases = [Case(u, frozenset()) for u in "abc"]
    r = run_eval(StubScorer({}, {"a": 0.01, "b": 0.02, "c": 0.03}), cases)
    lat = r["latency"]
    assert lat["max"] >= 0.03 and lat["p95"] == lat["max"]
    assert 0.015 < lat["median"] < 0.05


def test_run_eval_excludes_warmup():
    sc = StubScorer({}, {"워밍업": 0.2})
    r = run_eval(sc, [Case("a", frozenset())])
    assert sc.calls == ["워밍업", "a"]
    assert r["latency"]["max"] < 0.1


def test_format_report_contains_all_metric_labels():
    res = {"rot=1": {"top1_accuracy": 1.0, "set_match": 0.5, "none_accuracy": 1.0, "n": 2,
                     "latency": {"mean": .1, "median": .1, "p95": .2, "max": .2}}}
    out = format_report(res)
    for w in ("top1", "set_match", "none", "mean", "median", "p95", "max", "rot=1"):
        assert w in out
