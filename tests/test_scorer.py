import math

import pytest

from intent_scorer.config import Intent
from intent_scorer.scorer import build_prompt, label_probs

TWO = [Intent("정착지원", "정착 정의"), Intent("생활지원", "생활 정의")]


def test_build_prompt_lists_options_and_none_last():
    p = build_prompt("안녕", TWO)
    assert p.startswith("사용자 발화가 어떤 서비스를 원하는지 가장 가까운 것 하나를 고르세요.")
    order = [p.index(s) for s in ("A. 정착지원: ", "B. 생활지원: ", "C. 해당없음: 위 어느 것에도 해당하지 않거나 의도가 불분명함")]
    assert order == sorted(order)
    assert '발화: "안녕"' in p
    assert p.endswith("답은 알파벳 한 글자만.")


def test_build_prompt_keeps_quotes_and_newlines_inside_utterance():
    utt = '안녕 "하세요"\n다음줄'
    p = build_prompt(utt, TWO)
    assert utt in p
    assert p.splitlines()[-1] == "답은 알파벳 한 글자만."


def test_label_probs_normalizes():
    r = label_probs({"A": math.log(0.6), "B": math.log(0.2)}, 3)
    assert r == pytest.approx([0.75, 0.25, 0.0])


def test_label_probs_accepts_token_variants():
    r = label_probs({" a": math.log(0.5), "B ": math.log(0.5)}, 3)
    assert r == pytest.approx([0.5, 0.5, 0.0])


def test_label_probs_no_label_falls_back_to_none():
    assert label_probs({"x": -1.0}, 3) == [0.0, 0.0, 1.0]


def test_build_prompt_rejects_too_many_intents():
    with pytest.raises(ValueError):
        build_prompt("x", [Intent(f"i{k}", "d") for k in range(19)])


# ---- Task 3: IntentScorer ----
import re

import pytest as _pytest

from intent_scorer.config import NONE_LABEL, load_intents
from intent_scorer.scorer import IntentScorer, select_intents
from tests.conftest import FakeClient

INTENTS = load_intents("config/intents.toml")
NAMES = [i.name for i in INTENTS]


def first_option_name(prompt: str) -> str:
    return re.search(r"^A\. (.+?): ", prompt, re.M).group(1)


def always_a(prompt):
    return {"A": 0.0}


def make(client, **kw):
    return IntentScorer(client, "m", INTENTS, **kw)


def test_score_returns_distribution_keys_and_sums_to_one():
    s = make(FakeClient(always_a)).score("발화")
    assert list(s) == NAMES + [NONE_LABEL]
    assert sum(s.values()) == _pytest.approx(1.0)


def test_score_averages_by_intent_name_across_rotations():
    s = make(FakeClient(always_a), rotations=4).score("발화")
    assert [s[n] for n in NAMES] == _pytest.approx([0.25] * 4)
    assert s[NONE_LABEL] == 0.0


def test_score_uses_required_call_parameters():
    c = FakeClient(always_a)
    make(c, rotations=1).score("x")
    kw = c.kwargs[0]
    assert (kw["model"], kw["temperature"], kw["max_tokens"], kw["logprobs"], kw["top_logprobs"]) == ("m", 0, 1, True, 20)


def test_rotations_controls_call_count():
    c = FakeClient(always_a)
    make(c, rotations=1).score("x")
    assert len(c.calls) == 1
    c2 = FakeClient(always_a)
    make(c2).score("x")
    assert len(c2.calls) == 4


@_pytest.mark.parametrize("utt", ["", "  \n"])
def test_empty_or_whitespace_utterance_skips_model(utt):
    c = FakeClient(always_a)
    s = make(c).score(utt)
    assert s[NONE_LABEL] == 1.0 and c.calls == []


@_pytest.mark.parametrize("r", [0, 5])
def test_invalid_rotations_raise_value_error(r):
    with _pytest.raises(ValueError):
        make(FakeClient(always_a), rotations=r)


def test_client_error_propagates():
    def boom(prompt):
        raise RuntimeError("server down")

    with _pytest.raises(RuntimeError, match="server down"):
        make(FakeClient(boom)).score("x")


def test_concurrency_is_limited():
    c = FakeClient(always_a, delay=0.02)
    sc = make(c, max_concurrency=2)
    sc.score("x")
    assert c.max_in_flight <= 2


def test_select_intents():
    scores = {"정착지원": 0.5, "생활지원": 0.25, "소상공인지원": 0.05, "공장설립지원": 0.0, NONE_LABEL: 0.2}
    assert select_intents(scores) == ["정착지원", "생활지원"]
    assert select_intents({"정착지원": 0.3, "생활지원": 0.1, NONE_LABEL: 0.6}) == []
