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
