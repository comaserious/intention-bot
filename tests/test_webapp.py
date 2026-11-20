import pytest

from intent_scorer.config import load_intents
from intent_scorer.webapp import score_request
from tests.conftest import FakeClient

INTENTS = load_intents("config/intents.toml")


def fake():
    return FakeClient(lambda p: {"A": 0.0})


def test_score_request_returns_scores_selected_and_latency():
    out = score_request({"utterance": "안녕", "rotations": 4, "threshold": 0.2}, fake(), "m", INTENTS)
    assert set(out["scores"]) == {i.name for i in INTENTS} | {"해당없음"}
    assert out["selected"] == ["정착지원", "생활지원", "소상공인지원", "공장설립지원"]
    assert out["latency_ms"] >= 0 and out["calls"] == 4


def test_score_request_uses_custom_definitions():
    c = fake()
    score_request({"utterance": "x", "rotations": 1,
                   "intents": [{"name": "신규", "definition": "새 정의"}, {"name": "둘째", "definition": "d"}]}, c, "m", INTENTS)
    assert "A. 신규: 새 정의" in c.calls[0]


@pytest.mark.parametrize("bad", [{}, {"utterance": 3}, {"utterance": "x", "rotations": 99}, {"utterance": "x", "threshold": "a"}])
def test_score_request_rejects_bad_payload(bad):
    with pytest.raises(ValueError):
        score_request(bad, fake(), "m", INTENTS)
