import pytest

from intent_scorer.config import load_intents, load_settings


def test_load_intents_returns_four_in_file_order():
    intents = load_intents("config/intents.toml")
    assert [i.name for i in intents] == ["정착지원", "생활지원", "소상공인지원", "공장설립지원"]
    assert all(i.definition.strip() for i in intents)


def test_load_settings_reads_env():
    s = load_settings({"INTENT_BASE_URL": "http://x/v1", "INTENT_API_KEY": "k"})
    assert (s.base_url, s.api_key, s.model) == ("http://x/v1", "k", "google/gemma-4-26B-A4B-it")
    s = load_settings({"INTENT_BASE_URL": "http://x/v1", "INTENT_MODEL": "m"})
    assert (s.model, s.api_key) == ("m", "EMPTY_KEY")


def test_load_settings_missing_base_url_raises():
    with pytest.raises(ValueError, match="INTENT_BASE_URL"):
        load_settings({})
