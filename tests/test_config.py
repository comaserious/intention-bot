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


def test_load_settings_timeout_defaults_and_overrides():
    base = {"INTENT_BASE_URL": "http://x/v1"}
    assert load_settings(base).timeout == 30.0
    assert load_settings({**base, "INTENT_TIMEOUT": "5"}).timeout == 5.0


def test_load_settings_accepts_unprefixed_names_and_prefers_intent_prefix():
    s = load_settings({"BASE_URL": "http://x/v1", "API_KEY": "k", "MODEL": "m"})
    assert (s.base_url, s.api_key, s.model) == ("http://x/v1", "k", "m")
    s = load_settings({"BASE_URL": "http://plain/v1", "INTENT_BASE_URL": "http://intent/v1"})
    assert s.base_url == "http://intent/v1"


def test_load_settings_reads_dotenv_file_but_real_env_wins(tmp_path, monkeypatch):
    for k in ("INTENT_BASE_URL", "BASE_URL", "INTENT_API_KEY", "API_KEY", "INTENT_MODEL", "MODEL"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / ".env"
    f.write_text("BASE_URL=http://file/v1\nAPI_KEY=filekey\nMODEL=filemodel\n", encoding="utf-8")
    s = load_settings(dotenv_path=f)
    assert (s.base_url, s.api_key, s.model) == ("http://file/v1", "filekey", "filemodel")
    monkeypatch.setenv("BASE_URL", "http://real/v1")
    assert load_settings(dotenv_path=f).base_url == "http://real/v1"
