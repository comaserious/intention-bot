"""Intent definitions and connection settings."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

NONE_LABEL = "해당없음"
DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0


def load_intents(path: str | Path) -> list[Intent]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [Intent(i["name"], i["definition"]) for i in data["intent"]]


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    base_url = env.get("INTENT_BASE_URL")
    if not base_url:
        raise ValueError("환경변수 INTENT_BASE_URL 이 필요합니다 (예: http://host/v1)")
    return Settings(base_url, env.get("INTENT_API_KEY") or "EMPTY_KEY", env.get("INTENT_MODEL") or DEFAULT_MODEL,
                    float(env.get("INTENT_TIMEOUT") or 30.0))
