"""Intent definitions and connection settings."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
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


def _get(env: Mapping[str, str], name: str) -> str | None:
    """INTENT_<name> 이 우선, 없으면 접두사 없는 <name> (.env 에서 쓰는 BASE_URL 등)."""
    return env.get(f"INTENT_{name}") or env.get(name)


def load_settings(env: Mapping[str, str] | None = None, dotenv_path: str | Path = DOTENV_PATH) -> Settings:
    """env 가 없으면 `.env` 파일 + 실제 환경변수를 합쳐 읽는다 (실제 환경변수가 우선)."""
    if env is None:
        env = {**dotenv_values(dotenv_path), **os.environ}
    base_url = _get(env, "BASE_URL")
    if not base_url:
        raise ValueError("INTENT_BASE_URL(또는 BASE_URL) 이 필요합니다 — .env 파일이나 환경변수로 설정하세요")
    return Settings(base_url, _get(env, "API_KEY") or "EMPTY_KEY", _get(env, "MODEL") or DEFAULT_MODEL,
                    float(env.get("INTENT_TIMEOUT") or 30.0))
