"""Local test UI. Usage: python -m intent_scorer.webapp [--port 8000]  (needs INTENT_BASE_URL)."""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Intent, load_intents, load_settings
from .scorer import IntentScorer, select_intents

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def score_request(payload: dict, client, model: str, default_intents: list[Intent]) -> dict:
    utterance = payload.get("utterance")
    if not isinstance(utterance, str):
        raise ValueError("utterance(문자열)가 필요합니다")
    try:
        threshold = float(payload.get("threshold", 0.2))
    except (TypeError, ValueError):
        raise ValueError("threshold 는 숫자여야 합니다") from None
    raw = payload.get("intents")
    intents = default_intents if not raw else [Intent(str(i["name"]), str(i["definition"])) for i in raw]
    rotations = payload.get("rotations")
    if rotations is not None and (not isinstance(rotations, int) or isinstance(rotations, bool)):
        raise ValueError("rotations 는 정수여야 합니다")
    scorer = IntentScorer(client, model, intents, rotations=rotations)
    t0 = time.perf_counter()
    scores = scorer.score(utterance)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"scores": scores, "selected": select_intents(scores, threshold),
            "latency_ms": round(latency_ms, 1), "calls": scorer.rotations if utterance.strip() else 0}


def make_handler(client, model: str, intents: list[Intent]):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/intents":
                self._json(200, {"model": model, "intents": [{"name": i.name, "definition": i.definition} for i in intents]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/score":
                return self._json(404, {"error": "not found"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._json(200, score_request(payload, client, model, intents))
            except (ValueError, KeyError, TypeError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # upstream model server failure
                self._json(502, {"error": f"모델 서버 오류: {type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--intents", default="config/intents.toml")
    a = ap.parse_args(argv)
    from openai import OpenAI

    s = load_settings()
    client = OpenAI(base_url=s.base_url, api_key=s.api_key, timeout=s.timeout, max_retries=0)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(client, s.model, load_intents(a.intents)))
    print(f"http://127.0.0.1:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
