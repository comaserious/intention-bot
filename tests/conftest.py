import threading
import time
from types import SimpleNamespace


class FakeClient:
    """Mimics openai client.chat.completions.create for first-token logprobs."""

    def __init__(self, responder, delay: float = 0.0):
        self.responder = responder  # prompt -> {token: logprob}
        self.delay = delay
        self.calls: list[str] = []
        self.kwargs: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        prompt = kw["messages"][0]["content"]
        with self._lock:
            self.calls.append(prompt)
            self.kwargs.append(kw)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            top = [SimpleNamespace(token=t, logprob=lp) for t, lp in self.responder(prompt).items()]
        finally:
            with self._lock:
                self.in_flight -= 1
        content = [SimpleNamespace(top_logprobs=top)]
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])
