# intention_bot — 발화 의도 점수 래퍼

발화와 의도 정의를 받아 `{의도: 점수, "해당없음": 점수}`(합 1)를 반환합니다. 로컬 vLLM(OpenAI 호환)의 첫 토큰 logprob을 사용합니다.

## 준비
```powershell
conda activate intention-bot
pip install openai pytest
$env:INTENT_BASE_URL = "https://your-host.example/sllm/v1"   # 필수
$env:INTENT_API_KEY = "EMPTY_KEY"                                   # 선택
$env:INTENT_MODEL = "google/gemma-4-26B-A4B-it"                     # 선택
$env:INTENT_TIMEOUT = "30"                                          # 선택, 초 (기본 30)
```

## 사용
```python
from openai import OpenAI
from intent_scorer import IntentScorer, load_intents, load_settings, select_intents

s = load_settings()
scorer = IntentScorer(OpenAI(base_url=s.base_url, api_key=s.api_key,
                             timeout=s.timeout, max_retries=0), s.model,
                      load_intents("config/intents.toml"), rotations=4, max_concurrency=4)
scores = scorer.score("경북으로 이사 오는 청년도 월세 지원을 받을 수 있나요?")
select_intents(scores, threshold=0.2)   # 복합 질의면 여러 개, 해당없음이면 []
```
- `rotations`: 의도 순서 회전 횟수(1~의도 수). 클수록 정확, 호출 수 증가. `max_concurrency`: 동시 호출 상한.
- 의도 정의는 `config/intents.toml`에서 수정합니다.

## 평가 (정확도 + 소요 시간)
```powershell
python -m intent_scorer.evaluate --rotations 1 4
```
`eval/dataset.toml`(24 케이스) 기준 top1, set_match, none 정확도와 평균/중앙값/p95/최대 시간(초)을 표로 출력합니다. 테스트: `python -m pytest`.

설계: `docs/superpowers/specs/`, 계획: `docs/superpowers/plans/`. `spikes/`는 일회용 실험 코드입니다.
