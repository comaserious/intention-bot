# intention_bot — 발화 의도 점수 래퍼

발화와 의도 정의를 받아 `{의도: 점수, "해당없음": 점수}`(합 1)를 반환합니다. 로컬 vLLM(OpenAI 호환)의 첫 토큰 logprob을 사용합니다.

## 준비
```powershell
conda activate intention-bot
pip install openai pytest python-dotenv
Copy-Item .env.example .env     # 복사 후 .env 에 실제 주소를 입력 (.env 는 git 제외)
```
`.env` 항목: `BASE_URL`(필수), `API_KEY`(선택, 기본 `EMPTY_KEY`), `MODEL`(선택, 기본 `google/gemma-4-26B-A4B-it`).
`INTENT_BASE_URL` 처럼 `INTENT_` 접두사를 붙인 이름이나 실제 환경변수를 쓰면 `.env` 보다 우선합니다. `INTENT_TIMEOUT`(초, 기본 30)은 환경변수로만 지정합니다.

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

## 테스트 화면
```powershell
$env:INTENT_BASE_URL = "<서버 주소>/v1"
python -m intent_scorer.webapp --port 8000   # http://127.0.0.1:8000
```
발화를 넣어 점수 막대와 소요 시간을 보고, 순서 회전 횟수·기준값·의도 정의(화면에서만 적용)를 바꿔볼 수 있습니다.
