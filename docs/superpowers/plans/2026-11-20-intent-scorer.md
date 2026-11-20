# 의도 점수 래퍼(A4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자 발화와 의도 정의 목록을 받아 `{의도: 점수, "해당없음": 점수}`(합 1)를 돌려주는 래퍼와, 정확도·소요 시간 평가 스크립트를 만든다.

**Architecture:** 라벨(A/B/C…) 선택지 프롬프트를 `max_tokens=1`, logprobs로 호출해 첫 토큰 확률을 softmax 정규화한다. 의도 순서를 순환시켜 `rotations`번 호출하고 의도 이름 기준으로 평균낸다. 호출은 인스턴스 공유 스레드풀(`max_concurrency`)로 제한한다. 의도 정의는 TOML 설정, 서버 주소·키는 환경변수.

**Tech Stack:** Python 3.14 (conda `intention-bot`), `openai` SDK(동기), 표준 `tomllib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-20-intent-scorer-design.md`

## Global Constraints

- 모델 `google/gemma-4-26B-A4B-it`, OpenAI 호환 API, 임베딩 엔드포인트 없음
- 호출 파라미터: `temperature=0`, `max_tokens=1`, `logprobs=True`, `top_logprobs=20`
- 엔드포인트 주소·API 키 하드코딩 금지 (환경변수 `INTENT_BASE_URL`, `INTENT_API_KEY`, 선택 `INTENT_MODEL`)
- 의도 4개: 정착지원, 생활지원, 소상공인지원, 공장설립지원 + 라벨 `해당없음` (AI 민원실 제외)
- 의도 정의는 코드 밖 설정 파일에서 수정 가능
- 후보에 없는 라벨 토큰의 확률은 0, 점수는 합 1로 정규화
- 평가 항목에 소요 시간(평균, 중앙값, p95, 최대) 필수, 워밍업 호출 제외
- 로컬 모델 보호: 동시 호출 수는 `max_concurrency`(기본 4)로 제한
- 점수 기반 분기 처리(추가 질문, 담당기관 안내)는 범위 밖
- 실행 명령은 `conda run -n intention-bot ...`; 기존 `spike_ab*.py`는 일회용이라 래퍼에 이어 쓰지 않는다

## Review Focus

1. 빈 문자열/공백 발화 → 모델 호출 없이 `해당없음=1.0` 반환 (Task 3)
2. 라벨 토큰 변형(`" A"`, 소문자 `"a"`)이 top_logprobs에 오는 경우 → 앞뒤 공백 제거·대문자화 후 같은 라벨로 취급 (Task 2)
3. top_logprobs에 라벨이 하나도 없는 경우 → 0으로 나누지 않고 `해당없음=1.0` (Task 2)
4. 발화에 따옴표·줄바꿈이 있어도 프롬프트 구조가 깨지지 않음 (Task 2)
5. 서버 오류/타임아웃 → 조용히 점수를 만들어내지 말고 예외를 그대로 전파, `rotations`가 1 미만이거나 의도 수 초과면 `ValueError` (Task 3)

---

## File Structure

```
pyproject.toml            pytest 설정(pythonpath)
.gitignore
config/intents.toml       의도 정의 4개(RFP 기반)
eval/dataset.toml         평가 케이스(주신 16개 + 추가 8개)
intent_scorer/__init__.py 공개 API export
intent_scorer/config.py   Intent, load_intents, Settings, load_settings
intent_scorer/scorer.py   build_prompt, label_probs, IntentScorer, select_intents
intent_scorer/evaluate.py load_cases, run_eval, format_report, CLI
tests/conftest.py         FakeClient
tests/test_config.py  tests/test_scorer.py  tests/test_evaluate.py
spikes/                   기존 spike_ab*.py 이동(보관용)
```

### Task 1: 프로젝트 골격과 설정 로딩

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config/intents.toml`, `intent_scorer/__init__.py`, `intent_scorer/config.py`, `tests/test_config.py`
- Move: `spike_ab.py`, `spike_ab2.py`, `spike_ab3.py` → `spikes/`

**Interfaces:**
- Produces: `NONE_LABEL: str = "해당없음"`; `@dataclass(frozen=True) Intent(name: str, definition: str)`; `load_intents(path: str | Path) -> list[Intent]`; `@dataclass(frozen=True) Settings(base_url: str, api_key: str, model: str)`; `load_settings(env: Mapping[str, str] | None = None) -> Settings` (env 기본 `os.environ`)

- [ ] **Step 1: 환경 준비 및 git 초기화.** `conda run -n intention-bot pip install pytest openai` 실행, `git init`, 스파이크 파일을 `spikes/`로 이동. `.gitignore`에 `__pycache__/`, `.pytest_cache/`. `pyproject.toml`에 `[tool.pytest.ini_options] pythonpath = ["."]`, `testpaths = ["tests"]`.
- [ ] **Step 2: Write the failing tests** (`tests/test_config.py`)
  - `test_load_intents_returns_four_in_file_order`: `load_intents("config/intents.toml")`의 이름이 `["정착지원","생활지원","소상공인지원","공장설립지원"]`이고 모든 definition이 비어 있지 않음
  - `test_load_settings_reads_env`: `load_settings({"INTENT_BASE_URL":"http://x/v1","INTENT_API_KEY":"k"})` → `model == "google/gemma-4-26B-A4B-it"`; `INTENT_MODEL` 지정 시 그 값
  - `test_load_settings_missing_base_url_raises`: `INTENT_BASE_URL` 없으면 `KeyError` 대신 메시지에 변수명이 들어간 `ValueError`
- [ ] **Step 3: Run to verify fail**: `conda run -n intention-bot python -m pytest tests/test_config.py -v` → FAIL (모듈 없음)
- [ ] **Step 4: Implement** `config/intents.toml`(`[[intent]]` 배열, `name`/`definition`; 정의 문구는 설계 문서 5절 4개를 그대로), `intent_scorer/config.py`의 위 인터페이스(`tomllib` 사용, `INTENT_API_KEY` 없으면 `"EMPTY_KEY"`). `intent_scorer/__init__.py`는 일단 빈 파일.
- [ ] **Step 5: Run to verify pass**: 같은 명령 → 3 passed
- [ ] **Step 6: Commit**: `git add -A && git commit -m "feat: project scaffold and config loading"`

### Task 2: 프롬프트와 라벨 확률 계산

**Files:**
- Create: `intent_scorer/scorer.py`, `tests/test_scorer.py`

**Interfaces:**
- Consumes: `Intent`, `NONE_LABEL` (Task 1)
- Produces: `LABELS = "ABCDEFGHIJKLMNOPQRS"` (최대 19개, 의도 18 + 해당없음); `build_prompt(utterance: str, intents: Sequence[Intent]) -> str`; `label_probs(top_logprobs: Mapping[str, float], n_options: int) -> list[float]` (길이 `n_options`, 합 1, 마지막 원소가 해당없음)

- [ ] **Step 1: Write the failing tests** (`tests/test_scorer.py`)
  - `test_build_prompt_lists_options_and_none_last`: 의도 2개일 때 프롬프트에 `"A. 정착지원: "`, `"B. 생활지원: "`, `"C. 해당없음: 위 어느 것에도 해당하지 않거나 의도가 불분명함"`이 이 순서로 있고, `발화: "<utterance>"`와 `답은 알파벳 한 글자만.`으로 끝남. 첫 줄은 `사용자 발화가 어떤 서비스를 원하는지 가장 가까운 것 하나를 고르세요.`
  - `test_build_prompt_keeps_quotes_and_newlines_inside_utterance`: 발화 `'안녕 "하세요"\n다음줄'`이 그대로 포함되고 `답은 알파벳 한 글자만.`이 여전히 마지막 줄
  - `test_label_probs_normalizes`: `{"A": log(0.6), "B": log(0.2)}`, n=3 → `[0.75, 0.25, 0.0]` (근사)
  - `test_label_probs_accepts_token_variants`: `{" a": log(0.5), "B ": log(0.5)}` → A,B가 각각 0.5
  - `test_label_probs_no_label_falls_back_to_none`: `{"x": -1.0}`, n=3 → `[0.0, 0.0, 1.0]`
  - `test_build_prompt_rejects_too_many_intents`: 19개 의도 → `ValueError`
- [ ] **Step 2: Run to verify fail**: `... pytest tests/test_scorer.py -v` → FAIL
- [ ] **Step 3: Implement** 위 두 함수. 프롬프트는 스파이크(`spikes/spike_ab2.py`의 `score_a`) 문구와 동일하게. `label_probs`는 토큰을 `strip().upper()`로 정규화해 같은 라벨은 확률을 합산하고, 합이 0이면 해당없음에 1.0.
- [ ] **Step 4: Run to verify pass** → 6 passed
- [ ] **Step 5: Commit**: `git add -A && git commit -m "feat: prompt building and label probability normalization"`

### Task 3: IntentScorer (순서 회전, 병렬 제한, 선택 헬퍼)

**Files:**
- Modify: `intent_scorer/scorer.py`, `intent_scorer/__init__.py`, `tests/test_scorer.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `build_prompt`, `label_probs` (Task 2), `Intent`, `NONE_LABEL`
- Produces: `class IntentScorer(client, model: str, intents: Sequence[Intent], rotations: int | None = None, max_concurrency: int = 4)`; `IntentScorer.score(utterance: str) -> dict[str, float]` (키: 의도 이름들 + `해당없음`, 합 1); `select_intents(scores: Mapping[str, float], threshold: float = 0.2) -> list[str]`; `tests/conftest.py`의 `FakeClient(responder)` — `responder(prompt: str) -> dict[str, float]`(토큰→logprob)를 받아 OpenAI 응답 형태(`choices[0].logprobs.content[0].top_logprobs[*].token/.logprob`)를 흉내내고, `.calls`(프롬프트 목록)와 `.max_in_flight`를 기록
- `rotations=None`이면 의도 수. 순서 i는 `intents[i:] + intents[:i]` (i = 0..rotations-1).

- [ ] **Step 1: Write the failing tests**
  - `test_score_returns_distribution_keys_and_sums_to_one`: 항상 라벨 A에 몰아주는 responder → 키가 `[의도 4개 + 해당없음]`, 합 1
  - `test_score_averages_by_intent_name_across_rotations`: responder가 프롬프트 첫 옵션(라벨 A에 해당하는 의도 이름)에 확률 1을 주면, `rotations=4`에서 4개 의도가 각각 0.25
  - `test_rotations_controls_call_count`: `rotations=1` → `len(client.calls)==1`, 기본 → 4
  - `test_empty_or_whitespace_utterance_skips_model`: `""`, `"  \n"` → `해당없음 == 1.0`, `client.calls == []`
  - `test_invalid_rotations_raise_value_error`: `0`, `5`(의도 4개) → `ValueError`
  - `test_client_error_propagates`: responder가 `RuntimeError` → `score()`도 `RuntimeError` (삼키지 않음)
  - `test_concurrency_is_limited`: `max_concurrency=2`, 호출이 짧게 sleep → `client.max_in_flight <= 2`
  - `test_select_intents`: `{"정착지원":0.5,"생활지원":0.25,"소상공인지원":0.05,"공장설립지원":0.0,"해당없음":0.2}` → `["정착지원","생활지원"]`; 해당없음이 최고점이면 `[]`
- [ ] **Step 2: Run to verify fail** → FAIL
- [ ] **Step 3: Implement** `IntentScorer`: 생성자에서 `ThreadPoolExecutor(max_workers=max_concurrency)`를 한 번 만들어 모든 `score()` 호출이 공유. `score()`는 순서별 호출을 풀에 제출하고 결과를 의도 이름 기준으로 평균(`NONE_LABEL` 포함). 호출은 Global Constraints의 파라미터로 `client.chat.completions.create`. `select_intents`는 해당없음이 1위면 `[]`, 아니면 `threshold` 이상인 의도 이름을 점수 내림차순으로. `__init__.py`에서 `Intent, load_intents, load_settings, IntentScorer, select_intents, NONE_LABEL` export.
- [ ] **Step 4: Run to verify pass**: `... pytest -v` → 전부 통과
- [ ] **Step 5: Commit**: `git add -A && git commit -m "feat: IntentScorer with rotation averaging and concurrency limit"`

### Task 4: 평가 스크립트와 실서버 확인

**Files:**
- Create: `eval/dataset.toml`, `intent_scorer/evaluate.py`, `tests/test_evaluate.py`, `README.md`

**Interfaces:**
- Consumes: `IntentScorer`, `select_intents`, `load_intents`, `load_settings`
- Produces: `@dataclass Case(utterance: str, expected: frozenset[str])`; `load_cases(path) -> list[Case]`; `run_eval(scorer, cases, threshold: float = 0.2) -> dict` — 키: `top1_accuracy`(expected가 비어있지 않은 케이스 중 1위 의도가 expected에 속하는 비율), `set_match`(`select_intents` 결과 집합 == expected인 비율, 해당없음 케이스 포함), `none_accuracy`(expected가 빈 케이스 중 `select_intents==[]` 비율), `latency`(`mean`, `median`, `p95`, `max`: 초; p95는 정렬 후 `ceil(0.95*n)-1` 인덱스), `n`; `format_report(results: Mapping[str, dict]) -> str`(방식 이름→결과로 비교표); CLI: `python -m intent_scorer.evaluate [--rotations 1 4] [--dataset eval/dataset.toml] [--threshold 0.2]`가 rotations별 결과를 한 표로 출력

- [ ] **Step 1: Write the failing tests** (`tests/test_evaluate.py`, FakeClient 사용)
  - `test_load_cases_reads_dataset`: `eval/dataset.toml` → 24개, 첫 케이스가 정착지원 1개, 해당없음 케이스 expected가 빈 집합
  - `test_run_eval_metrics_on_known_scores`: 스코어러를 고정 점수 스텁으로 대체한 3개 케이스(정답 1, 복합 일치 1, 오답 1)에서 `top1_accuracy`, `set_match`가 손계산 값과 같음
  - `test_run_eval_none_accuracy`: expected 빈 케이스 2개 중 1개만 해당없음 1위 → `0.5`
  - `test_latency_stats`: 스코어러 스텁이 0.01초·0.02초·0.03초 sleep → `max >= 0.03`, `median`이 가운데 값 근처, `p95 == max`(n=3)
  - `test_run_eval_excludes_warmup`: 첫 호출을 시간 측정에서 제외하기 위해 평가 전에 `scorer.score("워밍업")`을 1회 호출함을 확인(`client.calls`의 첫 프롬프트에 `워밍업` 포함)
  - `test_format_report_contains_all_metric_labels`: 출력 문자열에 `top1`, `set_match`, `none`, `mean`, `median`, `p95`, `max`와 방식 이름이 들어 있음
- [ ] **Step 2: Run to verify fail** → FAIL
- [ ] **Step 3: Implement** `eval/dataset.toml`: `[[case]]` + `utterance`, `expected = ["정착지원"]` (해당없음은 `[]`). 주신 16개 발화(정착 4·생활 4·소상공인 4·공장 4)와 설계 근거의 추가 8개(복합 4: 정착+생활 2, 정착+소상공인 1, 소상공인+공장 1 / 모호·무관 4: "지원금 받을 수 있나요?", "안녕하세요", "오늘 점심 뭐 먹지", "주민등록등본은 어떻게 발급받나요?")는 `spikes/spike_ab2.py`의 `GIVEN`/`EXTRA`에서 옮긴다. `evaluate.py`는 위 인터페이스대로. CLI는 `load_settings()`로 `OpenAI(base_url=..., api_key=...)`를 만들고 rotations마다 `IntentScorer`를 새로 만들어 평가. 케이스는 순차 실행(요청당 end-to-end 시간 측정). `README.md`에 환경 준비, 환경변수, 사용 예(`IntentScorer(...).score("...")`), 평가 실행법을 짧게.
- [ ] **Step 4: Run to verify pass**: `conda run -n intention-bot python -m pytest -v` → 전부 통과
- [ ] **Step 5: 실서버 평가 실행.** PowerShell에서 `$env:INTENT_BASE_URL="https://your-host.example/sllm/v1"` 설정 후 `conda run -n intention-bot python -m intent_scorer.evaluate --rotations 1 4`. 기대: `rotations=4`에서 `top1_accuracy ≈ 16/16`, 평균 지연이 1초 미만 (스파이크: A1 15/16 0.06초, A4 16/16 0.29초). 크게 다르면 원인을 확인하고 결과를 보고한다. 출력된 표를 `eval/results-2026-11-20.txt`에 저장.
- [ ] **Step 6: Commit**: `git add -A && git commit -m "feat: evaluation script with latency metrics and dataset"`

---

## Self-Review

- **Spec coverage:** 점수 계산·순서 회전·정규화(Task 2, 3), 의도 정의 외부 설정(Task 1), 환경변수 설정(Task 1), 복합/해당없음 헬퍼(Task 3), `rotations`·`max_concurrency` 부하 제어(Task 3), 정확도·집합 일치·해당없음·소요 시간 평가와 방식 비교표(Task 4). B2·기준값 튜닝·분기 처리는 설계상 범위 밖이라 태스크 없음.
- **Type consistency:** `Intent`, `NONE_LABEL`, `label_probs`, `IntentScorer.score`, `select_intents`, `run_eval` 이름이 Task 간 일치.
- **Proportion:** 코드 본문 없이 시그니처·테스트 단언·스파이크 파일 참조 중심.
