# THROWAWAY SPIKE: compare A (label softmax) vs B (per-intent yes/no) scoring.
# Intent definitions and utterances below are PLACEHOLDER guesses, not user-provided.
import math, sys, io
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
c = OpenAI(base_url=__import__("dotenv").dotenv_values(".env")["BASE_URL"], api_key="EMPTY_KEY")
M = "google/gemma-4-26B-A4B-it"

INTENTS = {
    "정착지원": "새로 이주·입국·귀농귀촌한 사람이 새 지역에 자리잡는 데 필요한 초기 지원 (주택, 정착금, 지역 적응, 교육·상담)",
    "생활지원": "일상 생활 안정을 위한 지원 (생활비, 의료비, 돌봄, 복지, 공공요금, 긴급 생계)",
    "공장설비지원": "공장·제조업체의 설비 구입, 교체, 자동화, 시설 개선 비용 지원",
}
UTTS = [
    "귀농하려는데 집 구할 때 받을 수 있는 지원금 있나요?",
    "이번 달 생활비가 너무 부족한데 받을 수 있는 복지 있어요?",
    "저희 공장 프레스기계 교체하는데 보조금 신청하고 싶어요",
    "이사 와서 아직 일자리도 없고 생활비도 떨어졌어요",
    "공장 자동화 설비 들이고 싶은데 직원 기숙사도 지원되나요?",
    "오늘 날씨 어때요?",
    "외국에서 막 입국했는데 어디서부터 도움받을 수 있어요?",
]
LABELS = "ABCDEFGH"


def top(r):
    return {t.token.strip(): t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}


def score_a(utt, order):
    names = list(order)
    opts = "\n".join(f"{LABELS[i]}. {n}: {INTENTS[n]}" for i, n in enumerate(names))
    p = f"다음 의도 중 사용자 발화가 가장 가까운 것을 고르세요.\n{opts}\n\n발화: \"{utt}\"\n답은 알파벳 한 글자만."
    r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": p}], max_tokens=1,
                                  logprobs=True, top_logprobs=20, temperature=0)
    t = top(r)
    ps = {n: math.exp(t.get(LABELS[i], -50)) for i, n in enumerate(names)}
    s = sum(ps.values()) or 1
    return {n: v / s for n, v in ps.items()}


def score_a_avg(utt):
    names = list(INTENTS)
    orders = [names, names[::-1], names[1:] + names[:1]]
    out = {n: 0 for n in names}
    for o in orders:
        for n, v in score_a(utt, o).items():
            out[n] += v / len(orders)
    return out


def score_b_one(utt, n):
    p = (f"의도 정의: {n} = {INTENTS[n]}\n\n발화: \"{utt}\"\n\n이 발화가 위 의도에 해당합니까? "
         "yes 또는 no 한 단어만 답하세요.")
    r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": p}], max_tokens=1,
                                  logprobs=True, top_logprobs=20, temperature=0)
    t = top(r)
    y = sum(math.exp(v) for k, v in t.items() if k.lower() == "yes")
    no = sum(math.exp(v) for k, v in t.items() if k.lower() == "no")
    return y / (y + no) if y + no else float("nan")


def score_b(utt):
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda n: score_b_one(utt, n), INTENTS))
    return dict(zip(INTENTS, res))


fmt = lambda d: "  ".join(f"{k}={v:.2f}" for k, v in d.items())
for u in UTTS:
    print(u)
    print("  A(1order):", fmt(score_a(u, list(INTENTS))))
    print("  A(avg3)  :", fmt(score_a_avg(u)))
    print("  B        :", fmt(score_b(u)))
