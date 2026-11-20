# THROWAWAY SPIKE 2: RFP-based definitions, user-provided utterances + extra tests.
import math, sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

c = OpenAI(base_url=__import__("dotenv").dotenv_values(".env")["BASE_URL"], api_key="EMPTY_KEY")
M = "google/gemma-4-26B-A4B-it"

INTENTS = {
    "정착지원": "경상북도로 이주·전입했거나 하려는 사람이 정착하는 데 필요한 지원. 정주여건 지역 비교, 이사할 지역 추천, 전입 인센티브·정착지원금·월세 지원, 전입 지원정책과 신청 절차.",
    "생활지원": "경북 도민의 일상 생활 안정을 위한 복지·돌봄·주거 지원. 기초연금, 노인복지·노인맞춤돌봄, 아이돌봄, 주거복지의 안내와 자격 확인, 신청 서류.",
    "소상공인지원": "소상공인·예비창업자를 위한 지원. 창업 방법, 정책자금(금리·이차보전), 청년 창업 지원사업, 업종·매출 기반 지원사업 추천과 신청 준비.",
    "공장설립지원": "공장 설립을 준비하는 기업·수요자를 위한 지원. 설립 조건, 입지 규제와 용도지역 검토, 계획입지/개별입지, 배출시설 등 인허가 절차, 공장설립 AI 사전진단.",
}
NAMES = list(INTENTS)
# (utterance, expected set)
GIVEN = [
    ("30대, 김천혁신도시로 직장 이전 예정. 근처 이사할 지역 추천해줘", {"정착지원"}),
    ("전입 장려금(정착지원금)은 언제 지급되나요?", {"정착지원"}),
    ("경북으로 이사 오는 청년도 월세 지원을 받을 수 있나요?", {"정착지원"}),
    ("정착지원금을 받은 뒤 다른 지역으로 이사하면 돌려줘야 하나요?", {"정착지원"}),
    ("제가 75살이고 혼자 살고 있는데, 기초연금을 받을 수 있을까요?", {"생활지원"}),
    ("노인맞춤돌봄서비스는 어떻게 신청하나요?", {"생활지원"}),
    ("기초연금은 누가 받을 수 있고 얼마를 받나요?", {"생활지원"}),
    ("아이돌봄서비스 신청 후 대기가 길어요. 어떻게 해야 하나요?", {"생활지원"}),
    ("남편 직장 때문에 김천으로 이사 가는데, 음식점 창업하려면 어떻게 해야 해?", {"소상공인지원"}),
    ("경북 소상공인 정책자금의 금리와 이차보전은 어떻게 되나요?", {"소상공인지원"}),
    ("아직 사업자등록 전인 예비창업자도 지원사업에 신청할 수 있나요?", {"소상공인지원"}),
    ("청년 창업 지원사업의 \"청년\" 나이 기준은 어떻게 되나요?", {"소상공인지원"}),
    ("경북에 작은 식품 제조공장을 세우려고 하는데, 공장을 지으려면 어떤 조건이 필요한가요?", {"공장설립지원"}),
    ("공장설립 AI 사전진단 결과는 법적 효력이 있나요?", {"공장설립지원"}),
    ("계획입지(산업단지)와 개별입지는 어떻게 다른가요?", {"공장설립지원"}),
    ("도장이나 절삭 공정이 있으면 대기·폐수 배출시설 허가를 받아야 하나요?", {"공장설립지원"}),
]
EXTRA = [
    ("서울에서 경북으로 이사 가는데 아이 돌봄 지원도 받을 수 있나요?", {"정착지원", "생활지원"}),
    ("귀농해서 정착하려는데 식당 창업자금도 받을 수 있나요?", {"정착지원", "소상공인지원"}),
    ("소상공인인데 공장도 새로 지으려고 합니다. 받을 수 있는 보조금 알려주세요", {"소상공인지원", "공장설립지원"}),
    ("김천으로 이사 왔는데 혼자 사시는 어머니 돌봄서비스 신청하고 싶어요", {"정착지원", "생활지원"}),
    ("지원금 받을 수 있나요?", set()),
    ("안녕하세요", set()),
    ("오늘 점심 뭐 먹지", set()),
    ("주민등록등본은 어떻게 발급받나요?", set()),
]
LB = "ABCDE"


def top(r):
    return {t.token.strip(): t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}


def call(p):
    return top(c.chat.completions.create(model=M, messages=[{"role": "user", "content": p}], max_tokens=1,
                                         logprobs=True, top_logprobs=20, temperature=0))


def score_a(utt, order):
    opts = [f"{LB[i]}. {n}: {INTENTS[n]}" for i, n in enumerate(order)]
    opts.append(f"{LB[len(order)]}. 해당없음: 위 어느 것에도 해당하지 않거나 의도가 불분명함")
    p = "사용자 발화가 어떤 서비스를 원하는지 가장 가까운 것 하나를 고르세요.\n" + "\n".join(opts) + f"\n\n발화: \"{utt}\"\n답은 알파벳 한 글자만."
    t = call(p)
    ps = {n: math.exp(t.get(LB[i], -50)) for i, n in enumerate(order)}
    ps["해당없음"] = math.exp(t.get(LB[len(order)], -50))
    s = sum(ps.values()) or 1
    return {k: v / s for k, v in ps.items()}


def score_a_avg(utt):
    orders = [NAMES[i:] + NAMES[:i] for i in range(4)]
    out = {n: 0 for n in NAMES + ["해당없음"]}
    for o in orders:
        for k, v in score_a(utt, o).items():
            out[k] += v / len(orders)
    return out


def score_b_one(utt, n):
    p = (f"서비스: {n}\n정의: {INTENTS[n]}\n\n발화: \"{utt}\"\n\n사용자가 이 서비스에 관한 도움을 원합니까? "
         "yes 또는 no 한 단어만 답하세요.")
    t = call(p)
    y = sum(math.exp(v) for k, v in t.items() if k.lower() == "yes")
    no = sum(math.exp(v) for k, v in t.items() if k.lower() == "no")
    return y / (y + no) if y + no else float("nan")


def score_b(utt):
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(NAMES, ex.map(lambda n: score_b_one(utt, n), NAMES)))


short = {"정착지원": "정착", "생활지원": "생활", "소상공인지원": "소상", "공장설립지원": "공장", "해당없음": "없음"}
fmt = lambda d: " ".join(f"{short[k]}={v:.2f}" for k, v in d.items())

import time, statistics


def score_a1(utt):
    return score_a(utt, NAMES)


def score_b2_one(utt, n):
    defs = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    p = (f"서비스 목록:\n{defs}\n- 해당없음: 위 서비스와 무관하거나 의도가 불분명함\n\n발화: \"{utt}\"\n\n"
         f"이 발화에는 '{n}' 서비스를 이용하려는 의도가 명확히 들어 있습니까? 다른 서비스의 의도와 구분해서 판단하세요. yes 또는 no 한 단어만 답하세요.")
    t = call(p)
    y = sum(math.exp(v) for k, v in t.items() if k.lower() == "yes")
    no = sum(math.exp(v) for k, v in t.items() if k.lower() == "no")
    return y / (y + no) if y + no else float("nan")


def score_b2(utt):
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(NAMES, ex.map(lambda n: score_b2_one(utt, n), NAMES)))


METHODS = {"A1(1call)": (score_a1, 0.2), "A4(rot4)": (score_a_avg, 0.2), "B(old)": (score_b, 0.5), "B2(ctx)": (score_b2, 0.5)}
ALL = GIVEN + EXTRA
res = {m: [] for m in METHODS}
times = {m: [] for m in METHODS}
score_a1("워밍업")
for u, exp in ALL:
    for m, (f, th) in METHODS.items():
        t0 = time.perf_counter()
        sc = f(u)
        times[m].append(time.perf_counter() - t0)
        res[m].append(sc)
out = []
for m, (f, th) in METHODS.items():
    top1 = sum(max(NAMES, key=lambda k: r[k]) in exp for r, (u, exp) in zip(res[m][:16], GIVEN))

    def pred(r):
        if "해당없음" in r and max(r, key=r.get) == "해당없음":
            return set()
        return {k for k in NAMES if r[k] >= th}

    exact = sum(pred(r) == exp for r, (u, exp) in zip(res[m], ALL))
    ts = sorted(times[m])
    out.append(f"{m}: top1(given)={top1}/16 exactset(24)={exact}/24 latency mean={statistics.mean(ts):.2f}s "
               f"median={statistics.median(ts):.2f}s p95={ts[int(len(ts) * 0.95) - 1]:.2f}s max={ts[-1]:.2f}s")
out.append("")
for i, (u, exp) in enumerate(EXTRA):
    out.append(u + " exp=" + str(sorted(exp) or "없음"))
    for m in METHODS:
        out.append("  " + m + ": " + fmt(res[m][16 + i]))
for i, (u, exp) in enumerate(GIVEN):
    r = res["B2(ctx)"][i]
    if max(NAMES, key=lambda k: r[k]) not in exp:
        out.append("B2 miss: " + u + " " + fmt(r))
open(r"C:\Users\COM-NMOIM\AppData\Local\Temp\spike3.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
