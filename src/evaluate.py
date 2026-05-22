from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from tqdm import tqdm

from src.agent import MathAgent
from src.data import load_gsm8k_test


_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_FINAL_RE = re.compile(r"final answer[^\d-]*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _normalize_num(s: str) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def extract_gold(gsm8k_answer: str) -> str | None:
    m = _GOLD_RE.search(gsm8k_answer)
    if m:
        return _normalize_num(m.group(1))
    return None


def extract_predicted(agent_answer: str) -> str | None:
    m = _FINAL_RE.search(agent_answer)
    if m:
        n = _normalize_num(m.group(1))
        if n is not None:
            return n
    nums = _LAST_NUM_RE.findall(agent_answer)
    if nums:
        return _normalize_num(nums[-1])
    return None


@dataclass
class EvalReport:
    n: int
    correct: int
    accuracy: float
    by_route: dict
    examples: list


def evaluate(n: int = 50, k: int = 5, use_fallback: bool = True) -> EvalReport:
    test = load_gsm8k_test()
    sample = test.head(n).reset_index(drop=True)
    agent = MathAgent(k=k, use_fallback=use_fallback)

    correct = 0
    route_counts: Counter = Counter()
    route_correct: Counter = Counter()
    examples = []

    for _, row in tqdm(
        list(sample.iterrows()), total=len(sample), desc="eval", unit="q"
    ):
        q, gold_raw = row["question"], row["answer"]
        gold = extract_gold(gold_raw)
        try:
            result = agent.forward(q)
            pred = extract_predicted(result.answer)
            route = result.route
        except Exception as e:
            pred, route = None, f"error:{type(e).__name__}"

        route_counts[route] += 1
        ok = (gold is not None) and (pred == gold)
        if ok:
            correct += 1
            route_correct[route] += 1

        if len(examples) < 5:
            examples.append(
                {
                    "question": q[:120],
                    "gold": gold,
                    "predicted": pred,
                    "route": route,
                    "correct": ok,
                }
            )

    n_done = len(sample)
    by_route = {
        r: {"n": route_counts[r], "correct": route_correct[r]} for r in route_counts
    }
    return EvalReport(
        n=n_done,
        correct=correct,
        accuracy=correct / max(n_done, 1),
        by_route=by_route,
        examples=examples,
    )


def print_report(r: EvalReport) -> None:
    print(f"\nGSM8K eval: {r.correct}/{r.n} = {r.accuracy:.1%}")
    print("by route:")
    for route, stats in r.by_route.items():
        n = stats["n"]
        c = stats["correct"]
        acc = c / max(n, 1)
        print(f"  {route:>15s}: {c}/{n} = {acc:.1%}")
    print("\nfirst examples:")
    for ex in r.examples:
        mark = "OK" if ex["correct"] else "X "
        print(
            f"  {mark} [{ex['route']}] gold={ex['gold']} pred={ex['predicted']}"
            f"  q={ex['question']!r}"
        )
