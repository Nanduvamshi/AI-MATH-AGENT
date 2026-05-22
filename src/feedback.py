from __future__ import annotations

from src.agent import AgentResult, MathAgent
from src.index import add_correction


def _print_result(result: AgentResult) -> None:
    tag = "[grounded]" if result.route == "grounded" else "[web-fallback]"
    print(f"\n{tag} (judge said: {result.judge_verdict or 'n/a'})")
    print("-" * 60)
    print(result.answer)
    print("-" * 60)
    if result.route == "grounded" and result.sources:
        srcs = ", ".join(
            f"{s['source']}({s['score']:.2f})" for s in result.sources[:3]
        )
        print(f"top-3 retrieved sources: {srcs}")
    elif result.route == "web-fallback" and result.sources:
        urls = [s["web"].get("url", "") for s in result.sources if "web" in s]
        print(f"web sources: {len(urls)} results")


def ask_once(agent: MathAgent, question: str, *, interactive: bool = True) -> AgentResult:
    """Run one question through the agent. If interactive, run the HITL loop."""
    result = agent.forward(question)
    _print_result(result)

    if not interactive:
        return result

    try:
        ans = input("\nWas this correct? (y/n, blank=y): ").strip().lower()
    except EOFError:
        return result

    if ans == "n":
        try:
            correction = input("Please provide the correct answer: ").strip()
        except EOFError:
            return result
        if correction:
            pid = add_correction(question, correction, source="hitl")
            print(f"[hitl] upserted correction to index (id={pid[:8]}...)")
        else:
            print("[hitl] empty correction, skipping")
    else:
        print("[hitl] noted.")
    return result
