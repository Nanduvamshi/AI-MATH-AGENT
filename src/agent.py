from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    GROQ_API_KEY,
    GROQ_GEN_MODEL,
    GROQ_JUDGE_MODEL,
    RETRIEVAL_K,
)
from src.fallback import format_web_context, web_search
from src.index import retrieve

_gen_llm = None
_judge_llm = None


def _get_llm(model: str):
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env.")
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=0)


def get_gen_llm():
    global _gen_llm
    if _gen_llm is None:
        _gen_llm = _get_llm(GROQ_GEN_MODEL)
    return _gen_llm


def get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = _get_llm(GROQ_JUDGE_MODEL)
    return _judge_llm


GEN_PROMPT = """You are a math problem solver. Use the retrieved similar problems as context.

Retrieved problems:
{context}

Question: {question}

Solve the question step by step. End your answer with a single line:
Final Answer: <number or expression>
"""

WEB_PROMPT = """You are a math problem solver. Use the web search results as supporting context.

Web results:
{context}

Question: {question}

Solve the question step by step. End your answer with a single line:
Final Answer: <number or expression>
"""

JUDGE_PROMPT = """Given a math question and a candidate answer, decide whether the answer numerically resolves the question.

Question: {question}
Candidate answer: {answer}

Reply with exactly one word: YES or NO.
"""


def _format_retrieved(hits: list[dict]) -> str:
    if not hits:
        return "(none)"
    return "\n\n".join(
        f"[{i + 1}] (source={h['source']}, score={h['score']:.2f})\n"
        f"Q: {h['question']}\nA: {h['answer']}"
        for i, h in enumerate(hits)
    )


def _llm_invoke(llm, prompt: str) -> str:
    msg = llm.invoke(prompt)
    return getattr(msg, "content", str(msg)).strip()


@dataclass
class AgentResult:
    answer: str
    route: str  # "grounded" or "web-fallback"
    sources: list[dict] = field(default_factory=list)
    judge_verdict: str = ""


class MathAgent:
    """Retrieve -> generate -> LLM-judge -> (optional) web fallback."""

    def __init__(self, k: int = RETRIEVAL_K, use_fallback: bool = True):
        self.k = k
        self.use_fallback = use_fallback

    def forward(self, question: str) -> AgentResult:
        hits = retrieve(question, k=self.k)
        context = _format_retrieved(hits)
        gen_prompt = GEN_PROMPT.format(context=context, question=question)
        grounded_answer = _llm_invoke(get_gen_llm(), gen_prompt)

        judge_prompt = JUDGE_PROMPT.format(
            question=question, answer=grounded_answer
        )
        verdict = _llm_invoke(get_judge_llm(), judge_prompt).upper()
        ok = verdict.startswith("YES")

        if ok or not self.use_fallback:
            return AgentResult(
                answer=grounded_answer,
                route="grounded",
                sources=hits,
                judge_verdict=verdict,
            )

        snippets = web_search(question, num_results=3)
        web_ctx = format_web_context(snippets)
        web_prompt = WEB_PROMPT.format(context=web_ctx, question=question)
        web_answer = _llm_invoke(get_gen_llm(), web_prompt)
        return AgentResult(
            answer=web_answer,
            route="web-fallback",
            sources=[{"web": s} for s in snippets],
            judge_verdict=verdict,
        )

    __call__ = forward
