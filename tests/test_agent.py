from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_extract_predicted_finds_final_answer_pattern():
    from src.evaluate import extract_predicted

    out = "Step 1...\nStep 2: 12 - 5 = 7.\nFinal Answer: 7"
    assert extract_predicted(out) == "7"


def test_extract_predicted_falls_back_to_last_number():
    from src.evaluate import extract_predicted

    out = "The student bought 3 apples and 4 oranges, so 7 fruits."
    assert extract_predicted(out) == "7"


def test_extract_gold_parses_gsm8k_format():
    from src.evaluate import extract_gold

    s = "Janet sells 18 - 3 = <<18-3=15>>15 eggs.\n#### 18"
    assert extract_gold(s) == "18"


def test_normalize_num_handles_commas_and_floats():
    from src.evaluate import _normalize_num

    assert _normalize_num("1,000") == "1000"
    assert _normalize_num("3.0") == "3"
    assert _normalize_num("3.14") == "3.14"
    assert _normalize_num("not a number") is None


class _StubLLM:
    def __init__(self, text):
        self.text = text

    def invoke(self, _prompt):
        from types import SimpleNamespace

        return SimpleNamespace(content=self.text)


def test_routing_falls_back_when_judge_says_no(monkeypatch):
    """If the judge says NO, the agent must call the web fallback path."""
    from src import agent as agent_mod

    monkeypatch.setattr(agent_mod, "retrieve", lambda q, k=5: [])
    monkeypatch.setattr(agent_mod, "get_gen_llm", lambda: _StubLLM("garbage answer"))
    monkeypatch.setattr(agent_mod, "get_judge_llm", lambda: _StubLLM("NO"))

    called = {}

    def fake_web_search(q, num_results=3):
        called["yes"] = q
        return [{"title": "T", "url": "u", "text": "some web text"}]

    monkeypatch.setattr(agent_mod, "web_search", fake_web_search)
    monkeypatch.setattr(agent_mod, "format_web_context", lambda snips: "ctx")

    result = agent_mod.MathAgent(use_fallback=True).forward("nonsense question")
    assert called.get("yes") == "nonsense question"
    assert result.route == "web-fallback"


def test_routing_returns_grounded_when_judge_says_yes(monkeypatch):
    from src import agent as agent_mod

    monkeypatch.setattr(agent_mod, "retrieve", lambda q, k=5: [])
    monkeypatch.setattr(agent_mod, "get_gen_llm", lambda: _StubLLM("Final Answer: 42"))
    monkeypatch.setattr(agent_mod, "get_judge_llm", lambda: _StubLLM("YES"))

    def fail_web(*a, **kw):
        raise AssertionError("fallback must not be called when judge says YES")

    monkeypatch.setattr(agent_mod, "web_search", fail_web)
    result = agent_mod.MathAgent(use_fallback=True).forward("what is 6 * 7?")
    assert result.route == "grounded"
    assert "42" in result.answer
