"""Interactive REPL for the Math Agentic RAG.

Usage:
    python scripts/chat.py
    python scripts/chat.py --k 10
    python scripts/chat.py --no-fallback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable Exa web fallback (skip LLM-judge routing).",
    )
    args = parser.parse_args()

    from src.agent import MathAgent
    from src.config import assert_keys
    from src.feedback import ask_once

    assert_keys(require_exa=not args.no_fallback)

    agent = MathAgent(k=args.k, use_fallback=not args.no_fallback)
    print("Math Agentic RAG. Type a question. Empty line or Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            ask_once(agent, q, interactive=True)
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")
    print("bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
