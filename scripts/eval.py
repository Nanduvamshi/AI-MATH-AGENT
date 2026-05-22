"""Run GSM8K exact-match evaluation.

Usage:
    python scripts/eval.py --n 50
    python scripts/eval.py --n 200 --k 10
    python scripts/eval.py --no-fallback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    from src.config import assert_keys
    from src.evaluate import evaluate, print_report

    assert_keys(require_exa=not args.no_fallback)

    report = evaluate(n=args.n, k=args.k, use_fallback=not args.no_fallback)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
