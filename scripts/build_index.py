"""Build (or extend) the persistent Qdrant index from all available datasets.

Idempotent: re-running on the same data only embeds NEW rows.

Usage:
    python scripts/build_index.py                  # full corpus
    python scripts/build_index.py --subset 5000    # quick demo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_all
from src.index import upsert_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="Index only the first N rows after dedup (0 = all).",
    )
    args = parser.parse_args()

    df = load_all()
    if args.subset > 0:
        df = df.head(args.subset)
        print(f"[build_index] subset: keeping first {len(df):,} rows")

    t0 = time.time()
    written = upsert_rows(df)
    dt = time.time() - t0
    print(f"[build_index] done in {dt:.1f}s, {written:,} new points written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
