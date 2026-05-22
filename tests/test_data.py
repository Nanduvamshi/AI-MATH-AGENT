from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.data import _normalize, load_all


def test_normalize_collapses_whitespace_and_case():
    assert _normalize("  What is  TWO + 2?  ") == "what is two + 2?"


@pytest.mark.skipif(
    not (ROOT / "Datasets" / "raw" / "gsm8k" / "train-00000-of-00001.parquet").exists(),
    reason="run scripts/prepare_data.py first",
)
def test_load_all_has_required_columns():
    df = load_all()
    assert {"question", "answer", "source"}.issubset(df.columns)
    assert len(df) > 0
    assert df["question"].notna().all()
    assert df["answer"].notna().all()
    sources = set(df["source"].unique())
    assert sources.issubset({"gsm8k", "mathqa", "mathqsa", "orca"})
