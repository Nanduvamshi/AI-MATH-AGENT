from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture
def fresh_qdrant(monkeypatch, tmp_path):
    """Point Qdrant + collection at a tmp dir for the duration of the test.

    `index.py` does `from src.config import QDRANT_PATH`, so it has its own
    module-level binding — patching `config.QDRANT_PATH` alone is not enough.
    We also have to close any open client (Qdrant local takes a file lock).
    """
    from src import config, index

    if index._client is not None:
        try:
            index._client.close()
        except Exception:
            pass
        index._client = None

    tmp_db = tmp_path / "qdrant_db"
    monkeypatch.setattr(config, "QDRANT_PATH", tmp_db)
    monkeypatch.setattr(index, "QDRANT_PATH", tmp_db)
    monkeypatch.setattr(index, "_client", None)
    yield
    if index._client is not None:
        try:
            index._client.close()
        except Exception:
            pass
        index._client = None
    if tmp_db.exists():
        shutil.rmtree(tmp_db, ignore_errors=True)


def test_upsert_dedup_and_retrieve(fresh_qdrant):
    from src.index import retrieve, upsert_rows

    rows = pd.DataFrame(
        [
            {"question": "What is 2 + 2?", "answer": "4", "source": "test"},
            {"question": "What is 3 * 5?", "answer": "15", "source": "test"},
            {"question": "What is 10 - 7?", "answer": "3", "source": "test"},
            {"question": "What is the derivative of x squared?", "answer": "2x", "source": "test"},
            {"question": "What is 100 / 4?", "answer": "25", "source": "test"},
        ]
    )

    written_first = upsert_rows(rows, batch_size=2)
    assert written_first == 5

    written_again = upsert_rows(rows, batch_size=2)
    assert written_again == 0, "second call should be a no-op (dedup)"

    hits = retrieve("derivative of x^2", k=3)
    assert len(hits) >= 1
    top = hits[0]
    assert "derivative" in top["question"].lower()
    assert top["answer"] == "2x"


def test_add_correction_round_trip(fresh_qdrant):
    from src.index import add_correction, retrieve, upsert_rows

    upsert_rows(
        pd.DataFrame(
            [{"question": "What is 7 + 8?", "answer": "16", "source": "test"}]
        )
    )
    add_correction("What is 7 + 8?", "15", source="hitl")

    hits = retrieve("7 + 8", k=5)
    answers = [h["answer"] for h in hits]
    assert "15" in answers, "HITL correction must be retrievable after upsert"
