from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    GSM8K_DIR,
    MATHQA_DIR,
    MATHQSA_DIR,
    ORCA_PARQUET,
)


def _normalize(question: str) -> str:
    return " ".join(str(question).lower().split())


def load_gsm8k() -> pd.DataFrame:
    train = pd.read_parquet(GSM8K_DIR / "train-00000-of-00001.parquet")
    return train[["question", "answer"]].assign(source="gsm8k")


def load_mathqa() -> pd.DataFrame:
    with open(MATHQA_DIR / "train.json", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    return (
        df.rename(columns={"Problem": "question", "Rationale": "answer"})[
            ["question", "answer"]
        ]
        .assign(source="mathqa")
    )


def load_mathqsa() -> pd.DataFrame:
    df = pd.read_csv(MATHQSA_DIR / "train.csv")
    cols = {c.lower(): c for c in df.columns}
    q = cols.get("question") or cols.get("problem") or list(df.columns)[0]
    a = cols.get("answer") or cols.get("rationale") or list(df.columns)[1]
    return (
        df.rename(columns={q: "question", a: "answer"})[["question", "answer"]]
        .assign(source="mathqsa")
    )


def load_orca() -> pd.DataFrame:
    if not ORCA_PARQUET.exists():
        return pd.DataFrame(columns=["question", "answer", "source"])
    df = pd.read_parquet(ORCA_PARQUET)
    cols = {c.lower(): c for c in df.columns}
    q = cols.get("question") or list(df.columns)[0]
    a = cols.get("answer") or list(df.columns)[1]
    return (
        df.rename(columns={q: "question", a: "answer"})[["question", "answer"]]
        .assign(source="orca")
    )


def load_all() -> pd.DataFrame:
    frames = []
    for loader in (load_gsm8k, load_mathqa, load_mathqsa, load_orca):
        try:
            df = loader()
            if len(df):
                frames.append(df)
                print(f"[data] {loader.__name__}: {len(df):,} rows")
        except FileNotFoundError as e:
            print(f"[data] {loader.__name__}: missing ({e})")
    if not frames:
        raise RuntimeError(
            "No datasets loaded. Run `python scripts/prepare_data.py` first."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["question", "answer"])
    combined["question"] = combined["question"].astype(str)
    combined["answer"] = combined["answer"].astype(str)
    before = len(combined)
    combined["_norm_q"] = combined["question"].map(_normalize)
    combined = combined.drop_duplicates(subset=["_norm_q"]).drop(columns=["_norm_q"])
    print(f"[data] total: {before:,} -> after dedup: {len(combined):,}")
    return combined.reset_index(drop=True)


def load_gsm8k_test() -> pd.DataFrame:
    path = GSM8K_DIR / "test-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first."
        )
    return pd.read_parquet(path)[["question", "answer"]]
