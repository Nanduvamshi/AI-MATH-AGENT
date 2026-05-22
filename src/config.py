from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "Datasets"
RAW_DIR = DATASETS_DIR / "raw"

GSM8K_DIR = RAW_DIR / "gsm8k"
MATHQA_DIR = RAW_DIR / "mathqa"
MATHQSA_DIR = RAW_DIR / "mathqsa"
ORCA_DIR = RAW_DIR / "orca"
ORCA_PARQUET = ORCA_DIR / "200k.parquet"

QDRANT_PATH = ROOT / "qdrant_db"
COLLECTION_NAME = "math_qa"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 64

GROQ_GEN_MODEL = "llama-3.3-70b-versatile"
GROQ_JUDGE_MODEL = "llama-3.1-8b-instant"

RETRIEVAL_K = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")


def assert_keys(require_exa: bool = True) -> None:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if require_exa and not EXA_API_KEY:
        missing.append("EXA_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in."
        )
