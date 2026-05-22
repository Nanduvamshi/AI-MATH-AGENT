"""Unzip dataset archives and verify all data is in place.

Usage:
    python scripts/prepare_data.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATASETS_DIR,
    GSM8K_DIR,
    MATHQA_DIR,
    MATHQSA_DIR,
    ORCA_DIR,
    ORCA_PARQUET,
    RAW_DIR,
)


ZIP_TO_TARGET = [
    (DATASETS_DIR / "gsm8kzip.zip", "gsm8k", GSM8K_DIR),
    (DATASETS_DIR / "MathQAzip.zip", "MathQA", MATHQA_DIR),
    (DATASETS_DIR / "MathQSAzip.zip", "MathQSA", MATHQSA_DIR),
]


def _unzip_if_needed(zip_path: Path, zip_subdir: str, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] {target_dir} already populated")
        return
    if not zip_path.exists():
        print(f"[error] {zip_path} not found")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    extracted = RAW_DIR / zip_subdir
    if extracted.resolve() != target_dir.resolve() and extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        extracted.rmdir()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ORCA_DIR.mkdir(parents=True, exist_ok=True)

    for zip_path, subdir, target in ZIP_TO_TARGET:
        _unzip_if_needed(zip_path, subdir, target)

    print("\n[verify] expected files:")
    expected = [
        GSM8K_DIR / "train-00000-of-00001.parquet",
        GSM8K_DIR / "test-00000-of-00001.parquet",
        MATHQA_DIR / "train.json",
        MATHQA_DIR / "test.json",
        MATHQSA_DIR / "train.csv",
        MATHQSA_DIR / "test.csv",
    ]
    missing = []
    for p in expected:
        mark = "OK" if p.exists() else "MISSING"
        print(f"  {mark} {p.relative_to(ROOT)}")
        if not p.exists():
            missing.append(p)

    print(f"\n[verify] ORCA-Math (optional, ~200k rows):")
    if ORCA_PARQUET.exists():
        print(f"  OK      {ORCA_PARQUET.relative_to(ROOT)}")
    else:
        print(f"  MISSING {ORCA_PARQUET.relative_to(ROOT)} - download from:")
        print(f"    https://www.kaggle.com/datasets/johnsonhk88/microsoftorca-math-word-problems-200k")
        print(f"    Place the parquet at: {ORCA_PARQUET.relative_to(ROOT)}")

    if missing:
        print(f"\n[fail] {len(missing)} required file(s) missing")
        return 1
    print("\n[ok] datasets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
