from __future__ import annotations

import hashlib
import uuid
from typing import Iterable

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from src.config import (
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    QDRANT_PATH,
    RETRIEVAL_K,
)

_embedder = None
_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"batch_size": EMBED_BATCH_SIZE, "normalize_embeddings": True},
        )
    return _embedder


def _content_hash(question: str, answer: str) -> str:
    h = hashlib.sha1()
    h.update(question.encode("utf-8"))
    h.update(b"\x1f")
    h.update(answer.encode("utf-8"))
    return h.hexdigest()


def _hash_to_uuid(content_hash: str) -> str:
    return str(uuid.UUID(content_hash[:32]))


def ensure_collection() -> None:
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[index] created collection '{COLLECTION_NAME}'")
    else:
        n = client.count(COLLECTION_NAME, exact=True).count
        print(f"[index] collection '{COLLECTION_NAME}' exists with {n:,} points")


def existing_ids() -> set[str]:
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return set()
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10_000,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            ids.add(str(p.id))
        if offset is None:
            break
    return ids


def _batch(it: Iterable, n: int):
    chunk = []
    for x in it:
        chunk.append(x)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def upsert_rows(rows: pd.DataFrame, batch_size: int = EMBED_BATCH_SIZE) -> int:
    """Embed and upsert rows. Returns the number of NEW points written."""
    ensure_collection()
    client = get_client()
    embedder = get_embedder()

    already = existing_ids()

    pending = []
    for _, row in rows.iterrows():
        ch = _content_hash(row["question"], row["answer"])
        pid = _hash_to_uuid(ch)
        if pid in already:
            continue
        pending.append(
            (pid, row["question"], row["answer"], row.get("source", "unknown"))
        )

    if not pending:
        print(f"[index] no new rows to embed (input: {len(rows):,}, already indexed)")
        return 0

    print(
        f"[index] {len(pending):,} new rows to embed "
        f"(skipped {len(rows) - len(pending):,} already-indexed)"
    )

    written = 0
    for chunk in tqdm(
        list(_batch(pending, batch_size)),
        desc="embed+upsert",
        unit="batch",
    ):
        texts = [f"Q: {q}\nA: {a}" for (_pid, q, a, _src) in chunk]
        vectors = embedder.embed_documents(texts)
        points = [
            PointStruct(
                id=pid,
                vector=vec,
                payload={"question": q, "answer": a, "source": src},
            )
            for (pid, q, a, src), vec in zip(chunk, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        written += len(points)
    print(f"[index] wrote {written:,} new points")
    return written


def add_correction(question: str, correction: str, source: str = "hitl") -> str:
    """Insert a single (question, correction) pair. Returns the point id."""
    ensure_collection()
    client = get_client()
    embedder = get_embedder()

    ch = _content_hash(question, correction)
    pid = _hash_to_uuid(ch)
    vec = embedder.embed_documents([f"Q: {question}\nA: {correction}"])[0]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=pid,
                vector=vec,
                payload={"question": question, "answer": correction, "source": source},
            )
        ],
    )
    return pid


def retrieve(question: str, k: int = RETRIEVAL_K) -> list[dict]:
    client = get_client()
    embedder = get_embedder()
    vec = embedder.embed_query(question)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=k,
        with_payload=True,
    ).points
    return [
        {
            "score": float(h.score),
            "question": h.payload.get("question", ""),
            "answer": h.payload.get("answer", ""),
            "source": h.payload.get("source", "unknown"),
        }
        for h in hits
    ]
