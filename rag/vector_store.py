"""Step 2 of the RAG pipeline: store chunk embeddings and find the most similar ones.

We use FAISS (Facebook AI Similarity Search). Vectors are L2-normalised, so the
inner product computed by IndexFlatIP equals cosine similarity.

Each PDF's embeddings are cached on disk (keyed by a hash of the file), so a
document is only embedded once, even across app restarts.
"""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from .pdf_utils import Chunk

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def cache_key(pdf_bytes: bytes, *settings) -> str:
    h = hashlib.sha256(pdf_bytes)
    for s in settings:
        h.update(str(s).encode())
    return h.hexdigest()[:24]


def load_cached(key: str):
    vec_path, meta_path = CACHE_DIR / f"{key}.npy", CACHE_DIR / f"{key}.json"
    if vec_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return chunks, np.load(vec_path), meta["pages"]
    return None


def save_cached(key: str, chunks: list[Chunk], vectors: np.ndarray, pages: int):
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(CACHE_DIR / f"{key}.npy", vectors)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"pages": pages, "chunks": [c.to_dict() for c in chunks]}),
        encoding="utf-8",
    )


class VectorStore:
    """A FAISS index over the chunks of all currently selected PDFs."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        vectors = normalize(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        q = normalize(np.asarray(query_vector).reshape(1, -1))
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
