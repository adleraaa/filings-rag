"""Retrievers: BM25, dense, hybrid (reciprocal rank fusion) and hybrid + rerank.

Every retriever scores *all* chunks and then applies an optional document
filter. That makes "doc-scoped" and "corpus-wide" retrieval the same code path
with a different mask, so the comparison between them is apples to apples.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ingest import Chunk

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank", "dense_rerank")
RERANK_METHODS = ("hybrid_rerank", "dense_rerank")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were what which with how did does do give".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over an inverted index (term -> chunk ids, term frequencies)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.doc_len = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append((i, tf))
        self.avg_len = float(self.doc_len.mean()) if self.n else 0.0
        self.postings = {
            t: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float32))
            for t, p in postings.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        # The "+1" variant keeps IDF positive for very common terms.
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avg_len, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            out[ids] += self.idf(term) * tf * (self.k1 + 1) / (tf + norm[ids])
        return out


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per text."""
        ...


class SentenceTransformerEmbedder:
    # bge-*-v1.5 models are trained with this instruction on the query side only.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vecs, dtype=np.float32)


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, t) for t in texts]), dtype=np.float32)


def top_n(scores: np.ndarray, mask: np.ndarray | None, n: int) -> list[int]:
    """Indices of the n highest scores among allowed positions, best first."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
        n = min(n, int(mask.sum()))
    n = min(n, len(scores))
    if n == 0:
        return []
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")].tolist()


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank).

    RRF only looks at ranks, so BM25 scores (unbounded) and cosine scores
    (in [-1, 1]) can be fused without any score normalisation.
    Returns (item, fused score) pairs, best first; ties break on item id.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        candidates: int = 100,
        rerank_depth: int = 30,
    ):
        self.chunks = list(chunks)
        self.bm25 = BM25([c.text for c in self.chunks])
        self.embeddings = embeddings
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_depth = rerank_depth
        self._doc_of = np.array([c.doc for c in self.chunks])
        self.docs = sorted(set(self._doc_of.tolist()))

    def mask_for(self, docs: Sequence[str] | None) -> np.ndarray | None:
        if docs is None:
            return None
        return np.isin(self._doc_of, list(docs))

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None or self.embedder is None:
            raise RuntimeError("dense retrieval needs embeddings and an embedder")
        q = self.embedder.encode([query], is_query=True)[0]
        return self.embeddings @ q

    def search(
        self, query: str, method: str = "hybrid", k: int = 10, docs: Sequence[str] | None = None
    ) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        mask = self.mask_for(docs)

        if method == "bm25":
            s = self.bm25.scores(query)
            return [Hit(self.chunks[i], float(s[i])) for i in top_n(s, mask, k)]
        if method in ("dense", "dense_rerank"):
            s = self._dense_scores(query)
            ranked = [(i, float(s[i])) for i in top_n(s, mask, max(k, self.rerank_depth))]
        else:
            lexical = top_n(self.bm25.scores(query), mask, self.candidates)
            semantic = top_n(self._dense_scores(query), mask, self.candidates)
            ranked = rrf([lexical, semantic])
        if method in ("dense", "hybrid"):
            return [Hit(self.chunks[i], s) for i, s in ranked[:k]]

        if self.reranker is None:
            raise RuntimeError(f"{method} needs a reranker")
        pool = [i for i, _ in ranked[: self.rerank_depth]]
        s = self.reranker.score(query, [self.chunks[i].text for i in pool])
        order = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.chunks[pool[j]], float(s[j])) for j in order]
