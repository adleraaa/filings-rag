"""Tiny synthetic filings plus deterministic stand-ins for the ML models.

The page texts are invented for tests; they are not FinanceBench data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from filings_rag.index import Index
from filings_rag.ingest import Page, chunk_pages
from filings_rag.retrieval import Retriever, tokenize

PAGES = [
    Page("ACME_2022_10K", 1, "Acme Corporation annual report fiscal 2022. Table of contents."),
    Page(
        "ACME_2022_10K",
        2,
        "Consolidated statement of cash flows. Purchases of property, plant and "
        "equipment (1,577) million in 2022 and (1,373) million in 2021.",
    ),
    Page("ACME_2022_10K", 3, "Risk factors: supply chain disruption and litigation may affect results."),
    Page("GLOBEX_2021_10K", 1, "Globex Inc annual report fiscal 2021. Revenue was 8,700 million."),
    Page(
        "GLOBEX_2021_10K",
        2,
        "Dividends declared per share were 2.40 dollars. The company repurchased shares worth 500 million.",
    ),
    Page("GLOBEX_2021_10K", 3, "Purchases of property, plant and equipment were 910 million."),
]


class HashEmbedder:
    """Bag-of-words hashed into 64 dims: similar wording -> similar vectors."""

    dim = 64

    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OverlapReranker:
    """Scores a passage by how many distinct query tokens it contains."""

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) for t in texts], dtype=np.float32)


class FakeLLM:
    """Returns queued replies in order and records every prompt it was sent."""

    def __init__(self, replies: Sequence[str]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.prompts.append((system, user))
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


@pytest.fixture
def index() -> Index:
    chunks = chunk_pages(PAGES, size=40, overlap=10)
    embedder = HashEmbedder()
    embeddings = embedder.encode([c.text for c in chunks], is_query=False)
    retriever = Retriever(chunks, embeddings, embedder, OverlapReranker(), candidates=10, rerank_depth=10)
    return Index({(p.doc, p.page): p for p in PAGES}, retriever)
