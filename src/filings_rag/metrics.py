"""Page-level retrieval metrics.

Retrievers return chunks, but FinanceBench labels evidence *pages*, so a
chunk ranking is first collapsed to a ranking of distinct (doc, page) pairs.
Several chunks from one page therefore count once, at the best rank.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

PageKey = tuple[str, int]


def pages_in_order(keys: Iterable[PageKey]) -> list[PageKey]:
    seen: set[PageKey] = set()
    out = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def hit_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int) -> float:
    """1.0 if any gold page is in the top k, else 0.0."""
    return float(any(p in gold for p in ranked[:k]))


def recall_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int) -> float:
    """Fraction of gold pages found in the top k."""
    if not gold:
        raise ValueError("gold set is empty")
    return len(gold.intersection(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: Sequence[PageKey], gold: set[PageKey]) -> float:
    for rank, p in enumerate(ranked, start=1):
        if p in gold:
            return 1.0 / rank
    return 0.0
