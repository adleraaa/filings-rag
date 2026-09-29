import numpy as np
import pytest

from filings_rag.metrics import hit_at_k, pages_in_order, recall_at_k, reciprocal_rank
from filings_rag.retrieval import BM25, rrf, tokenize, top_n


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the FY2018 CapEx of 3M?") == ["fy2018", "capex", "3m"]


def test_bm25_prefers_rare_matching_terms():
    bm = BM25(["apple banana", "apple cherry", "apple apple durian"])
    assert bm.idf("durian") > bm.idf("apple") > 0
    scores = bm.scores("apple durian")
    assert int(np.argmax(scores)) == 2
    assert bm.scores("zzz").sum() == 0


def test_bm25_length_normalisation():
    # Same single match; the shorter document should score higher.
    bm = BM25(["revenue", "revenue " + "filler " * 50])
    s = bm.scores("revenue")
    assert s[0] > s[1] > 0


def test_top_n_respects_mask_and_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_n(scores, None, 2) == [1, 3]
    mask = np.array([True, False, True, False])
    assert top_n(scores, mask, 3) == [2, 0]  # only 2 allowed items, even though n=3
    assert top_n(scores, np.zeros(4, dtype=bool), 3) == []


def test_rrf_combines_ranks():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == 1  # ranks 1 and 2
    assert ids[1] == 3  # ranks 3 and 1
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_doc_scoped_search_only_returns_that_doc(index):
    for method in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        hits = index.retriever.search(
            "purchases of property plant equipment", method, k=5, docs=["GLOBEX_2021_10K"]
        )
        assert hits and {h.chunk.doc for h in hits} == {"GLOBEX_2021_10K"}


def test_corpus_search_finds_the_capex_pages(index):
    hits = index.retriever.search("purchases of property plant and equipment", "hybrid_rerank", k=3)
    top_pages = {(h.chunk.doc, h.chunk.page) for h in hits[:2]}
    assert top_pages == {("ACME_2022_10K", 2), ("GLOBEX_2021_10K", 3)}


def test_unknown_method_raises(index):
    with pytest.raises(ValueError):
        index.retriever.search("x", method="magic")


def test_page_metrics():
    ranked = pages_in_order([("A", 1), ("A", 1), ("A", 5), ("B", 2), ("A", 9)])
    assert ranked == [("A", 1), ("A", 5), ("B", 2), ("A", 9)]
    gold = {("A", 5), ("A", 9)}
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 3) == 0.5
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert reciprocal_rank(ranked, gold) == 0.5
    assert reciprocal_rank(ranked, {("C", 1)}) == 0.0
    # Same page number in a different filing is not a hit.
    assert hit_at_k(ranked, {("B", 1)}, 4) == 0.0
