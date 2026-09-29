"""Retrieval ablation: every method x {doc-scoped, corpus-wide} on FinanceBench."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from .data import Question
from .metrics import hit_at_k, pages_in_order, recall_at_k, reciprocal_rank
from .retrieval import METHODS, Retriever

KS = (1, 3, 5, 10)
CHUNK_DEPTH = 30  # chunks fetched per query; equals the rerank pool so all methods see the same depth
PAGE_DEPTH = 10  # MRR is computed over the top 10 distinct pages (MRR@10)


def run_config(
    retriever: Retriever, questions: Sequence[Question], method: str, scope: str
) -> tuple[dict, list[dict]]:
    per_q = []
    latencies = []
    for q in questions:
        docs = [q.doc] if scope == "doc" else None
        t0 = time.perf_counter()
        hits = retriever.search(q.question, method=method, k=CHUNK_DEPTH, docs=docs)
        latencies.append(time.perf_counter() - t0)
        ranked = pages_in_order((h.chunk.doc, h.chunk.page) for h in hits)[:PAGE_DEPTH]
        gold = {(q.doc, p) for p in q.evidence_pages}
        per_q.append(
            {
                "qid": q.qid,
                "question_type": q.question_type,
                "gold": sorted(gold),
                "ranked_pages": ranked,
                **{f"hit@{k}": hit_at_k(ranked, gold, k) for k in KS},
                **{f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS},
                "rr": reciprocal_rank(ranked, gold),
            }
        )
    summary = {
        "method": method,
        "scope": scope,
        "n": len(per_q),
        **{f"hit@{k}": _mean(per_q, f"hit@{k}") for k in KS},
        **{f"recall@{k}": _mean(per_q, f"recall@{k}") for k in KS},
        "mrr@10": _mean(per_q, "rr"),
        "median_latency_ms": round(1000 * statistics.median(latencies), 1),
    }
    return summary, per_q


def _mean(rows: list[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0


def run_ablation(
    retriever: Retriever,
    questions: Sequence[Question],
    out_dir: Path,
    methods: Sequence[str] = METHODS,
    scopes: Sequence[str] = ("doc", "corpus"),
) -> list[dict]:
    summaries, runs = [], {}
    for scope in scopes:
        for method in methods:
            summary, per_q = run_config(retriever, questions, method, scope)
            print(f"{scope:6s} {method:14s} hit@5={summary['hit@5']:.3f} mrr={summary['mrr@10']:.3f}")
            summaries.append(summary)
            runs[f"{scope}/{method}"] = per_q
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_ablation.json").write_text(
        json.dumps({"configs": summaries, "chunk_depth": CHUNK_DEPTH}, indent=2), encoding="utf-8"
    )
    (out_dir / "retrieval_ablation.md").write_text(to_markdown(summaries), encoding="utf-8")
    (out_dir / "retrieval_per_question.json").write_text(json.dumps(runs), encoding="utf-8")
    return summaries


def to_markdown(summaries: Sequence[dict]) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "mrr@10", "median_latency_ms"]
    lines = [
        "| scope | method | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for s in summaries:
        cells = [f"{s[c]:.3f}" if c != "median_latency_ms" else f"{s[c]:.0f}" for c in cols]
        lines.append(f"| {s['scope']} | {s['method']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
