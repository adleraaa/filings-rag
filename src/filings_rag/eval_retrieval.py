"""Retrieval ablation: every method x {doc-scoped, corpus-wide} on FinanceBench."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from .data import Question
from .metrics import hit_at_k, pages_in_order, recall_at_k, reciprocal_rank
from .retrieval import METHODS, RERANK_METHODS, Reranker, Retriever

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
                # Diagnostic for corpus-wide runs: did we at least reach the right
                # filing? Separates "wrong document" from "right document, wrong page".
                "doc_hit@5": float(any(doc == q.doc for doc, _ in ranked[:5])),
            }
        )
    summary = {
        "method": method,
        "scope": scope,
        "n": len(per_q),
        **{f"hit@{k}": _mean(per_q, f"hit@{k}") for k in KS},
        **{f"recall@{k}": _mean(per_q, f"recall@{k}") for k in KS},
        "mrr@10": _mean(per_q, "rr"),
        "doc_hit@5": _mean(per_q, "doc_hit@5"),
        "hit@5_by_question_type": {
            t: _mean([r for r in per_q if r["question_type"] == t], "hit@5")
            for t in sorted({r["question_type"] for r in per_q})
        },
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
    rerankers: dict[str, Reranker] | None = None,
) -> list[dict]:
    """Run every (scope, method) pair. With `rerankers`, each rerank method runs
    once per named reranker and is labelled e.g. "hybrid_rerank[minilm]".

    Results are merged into existing files in out_dir by (scope, label), so a
    subset can be re-run without repeating slow configs."""
    variants: list[tuple[str, str, Reranker | None]] = []
    for method in methods:
        if method in RERANK_METHODS and rerankers:
            variants += [(method, f"{method}[{name}]", r) for name, r in rerankers.items()]
        else:
            variants.append((method, method, retriever.reranker))

    summaries, runs = [], {}
    for scope in scopes:
        for method, label, reranker in variants:
            retriever.reranker = reranker
            summary, per_q = run_config(retriever, questions, method, scope)
            summary["method"] = label
            print(f"{scope:6s} {label:22s} hit@5={summary['hit@5']:.3f} mrr={summary['mrr@10']:.3f}")
            summaries.append(summary)
            runs[f"{scope}/{label}"] = per_q
    summaries, runs = _merge(out_dir, summaries, runs)
    (out_dir / "retrieval_ablation.json").write_text(
        json.dumps({"configs": summaries, "chunk_depth": CHUNK_DEPTH}, indent=2), encoding="utf-8"
    )
    (out_dir / "retrieval_ablation.md").write_text(to_markdown(summaries), encoding="utf-8")
    (out_dir / "retrieval_per_question.json").write_text(json.dumps(runs), encoding="utf-8")
    return summaries


def _merge(out_dir: Path, new: list[dict], new_runs: dict) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "retrieval_ablation.json"
    runs_file = out_dir / "retrieval_per_question.json"
    if not (summary_file.exists() and runs_file.exists()):
        return new, new_runs
    old = json.loads(summary_file.read_text(encoding="utf-8"))["configs"]
    runs = json.loads(runs_file.read_text(encoding="utf-8"))
    by_key = {(s["scope"], s["method"]): s for s in old}
    by_key.update({(s["scope"], s["method"]): s for s in new})
    runs.update(new_runs)
    # Stable order: doc scope first, then methods in the order they were first seen.
    order = {k: i for i, k in enumerate([*by_key])}
    merged = sorted(by_key.values(), key=lambda s: (s["scope"] != "doc", order[(s["scope"], s["method"])]))
    return merged, runs


def to_markdown(summaries: Sequence[dict]) -> str:
    cols = [
        *(f"hit@{k}" for k in KS),
        *(f"recall@{k}" for k in KS),
        "mrr@10",
        "doc_hit@5",
        "median_latency_ms",
    ]
    lines = [
        "| scope | method | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for s in summaries:
        cells = [f"{s[c]:.3f}" if c != "median_latency_ms" else f"{s[c]:.0f}" for c in cols]
        lines.append(f"| {s['scope']} | {s['method']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
