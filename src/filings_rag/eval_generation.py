"""Generation experiment: answer, judge, and citation-check every question.

Results are appended one JSON line per question, and questions already in the
output file are skipped, so an interrupted run resumes without paying twice.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .citation_check import check_answer
from .data import Question
from .generate import _BRACKET, Answer, answer_question
from .index import Index
from .judge import judge
from .llm import ChatModel
from .retrieval import Embedder


def run_generation(
    index: Index,
    questions: Sequence[Question],
    answer_llm: ChatModel,
    judge_llm: ChatModel,
    out_path: Path,
    method: str,
    scope: str,
    k: int = 5,
) -> list[dict]:
    done = set()
    if out_path.exists():
        done = {json.loads(line)["qid"] for line in out_path.open(encoding="utf-8")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for q in questions:
            if q.qid in done:
                continue
            docs = [q.doc] if scope == "doc" else None
            hits = index.retriever.search(q.question, method=method, k=k, docs=docs)
            ans = answer_question(answer_llm, q.question, hits)
            verdict = judge(judge_llm, q.question, q.answer, ans.text)
            gold = {(q.doc, p) for p in q.evidence_pages}
            row = {
                "qid": q.qid,
                "question_type": q.question_type,
                "question": q.question,
                "gold_answer": q.answer,
                "gold_pages": sorted(gold),
                "context_pages": list(ans.context_pages),
                "gold_in_context": bool(gold & set(ans.context_pages)),
                "answer": ans.text,
                "citations": list(ans.citations),
                "refusal": ans.is_refusal,
                "verdict": verdict.verdict,
                "judge_reason": verdict.reason,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{q.qid} {verdict.verdict}")
    return [json.loads(line) for line in out_path.open(encoding="utf-8")]


CHECK_VARIANTS = {"verbatim": False, "arithmetic": True}


def apply_checks(rows: list[dict], index: Index) -> None:
    """Run both citation-check variants on stored answers (no API calls)."""
    for r in rows:
        ans = Answer(
            r["answer"],
            tuple(tuple(c) for c in r["citations"]),
            tuple(tuple(c) for c in r["context_pages"]),
        )
        r["checks"] = {}
        for name, arithmetic in CHECK_VARIANTS.items():
            c = check_answer(ans, index.page_text, arithmetic=arithmetic)
            r["checks"][name] = {
                "flagged": c.flagged,
                "reasons": c.reasons,
                "numbers": c.numbers,
                "supported": c.supported,
                "unsupported_values": c.unsupported_values,
            }


def add_similarity(rows: list[dict], index: Index, embedder: Embedder) -> None:
    """Cosine similarity between the answer (citations removed) and its cited pages.

    Pages are longer than the embedder's 512-token window, so each cited page is
    represented by its chunks and the best-matching chunk counts.
    """
    by_page: dict[tuple[str, int], list[str]] = {}
    for c in index.retriever.chunks:
        by_page.setdefault((c.doc, c.page), []).append(c.text)
    for r in rows:
        texts = [t for c in r["citations"] for t in by_page.get(tuple(c), [])]
        if r["refusal"] or not texts:
            r["similarity"] = None
            continue
        a = embedder.encode([_BRACKET.sub(" ", r["answer"])], is_query=False)[0]
        r["similarity"] = round(float(np.max(embedder.encode(texts, is_query=False) @ a)), 4)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability a random positive outranks a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def _acc(sub: list[dict]) -> dict:
    n = len(sub)
    correct = sum(r["verdict"] == "correct" for r in sub)
    return {
        "n": n,
        "correct": correct,
        "incorrect": sum(r["verdict"] == "incorrect" for r in sub),
        "refusal": sum(r["verdict"] == "refusal" for r in sub),
        "accuracy": round(correct / n, 4) if n else None,
    }


def _check_summary(answered: list[dict], variant: str) -> dict:
    flagged = [r for r in answered if r["checks"][variant]["flagged"]]
    unflagged = [r for r in answered if not r["checks"][variant]["flagged"]]
    reasons: dict[str, int] = {}
    for r in flagged:
        for reason in r["checks"][variant]["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "flagged": len(flagged),
        "flag_rate": round(len(flagged) / len(answered), 4) if answered else None,
        "flag_reasons": reasons,
        "flagged_judge": _acc(flagged),
        "unflagged_judge": _acc(unflagged),
        # How well the flag separates wrong from right answers (0.5 = chance).
        "auroc_flag_predicts_wrong": auroc(
            [float(r["checks"][variant]["flagged"]) for r in answered],
            [r["verdict"] != "correct" for r in answered],
        ),
    }


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["refusal"]]
    with_sim = [r for r in answered if r.get("similarity") is not None]
    return {
        "overall": _acc(rows),
        "by_question_type": {
            t: _acc([r for r in rows if r["question_type"] == t])
            for t in sorted({r["question_type"] for r in rows})
        },
        "gold_page_in_context": _acc([r for r in rows if r["gold_in_context"]]),
        "gold_page_not_in_context": _acc([r for r in rows if not r["gold_in_context"]]),
        "citation_check": {
            "answered": len(answered),
            **{v: _check_summary(answered, v) for v in CHECK_VARIANTS},
            "auroc_low_similarity_predicts_wrong": auroc(
                [-r["similarity"] for r in with_sim], [r["verdict"] != "correct" for r in with_sim]
            ),
        },
    }


def to_markdown(s: dict) -> str:
    def row(name: str, a: dict) -> str:
        pct = f"{a['accuracy']:.3f}" if a["accuracy"] is not None else "-"
        return f"| {name} | {a['n']} | {a['correct']} | {a['incorrect']} | {a['refusal']} | {pct} |"

    lines = ["| subset | n | correct | incorrect | refusal | accuracy |", "|---|---|---|---|---|---|"]
    lines.append(row("all", s["overall"]))
    for t, a in s["by_question_type"].items():
        lines.append(row(t, a))
    lines.append(row("gold page retrieved", s["gold_page_in_context"]))
    lines.append(row("gold page not retrieved", s["gold_page_not_in_context"]))
    cc = s["citation_check"]
    variants = list(CHECK_VARIANTS)
    lines += ["", f"Citation check on {cc['answered']} non-refusal answers:", ""]
    lines += ["| metric | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]

    def metric(name: str, get) -> None:
        lines.append(f"| {name} | " + " | ".join(str(get(cc[v])) for v in variants) + " |")

    metric("flagged", lambda c: f"{c['flagged']} ({c['flag_rate']})")
    metric("judge accuracy, flagged", lambda c: c["flagged_judge"]["accuracy"])
    metric("judge accuracy, unflagged", lambda c: c["unflagged_judge"]["accuracy"])
    metric("AUROC flag -> wrong", lambda c: c["auroc_flag_predicts_wrong"])
    lines += ["", f"AUROC low answer/page similarity -> wrong: {cc['auroc_low_similarity_predicts_wrong']}"]
    return "\n".join(lines) + "\n"
