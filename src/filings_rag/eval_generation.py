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
from .generate import _BRACKET, answer_question
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
            check = check_answer(ans, index.page_text)
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
                "flagged": check.flagged,
                "flag_reasons": check.reasons,
                "numbers": check.numbers,
                "numbers_supported": check.supported,
                "unsupported_values": check.unsupported_values,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{q.qid} {verdict.verdict:9s} flagged={check.flagged}")
    return [json.loads(line) for line in out_path.open(encoding="utf-8")]


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


def summarize(rows: list[dict]) -> dict:
    def acc(sub: list[dict]) -> dict:
        n = len(sub)
        return {
            "n": n,
            "correct": sum(r["verdict"] == "correct" for r in sub),
            "incorrect": sum(r["verdict"] == "incorrect" for r in sub),
            "refusal": sum(r["verdict"] == "refusal" for r in sub),
            "accuracy": round(sum(r["verdict"] == "correct" for r in sub) / n, 4) if n else None,
        }

    answered = [r for r in rows if not r["refusal"]]
    flagged = [r for r in answered if r["flagged"]]
    unflagged = [r for r in answered if not r["flagged"]]
    reasons: dict[str, int] = {}
    for r in flagged:
        for reason in r["flag_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    with_sim = [r for r in answered if r.get("similarity") is not None]
    wrong = [r["verdict"] != "correct" for r in with_sim]
    return {
        "overall": acc(rows),
        "by_question_type": {
            t: acc([r for r in rows if r["question_type"] == t])
            for t in sorted({r["question_type"] for r in rows})
        },
        "gold_page_in_context": acc([r for r in rows if r["gold_in_context"]]),
        "gold_page_not_in_context": acc([r for r in rows if not r["gold_in_context"]]),
        "citation_check": {
            "answered": len(answered),
            "flagged": len(flagged),
            "flag_rate": round(len(flagged) / len(answered), 4) if answered else None,
            "flag_reasons": reasons,
            "flagged_judge": acc(flagged),
            "unflagged_judge": acc(unflagged),
            # How well each signal separates wrong from right answers (0.5 = chance).
            "auroc_flag_predicts_wrong": auroc(
                [float(r["flagged"]) for r in answered], [r["verdict"] != "correct" for r in answered]
            ),
            "auroc_low_similarity_predicts_wrong": auroc([-r["similarity"] for r in with_sim], wrong),
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
    lines += ["", "| citation check | value |", "|---|---|"]
    lines.append(f"| non-refusal answers | {cc['answered']} |")
    lines.append(f"| flagged | {cc['flagged']} ({cc['flag_rate']}) |")
    for reason, count in sorted(cc["flag_reasons"].items()):
        lines.append(f"| reason: {reason} | {count} |")
    lines.append(f"| judge accuracy, flagged | {cc['flagged_judge']['accuracy']} |")
    lines.append(f"| judge accuracy, unflagged | {cc['unflagged_judge']['accuracy']} |")
    lines.append(f"| AUROC flag -> wrong | {cc['auroc_flag_predicts_wrong']} |")
    lines.append(f"| AUROC low similarity -> wrong | {cc['auroc_low_similarity_predicts_wrong']} |")
    return "\n".join(lines) + "\n"
