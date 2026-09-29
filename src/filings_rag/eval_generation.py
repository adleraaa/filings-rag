"""Generation experiment: answer, judge, and citation-check every question.

Results are appended one JSON line per question, and questions already in the
output file are skipped, so an interrupted run resumes without paying twice.
The run's config is stored next to the output, and resuming with a different
config is refused, so one results file never mixes two configurations.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .citation_check import MODES, check_answer
from .claims import BRACKET
from .data import Question
from .generate import Answer, answer_question
from .index import Index
from .judge import judge
from .llm import ChatModel
from .null_baseline import MAGNITUDES, PERTURBATIONS, false_accepts
from .retrieval import Embedder

STATUSES = ("answered", "partial", "refusal")
VERDICTS = ("correct", "incorrect", "refusal")


def config_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".config.json")


def _check_config(out_path: Path, config: dict) -> None:
    """Refuse to append to a results file that was produced with another config."""
    cfg_file = config_path(out_path)
    has_rows = out_path.exists() and out_path.stat().st_size > 0
    if has_rows:
        stored = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else None
        if stored != config:
            raise ValueError(
                f"{out_path} was produced with config {stored}, not {config}; "
                "use another output path or delete the file to start over"
            )
    else:
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_generation(
    index: Index,
    questions: Sequence[Question],
    answer_llm: ChatModel,
    judge_llm: ChatModel,
    out_path: Path,
    method: str,
    scope: str,
    k: int = 5,
    label: str | None = None,
) -> list[dict]:
    """`label` names the method in the stored config (e.g. "hybrid_rerank[bge]")."""
    config = {
        "method": label or method,
        "scope": scope,
        "k_chunks": k,
        "answer_model": getattr(answer_llm, "model", None),
        "judge_model": getattr(judge_llm, "model", None),
    }
    _check_config(out_path, config)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["qid"] for line in out_path.open(encoding="utf-8")}
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
                "status": ans.status,
                "verdict": verdict.verdict,
                "judge_reason": verdict.reason,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{q.qid} {verdict.verdict}")
    return [json.loads(line) for line in out_path.open(encoding="utf-8")]


def _answer(r: dict) -> Answer:
    return Answer(
        r["answer"],
        tuple(tuple(c) for c in r["citations"]),
        tuple(tuple(c) for c in r["context_pages"]),
    )


def apply_checks(rows: list[dict], index: Index) -> None:
    """Classify each stored answer and run every citation-check mode (no API calls).

    The status is recomputed from the answer text rather than read from the
    row, so a change to the classifier applies to old results too.
    """
    for r in rows:
        ans = _answer(r)
        r.pop("refusal", None)  # legacy field from before the "partial" status existed
        r["status"] = ans.status
        r["checks"] = {}
        for mode in MODES:
            c = check_answer(ans, index.page_text, mode=mode)
            r["checks"][mode] = {
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
        if r["status"] == "refusal" or not texts:
            r["similarity"] = None
            continue
        a = embedder.encode([BRACKET.sub(" ", r["answer"])], is_query=False)[0]
        r["similarity"] = round(float(np.max(embedder.encode(texts, is_query=False) @ a)), 4)


def add_null_baseline(summary: dict, rows: list[dict], page_text: Callable[[str, int], str | None]) -> None:
    """False-accept rate of every check mode on corrupted copies of the real answers."""
    answers = [_answer(r) for r in rows if r["status"] != "refusal"]
    for mode in MODES:
        summary["citation_check"][mode]["null_baseline"] = false_accepts(answers, page_text, mode)


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


def _wrong(r: dict) -> bool:
    # "Wrong" = not judged correct: judged incorrect, or judged a refusal although
    # the text states numbers (a hedged partial answer).
    return r["verdict"] != "correct"


def _flag_stats(rows: list[dict], mode: str) -> tuple[float | None, float | None]:
    flags = [r["checks"][mode]["flagged"] for r in rows]
    flagged = [r for r, f in zip(rows, flags, strict=True) if f]
    unflagged = [r for r, f in zip(rows, flags, strict=True) if not f]
    gap = None
    if flagged and unflagged:
        gap = _acc(flagged)["accuracy"] - _acc(unflagged)["accuracy"]
    return auroc([float(f) for f in flags], [_wrong(r) for r in rows]), gap


def _bootstrap_ci(rows: list[dict], mode: str, n_resamples: int = 2000, seed: int = 0) -> dict:
    """95% percentile intervals over question resamples; resamples where a
    statistic is undefined (no flagged or no wrong answer drawn) are skipped."""
    rng = np.random.default_rng(seed)
    aurocs, gaps = [], []
    for _ in range(n_resamples):
        sample = [rows[i] for i in rng.integers(0, len(rows), size=len(rows))]
        a, g = _flag_stats(sample, mode)
        if a is not None:
            aurocs.append(a)
        if g is not None:
            gaps.append(g)

    def ci(xs: list[float]) -> list[float] | None:
        return [round(float(np.percentile(xs, q)), 4) for q in (2.5, 97.5)] if xs else None

    return {"auroc_ci95": ci(aurocs), "accuracy_gap_ci95": ci(gaps), "resamples": n_resamples}


def _check_summary(checked: list[dict], mode: str) -> dict:
    flagged = [r for r in checked if r["checks"][mode]["flagged"]]
    unflagged = [r for r in checked if not r["checks"][mode]["flagged"]]
    reasons: dict[str, int] = {}
    for r in flagged:
        for reason in r["checks"][mode]["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    auroc_value, gap = _flag_stats(checked, mode)
    return {
        "flagged": len(flagged),
        "flag_rate": round(len(flagged) / len(checked), 4) if checked else None,
        "flag_reasons": reasons,
        "wrong_flagged": sum(_wrong(r) for r in flagged),
        "wrong_unflagged": sum(_wrong(r) for r in unflagged),
        "flagged_judge": _acc(flagged),
        "unflagged_judge": _acc(unflagged),
        # Accuracy of flagged minus unflagged answers; a useful check makes this negative.
        "accuracy_gap_flagged_minus_unflagged": None if gap is None else round(gap, 4),
        # How well the flag separates wrong (= not judged correct) from correct answers; 0.5 = chance.
        "auroc_flag_predicts_wrong": auroc_value,
        **_bootstrap_ci(checked, mode),
        "flag_rate_by_question_type": {
            t: round(sum(r["checks"][mode]["flagged"] for r in sub) / len(sub), 4)
            for t in sorted({r["question_type"] for r in checked})
            if (sub := [r for r in checked if r["question_type"] == t])
        },
    }


def summarize(rows: list[dict]) -> dict:
    checked = [r for r in rows if r["status"] != "refusal"]
    with_sim = [r for r in checked if r.get("similarity") is not None]
    return {
        "overall": _acc(rows),
        "by_question_type": {
            t: _acc([r for r in rows if r["question_type"] == t])
            for t in sorted({r["question_type"] for r in rows})
        },
        "gold_page_in_context": _acc([r for r in rows if r["gold_in_context"]]),
        "gold_page_not_in_context": _acc([r for r in rows if not r["gold_in_context"]]),
        # Rule-based answer status (generate.Answer.status) against the judge's verdict.
        "status_vs_verdict": {
            s: {v: sum(r["status"] == s and r["verdict"] == v for r in rows) for v in VERDICTS}
            for s in STATUSES
        },
        "citation_check": {
            "checked": len(checked),
            "checked_wrong": sum(_wrong(r) for r in checked),
            **{mode: _check_summary(checked, mode) for mode in MODES},
            "auroc_low_similarity_predicts_wrong": auroc(
                [-r["similarity"] for r in with_sim], [_wrong(r) for r in with_sim]
            ),
        },
    }


def to_markdown(s: dict) -> str:
    def row(name: str, a: dict) -> str:
        pct = f"{a['accuracy']:.3f}" if a["accuracy"] is not None else "-"
        return f"| {name} | {a['n']} | {a['correct']} | {a['incorrect']} | {a['refusal']} | {pct} |"

    lines = ["Judge verdicts:", "", "| subset | n | correct | incorrect | refusal | accuracy |"]
    lines.append("|---|---|---|---|---|---|")
    lines.append(row("all", s["overall"]))
    for t, a in s["by_question_type"].items():
        lines.append(row(t, a))
    lines.append(row("gold page retrieved", s["gold_page_in_context"]))
    lines.append(row("gold page not retrieved", s["gold_page_not_in_context"]))

    lines += ["", "Answer status (rule) vs judge verdict:", ""]
    lines += ["| status | " + " | ".join(VERDICTS) + " |", "|---|" + "---|" * len(VERDICTS)]
    for status, counts in s["status_vs_verdict"].items():
        lines.append(f"| {status} | " + " | ".join(str(counts[v]) for v in VERDICTS) + " |")

    cc = s["citation_check"]
    lines += [
        "",
        f"Citation check on {cc['checked']} answered or partial answers "
        f"({cc['checked_wrong']} not judged correct):",
        "",
    ]
    lines += ["| metric | " + " | ".join(MODES) + " |", "|---|" + "---|" * len(MODES)]

    def metric(name: str, get) -> None:
        lines.append(f"| {name} | " + " | ".join(str(get(cc[m])) for m in MODES) + " |")

    metric("flagged", lambda c: f"{c['flagged']} ({c['flag_rate']})")
    metric("wrong answers flagged / unflagged", lambda c: f"{c['wrong_flagged']} / {c['wrong_unflagged']}")
    metric("judge accuracy, flagged", lambda c: c["flagged_judge"]["accuracy"])
    metric("judge accuracy, unflagged", lambda c: c["unflagged_judge"]["accuracy"])
    metric(
        "accuracy gap [95% CI]",
        lambda c: f"{c['accuracy_gap_flagged_minus_unflagged']} {c['accuracy_gap_ci95']}",
    )
    metric("AUROC flag -> wrong [95% CI]", lambda c: f"{c['auroc_flag_predicts_wrong']} {c['auroc_ci95']}")
    if "null_baseline" in cc[MODES[0]]:
        for kind in PERTURBATIONS:
            for bucket in (None, *MAGNITUDES):

                def cell(c: dict, kind: str = kind, bucket: str | None = bucket) -> str:
                    nb = c["null_baseline"][kind]
                    nb = nb if bucket is None else nb["by_magnitude"][bucket]
                    return f"{nb['false_accept_rate']} ({nb['accepted']}/{nb['numbers']})"

                where = "all numbers" if bucket is None else f"numbers {bucket}"
                metric(f"false-accept rate, {kind}, {where}", cell)
    lines += ["", f"AUROC low answer/page similarity -> wrong: {cc['auroc_low_similarity_predicts_wrong']}"]
    return "\n".join(lines) + "\n"
