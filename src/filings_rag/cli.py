"""Command-line entry point: filings-rag <command> [options]."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from .data import download, load_questions
from .retrieval import METHODS, RERANK_METHODS

DATA = Path("data")
QUESTIONS = DATA / "financebench_open_source.jsonl"
INDEX = DATA / "index"
RESULTS = Path("results")


def cmd_download(args: argparse.Namespace) -> None:
    paths = download(args.data_dir)
    print(f"{len(paths)} PDFs in {args.data_dir / 'pdfs'}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .index import DEFAULT_EMBED_MODEL, build_index
    from .ingest import read_chunks, read_pages
    from .retrieval import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(DEFAULT_EMBED_MODEL)
    if args.stats_only:
        pages = read_pages(args.index_dir / "pages.jsonl")
        chunks = read_chunks(args.index_dir / "chunks.jsonl")
    else:
        pages, chunks = build_index(args.pdf_dir, args.index_dir, embedder, args.chunk_words, args.overlap)
    print(f"{len(pages)} pages -> {len(chunks)} chunks in {args.index_dir}")

    # Stats that tell us whether the chunking choice is sane for the embedder.
    lengths = [len(ids) for ids in embedder.model.tokenizer([c.text for c in chunks])["input_ids"]]
    stats = {
        "documents": len({p.doc for p in pages}),
        "pages": len(pages),
        "empty_pages": sum(not p.text for p in pages),
        "chunks": len(chunks),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap,
        "chunk_tokens_median": statistics.median(lengths),
        "chunk_tokens_p90": sorted(lengths)[int(0.9 * len(lengths))],
        "chunks_over_512_tokens": round(sum(n > 512 for n in lengths) / len(lengths), 4),
        "evidence_page_check": evidence_page_check(args.questions, {(p.doc, p.page): p.text for p in pages}),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def evidence_page_check(qfile: Path, page_text: dict) -> dict:
    """Where FinanceBench's evidence snippet is found relative to the labelled page.

    Guards against an off-by-one in page numbering, which would silently turn
    every retrieval metric into noise: the labelled page (0-based index + 1)
    should win by a wide margin over its neighbours.
    """

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    def alnum(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    counts = {"labelled_page": 0, "page_before": 0, "page_after": 0}
    # The three page counts can overlap (a snippet repeated on two pages), so
    # items found on none of them are counted directly rather than by subtraction.
    not_found = 0
    # Of those, how many match the labelled page once spaces and punctuation are
    # ignored, i.e. differ only in how the two PDF extractors laid out the text.
    not_found_but_alnum_match = 0
    total = 0
    for line in qfile.open(encoding="utf-8"):
        for ev in json.loads(line)["evidence"]:
            snippet = norm(ev["evidence_text_full_page"])[50:150]
            page = ev["evidence_page_num"] + 1
            total += 1
            found = False
            for name, p in (("labelled_page", page), ("page_before", page - 1), ("page_after", page + 1)):
                hit = snippet in norm(page_text.get((ev["doc_name"], p), ""))
                counts[name] += hit
                found |= hit
            if not found:
                not_found += 1
                # Drop 5 characters at each end: the 100-character cut may split a word.
                core = alnum(snippet)[5:-5]
                not_found_but_alnum_match += core in alnum(page_text.get((ev["doc_name"], page), ""))
    return {
        "evidence_items": total,
        "snippet_found_on": counts,
        "found_on_none_of_the_three": not_found,
        "of_which_on_labelled_page_ignoring_spaces_and_punctuation": not_found_but_alnum_match,
    }


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    from .eval_retrieval import run_ablation, to_markdown
    from .index import RERANKERS, load_index
    from .retrieval import CrossEncoderReranker

    if args.markdown_only:
        # Re-render the table from saved results without running retrieval again.
        configs = json.loads((args.out / "retrieval_ablation.json").read_text(encoding="utf-8"))["configs"]
        (args.out / "retrieval_ablation.md").write_text(to_markdown(configs), encoding="utf-8")
        return
    index = load_index(args.index_dir, reranker=None)
    rerankers = {name: CrossEncoderReranker(RERANKERS[name]) for name in args.rerankers}
    questions = load_questions(args.questions)
    run_ablation(index.retriever, questions, args.out, args.methods, args.scopes, rerankers)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    from .bootstrap import run

    for row in run(args.out):
        lo, hi = row["ci95"]
        print(f"{row['config']:28s} vs {row['reference']}: {row['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def parse_method(label: str) -> tuple[str, str | None]:
    """ "hybrid_rerank[bge]" -> ("hybrid_rerank", "bge"); "bm25" -> ("bm25", None)."""
    from .index import DEFAULT_RERANKER, RERANKERS

    m = re.fullmatch(r"(\w+)(?:\[(\w+)\])?", label)
    if not m or m.group(1) not in METHODS or (m.group(2) and m.group(2) not in RERANKERS):
        raise argparse.ArgumentTypeError(f"unknown method {label!r}")
    method, reranker = m.groups()
    if method in RERANK_METHODS:
        return method, reranker or DEFAULT_RERANKER
    return method, None


def method_label(label: str) -> str:
    """argparse type: validate a method label and keep it as a string."""
    parse_method(label)
    return label


def best_config(results_dir: Path, scope: str) -> str:
    configs = json.loads((results_dir / "retrieval_ablation.json").read_text())["configs"]
    candidates = [c for c in configs if c["scope"] == scope]
    return max(candidates, key=lambda c: (c["hit@5"], c["mrr@10"]))["method"]


def cmd_eval_generation(args: argparse.Namespace) -> None:
    from .eval_generation import (
        add_null_baseline,
        add_similarity,
        apply_checks,
        config_path,
        run_generation,
        summarize,
        to_markdown,
    )
    from .index import load_index
    from .llm import DeepSeekChat, SpendTracker

    label = args.method or best_config(args.out, args.scope)
    method, reranker = parse_method(label)
    print(f"generation with method={label} scope={args.scope} k={args.k}")
    index = load_index(args.index_dir, reranker=reranker)
    tracker = SpendTracker.load(args.out / "spend.json", cap_cny=args.cap)
    questions = load_questions(args.questions)[: args.limit]
    out = args.out / "generation.jsonl"
    rows = run_generation(
        index,
        questions,
        DeepSeekChat(tracker, "answer"),
        DeepSeekChat(tracker, "judge"),
        out,
        method,
        args.scope,
        args.k,
        label=label,
    )
    apply_checks(rows, index)
    add_similarity(rows, index, index.retriever.embedder)
    summary = summarize(rows)
    add_null_baseline(summary, rows, index.page_text)
    summary["config"] = json.loads(config_path(out).read_text(encoding="utf-8"))
    # Spend is cumulative over every command that calls the API (including `ask`),
    # so it lives in one file instead of a copy that goes stale.
    summary["spend_file"] = "spend.json"
    (args.out / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "generation_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (args.out / "generation_with_similarity.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    print(to_markdown(summary))


def cmd_ask(args: argparse.Namespace) -> None:
    from .citation_check import check_answer
    from .generate import answer_question
    from .index import load_index
    from .llm import DeepSeekChat, SpendTracker
    from .mcp_server import company_names, docs_for_company

    method, reranker = parse_method(args.method)
    index = load_index(args.index_dir, reranker=reranker)
    docs = [args.doc] if args.doc else (docs_for_company(index, args.company) if args.company else None)
    if docs == [] or (args.doc and args.doc not in index.retriever.docs):
        # Without this, retrieval returns nothing and we would pay for an empty-context answer.
        sys.exit(f"no filing matches; companies in the index: {', '.join(company_names(index))}")
    hits = index.retriever.search(args.question, method=method, k=args.k, docs=docs)
    tracker = SpendTracker.load(RESULTS / "spend.json", cap_cny=args.cap)
    ans = answer_question(DeepSeekChat(tracker, "ask"), args.question, hits)
    print(ans.text)
    print(f"\ncontext pages: {', '.join(f'{d} p.{p}' for d, p in ans.context_pages)}")
    for name in ("verbatim", "arithmetic"):
        check = check_answer(ans, index.page_text, mode=name)
        status = f"FLAGGED {', '.join(check.reasons)} {check.unsupported_values}" if check.flagged else "ok"
        print(f"citation check ({name}): {status}")


def cmd_serve_mcp(args: argparse.Namespace) -> None:
    from .index import load_index
    from .mcp_server import build_server

    method, reranker = parse_method(args.method)
    build_server(load_index(args.index_dir, reranker=reranker), method=method).run()


def export_explorer(results: Path) -> dict:
    """Join retrieval runs and generation rows into one JSON for the static page."""
    ablation = json.loads((results / "retrieval_ablation.json").read_text(encoding="utf-8"))["configs"]
    runs = json.loads((results / "retrieval_per_question.json").read_text(encoding="utf-8"))
    gen_path = results / "generation_with_similarity.jsonl"
    generated = {}
    if gen_path.exists():
        generated = {r["qid"]: r for r in map(json.loads, gen_path.open(encoding="utf-8"))}
    # Index every run by question id: runs may come from merged subset reruns,
    # so their row order is not guaranteed to match.
    by_qid = {cfg: {r["qid"]: r for r in rows} for cfg, rows in runs.items()}
    first_seen: dict[str, dict] = {}
    for rows in runs.values():
        for r in rows:
            first_seen.setdefault(r["qid"], r)
    questions = []
    for qid, first in first_seen.items():
        g = generated.get(qid, {})
        questions.append(
            {
                "qid": qid,
                "question_type": first["question_type"],
                "doc": first["gold"][0][0],
                "gold_pages": first["gold"],
                "question": g.get("question", ""),
                "gold_answer": g.get("gold_answer", ""),
                "answer": g.get("answer", ""),
                "status": g.get("status", ""),
                "verdict": g.get("verdict", "not run"),
                "judge_reason": g.get("judge_reason", ""),
                "checks": g.get("checks", {}),
                # Top-5 pages of every config that retrieved for this question.
                "retrieval": {
                    cfg: rows[qid]["ranked_pages"][:5] for cfg, rows in by_qid.items() if qid in rows
                },
            }
        )
    return {"ablation": ablation, "questions": questions}


def cmd_export_explorer(args: argparse.Namespace) -> None:
    data = export_explorer(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {len(data['questions'])} questions to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="filings-rag", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="fetch FinanceBench questions and the PDFs they reference")
    s.add_argument("--data-dir", type=Path, default=DATA)
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("ingest", help="extract pages, chunk, embed")
    s.add_argument("--pdf-dir", type=Path, default=DATA / "pdfs")
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--questions", type=Path, default=QUESTIONS)
    s.add_argument("--chunk-words", type=int, default=250)
    s.add_argument("--overlap", type=int, default=50)
    s.add_argument("--stats-only", action="store_true", help="recompute ingest_stats.json only")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("eval-retrieval", help="retrieval ablation (no LLM calls)")
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--questions", type=Path, default=QUESTIONS)
    s.add_argument("--out", type=Path, default=RESULTS)
    s.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    s.add_argument("--scopes", nargs="+", choices=("doc", "corpus"), default=["doc", "corpus"])
    s.add_argument("--rerankers", nargs="+", default=["minilm", "bge"], help="keys of index.RERANKERS")
    s.add_argument("--markdown-only", action="store_true", help="re-render the .md table from the saved JSON")
    s.set_defaults(func=cmd_eval_retrieval)

    s = sub.add_parser("bootstrap", help="paired bootstrap CIs of hit@5 vs dense retrieval")
    s.add_argument("--out", type=Path, default=RESULTS)
    s.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("eval-generation", help="answer + judge + citation check (DeepSeek, costs money)")
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--questions", type=Path, default=QUESTIONS)
    s.add_argument("--out", type=Path, default=RESULTS)
    s.add_argument("--method", type=method_label, default=None, help="default: best doc-scoped hit@5")
    s.add_argument("--scope", choices=("doc", "corpus"), default="doc")
    s.add_argument("--k", type=int, default=5, help="chunks passed to the model")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--cap", type=float, default=3.0, help="hard spend cap in CNY across runs")
    s.set_defaults(func=cmd_eval_generation)

    s = sub.add_parser("ask", help="answer one question with citations")
    s.add_argument("question")
    s.add_argument("--company", default=None)
    s.add_argument("--doc", default=None, help="restrict to one filing, e.g. 3M_2018_10K")
    s.add_argument("--method", type=method_label, default="dense")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--cap", type=float, default=3.0)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve-mcp", help="run the MCP server over stdio")
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--method", type=method_label, default="dense")
    s.set_defaults(func=cmd_serve_mcp)

    s = sub.add_parser("export-explorer", help="write explorer/explorer.json for the static site")
    s.add_argument("--results", type=Path, default=RESULTS)
    s.add_argument("--out", type=Path, default=Path("explorer") / "explorer.json")
    s.set_defaults(func=cmd_export_explorer)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
