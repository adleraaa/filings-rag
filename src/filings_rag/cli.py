"""Command-line entry point: filings-rag <command> [options]."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from .data import download, load_questions
from .retrieval import METHODS

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

    counts = {"labelled_page": 0, "page_before": 0, "page_after": 0}
    total = 0
    for line in qfile.open(encoding="utf-8"):
        for ev in json.loads(line)["evidence"]:
            snippet = norm(ev["evidence_text_full_page"])[50:150]
            page = ev["evidence_page_num"] + 1
            total += 1
            for name, p in (("labelled_page", page), ("page_before", page - 1), ("page_after", page + 1)):
                counts[name] += snippet in norm(page_text.get((ev["doc_name"], p), ""))
    return {"evidence_items": total, "snippet_found_on": counts}


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    from .eval_retrieval import run_ablation
    from .index import RERANKERS, load_index
    from .retrieval import CrossEncoderReranker

    index = load_index(args.index_dir, reranker=None)
    rerankers = {name: CrossEncoderReranker(model) for name, model in RERANKERS.items()}
    questions = load_questions(args.questions)
    run_ablation(index.retriever, questions, args.out, rerankers=rerankers)


def parse_method(label: str) -> tuple[str, str | None]:
    """ "hybrid_rerank[bge]" -> ("hybrid_rerank", "bge"); "bm25" -> ("bm25", None)."""
    from .index import DEFAULT_RERANKER, RERANKERS

    m = re.fullmatch(r"(\w+)(?:\[(\w+)\])?", label)
    if not m or m.group(1) not in METHODS or (m.group(2) and m.group(2) not in RERANKERS):
        raise argparse.ArgumentTypeError(f"unknown method {label!r}")
    method, reranker = m.groups()
    if method == "hybrid_rerank":
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
    from .eval_generation import add_similarity, run_generation, summarize, to_markdown
    from .index import load_index
    from .llm import DeepSeekChat, SpendTracker

    label = args.method or best_config(args.out, args.scope)
    method, reranker = parse_method(label)
    print(f"generation with method={label} scope={args.scope} k={args.k}")
    index = load_index(args.index_dir, reranker=reranker)
    tracker = SpendTracker.load(args.out / "spend.json", cap_cny=args.cap)
    questions = load_questions(args.questions)[: args.limit]
    rows = run_generation(
        index,
        questions,
        DeepSeekChat(tracker, "answer"),
        DeepSeekChat(tracker, "judge"),
        args.out / "generation.jsonl",
        method,
        args.scope,
        args.k,
    )
    add_similarity(rows, index, index.retriever.embedder)
    summary = summarize(rows)
    summary["config"] = {"method": label, "scope": args.scope, "k_chunks": args.k}
    summary["spend"] = tracker.to_dict()
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
    from .mcp_server import docs_for_company

    method, reranker = parse_method(args.method)
    index = load_index(args.index_dir, reranker=reranker)
    docs = [args.doc] if args.doc else (docs_for_company(index, args.company) if args.company else None)
    hits = index.retriever.search(args.question, method=method, k=args.k, docs=docs)
    tracker = SpendTracker.load(RESULTS / "spend.json", cap_cny=args.cap)
    ans = answer_question(DeepSeekChat(tracker, "ask"), args.question, hits)
    check = check_answer(ans, index.page_text)
    print(ans.text)
    print(f"\ncontext pages: {', '.join(f'{d} p.{p}' for d, p in ans.context_pages)}")
    print(f"citation check: {'FLAGGED ' + ', '.join(check.reasons) if check.flagged else 'ok'}")


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
    questions = []
    for i, first in enumerate(runs["doc/bm25"]):
        g = generated.get(first["qid"], {})
        questions.append(
            {
                "qid": first["qid"],
                "question_type": first["question_type"],
                "doc": first["gold"][0][0],
                "gold_pages": first["gold"],
                "question": g.get("question", ""),
                "gold_answer": g.get("gold_answer", ""),
                "answer": g.get("answer", ""),
                "verdict": g.get("verdict", "not run"),
                "judge_reason": g.get("judge_reason", ""),
                "flagged": g.get("flagged", False),
                "flag_reasons": g.get("flag_reasons", []),
                # Top-5 pages of every config for this question (runs are in question order).
                "retrieval": {cfg: rows[i]["ranked_pages"][:5] for cfg, rows in runs.items()},
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
    s.set_defaults(func=cmd_eval_retrieval)

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
    s.add_argument("--method", type=method_label, default="hybrid_rerank[minilm]")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--cap", type=float, default=3.0)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve-mcp", help="run the MCP server over stdio")
    s.add_argument("--index-dir", type=Path, default=INDEX)
    s.add_argument("--method", type=method_label, default="hybrid_rerank[minilm]")
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
