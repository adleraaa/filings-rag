import argparse
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from filings_rag.cli import best_config, build_parser, evidence_page_check, export_explorer, parse_method
from filings_rag.data import Question
from filings_rag.eval_generation import (
    add_null_baseline,
    add_similarity,
    apply_checks,
    auroc,
    run_generation,
    summarize,
    to_markdown,
)
from filings_rag.eval_retrieval import run_ablation
from filings_rag.index import Index
from filings_rag.ingest import Page, chunk_pages
from filings_rag.llm import BudgetExceeded, DeepSeekChat, SpendTracker
from filings_rag.mcp_server import build_server, docs_for_company
from filings_rag.retrieval import METHODS, Retriever

from .conftest import FakeLLM, HashEmbedder, OverlapReranker


class ReverseReranker:
    def score(self, query, texts):
        return -OverlapReranker().score(query, texts)


QUESTIONS = [
    Question(
        "q1",
        "Acme",
        "ACME_2022_10K",
        "What were purchases of property, plant and equipment in 2022?",
        "$1577 million",
        "metrics-generated",
        (2,),
    ),
    Question(
        "q2",
        "Globex",
        "GLOBEX_2021_10K",
        "What dividends per share were declared?",
        "$2.40",
        "domain-relevant",
        (2,),
    ),
]


def test_spend_tracker_cost_and_cap(tmp_path):
    t = SpendTracker(cap_cny=0.01, path=tmp_path / "spend.json")
    t.record({"prompt_tokens": 1000, "prompt_cache_hit_tokens": 400, "completion_tokens": 100}, "answer")
    # 400 hits * 0.04 + 600 misses * 2 + 100 out * 8, per million tokens
    assert t.cost_cny == pytest.approx((400 * 0.04 + 600 * 2 + 100 * 8) / 1e6)
    with pytest.raises(BudgetExceeded):
        t.check(est_prompt_tokens=1000, max_tokens=2000)  # worst case 0.018 CNY > 0.01 cap
    t.check(est_prompt_tokens=100, max_tokens=100)
    resumed = SpendTracker.load(tmp_path / "spend.json", cap_cny=0.01)
    assert resumed.cost_cny == pytest.approx(t.cost_cny)
    assert resumed.by_stage["answer"]["prompt_tokens"] == 1000


def test_eval_retrieval_writes_results(index, tmp_path):
    summaries = run_ablation(index.retriever, QUESTIONS, tmp_path)
    assert len(summaries) == 10  # 5 methods x 2 scopes
    by_key = {(s["scope"], s["method"]): s for s in summaries}
    assert by_key[("doc", "bm25")]["hit@10"] == 1.0  # a 3-page filing: top 10 covers everything
    assert by_key[("doc", "dense")]["doc_hit@5"] == 1.0  # doc-scoped can only return the right filing
    assert json.loads((tmp_path / "retrieval_ablation.json").read_text())["configs"]
    md = (tmp_path / "retrieval_ablation.md").read_text()
    assert "| doc | hybrid_rerank |" in md
    assert all(f"recall@{k}" in md for k in (1, 3, 5, 10))


def test_best_config_uses_hit5_then_mrr_within_scope(tmp_path):
    # On the tiny fixture every method ties, so test the choice on explicit numbers.
    configs = [
        {"scope": "doc", "method": "bm25", "hit@5": 0.3, "mrr@10": 0.9},
        {"scope": "doc", "method": "dense", "hit@5": 0.6, "mrr@10": 0.4},
        {"scope": "doc", "method": "hybrid", "hit@5": 0.6, "mrr@10": 0.5},
        {"scope": "corpus", "method": "dense_rerank", "hit@5": 0.9, "mrr@10": 0.9},
    ]
    (tmp_path / "retrieval_ablation.json").write_text(json.dumps({"configs": configs}))
    assert best_config(tmp_path, "doc") == "hybrid"
    assert best_config(tmp_path, "corpus") == "dense_rerank"


def test_generation_run_resumes_and_summarises(index, tmp_path):
    answers = FakeLLM(
        [
            "Purchases were $1,577 million [ACME_2022_10K p.2].",
            "Dividends were $9.99 per share [GLOBEX_2021_10K p.2].",
        ]
    )
    judge = FakeLLM(['{"verdict": "correct", "reason": "ok"}', '{"verdict": "incorrect", "reason": "no"}'])
    out = tmp_path / "gen.jsonl"
    rows = run_generation(index, QUESTIONS, answers, judge, out, "hybrid_rerank", "doc", k=3)
    assert [r["verdict"] for r in rows] == ["correct", "incorrect"]
    assert all(r["gold_in_context"] for r in rows)
    apply_checks(rows, index)
    for variant in ("verbatim", "arithmetic", "arithmetic_loose"):
        # 9.99 is not on the cited page and cannot be derived from 2.40 alone.
        assert [r["checks"][variant]["flagged"] for r in rows] == [False, True]

    # A second run must not call the models again.
    again = FakeLLM(["should not be used"])
    run_generation(index, QUESTIONS, again, again, out, "hybrid_rerank", "doc", k=3)
    assert again.prompts == []
    # ...and resuming with another config must fail instead of mixing results.
    with pytest.raises(ValueError, match="k_chunks"):
        run_generation(index, QUESTIONS, again, again, out, "hybrid_rerank", "doc", k=8)
    with pytest.raises(ValueError):
        run_generation(index, QUESTIONS, again, again, out, "dense", "doc", k=3)
    assert again.prompts == []

    add_similarity(rows, index, HashEmbedder())
    assert all(0 < r["similarity"] <= 1 for r in rows)
    s = summarize(rows)
    assert s["overall"]["accuracy"] == 0.5
    assert s["status_vs_verdict"]["answered"] == {"correct": 1, "incorrect": 1, "refusal": 0}
    cc = s["citation_check"]["arithmetic"]
    assert cc["flagged_judge"]["accuracy"] == 0.0
    assert cc["unflagged_judge"]["accuracy"] == 1.0
    assert cc["wrong_flagged"] == 1 and cc["wrong_unflagged"] == 0
    assert cc["auroc_flag_predicts_wrong"] == 1.0
    assert cc["accuracy_gap_flagged_minus_unflagged"] == -1.0
    add_null_baseline(s, rows, index.page_text)
    # 1,577 x 1.1 = 1,735 is not on the page, so the check rejects it.
    assert s["citation_check"]["verbatim"]["null_baseline"]["x1.1"]["accepted"] == 0
    md = to_markdown(s)
    assert "| all | 2 | 1 | 1 | 0 | 0.500 |" in md
    assert "| flagged | 1 (0.5) | 1 (0.5) | 1 (0.5) |" in md
    assert "false-accept rate, x1.1, all numbers" in md


def test_export_explorer_joins_retrieval_and_generation(index, tmp_path):
    run_ablation(index.retriever, QUESTIONS, tmp_path)
    # A merged rerun may list questions in another order: the join must use qids.
    runs = json.loads((tmp_path / "retrieval_per_question.json").read_text())
    runs["doc/dense"].reverse()
    (tmp_path / "retrieval_per_question.json").write_text(json.dumps(runs))
    answers = FakeLLM(["$1,577 million [ACME_2022_10K p.2].", "Not found in the provided pages."])
    judge = FakeLLM(['{"verdict": "correct", "reason": "ok"}', '{"verdict": "refusal", "reason": "-"}'])
    rows = run_generation(index, QUESTIONS, answers, judge, tmp_path / "g.jsonl", "bm25", "doc", k=2)
    apply_checks(rows, index)
    (tmp_path / "generation_with_similarity.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = export_explorer(tmp_path)
    assert [q["qid"] for q in data["questions"]] == ["q1", "q2"]
    q2 = data["questions"][1]
    assert q2["verdict"] == "refusal" and q2["doc"] == "GLOBEX_2021_10K"
    assert set(q2["retrieval"]) == {f"{s}/{m}" for s in ("doc", "corpus") for m in METHODS}
    assert all(len(pages) <= 5 for pages in q2["retrieval"].values())
    assert q2["retrieval"]["doc/dense"] == runs["doc/dense"][0]["ranked_pages"][:5]
    assert all(doc == "GLOBEX_2021_10K" for doc, _ in q2["retrieval"]["doc/dense"])


def test_auroc():
    assert auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([1.0], [True]) is None


def test_mcp_tools_in_process(index):
    server = build_server(index, method="hybrid")
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert tools == {"search_filings", "get_page"}

    def call(name, args):
        result = asyncio.run(server.call_tool(name, args))
        # FastMCP returns (content blocks, structured output) for typed tools.
        return result[1]["result"] if isinstance(result, tuple) else result

    hits = call("search_filings", {"query": "dividends per share", "company": "Globex", "k": 2})
    assert hits and all(h["doc"] == "GLOBEX_2021_10K" for h in hits)
    # An unknown company is an error that names the companies we do have.
    with pytest.raises(Exception, match="known: ACME, GLOBEX"):
        call("search_filings", {"query": "x", "company": "Initech"})

    # A plain dict return comes back as one JSON text block.
    blocks = asyncio.run(server.call_tool("get_page", {"doc": "ACME_2022_10K", "page": 2}))
    page = json.loads(blocks[0].text)
    assert page["page"] == 2 and "1,577" in page["text"]
    with pytest.raises(Exception, match="no page 99"):
        asyncio.run(server.call_tool("get_page", {"doc": "ACME_2022_10K", "page": 99}))


def test_company_matching(index):
    assert docs_for_company(index, "acme") == ["ACME_2022_10K"]
    assert docs_for_company(index, "Acme Corp") == ["ACME_2022_10K"]
    assert docs_for_company(index, "Ac") == []  # no prefix matching
    assert docs_for_company(index, "") == []


def test_company_matching_multi_token_names():
    # Real FinanceBench filing names that the first-token rule used to miss.
    pages = [
        Page("JOHNSON_JOHNSON_2022_10K", 1, "Net earnings 17,941"),
        Page("JOHNSON_JOHNSON_2023Q2_10Q", 1, "Sales"),
        Page("AES_2022_10K", 1, "Revenue"),
        Page("COCACOLA_2022_10K", 1, "Revenue"),
    ]
    chunks = chunk_pages(pages, size=40, overlap=10)
    emb = HashEmbedder()
    retriever = Retriever(chunks, emb.encode([c.text for c in chunks], is_query=False), emb, None)
    idx = Index({(p.doc, p.page): p for p in pages}, retriever)
    jnj = ["JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2023Q2_10Q"]
    assert docs_for_company(idx, "Johnson & Johnson") == jnj
    assert docs_for_company(idx, "AES Corporation") == ["AES_2022_10K"]
    assert docs_for_company(idx, "The Coca-Cola Company") == ["COCACOLA_2022_10K"]


def test_evidence_page_check_counts_each_page_and_misses(tmp_path):
    filler = "x" * 60  # the check compares characters 50-150 of the normalised evidence text
    evidence = [
        {
            "doc_name": "D",
            "evidence_page_num": 0,
            "evidence_text_full_page": filler + " capex was 1,577 million",
        },
        {"doc_name": "D", "evidence_page_num": 1, "evidence_text_full_page": filler + " dividends were 2.40"},
        {
            "doc_name": "D",
            "evidence_page_num": 0,
            "evidence_text_full_page": filler + " text nobody extracted",
        },
    ]
    qfile = tmp_path / "q.jsonl"
    qfile.write_text(json.dumps({"evidence": evidence}) + "\n")
    pages = {
        ("D", 1): filler + " Capex was 1,577   million",  # 0-based index 0 -> page 1
        ("D", 2): filler + " capex was 1,577 million",  # the same text repeated on page 2
        ("D", 3): filler + " Dividends were 2.40",  # labelled page 2, found one page later
    }
    out = evidence_page_check(qfile, pages)
    assert out["evidence_items"] == 3
    assert out["snippet_found_on"] == {"labelled_page": 1, "page_before": 0, "page_after": 2}
    assert out["found_on_none_of_the_three"] == 1
    assert out["of_which_on_labelled_page_ignoring_spaces_and_punctuation"] == 0

    # Same words, different punctuation/spacing from another PDF extractor.
    pages[("D", 1)] += " " + filler + " text-nobody  extracted"
    assert evidence_page_check(qfile, pages)["of_which_on_labelled_page_ignoring_spaces_and_punctuation"] == 1


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 10})
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def test_deepseek_chat_checks_budget_before_calling_the_api(tmp_path):
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    tracker = SpendTracker(cap_cny=0.001, path=tmp_path / "spend.json")
    chat = DeepSeekChat(tracker, "answer", client=client)
    assert chat.complete("system", "user", max_tokens=10) == "ok"
    assert completions.calls == 1 and tracker.calls == 1
    assert json.loads((tmp_path / "spend.json").read_text())["calls"] == 1
    with pytest.raises(BudgetExceeded):
        chat.complete("system", "user " * 1000, max_tokens=1000)  # worst case about 0.011 CNY > cap
    assert completions.calls == 1  # the over-budget request never reached the client


def test_cli_parser():
    args = build_parser().parse_args(["ask", "What was capex?", "--company", "3M", "--k", "3"])
    assert args.company == "3M" and args.k == 3 and args.method == "dense"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval-generation", "--method", "magic"])


def test_parse_method_labels():
    assert parse_method("bm25") == ("bm25", None)
    assert parse_method("hybrid_rerank") == ("hybrid_rerank", "minilm")
    assert parse_method("hybrid_rerank[bge]") == ("hybrid_rerank", "bge")
    for bad in ("magic", "hybrid_rerank[gpt]", "bm25 "):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_method(bad)


def test_ablation_runs_each_named_reranker(index, tmp_path):
    rerankers = {"overlap": OverlapReranker(), "reverse": ReverseReranker()}
    summaries = run_ablation(index.retriever, QUESTIONS, tmp_path, ["hybrid_rerank"], ["doc"], rerankers)
    assert [s["method"] for s in summaries] == ["hybrid_rerank[overlap]", "hybrid_rerank[reverse]"]
    # The rerankers disagree, so the two runs must not be identical.
    assert summaries[0]["mrr@10"] != summaries[1]["mrr@10"]


def test_ablation_subset_rerun_merges_into_existing_results(index, tmp_path):
    run_ablation(index.retriever, QUESTIONS, tmp_path, ["bm25", "dense"], ["doc"])
    run_ablation(index.retriever, QUESTIONS, tmp_path, ["dense_rerank"], ["doc", "corpus"])
    saved = json.loads((tmp_path / "retrieval_ablation.json").read_text())["configs"]
    assert [(s["scope"], s["method"]) for s in saved] == [
        ("doc", "bm25"),
        ("doc", "dense"),
        ("doc", "dense_rerank"),
        ("corpus", "dense_rerank"),
    ]
    runs = json.loads((tmp_path / "retrieval_per_question.json").read_text())
    assert set(runs) == {"doc/bm25", "doc/dense", "doc/dense_rerank", "corpus/dense_rerank"}


def test_paired_bootstrap():
    from filings_rag.bootstrap import compare_to_reference, paired_bootstrap

    a = np.array([1.0] * 50 + [0.0] * 50)
    mean, lo, hi = paired_bootstrap(a, a)
    assert mean == lo == hi == 0.0
    better = np.ones(100)
    mean, lo, hi = paired_bootstrap(better, a)
    assert mean == 0.5 and 0.35 < lo < 0.5 < hi < 0.65
    runs = {
        "doc/dense": [{"hit@5": x} for x in a],
        "doc/bm25": [{"hit@5": 0.0} for _ in a],
        "corpus/dense": [{"hit@5": 0.0} for _ in a],
    }
    rows = compare_to_reference(runs, "doc/dense")
    assert [r["config"] for r in rows] == ["doc/bm25"]  # other scopes are not compared
    assert rows[0]["diff"] == -0.5 and rows[0]["significant"]
