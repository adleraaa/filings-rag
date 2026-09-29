import asyncio
import json

import pytest

from filings_rag.cli import best_config, build_parser
from filings_rag.data import Question
from filings_rag.eval_generation import add_similarity, auroc, run_generation, summarize, to_markdown
from filings_rag.eval_retrieval import run_ablation
from filings_rag.llm import BudgetExceeded, SpendTracker
from filings_rag.mcp_server import build_server, docs_for_company

from .conftest import FakeLLM, HashEmbedder

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
    assert len(summaries) == 8
    by_key = {(s["scope"], s["method"]): s for s in summaries}
    assert by_key[("doc", "bm25")]["hit@10"] == 1.0  # a 3-page filing: top 10 covers everything
    assert json.loads((tmp_path / "retrieval_ablation.json").read_text())["configs"]
    assert "| doc | hybrid_rerank |" in (tmp_path / "retrieval_ablation.md").read_text()
    assert best_config(tmp_path, "doc") in {"bm25", "dense", "hybrid", "hybrid_rerank"}


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
    assert [r["flagged"] for r in rows] == [False, True]  # 9.99 is not on the cited page
    assert all(r["gold_in_context"] for r in rows)

    # A second run must not call the models again.
    again = FakeLLM(["should not be used"])
    run_generation(index, QUESTIONS, again, again, out, "hybrid_rerank", "doc", k=3)
    assert again.prompts == []

    add_similarity(rows, index, HashEmbedder())
    assert all(0 < r["similarity"] <= 1 for r in rows)
    s = summarize(rows)
    assert s["overall"]["accuracy"] == 0.5
    assert s["citation_check"]["flagged_judge"]["accuracy"] == 0.0
    assert s["citation_check"]["unflagged_judge"]["accuracy"] == 1.0
    assert s["citation_check"]["auroc_flag_predicts_wrong"] == 1.0
    assert "| all | 2 | 1 | 1 | 0 | 0.500 |" in to_markdown(s)


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
    assert call("search_filings", {"query": "x", "company": "Initech"}) == []

    # A plain dict return comes back as one JSON text block.
    blocks = asyncio.run(server.call_tool("get_page", {"doc": "ACME_2022_10K", "page": 2}))
    page = json.loads(blocks[0].text)
    assert page["page"] == 2 and "1,577" in page["text"]
    with pytest.raises(Exception, match="no page 99"):
        asyncio.run(server.call_tool("get_page", {"doc": "ACME_2022_10K", "page": 99}))


def test_company_matching(index):
    assert docs_for_company(index, "acme") == ["ACME_2022_10K"]
    assert docs_for_company(index, "Acme Corp") == []


def test_cli_parser():
    args = build_parser().parse_args(["ask", "What was capex?", "--company", "3M", "--k", "3"])
    assert args.company == "3M" and args.k == 3 and args.method == "hybrid_rerank"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval-generation", "--method", "magic"])
