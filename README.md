# filings-rag

Retrieval-augmented question answering over SEC filings (10-K, 10-Q, 8-K, earnings releases),
evaluated on the 150-question open-source sample of [FinanceBench](https://github.com/patronus-ai/financebench).
The project asks two narrow questions: **which retrieval setup actually puts the analyst-labelled
evidence page in front of the model**, and **can a cheap, LLM-free citation check catch answers that
are not supported by the pages they cite?** Everything runs on CPU except answer generation and
judging, which use the DeepSeek API under a hard 3 CNY budget.

Results explorer (static page, every question with retrieved pages, answer, verdict and flags):
https://adleraaa.github.io/filings-rag/

## Results

Short version: plain dense retrieval (`bge-small-en-v1.5`) was the best setup; adding BM25 through
reciprocal rank fusion made it significantly worse, and cross-encoder reranking did not help
significantly. With the right page in context DeepSeek answered 76% of questions correctly
(LLM-judged), without it 22%. The citation check does **not** catch wrong answers: the wrong
answers cite numbers that really are on the cited pages, but pick the wrong line item or draw the
wrong conclusion.

All numbers below are copied from files in `results/`, produced by the commands in
[Reproduce](#quickstart--reproduce). Hardware: laptop, Intel i9-14900HX, CPU only for retrieval
(other jobs were running at the same time, so latencies are rough); DeepSeek API for generation.

### 1. Retrieval: which setup finds the evidence page? (`results/retrieval_ablation.md`)

150 questions. **hit@k** = share of questions with at least one gold evidence page among the top k
distinct pages (chunk rankings are collapsed to pages). **doc-scoped** filters to the question's
filing; **corpus-wide** searches all 84 filings, where a page counts only if it is in the right
filing. `doc_hit@5` = the right filing appears at all in the top 5 pages.

| scope | method | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | doc_hit@5 | median latency |
|---|---|---|---|---|---|---|---|---|
| doc | BM25 | 0.147 | 0.253 | 0.313 | 0.420 | 0.218 | 1.000 | 1 ms |
| doc | **dense** | **0.287** | 0.487 | **0.587** | 0.673 | **0.408** | 1.000 | 18 ms |
| doc | hybrid (RRF) | 0.227 | 0.360 | 0.447 | 0.640 | 0.324 | 1.000 | 24 ms |
| doc | hybrid + MiniLM rerank | 0.233 | 0.473 | 0.540 | 0.620 | 0.361 | 1.000 | 958 ms |
| doc | hybrid + bge-reranker-base | 0.187 | 0.407 | 0.513 | 0.633 | 0.322 | 1.000 | 5946 ms |
| doc | dense + MiniLM rerank | 0.253 | **0.507** | 0.573 | **0.680** | 0.392 | 1.000 | 996 ms |
| corpus | BM25 | 0.073 | 0.093 | 0.113 | 0.153 | 0.091 | 0.607 | 1 ms |
| corpus | **dense** | **0.153** | 0.287 | **0.373** | 0.487 | **0.244** | 0.800 | 19 ms |
| corpus | hybrid (RRF) | 0.107 | 0.173 | 0.227 | 0.333 | 0.162 | 0.753 | 18 ms |
| corpus | hybrid + MiniLM rerank | 0.107 | 0.240 | 0.300 | 0.467 | 0.199 | 0.780 | 995 ms |
| corpus | hybrid + bge-reranker-base | 0.127 | 0.253 | 0.347 | 0.460 | 0.221 | 0.847 | 6285 ms |
| corpus | dense + MiniLM rerank | 0.120 | **0.293** | 0.367 | **0.513** | 0.229 | **0.873** | 1030 ms |

Is the gap real? Paired bootstrap over questions, hit@5 minus dense hit@5, 95% interval
(`results/retrieval_bootstrap.json`):

| config (vs dense, same scope) | doc-scoped | corpus-wide |
|---|---|---|
| BM25 | -0.273 [-0.360, -0.180] | -0.260 [-0.333, -0.187] |
| hybrid (RRF) | -0.140 [-0.207, -0.073] | -0.147 [-0.213, -0.073] |
| hybrid + MiniLM rerank | -0.047 [-0.120, +0.027] | -0.073 [-0.133, -0.013] |
| hybrid + bge-reranker-base | -0.073 [-0.153, +0.007] | -0.027 [-0.093, +0.040] |
| dense + MiniLM rerank | -0.013 [-0.087, +0.053] | -0.007 [-0.067, +0.053] |

What this says:
- BM25 is weak here. FinanceBench questions use analyst vocabulary ("capital expenditure",
  "quick ratio") while the evidence is a financial statement that says "Purchases of property,
  plant and equipment" and is mostly numbers.
- Fusing a weak ranker with a strong one using equal-weight RRF hurts: hybrid is 14 points below
  dense in both scopes, and the interval excludes zero.
- Rerankers recover part of that loss but never beat plain dense by a significant margin. The
  larger `bge-reranker-base` was worse than MiniLM doc-scoped and better corpus-wide, at about 6x
  the CPU time. `dense_rerank` was run with MiniLM only, to save CPU time after `bge-reranker-base`
  gave no gain on the hybrid pools.
- Corpus-wide, a large part of the loss is picking the wrong filing: dense reaches the right filing in
  the top 5 for 80% of questions. FinanceBench has several filings per company (for example 3M 2018,
  2022 and 2023 Q2), and the same number often appears in two years' reports.

Index (`results/ingest_stats.json`): 84 filings, 12,013 pages (65 with no extractable text),
33,998 chunks; median chunk length 308 tokens (bge tokenizer), 1.4% of chunks exceed the 512-token
model window and are truncated. FinanceBench `evidence_page_num` is a 0-based index: the evidence
snippet was found on the labelled page (index + 1) for 157 of 189 evidence items, and on the
neighbouring pages for 7; the remaining 25 did not match because PyMuPDF and the dataset's
extractor break text differently.

### 2. Generation with citations (`results/generation_summary.md`)

Config: dense, doc-scoped, top 8 chunks, `deepseek-v4-flash` (thinking disabled, temperature 0).
The model must cite `[DOC p.N]` and reply "Not found in the provided pages." when the excerpts do
not support an answer. Answers are graded by a second DeepSeek call against the gold answer with a
strict rubric (numbers within 1% or rounding; unit changes allowed).

| subset | n | correct | incorrect | refusal | accuracy |
|---|---|---|---|---|---|
| all | 150 | 83 | 15 | 52 | 0.553 |
| metrics-generated | 50 | 38 | 2 | 10 | 0.760 |
| novel-generated | 50 | 27 | 6 | 17 | 0.540 |
| domain-relevant | 50 | 18 | 7 | 25 | 0.360 |
| gold page among the 8 chunks | 92 | 70 | 9 | 13 | 0.761 |
| gold page not retrieved | 58 | 13 | 6 | 39 | 0.224 |

The model refuses much more often than it hallucinates (52 refusals vs 15 wrong answers). Most
refusals (39 of 52) happen when retrieval missed the gold page, so retrieval is the bottleneck.
Some answers are correct without the labelled page because another page repeats the figure: 3M's
2018 capital spending ($1,577M) is labelled on the cash-flow statement (p.60) but also appears in
the MD&A on p.39, which is what the model cited.

**Caveat on accuracy:** verdicts come from an LLM judge (the same model family that wrote the
answers) and were not checked by a human. Read them as a relative signal. Every verdict and its
one-line reason is in `results/generation.jsonl` and in the explorer.

### 3. Does a cheap citation check catch unsupported answers?

The check (`citation_check.py`, no LLM) flags a non-refusal answer if it has no citation, cites a
page that was not in its context, or states a number that cannot be traced to a cited page
(rounding and thousand/million/percent scaling allowed). Two variants: **verbatim** (every number
must be on the page) and **arithmetic** (a number may also be one arithmetic step, applied up to
twice, from numbers already traced: sum, difference, product, ratio, average, relative change).

| on 96 non-refusal answers | verbatim | arithmetic |
|---|---|---|
| flagged | 35 (36%) | 3 (3%) |
| judge accuracy of flagged answers | 0.914 | 1.000 |
| judge accuracy of unflagged answers | 0.820 | 0.850 |
| AUROC, flag predicts a wrong answer | 0.41 | 0.48 |
| AUROC, low answer-to-cited-chunk cosine similarity predicts a wrong answer | 0.43 | |

Answer: **no, not on this data.** Every verbatim flag came from an unsupported number, and 32 of the 35
flagged answers were judged correct: they were computed ratios and averages, which the verbatim
rule cannot trace. The arithmetic variant removes those false alarms but still catches no wrong
answers. Reading the 15 wrong answers (all in the explorer) shows why: their numbers are real and on the
cited page, but the answer uses the wrong line item (segment store counts instead of the total),
leaves out part of what was asked, or reaches the wrong conclusion. No citation was missing or pointed outside the
context. A check that only asks "is this number on the page?" cannot see those errors. Embedding
similarity between the answer and its cited chunks did no better than chance either (AUROC 0.43).

Spend (`results/spend.json`, all calls ever made by this repo): 311 API calls (150 answers, 150
judgements, a smoke test, a 4-question pilot and one `ask` demo), 536,611 prompt tokens and 18,183
completion tokens, **1.17 CNY** at DeepSeek's peak list price (an upper bound; off-peak is half
price).

## Architecture

```
PDFs (84 filings) --PyMuPDF--> pages (1-based) --250-word windows, 50 overlap--> chunks (doc, page)
                                                                                   |
               +--------------------------+----------------------------------------+
               |                          |
        BM25 (own inverted index)   bge-small-en-v1.5 embeddings (cosine)
               |                          |
               +---- reciprocal rank fusion (k=60) ----+
                                                       |
         optional cross-encoder rerank of the top 30 (of hybrid or dense; MiniLM or bge-reranker-base)
                                                       |
                         top-k chunks, labelled [DOC p.N] --> DeepSeek answer with citations
                                                       |
                        +------------------------------+------------------------------+
                        |                                                             |
          LLM judge vs gold answer (DeepSeek)                citation check (no LLM): citations present,
                                                             cited pages in context, every number found
                                                             on a cited page (rounding/unit aware)
```

Modules (`src/filings_rag/`): `data` (questions, download), `ingest` (PDF to chunks), `retrieval`
(BM25, dense, RRF, rerank), `metrics`, `eval_retrieval`, `generate`, `judge`, `citation_check`,
`llm` (DeepSeek client, spend cap), `eval_generation`, `bootstrap`, `mcp_server`, `cli`.

## Quickstart / reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml,dev]"

filings-rag download          # FinanceBench questions + the 84 PDFs they reference (~160 MB) into data/
filings-rag ingest            # pages, chunks, embeddings -> data/index/, stats -> results/ingest_stats.json
filings-rag eval-retrieval --methods bm25 dense hybrid hybrid_rerank   # both rerankers, both scopes
filings-rag eval-retrieval --methods dense_rerank --rerankers minilm    # merged into the same files
filings-rag bootstrap         # paired bootstrap CIs -> results/retrieval_bootstrap.json
echo "DEEPSEEK_API_KEY=..." > .env
filings-rag eval-generation --k 8   # best doc-scoped config (dense) -> results/generation_*, results/spend.json
filings-rag export-explorer   # explorer/explorer.json for the static page

filings-rag ask "What was 3M's FY2018 capital expenditure?" --company 3M   # default method: dense
pytest && ruff check .        # 38 offline tests; no models, data or API key needed
```

MCP server (stdio), for Claude Desktop or any MCP client:

```json
{"mcpServers": {"filings": {"command": "filings-rag", "args": ["serve-mcp"], "cwd": "/path/to/filings-rag"}}}
```

It exposes `search_filings(query, company=None, k=5)` and `get_page(doc, page)`.
`python scripts/mcp_smoke.py` starts the server as a subprocess and calls both tools over stdio.

## Design decisions

- **Page-level evaluation, chunks that never cross a page.** FinanceBench labels evidence pages, and
  a citation must point at one page. Chunks are 250-word windows with 50 words of overlap inside a
  page, and chunk rankings are collapsed to distinct pages before scoring, so several chunks from
  one page are not counted as several hits.
- **One code path for doc-scoped and corpus-wide search.** Every retriever scores all chunks and
  then applies a document mask. The two scopes differ only in the mask, so their numbers are
  directly comparable.
- **BM25 written by hand (about 40 lines, inverted index + numpy).** It is small enough to explain
  line by line and fast enough (1 ms per query over 34k chunks) that it was never the bottleneck.
- **Reciprocal rank fusion instead of score blending.** RRF only uses ranks, so unbounded BM25
  scores and cosine similarities need no normalisation. The results show its weakness too: equal
  weights let a weak ranker drag down a strong one.
- **Hard budget in code, not in a spreadsheet.** `SpendTracker` refuses any call whose worst case
  (all input uncached, `max_tokens` output) would cross the cap. It resumes from `results/spend.json`,
  so the cap covers all runs together. Generation results are appended per question and a
  re-run skips finished questions, so a crash never pays twice, and re-running
  `eval-generation` after a completed run recomputes the checks with zero API calls.
- **Models behind small interfaces (`Embedder`, `Reranker`, `ChatModel`).** Tests plug in a hashed
  bag-of-words embedder, a word-overlap reranker and a scripted LLM, so the 38 tests and CI need no
  PyTorch, no model downloads, no dataset and no API key.

## Limitations

- 150 questions is small. The bootstrap intervals are 12 to 18 points wide; apart from dense vs
  BM25 and dense vs RRF hybrid, only corpus-wide hybrid + MiniLM is (barely) below dense.
- Accuracy comes from an LLM judge from the same provider as the answering model, with no human
  check. It may share the answering model's blind spots.
- Page labels are strict: a correct figure on a different page than the labelled one counts as a
  retrieval miss (see the 3M example above), so hit@k understates how often usable evidence was
  retrieved.
- Only one embedding model, one chunk size and one RRF constant were tried, none tuned. There is
  no separate dev set: the generation config was chosen by its retrieval score on the same 150
  questions. Tables are extracted as flat text, which loses row/column structure.
- Generation was run on one configuration (dense, doc-scoped, 8 chunks). Corpus-wide generation
  and the effect of `k` on accuracy were not measured.
- The citation check only reasons about numbers. It cannot detect a wrong line item, a wrong period
  or a wrong conclusion, which is where this system's errors actually were.
- Latencies were measured on a shared laptop CPU while other jobs ran; treat them as orders of
  magnitude.

## Data and license

- FinanceBench open-source sample, Patronus AI, https://github.com/patronus-ai/financebench
  (paper: Islam et al., 2023, arXiv:2311.11944). The GitHub repo has no license file; the
  Hugging Face dataset card (https://huggingface.co/datasets/PatronusAI/financebench) lists
  **CC BY-NC 4.0**. This repo does not commit the questions or PDFs; `filings-rag download` fetches
  them. Files under `results/` and `explorer/explorer.json` quote FinanceBench questions and gold
  answers for non-commercial evaluation purposes, with this attribution.
- Models: `BAAI/bge-small-en-v1.5` (MIT), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0),
  `BAAI/bge-reranker-base` (MIT), downloaded from Hugging Face at run time.
- Code: MIT, Copyright (c) 2026 Yunlong Lu. See `LICENSE`.
