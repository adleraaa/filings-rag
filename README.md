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
(LLM-judged), without it 22%. The citation check does **not** catch wrong answers: most wrong
answers cite numbers that really are on the cited pages, but pick the wrong line item or draw the
wrong conclusion. A corruption test shows what the check can do: it rejects 97-99% of corrupted
large numbers (100 and up) but still accepts 33-44% of corrupted small ones (percentages,
ratios, per-share values).

All numbers below are copied from files in `results/`, produced by the commands in
[Reproduce](#quickstart--reproduce). Hardware: laptop, Intel i9-14900HX, CPU only for retrieval
(other jobs were running at the same time, so latencies are rough); DeepSeek API for generation.

### 1. Retrieval: which setup finds the evidence page? (`results/retrieval_ablation.md`)

150 questions. Each query retrieves the top 30 chunks, which are collapsed to their first 10
distinct pages. **hit@k** = share of questions with at least one gold evidence page among the top
k pages. **recall@k** = share of a question's gold pages found in the top k, averaged over
questions; it is lower than hit@k because 35 questions have more than one gold page (dense,
doc-scoped: recall@5 0.540 vs hit@5 0.587). Recall@1/3/5/10 for every config are in
`results/retrieval_ablation.md`; the table below shows hit@k. **doc-scoped** filters to the
question's filing; **corpus-wide** searches all 84 filings, where a page counts only if it is in
the right filing. `doc_hit@5` = the right filing appears at all in the top 5 pages.

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
model window and are truncated. FinanceBench `evidence_page_num` is a 0-based index: a 100-character
evidence snippet was found verbatim on the labelled page (index + 1) for 157 of 189 evidence items.
It was also found on the page before (2) or after (5), but only for items that were also on the
labelled page (repeated text). The other 32 items were found on none of the three pages; all 32
match the labelled page once spaces and punctuation are ignored, so the two PDF text extractors
differ in layout, not in page numbering.

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

The columns are the judge's verdicts. The model refuses much more often than it answers wrongly
(52 judged refusals vs 15 judged incorrect). Most refusals (39 of 52) happen when retrieval missed
the gold page, so retrieval is the bottleneck. Some answers are correct without the labelled
page because another page repeats the figure: 3M's 2018 capital spending ($1,577M) is labelled
on the cash-flow statement (p.60) but also appears in the MD&A on p.39, which is what the model
cited.

The model often hedges: it states some figures with citations and then adds the "Not found"
sentence for the rest. A rule (`Answer.status`) labels an answer **partial** when it contains that
sentence and also states a number (years excluded), and **refusal** when it contains the sentence
and no number. Rule vs judge (`status_vs_verdict` in `results/generation_summary.json`):

| rule status | judged correct | judged incorrect | judged refusal |
|---|---|---|---|
| answered (96) | 82 | 13 | 1 |
| partial (7) | 1 | 2 | 4 |
| refusal (47) | 0 | 0 | 47 |

The rule and the judge disagree on whether 5 answers are refusals: the judge calls 1 answered
text a refusal (it explains why a quick ratio cannot be computed) and 4 of the 7 partial answers;
it grades the other 3 partial answers 1 correct and 2 incorrect. Before this split, all 7 partial
answers were counted as refusals and skipped by the citation check.

**Caveat on accuracy:** verdicts come from an LLM judge (the same model family that wrote the
answers) and were not checked by a human. Read them as a relative signal. Every verdict and its
one-line reason is in `results/generation.jsonl` and in the explorer.

### 3. Does a cheap citation check catch unsupported answers?

The check (`citation_check.py`, no LLM) flags an answered or partial answer if it has no
citation, cites a page that was not in its context, or states a number that cannot be traced to a
cited page (rounding and thousand/million/percent scaling allowed). Modes:

- **verbatim**: every number must be on a cited page.
- **arithmetic**: a number may also be one difference, ratio or relative change (optionally as a
  percentage) of two numbers the answer already traced to the page.
- **arithmetic_loose**: the first version of the arithmetic mode, kept only for comparison. It also
  allowed sums, products and averages of any traced pair, applied twice, with thousand/million
  rescaling of the derived values, so each answer produced thousands of candidate values.

"Wrong" below means *not judged correct* (judged incorrect, or judged a refusal although the text
states numbers): 20 of the 103 checked answers (15 incorrect, 5 judged refusal). Intervals are 95%
paired bootstrap intervals over answers (2,000 resamples).

| on 103 answered or partial answers | verbatim | arithmetic | arithmetic_loose |
|---|---|---|---|
| flagged | 36 (35%) | 13 (13%) | 3 (3%) |
| wrong answers flagged / not flagged | 4 / 16 | 1 / 19 | 0 / 20 |
| judge accuracy, flagged | 0.889 | 0.923 | 1.000 |
| judge accuracy, not flagged | 0.761 | 0.789 | 0.800 |
| accuracy gap, flagged minus not flagged | +0.13 [-0.02, +0.27] | +0.13 [-0.07, +0.27] | +0.20 [+0.12, +0.28] |
| AUROC, flag predicts a wrong answer | 0.41 [0.31, 0.52] | 0.45 [0.40, 0.52] | 0.48 [0.46, 0.50] |

**Is the check tight enough to catch a wrong number at all? (false-accept baseline,
`null_baseline.py`)** For every number a mode accepted in the real answers, one at a time, the
number is corrupted and the check re-run. Share of corrupted numbers still accepted:

| corruption | numbers | verbatim | arithmetic | arithmetic_loose |
|---|---|---|---|---|
| x1.1 (10% too high) | all | 0.13 (60/453) | 0.14 (70/495) | 0.16 (83/521) |
| | below 100 | 0.37 (57/155) | 0.35 (65/186) | 0.41 (77/190) |
| | 100 and up | 0.010 (3/298) | 0.016 (5/309) | 0.018 (6/331) |
| adjacent digits swapped | all | 0.11 (43/406) | 0.13 (56/447) | 0.15 (69/473) |
| | below 100 | 0.34 (40/117) | 0.33 (49/148) | 0.39 (60/152) |
| | 100 and up | 0.010 (3/289) | 0.023 (7/299) | 0.028 (9/321) |
| random, same digit count | all | 0.17 (82/484) | 0.17 (88/531) | 0.18 (101/555) |
| | below 100 | 0.41 (77/186) | 0.39 (86/222) | 0.44 (98/224) |
| | 100 and up | 0.017 (5/298) | 0.007 (2/309) | 0.009 (3/331) |

Answer: **no, not on this data, and the check is only meaningful for large numbers.**

- For dollar amounts and other numbers of 100 and up, every mode rejects 97-99% of corrupted
  values, so a wrong large figure would be caught. For numbers below 100 (percentages, ratios,
  per-share values) even the verbatim mode accepts 34-41% of corrupted values: a financial
  statement page holds hundreds of small numbers, and rounding plus unit/percent scaling makes a
  chance match likely. A clean result on a percentage says little.
- The loose arithmetic mode was 1-5 points more permissive than verbatim; the strict mode is
  close to verbatim on false accepts while flagging 13% of answers instead of 35%. Most verbatim
  flags (32 of 36) are on answers judged correct: computed ratios the verbatim rule cannot trace.
  Verbatim flag rate is 65% on metrics-generated questions (mostly computed ratios) and 12-21% on
  the other types, so its flags mostly track question type.
- With 20 wrong answers, of which the best mode flags 4, the data cannot support any directional
  claim: both AUROC intervals of the usable modes include 0.5, and both accuracy-gap intervals
  include 0. The loose mode's interval excludes 0 only because its 3 flags all landed on correct
  answers.
- Why the wrong answers pass: no checked answer lacked a citation or cited a page outside its
  context, and 12 of the 15 answers judged incorrect pass even the verbatim check: every number
  they state matches a number on a cited page. Reading them (all in the explorer) shows a wrong line item
  (segment store counts instead of the total), a missing part of what was asked, or a wrong
  conclusion. A check that only asks "is this number on the page?" cannot see those errors.
- Embedding similarity between the answer and its cited chunks did no better than chance either
  (AUROC 0.42 for low similarity predicting a wrong answer).

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
                                                                          |
                                                             false-accept baseline: corrupt one number,
                                                             re-check, count how often it still passes
```

Modules (`src/filings_rag/`): `data` (questions, download), `ingest` (PDF to chunks), `retrieval`
(BM25, dense, RRF, rerank), `metrics`, `eval_retrieval`, `generate`, `judge`, `citation_check`,
`claims` (numbers in text), `null_baseline` (false-accept test), `llm` (DeepSeek client, spend
cap), `eval_generation`, `bootstrap`, `mcp_server`, `cli`.

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
# Re-running eval-generation on a complete results/generation.jsonl makes no API calls; it re-classifies
# answers and recomputes checks, bootstrap intervals and the false-accept baseline. It refuses to resume
# a file produced with another method/scope/k (see results/generation.config.json).
filings-rag export-explorer   # explorer/explorer.json for the static page

filings-rag ask "What was 3M's FY2018 capital expenditure?" --company 3M   # default method: dense
pytest && ruff check .        # 47 offline tests; no models, data or API key needed
```

MCP server (stdio), for Claude Desktop or any MCP client:

```json
{"mcpServers": {"filings": {"command": "filings-rag", "args": ["serve-mcp"], "cwd": "/path/to/filings-rag"}}}
```

It exposes `search_filings(query, company=None, k=5)` and `get_page(doc, page)`. `company` is
matched against the filing-name prefix, ignoring case, punctuation and legal suffixes
("Johnson & Johnson" -> `JOHNSON_JOHNSON_*`, "AES Corporation" -> `AES_2022_10K`); an unknown
company returns an error that lists the known ones.
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
  `eval-generation` after a completed run recomputes the checks with zero API calls. The run's
  config is stored next to the results, so a resume with a different `--k` or method fails
  instead of silently mixing two configurations.
- **Measure the checker, not only the answers.** A check that flags few answers can be either
  precise or blind. Corrupting numbers the check accepted and re-running it gives a false-accept
  rate that says which: here it is blind to small numbers and reliable for large ones.
- **Models behind small interfaces (`Embedder`, `Reranker`, `ChatModel`).** Tests plug in a hashed
  bag-of-words embedder, a word-overlap reranker and a scripted LLM, so the 47 tests and CI need no
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
  or a wrong conclusion, which is where this system's errors actually were. It ignores signs
  (one wrong answer reported -0.6% where the gold is 0.62%), and for numbers below 100 it accepts
  about a third or more of corrupted values (section 3).
- The partial/refusal split is a rule (does the text state a number?), not a judgement; it
  disagrees with the LLM judge about refusals on 5 of 150 answers. The raw `results/generation.jsonl` still has
  the older boolean `refusal` field written at generation time; the status used in the tables is
  recomputed from the answer text.
- The false-accept baseline uses synthetic corruptions (x1.1, digit swap, random); real model
  errors (a wrong line item) are different and, as section 3 shows, mostly pass the check.
- Latencies were measured on a shared laptop CPU while other jobs ran; treat them as orders of
  magnitude.

## Data and license

- FinanceBench open-source sample, Patronus AI, https://github.com/patronus-ai/financebench
  (paper: Islam et al., 2023, arXiv:2311.11944). The GitHub repo has no license file; the
  Hugging Face dataset card (https://huggingface.co/datasets/PatronusAI/financebench) lists
  **CC BY-NC 4.0**. This repo does not commit the raw question file or the PDFs;
  `filings-rag download` fetches them. However, `results/generation.jsonl`,
  `results/generation_with_similarity.jsonl` and `explorer/explorer.json` (also served on the
  Pages site) contain all 150 questions, gold answers and evidence page numbers (converted to
  1-based). That content stays under CC BY-NC 4.0 (non-commercial use, attribution to Patronus AI)
  and is **not** covered by this repo's MIT licence; see `results/NOTICE`.
- Answers and judge verdicts in those files were generated by DeepSeek (`deepseek-v4-flash`).
- Models: `BAAI/bge-small-en-v1.5` (MIT), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0),
  `BAAI/bge-reranker-base` (MIT), downloaded from Hugging Face at run time.
- Code: MIT, Copyright (c) 2026 Yunlong Lu. See `LICENSE`. The MIT licence covers the code only,
  not the FinanceBench content listed above.
