import random

import pytest

from filings_rag.citation_check import Number, check_answer, extract_numbers, is_supported
from filings_rag.generate import NOT_FOUND, Answer, answer_question, parse_citations
from filings_rag.judge import parse_verdict
from filings_rag.null_baseline import false_accepts, perturb

from .conftest import FakeLLM


def test_parse_citations_formats():
    text = (
        "Capex was $1,577M [3M_2018_10K p.60]. It rose [3M_2018_10K p.60; 3M_2018_10K p.61] "
        "per [AMCOR_2022_8K_dated-2022-07-01 p. 3]."
    )
    assert parse_citations(text) == [
        ("3M_2018_10K", 60),
        ("3M_2018_10K", 61),
        ("AMCOR_2022_8K_dated-2022-07-01", 3),
    ]
    assert parse_citations("no brackets here") == []


def test_extract_numbers_skips_years_and_labels():
    nums = extract_numbers("In FY2018 capex was $1,577.00 million, up 12.5% from 2017 (Q4 note 3).")
    assert nums == [Number(1577.0, 2), Number(12.5, 1), Number(3.0, 0)]


def test_is_supported_handles_rounding_and_units():
    page = {1577.0, 0.125}
    assert is_supported(Number(1577.0, 0), page)
    assert is_supported(Number(1.6, 1), page)  # $1.6 billion vs 1,577 million
    assert is_supported(Number(12.5, 1), page)  # 12.5% vs 0.125
    assert not is_supported(Number(1.5, 1), page)
    assert not is_supported(Number(1600.0, 0), page)


def _page_text(doc, page):
    return {("D", 1): "Purchases of PP&E (1,577) (1,373)", ("D", 2): "Net income 5,363"}.get((doc, page))


def test_check_answer_passes_grounded_answer():
    ans = Answer("Capex was $1,577 million in 2018 [D p.1].", (("D", 1),), (("D", 1), ("D", 2)))
    res = check_answer(ans, _page_text)
    assert not res.flagged and res.numbers == 1 and res.supported == 1


def test_check_answer_flags_number_from_uncited_page():
    # 5,363 is on page 2, but the answer only cites page 1.
    ans = Answer("Net income was 5,363 [D p.1].", (("D", 1),), (("D", 1), ("D", 2)))
    res = check_answer(ans, _page_text)
    assert res.flagged and res.reasons == ["unsupported_number"] and res.unsupported_values == [5363.0]


def test_check_answer_flags_missing_and_foreign_citations():
    assert check_answer(Answer("It was 1,577.", (), (("D", 1),)), _page_text).reasons == [
        "no_citation",
        "unsupported_number",
    ]
    res = check_answer(Answer("It was 1,577 [D p.9].", (("D", 9),), (("D", 1),)), _page_text)
    assert "citation_not_in_context" in res.reasons


def test_refusal_is_not_flagged():
    ans = Answer(NOT_FOUND, (), (("D", 1),))
    assert ans.is_refusal and not check_answer(ans, _page_text).flagged


def test_answer_status_separates_hedged_partial_answers():
    ctx = (("D", 1),)
    assert Answer("Capex was 1,577 [D p.1].", (("D", 1),), ctx).status == "answered"
    # Explains what is missing, cites pages, but claims no number: still a refusal.
    explained = Answer(f"The excerpts lack FY2022 current assets [D p.1]. {NOT_FOUND}", (("D", 1),), ctx)
    assert explained.status == "refusal"
    # States a figure, then hedges: partial, and it must still be checked.
    hedged = Answer(f"Net income was 5,363 [D p.1]. {NOT_FOUND}", (("D", 1),), ctx)
    assert hedged.status == "partial" and not hedged.is_refusal
    res = check_answer(hedged, _page_text)
    assert res.flagged and res.unsupported_values == [5363.0]


def test_answer_question_sends_labelled_context(index):
    hits = index.retriever.search("dividends per share", "bm25", k=2, docs=["GLOBEX_2021_10K"])
    llm = FakeLLM(["Dividends were $2.40 per share [GLOBEX_2021_10K p.2]."])
    ans = answer_question(llm, "What were dividends per share?", hits)
    _, user = llm.prompts[0]
    assert "--- [GLOBEX_2021_10K p.2] ---" in user
    assert ans.citations == (("GLOBEX_2021_10K", 2),)
    assert ("GLOBEX_2021_10K", 2) in ans.context_pages
    assert not check_answer(ans, index.page_text).flagged


def test_parse_verdict():
    assert parse_verdict('{"verdict": "Correct", "reason": "matches"}').verdict == "correct"
    assert parse_verdict('Sure! ```json\n{"verdict": "refusal", "reason": "x"}\n```').verdict == "refusal"
    assert parse_verdict("looks right to me").verdict == "incorrect"
    assert parse_verdict('{"verdict": "mostly"}').verdict == "incorrect"


def test_arithmetic_mode_accepts_computed_values_only():
    page = {("D", 1): "Operating income 1,493,602 and 903,095. Current assets 5,121.3 liabilities 7,491.5"}
    text = (
        "Operating income rose from $903,095 to $1,493,602 thousand [D p.1], "
        "a change of $590,507 thousand or 65.4% [D p.1]."
    )
    ans = Answer(text, (("D", 1),), (("D", 1),))
    verbatim = check_answer(ans, lambda d, p: page.get((d, p)))
    assert verbatim.flagged and verbatim.unsupported_values == [590507.0, 65.4]
    assert not check_answer(ans, lambda d, p: page.get((d, p)), mode="arithmetic").flagged

    # A number that no single step produces is still flagged.
    wrong = Answer("Operating income was 903,095 and margin 12.34% [D p.1].", (("D", 1),), (("D", 1),))
    res = check_answer(wrong, lambda d, p: page.get((d, p)), mode="arithmetic")
    assert res.flagged and res.unsupported_values == [12.34]


def test_arithmetic_mode_rejects_a_corrupted_number_the_loose_mode_accepted():
    # 649,558 is 590,507 x 1.1 (a 10% error), and also 590,507 + 59,051. The loose
    # mode's sums of arbitrary pairs accepted it; the strict mode must not.
    page = {("D", 1): "Revenue 590,507. Other income 59,051. Segments 3."}
    text = "Revenue was 590,507 [D p.1] and other income 59,051 [D p.1]; revenue grew to 649,558 [D p.1]."
    ans = Answer(text, (("D", 1),), (("D", 1),))
    strict = check_answer(ans, lambda d, p: page.get((d, p)), mode="arithmetic")
    assert strict.flagged and strict.unsupported_values == [649558.0]
    assert not check_answer(ans, lambda d, p: page.get((d, p)), mode="arithmetic_loose").flagged


def test_arithmetic_mode_does_not_rescale_derived_values():
    # 590,507 / 903,095 = 0.6539; shown as a percentage (65.4) is fine, but the
    # same ratio "in thousands" (653.9) is not a meaningful derivation.
    page = {("D", 1): "Operating income 1,493,602 and 903,095."}
    ok = Answer("From 903,095 to 1,493,602, up 65.4% [D p.1].", (("D", 1),), (("D", 1),))
    assert not check_answer(ok, lambda d, p: page.get((d, p)), mode="arithmetic").flagged
    bad = Answer("From 903,095 to 1,493,602, up 653.9 [D p.1].", (("D", 1),), (("D", 1),))
    assert check_answer(bad, lambda d, p: page.get((d, p)), mode="arithmetic").flagged
    with pytest.raises(ValueError):
        check_answer(ok, lambda d, p: page.get((d, p)), mode="fuzzy")


def test_perturbations_keep_format_and_change_the_value():
    rng = random.Random(0)
    assert perturb("590,507", "x1.1", rng) == "649,558"
    assert perturb("12.5", "x1.1", rng) == "13.8"
    assert perturb("3", "x1.1", rng) is None  # 3.3 rounds back to 3
    swapped = perturb("1,577", "digit_swap", rng)
    assert swapped != "1,577" and sorted(swapped) == sorted("1,577")
    assert perturb("11", "digit_swap", rng) is None
    assert perturb("10", "digit_swap", rng) is None  # "01" would drop a digit
    r = perturb("2.40", "random", rng)
    assert len(r) == 4 and r[1] == "." and r[0] != "0"


def test_false_accept_rate_counts_only_numbers_the_check_accepted():
    page = {("D", 1): "Revenue 590,507. Margin 12.5 percent. Other 13.8 and 590,570"}
    ans = Answer("Revenue 590,507 and margin 12.5% [D p.1].", (("D", 1),), (("D", 1),))
    out = false_accepts([ans], lambda d, p: page.get((d, p)), "verbatim")
    # x1.1: 649,558 is rejected, but 13.8 happens to be on the page, so it is accepted.
    assert out["x1.1"]["numbers"] == 2 and out["x1.1"]["accepted"] == 1
    assert out["x1.1"]["by_magnitude"]["<100"]["accepted"] == 1
    assert out["x1.1"]["by_magnitude"][">=100"] == {"numbers": 1, "accepted": 0, "false_accept_rate": 0.0}
