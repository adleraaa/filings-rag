"""Cheap post-hoc check: are an answer's numbers actually on the pages it cites?

No LLM is involved. An answer is flagged when it (a) makes a claim without a
citation, (b) cites a page that was not in its context, or (c) contains a
number that cannot be traced to a cited page, allowing for rounding and for
unit changes (thousands/millions/billions, percent vs fraction).

Two variants of (c):
- verbatim: every number must appear on a cited page.
- arithmetic: a number may also be the result of one arithmetic step
  (+, -, *, /, average, percent change) on numbers already traced, repeated
  for up to two rounds. Financial answers are full of computed ratios and
  averages, which the verbatim rule flags even when they are right.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .generate import _BRACKET, Answer

_NUM = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
# A value in the answer may be the page value scaled by one of these factors.
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)
ARITHMETIC_ROUNDS = 2


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int


@dataclass
class CheckResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    numbers: int = 0
    supported: int = 0
    unsupported_values: list[float] = field(default_factory=list)


def extract_numbers(text: str, skip_years: bool = True) -> list[Number]:
    out = []
    for int_part, frac in _NUM.findall(text):
        value = float(int_part.replace(",", "") + frac)
        is_year = not frac and "," not in int_part and 1990 <= value <= 2035
        if skip_years and is_year:
            continue  # fiscal years are part of the question, not a claim to verify
        out.append(Number(value, len(frac) - 1 if frac else 0))
    return out


def is_supported(n: Number, values: Iterable[float]) -> bool:
    """True if some value, scaled and rounded to n's precision, equals n."""
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    for y in values:
        for s in SCALES:
            # Rounding tolerance: "1.6 billion" matches 1,577 million (1.577 -> 1.6).
            if abs(y * s - n.value) <= tol:
                return True
    return False


def one_step_results(values: Iterable[float]) -> set[float]:
    """Every result of one arithmetic step on a pair of values."""
    vals = sorted(set(values))
    out = set()
    for i, a in enumerate(vals):
        for b in vals[i:]:
            out.update((a + b, abs(a - b), a * b, (a + b) / 2))
            for num, den in ((a, b), (b, a)):
                if den:
                    out.update((num / den, (num - den) / den))  # ratio and relative change
    return out


def check_answer(
    answer: Answer, page_text: Callable[[str, int], str | None], arithmetic: bool = False
) -> CheckResult:
    if answer.is_refusal:
        return CheckResult(flagged=False)
    reasons = []
    if not answer.citations:
        reasons.append("no_citation")
    if any(c not in answer.context_pages for c in answer.citations):
        reasons.append("citation_not_in_context")

    cited_text = " ".join(t for c in answer.citations if (t := page_text(*c)))
    page_values = {n.value for n in extract_numbers(cited_text, skip_years=False)}
    claims = extract_numbers(_BRACKET.sub(" ", answer.text))

    unsupported = [n for n in claims if not is_supported(n, page_values)]
    if arithmetic:
        # Only numbers the answer itself traced to the page are used as operands,
        # so the candidate set stays small and tied to what the answer claims.
        traced = [n.value for n in claims if n not in unsupported]
        for _ in range(ARITHMETIC_ROUNDS):
            if not unsupported:
                break
            derived = one_step_results(traced)
            newly = [n for n in unsupported if is_supported(n, derived)]
            traced += [n.value for n in newly]
            unsupported = [n for n in unsupported if n not in newly]

    if unsupported:
        reasons.append("unsupported_number")
    return CheckResult(
        flagged=bool(reasons),
        reasons=reasons,
        numbers=len(claims),
        supported=len(claims) - len(unsupported),
        unsupported_values=[n.value for n in unsupported],
    )
