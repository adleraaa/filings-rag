"""Cheap post-hoc check: are an answer's numbers actually on the pages it cites?

No LLM is involved. An answer is flagged when it (a) makes a claim without a
citation, (b) cites a page that was not in its context, or (c) contains a
number that cannot be found on any cited page, allowing for rounding and for
unit changes (thousands/millions/billions, percent vs fraction).

Known blind spot: a correctly *computed* number (a ratio, a growth rate) does
not appear verbatim on any page, so it is flagged too. The README reports how
often that happens instead of hiding it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .generate import _BRACKET, Answer

_NUM = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
# A value in the answer may be the page value scaled by one of these factors.
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)


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


def is_supported(n: Number, page_values: set[float]) -> bool:
    """True if some page value, scaled and rounded to n's precision, equals n."""
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    for y in page_values:
        for s in SCALES:
            # Rounding tolerance: "1.6 billion" matches 1,577 million (1.577 -> 1.6).
            if abs(y * s - n.value) <= tol:
                return True
    return False


def check_answer(answer: Answer, page_text: Callable[[str, int], str | None]) -> CheckResult:
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
    unsupported = [n.value for n in claims if not is_supported(n, page_values)]
    if unsupported:
        reasons.append("unsupported_number")
    return CheckResult(
        flagged=bool(reasons),
        reasons=reasons,
        numbers=len(claims),
        supported=len(claims) - len(unsupported),
        unsupported_values=unsupported,
    )
