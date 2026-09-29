"""Cheap post-hoc check: are an answer's numbers actually on the pages it cites?

No LLM is involved. An answer is flagged when it (a) makes a claim without a
citation, (b) cites a page that was not in its context, or (c) contains a
number that cannot be traced to a cited page, allowing for rounding and for
unit changes (thousands/millions/billions, percent vs fraction).

Modes for (c):
- verbatim: every number must appear on a cited page.
- arithmetic: a number may also be one difference, ratio or relative change of
  two numbers the answer already traced to the page. Financial answers are
  full of computed ratios, which the verbatim rule flags even when right.
- arithmetic_loose: the first version of the arithmetic mode, kept only so its
  false-accept rate can be measured (see null_baseline.py). It also allowed
  sums, products and averages, two rounds, and unit rescaling of derived
  values. With a few dozen traced numbers that produces thousands of candidate
  values, so it accepts almost any number; do not use it as a check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from .claims import Number, claims, extract_numbers
from .generate import Answer

__all__ = ["MODES", "CheckResult", "Number", "check_answer", "extract_numbers", "is_supported"]

MODES = ("verbatim", "arithmetic", "arithmetic_loose")
# A value in the answer may be the page value scaled by one of these factors.
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)
# A derived value may only be shown as a percentage: rescaling a ratio or a
# difference by 1e3 or 1e6 has no financial meaning and multiplies false accepts.
DERIVED_SCALES = (1.0, 100.0)


@dataclass
class CheckResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    numbers: int = 0
    supported: int = 0
    unsupported_values: list[float] = field(default_factory=list)


def is_supported(n: Number, values: Iterable[float], scales: Iterable[float] = SCALES) -> bool:
    """True if some value, scaled and rounded to n's precision, equals n."""
    vals = np.fromiter(values, dtype=float)
    if vals.size == 0:
        return False
    # Rounding tolerance: "1.6 billion" matches 1,577 million (1.577 -> 1.6).
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    candidates = np.outer(vals, np.fromiter(scales, dtype=float))
    return bool(np.any(np.abs(candidates - n.value) <= tol))


def one_step_results(values: Iterable[float]) -> set[float]:
    """Difference, ratio and relative change of every pair of values."""
    vals = sorted(set(values))
    out = set()
    for i, a in enumerate(vals):
        for b in vals[i + 1 :]:
            out.add(b - a)
            for num, den in ((a, b), (b, a)):
                if den:
                    out.update((num / den, (num - den) / den))
    return out


def _loose_one_step_results(values: Iterable[float]) -> set[float]:
    """The original, permissive operation set (see module docstring)."""
    vals = sorted(set(values))
    out = set()
    for i, a in enumerate(vals):
        for b in vals[i:]:
            out.update((a + b, abs(a - b), a * b, (a + b) / 2))
            for num, den in ((a, b), (b, a)):
                if den:
                    out.update((num / den, (num - den) / den))
    return out


def _trace_arithmetic(claimed: list[Number], unsupported: list[Number], loose: bool) -> list[Number]:
    """Return the claimed numbers that stay unsupported after arithmetic tracing."""
    # Only numbers the answer itself traced to the page are used as operands,
    # so the candidate set stays small and tied to what the answer claims.
    traced = [n.value for n in claimed if n not in unsupported]
    if not loose:
        derived = one_step_results(traced)
        return [n for n in unsupported if not is_supported(n, derived, DERIVED_SCALES)]
    for _ in range(2):
        if not unsupported:
            break
        derived = _loose_one_step_results(traced)
        newly = [n for n in unsupported if is_supported(n, derived)]
        traced += [n.value for n in newly]
        unsupported = [n for n in unsupported if n not in newly]
    return unsupported


def check_answer(
    answer: Answer, page_text: Callable[[str, int], str | None], mode: str = "verbatim"
) -> CheckResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if answer.is_refusal:
        return CheckResult(flagged=False)
    reasons = []
    if not answer.citations:
        reasons.append("no_citation")
    if any(c not in answer.context_pages for c in answer.citations):
        reasons.append("citation_not_in_context")

    cited_text = " ".join(t for c in answer.citations if (t := page_text(*c)))
    page_values = [n.value for n in extract_numbers(cited_text, skip_years=False)]
    claimed = [c.number for c in claims(answer.text)]

    unsupported = [n for n in claimed if not is_supported(n, page_values)]
    if mode != "verbatim" and unsupported:
        unsupported = _trace_arithmetic(claimed, unsupported, loose=mode == "arithmetic_loose")

    if unsupported:
        reasons.append("unsupported_number")
    return CheckResult(
        flagged=bool(reasons),
        reasons=reasons,
        numbers=len(claimed),
        supported=len(claimed) - len(unsupported),
        unsupported_values=[n.value for n in unsupported],
    )
