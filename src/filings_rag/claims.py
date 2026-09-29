"""Find the numbers in a piece of text: on a filing page, or claimed in an answer.

Shared by the refusal classifier (generate.py) and the citation check, so both
agree on what counts as "the answer states a number".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# One bracket may hold several references separated by ';' or ','.
BRACKET = re.compile(r"\[([^\[\]]+)\]")
# A number not glued to a word ("FY22", "Q4" and "p.34" do not match).
NUM = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int


@dataclass(frozen=True)
class Claim:
    """A number stated in an answer, with its position so it can be edited."""

    start: int
    end: int
    number: Number


def _parse(int_part: str, frac: str) -> Number:
    return Number(float(int_part.replace(",", "") + frac), len(frac) - 1 if frac else 0)


def _is_year(int_part: str, frac: str) -> bool:
    return not frac and "," not in int_part and 1990 <= int(int_part) <= 2035


def extract_numbers(text: str, skip_years: bool = True) -> list[Number]:
    out = []
    for int_part, frac in NUM.findall(text):
        if skip_years and _is_year(int_part, frac):
            continue  # fiscal years are part of the question, not a claim to verify
        out.append(_parse(int_part, frac))
    return out


def claims(answer_text: str) -> list[Claim]:
    """Numbers an answer states, excluding years and anything inside [DOC p.N] citations."""
    cited = [m.span() for m in BRACKET.finditer(answer_text)]
    out = []
    for m in NUM.finditer(answer_text):
        if any(s <= m.start() < e for s, e in cited) or _is_year(m.group(1), m.group(2) or ""):
            continue
        out.append(Claim(m.start(), m.end(), _parse(m.group(1), m.group(2) or "")))
    return out
