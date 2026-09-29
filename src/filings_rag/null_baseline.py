"""How often would a citation check accept a *wrong* number? (false-accept rate)

A check that flags few real answers is only informative if it would have
flagged wrong ones. For every number a check accepted in a real answer, we
replace that one number with a corrupted value, re-run the check, and count
how often the corrupted value is still accepted. Corruptions:

- x1.1: the value is 10% too high (same decimals and thousands separators).
- digit_swap: two adjacent, different digits are transposed (a typo).
- random: a random number with the same digit count and decimals.

No LLM and no API calls; it only needs the stored answers and page texts.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .citation_check import check_answer
from .claims import Claim, claims
from .generate import Answer

PERTURBATIONS = ("x1.1", "digit_swap", "random")
MAGNITUDES = ("<100", ">=100")


def _format_like(value: float, token: str) -> str:
    decimals = len(token.split(".")[1]) if "." in token else 0
    return f"{value:,.{decimals}f}" if "," in token else f"{value:.{decimals}f}"


def perturb(token: str, kind: str, rng: random.Random) -> str | None:
    """A corrupted version of a number token, or None if this kind cannot change it."""
    if kind == "x1.1":
        new = _format_like(float(token.replace(",", "")) * 1.1, token)
    elif kind == "digit_swap":
        pos = [i for i, ch in enumerate(token) if ch.isdigit()]
        pairs = [(i, j) for i, j in zip(pos, pos[1:], strict=False) if token[i] != token[j]]
        # A swap that moves a 0 to the front would change the digit count.
        pairs = [(i, j) for i, j in pairs if not (i == pos[0] and token[j] == "0")]
        if not pairs:
            return None
        i, j = rng.choice(pairs)
        chars = list(token)
        chars[i], chars[j] = chars[j], chars[i]
        new = "".join(chars)
    elif kind == "random":
        digits = [ch for ch in token if ch.isdigit()]
        out, k = [], 0
        for ch in token:
            if ch.isdigit():
                lead = k == 0 and len(digits) > 1 and token[0] != "0"
                out.append(str(rng.randint(1, 9) if lead else rng.randint(0, 9)))
                k += 1
            else:
                out.append(ch)
        new = "".join(out)
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    return new if new != token else None


def false_accepts(
    answers: list[Answer], page_text: Callable[[str, int], str | None], mode: str, seed: int = 0
) -> dict:
    """False-accept rate of one check mode for every perturbation kind."""
    rng = random.Random(seed)
    # Small numbers (percentages, ratios, per-share values) are much easier to
    # match by accident than large dollar amounts, so they are counted apart.
    counts = {kind: {b: {"numbers": 0, "accepted": 0} for b in MAGNITUDES} for kind in PERTURBATIONS}
    for ans in answers:
        if ans.is_refusal:
            continue
        base = check_answer(ans, page_text, mode=mode)
        for claim in claims(ans.text):
            if claim.number.value in base.unsupported_values:
                continue  # already rejected; corrupting it tells us nothing
            bucket = "<100" if claim.number.value < 100 else ">=100"
            for kind in PERTURBATIONS:
                accepted = _accepts_corrupted(ans, claim, kind, page_text, mode, rng)
                if accepted is None:
                    continue
                counts[kind][bucket]["numbers"] += 1
                counts[kind][bucket]["accepted"] += accepted
    return {kind: _rates(by_bucket) for kind, by_bucket in counts.items()}


def _rates(by_bucket: dict) -> dict:
    def rate(c: dict) -> dict:
        return {**c, "false_accept_rate": round(c["accepted"] / c["numbers"], 4) if c["numbers"] else None}

    total = {
        "numbers": sum(c["numbers"] for c in by_bucket.values()),
        "accepted": sum(c["accepted"] for c in by_bucket.values()),
    }
    return {**rate(total), "by_magnitude": {b: rate(c) for b, c in by_bucket.items()}}


def _accepts_corrupted(
    ans: Answer,
    claim: Claim,
    kind: str,
    page_text: Callable[[str, int], str | None],
    mode: str,
    rng: random.Random,
) -> bool | None:
    token = ans.text[claim.start : claim.end]
    new = perturb(token, kind, rng)
    if new is None:
        return None
    text = ans.text[: claim.start] + new + ans.text[claim.end :]
    result = check_answer(Answer(text, ans.citations, ans.context_pages), page_text, mode=mode)
    return float(new.replace(",", "")) not in result.unsupported_values
