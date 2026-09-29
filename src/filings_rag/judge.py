"""LLM-as-judge: compare a candidate answer with the FinanceBench gold answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import ChatModel

VERDICTS = ("correct", "incorrect", "refusal")

JUDGE_PROMPT = """You grade answers to financial questions against a gold answer written by an analyst.
Be strict. Output one JSON object:
{"verdict": "correct" | "incorrect" | "refusal", "reason": "<one sentence>"}
- "refusal": the candidate says the information was not found or declines to answer.
- "correct": the candidate's final answer agrees with the gold answer on every key number and conclusion.
  Numbers may differ by rounding or by up to 1% relative, and may use different units if equivalent
  (e.g. $1.58 billion vs $1577 million). Extra correct detail is fine.
- "incorrect": anything else, including a wrong number, a wrong yes/no conclusion, a missing key
  element the question asks for, or an answer about a different period or company.
Judge only the final answer, not the citations."""


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str


def parse_verdict(raw: str) -> Verdict:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            v = str(obj.get("verdict", "")).strip().lower()
            if v in VERDICTS:
                return Verdict(v, str(obj.get("reason", "")))
        except json.JSONDecodeError:
            pass
    # An unparseable judgement is never counted as correct.
    return Verdict("incorrect", f"unparseable judge output: {raw[:200]}")


def judge(llm: ChatModel, question: str, gold: str, candidate: str) -> Verdict:
    user = f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {candidate}"
    return parse_verdict(llm.complete(JUDGE_PROMPT, user, max_tokens=150))
