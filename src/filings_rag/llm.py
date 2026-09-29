"""DeepSeek chat client with token accounting and a hard spend cap.

Prices are DeepSeek's *peak* list prices in CNY per 1M tokens for the flash
model (https://api-docs.deepseek.com/zh-cn/quick_start/pricing, checked
2026-09-28). Off-peak is half price, so the recorded spend is an upper bound.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PRICE_CNY_PER_M = {"input_cache_hit": 0.04, "input_cache_miss": 2.0, "output": 8.0}
DEFAULT_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"


class BudgetExceeded(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


@dataclass
class SpendTracker:
    cap_cny: float
    path: Path | None = None
    calls: int = 0
    prompt_cache_hit: int = 0
    prompt_cache_miss: int = 0
    completion: int = 0
    by_stage: dict = field(default_factory=dict)

    @property
    def cost_cny(self) -> float:
        p = PRICE_CNY_PER_M
        return (
            self.prompt_cache_hit * p["input_cache_hit"]
            + self.prompt_cache_miss * p["input_cache_miss"]
            + self.completion * p["output"]
        ) / 1e6

    def check(self, est_prompt_tokens: int, max_tokens: int) -> None:
        """Refuse a call whose worst case (all cache misses, max output) would cross the cap."""
        worst = (
            est_prompt_tokens * PRICE_CNY_PER_M["input_cache_miss"] + max_tokens * PRICE_CNY_PER_M["output"]
        ) / 1e6
        if self.cost_cny + worst > self.cap_cny:
            raise BudgetExceeded(
                f"spent {self.cost_cny:.4f} CNY; next call could add {worst:.4f}, cap {self.cap_cny}"
            )

    def record(self, usage: dict, stage: str) -> None:
        hit = int(usage.get("prompt_cache_hit_tokens", 0))
        total_prompt = int(usage.get("prompt_tokens", 0))
        # DeepSeek reports cache hits separately; if absent, treat everything as a miss.
        miss = int(usage.get("prompt_cache_miss_tokens", total_prompt - hit))
        out = int(usage.get("completion_tokens", 0))
        self.calls += 1
        self.prompt_cache_hit += hit
        self.prompt_cache_miss += miss
        self.completion += out
        s = self.by_stage.setdefault(stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        s["calls"] += 1
        s["prompt_tokens"] += hit + miss
        s["completion_tokens"] += out
        self.save()

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_cache_hit_tokens": self.prompt_cache_hit,
            "prompt_cache_miss_tokens": self.prompt_cache_miss,
            "completion_tokens": self.completion,
            "cost_cny_upper_bound": round(self.cost_cny, 4),
            "cap_cny": self.cap_cny,
            "price_cny_per_million_tokens": PRICE_CNY_PER_M,
            "by_stage": self.by_stage,
        }

    def save(self) -> None:
        """Write via a temp file and rename, so a crash never leaves a torn spend.json.

        Two processes calling the API at once would still overwrite each other's
        totals; run paid commands one at a time.
        """
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: Path, cap_cny: float) -> SpendTracker:
        """Resume from a previous run so the cap covers all runs, not just this one."""
        t = cls(cap_cny=cap_cny, path=path)
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            t.calls = d["calls"]
            t.prompt_cache_hit = d["prompt_cache_hit_tokens"]
            t.prompt_cache_miss = d["prompt_cache_miss_tokens"]
            t.completion = d["completion_tokens"]
            t.by_stage = d.get("by_stage", {})
        return t


def read_api_key(env_file: Path = Path(".env")) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("set DEEPSEEK_API_KEY or put it in an untracked .env file")
    return key


class DeepSeekChat:
    def __init__(self, tracker: SpendTracker, stage: str, model: str = DEFAULT_MODEL, client=None):
        """`client` is any object with the OpenAI client's chat.completions.create (tests pass a fake)."""
        if client is None:
            from openai import OpenAI

            # The SDK retries 429s, 5xx and connection errors with exponential
            # backoff; we only make the timeout and retry count explicit.
            client = OpenAI(api_key=read_api_key(), base_url=BASE_URL, timeout=60.0, max_retries=4)
        self.client = client
        self.tracker = tracker
        self.stage = stage
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        # ~3.5 characters per token is a conservative estimate for English text
        # with many numbers; it only feeds the pre-call budget check.
        self.tracker.check(int((len(system) + len(user)) / 3.5), max_tokens)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
            # Thinking mode is on by default and would multiply output tokens.
            extra_body={"thinking": {"type": "disabled"}},
        )
        self.tracker.record(resp.usage.model_dump(), self.stage)
        return resp.choices[0].message.content or ""
