"""Paired bootstrap confidence intervals for retrieval metric differences.

With only 150 questions, a 2-4 point gap in hit@5 can be noise. Resampling
questions (the same resample for both configs, hence "paired") gives a 95%
interval for the difference between each config and a reference config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) and its 95% percentile interval over question resamples."""
    if a.shape != b.shape:
        raise ValueError("configs must be scored on the same questions")
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_to_reference(runs: dict, reference: str, metric: str = "hit@5") -> list[dict]:
    """For every config in the same scope as `reference`, the paired difference vs the reference."""
    scope = reference.split("/")[0]
    ref = np.array([r[metric] for r in runs[reference]])
    out = []
    for name, rows in runs.items():
        if name == reference or not name.startswith(scope + "/"):
            continue
        mean, lo, hi = paired_bootstrap(np.array([r[metric] for r in rows]), ref)
        out.append(
            {
                "config": name,
                "reference": reference,
                "metric": metric,
                "diff": round(mean, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": not (lo <= 0 <= hi),
            }
        )
    return out


def run(results_dir: Path, references: tuple[str, ...] = ("doc/dense", "corpus/dense")) -> list[dict]:
    runs = json.loads((results_dir / "retrieval_per_question.json").read_text(encoding="utf-8"))
    rows = [row for ref in references for row in compare_to_reference(runs, ref)]
    (results_dir / "retrieval_bootstrap.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
