"""FinanceBench question loading and PDF download.

The FinanceBench open-source sample (https://github.com/patronus-ai/financebench)
is distributed under CC BY-NC 4.0 (per its Hugging Face dataset card). The raw
question file and the PDFs are not committed; `download` fetches them into
data/. Result files that quote questions and gold answers are covered by
results/NOTICE.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{RAW_BASE}/data/financebench_open_source.jsonl"


@dataclass(frozen=True)
class Question:
    qid: str
    company: str
    doc: str
    question: str
    answer: str
    question_type: str
    # Gold evidence pages as 1-based page numbers (FinanceBench stores 0-based).
    evidence_pages: tuple[int, ...]


def parse_question(row: dict) -> Question:
    pages = sorted({int(e["evidence_page_num"]) + 1 for e in row["evidence"]})
    return Question(
        qid=row["financebench_id"],
        company=row["company"],
        doc=row["doc_name"],
        question=row["question"],
        answer=row["answer"],
        question_type=row["question_type"],
        evidence_pages=tuple(pages),
    )


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [parse_question(json.loads(line)) for line in f if line.strip()]


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    with tmp.open("rb") as f:
        head = f.read(5)
    if dest.suffix == ".pdf" and head != b"%PDF-":
        # raw.githubusercontent.com serves a small text pointer for Git LFS files.
        tmp.unlink()
        raise RuntimeError(f"{url} did not return a PDF (a Git LFS pointer?)")
    tmp.replace(dest)  # atomic rename so an interrupted download is never "done"


def download(data_dir: Path) -> list[Path]:
    """Download the question file and only the PDFs the 150 questions reference."""
    qfile = data_dir / "financebench_open_source.jsonl"
    if not qfile.exists():
        _fetch(QUESTIONS_URL, qfile)
    docs = sorted({q.doc for q in load_questions(qfile)})
    paths = []
    for doc in docs:
        pdf = data_dir / "pdfs" / f"{doc}.pdf"
        if not pdf.exists():
            print(f"downloading {doc}.pdf")
            _fetch(f"{RAW_BASE}/pdfs/{doc}.pdf", pdf)
        paths.append(pdf)
    return paths
