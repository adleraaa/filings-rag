"""PDF -> page text -> overlapping chunks, each tagged with (doc, page)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    doc: str
    page: int  # 1-based, matches what a human sees in a PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    doc: str
    page: int
    text: str


def extract_pages(pdf_path: Path) -> list[Page]:
    import pymupdf  # imported lazily so tests that use fixtures do not need it

    doc_name = pdf_path.stem
    with pymupdf.open(pdf_path) as pdf:
        return [Page(doc_name, i + 1, _clean(p.get_text())) for i, p in enumerate(pdf)]


def _clean(text: str) -> str:
    # PyMuPDF emits one line per text span; tables become long runs of short
    # lines. Collapsing whitespace keeps numbers next to their row labels.
    return re.sub(r"\s+", " ", text).strip()


def chunk_page(page: Page, size: int, overlap: int) -> Iterator[tuple[str, int, int]]:
    """Yield (doc, page, text) windows of `size` words with `overlap` words shared.

    Chunks never cross a page boundary: the evaluation unit is the evidence
    page, and a chunk spanning two pages would have an ambiguous citation.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = page.text.split()
    if not words:
        return
    step = size - overlap
    start = 0
    while True:
        window = words[start : start + size]
        yield page.doc, page.page, " ".join(window)
        if start + size >= len(words):
            break
        start += step


def chunk_pages(pages: Iterable[Page], size: int = 250, overlap: int = 50) -> list[Chunk]:
    chunks = []
    for page in pages:
        for doc, pno, text in chunk_page(page, size, overlap):
            chunks.append(Chunk(len(chunks), doc, pno, text))
    return chunks


def write_jsonl(path: Path, items: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_pages(path: Path) -> list[Page]:
    with path.open(encoding="utf-8") as f:
        return [Page(**json.loads(line)) for line in f]


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
