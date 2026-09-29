"""Build and load the on-disk index: pages.jsonl, chunks.jsonl, embeddings.npy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Chunk, Page, chunk_pages, extract_pages, read_chunks, read_pages, write_jsonl
from .retrieval import CrossEncoderReranker, Embedder, Retriever, SentenceTransformerEmbedder

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Index:
    pages: dict[tuple[str, int], Page]
    retriever: Retriever

    def page_text(self, doc: str, page: int) -> str | None:
        p = self.pages.get((doc, page))
        return p.text if p else None


def build_index(
    pdf_dir: Path,
    out_dir: Path,
    embedder: Embedder | None,
    size: int = 250,
    overlap: int = 50,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> tuple[list[Page], list[Chunk]]:
    pages: list[Page] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        pages.extend(extract_pages(pdf))
    chunks = chunk_pages(pages, size, overlap)
    write_index(out_dir, pages, chunks, embedder, size, overlap, model_name)
    return pages, chunks


def write_index(
    out_dir: Path,
    pages: list[Page],
    chunks: list[Chunk],
    embedder: Embedder | None,
    size: int,
    overlap: int,
    model_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pages.jsonl", pages)
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    meta = {
        "chunk_words": size,
        "overlap_words": overlap,
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "embed_model": None,
    }
    if embedder is not None:
        vecs = embedder.encode([c.text for c in chunks], is_query=False)
        np.save(out_dir / "embeddings.npy", vecs)
        meta["embed_model"] = model_name
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(
    index_dir: Path,
    embedder: Embedder | None = None,
    reranker=None,
    load_models: bool = True,
) -> Index:
    """Load an index. With load_models=True, missing embedder/reranker are
    created from the model names recorded at build time."""
    pages = read_pages(index_dir / "pages.jsonl")
    chunks = read_chunks(index_dir / "chunks.jsonl")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    emb_path = index_dir / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    if load_models and embeddings is not None and embedder is None:
        embedder = SentenceTransformerEmbedder(meta["embed_model"])
    if load_models and reranker is None:
        reranker = CrossEncoderReranker(DEFAULT_RERANK_MODEL)
    retriever = Retriever(chunks, embeddings, embedder, reranker)
    return Index({(p.doc, p.page): p for p in pages}, retriever)
