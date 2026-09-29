"""MCP server exposing the filings index to any MCP client over stdio.

Tools:
  search_filings(query, company=None, k=5) -> ranked chunks with doc/page labels
  get_page(doc, page)                      -> full text of one page
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .index import Index, load_index

# Legal-form words that users add but filing names leave out ("AES Corporation" -> AES_2022_10K).
_SUFFIXES = re.compile(r"\b(?:CORPORATION|CORP|INCORPORATED|INC|COMPANY|CO|PLC|LTD|LLC|HOLDINGS|GROUP|THE)\b")


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", _SUFFIXES.sub(" ", s.upper().replace("_", " ")))


def company_of(doc: str) -> str:
    """The company part of a filing name: everything before the first _YYYY
    ("JOHNSON_JOHNSON_2022_10K" -> "JOHNSON_JOHNSON")."""
    return re.split(r"_\d{4}", doc, maxsplit=1)[0]


def company_names(index: Index) -> list[str]:
    return sorted({company_of(d) for d in index.retriever.docs})


def docs_for_company(index: Index, company: str) -> list[str]:
    """Match "American Express" to AMERICANEXPRESS_2022_10K, "Johnson & Johnson"
    to JOHNSON_JOHNSON_2022_10K, and so on. Exact match after normalising, so
    "Pfizer" never picks up a different company whose name merely starts with it."""
    key = _norm(company)
    return [d for d in index.retriever.docs if key and _norm(company_of(d)) == key]


def build_server(index: Index, method: str = "dense") -> FastMCP:
    mcp = FastMCP("filings-rag")

    @mcp.tool()
    def search_filings(query: str, company: str | None = None, k: int = 5) -> list[dict]:
        """Search SEC filings (10-K, 10-Q, 8-K, earnings releases) and return the
        best-matching passages with their document name and 1-based page number.
        Pass `company` (e.g. "3M", "American Express") to search only its filings."""
        k = max(1, min(k, 20))
        docs = None
        if company:
            docs = docs_for_company(index, company)
            if not docs:
                # An error the client can read beats an empty list it cannot explain.
                raise ValueError(
                    f"no filings for company {company!r}; known: {', '.join(company_names(index))}"
                )
        hits = index.retriever.search(query, method=method, k=k, docs=docs)
        return [
            {"doc": h.chunk.doc, "page": h.chunk.page, "score": round(h.score, 4), "text": h.chunk.text}
            for h in hits
        ]

    @mcp.tool()
    def get_page(doc: str, page: int) -> dict:
        """Return the full extracted text of one page (1-based) of a filing."""
        text = index.page_text(doc, page)
        if text is None:
            raise ValueError(f"no page {page} in document {doc!r}")
        return {"doc": doc, "page": page, "text": text}

    return mcp


def main() -> None:
    index_dir = Path(os.environ.get("FILINGS_RAG_INDEX", "data/index"))
    # Dense retrieval was the best configuration in results/retrieval_ablation.md.
    build_server(load_index(index_dir, reranker=None)).run()  # stdio transport by default


if __name__ == "__main__":
    main()
