"""Answer a question from retrieved chunks, with mandatory [DOC p.N] citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .claims import BRACKET, claims
from .llm import ChatModel
from .retrieval import Hit

NOT_FOUND = "Not found in the provided pages."

SYSTEM_PROMPT = f"""You answer questions about SEC filings using ONLY the excerpts provided.
Rules:
- Every sentence that states a fact or number must end with a citation in the exact form [DOC p.N],
  using a DOC and page label shown in the excerpt headers. Several are allowed: [DOC p.3] [DOC p.7].
- If you compute something (a ratio, a change), show the inputs with their citations, then the result.
- Do not use outside knowledge. If the excerpts do not contain what is needed, reply exactly: {NOT_FOUND}
- Be brief: at most 4 sentences."""

_REF = re.compile(r"([A-Za-z0-9_.&-]+)\s+p\.?\s*(\d+)")


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[tuple[str, int], ...]
    context_pages: tuple[tuple[str, int], ...]

    @property
    def status(self) -> str:
        """One of "answered", "partial" or "refusal".

        The model often hedges: it states some figures with citations and then
        adds the NOT_FOUND sentence for the rest. Those answers make checkable
        claims, so only an answer with the sentence and no numeric claim (apart
        from years) is a refusal. Numbers are the criterion because numbers are
        what the citation check verifies.
        """
        if NOT_FOUND.lower().rstrip(".") not in self.text.lower():
            return "answered"
        return "partial" if claims(self.text) else "refusal"

    @property
    def is_refusal(self) -> bool:
        return self.status == "refusal"


def format_context(hits: Sequence[Hit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"--- [{h.chunk.doc} p.{h.chunk.page}] ---\n{h.chunk.text}")
    return "\n\n".join(blocks)


def parse_citations(text: str) -> list[tuple[str, int]]:
    refs = []
    for bracket in BRACKET.findall(text):
        for doc, page in _REF.findall(bracket):
            ref = (doc, int(page))
            if ref not in refs:
                refs.append(ref)
    return refs


def answer_question(llm: ChatModel, question: str, hits: Sequence[Hit], max_tokens: int = 350) -> Answer:
    user = f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM_PROMPT, user, max_tokens=max_tokens).strip()
    context = tuple(dict.fromkeys((h.chunk.doc, h.chunk.page) for h in hits))
    return Answer(text, tuple(parse_citations(text)), context)
