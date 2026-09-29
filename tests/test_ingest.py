import pytest

from filings_rag.data import parse_question
from filings_rag.ingest import Page, chunk_page, chunk_pages, extract_pages


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_chunk_windows_overlap_and_cover_the_page():
    chunks = list(chunk_page(Page("D", 4, words(25)), size=10, overlap=3))
    texts = [c[2].split() for c in chunks]
    assert [len(t) for t in texts] == [10, 10, 10, 4]
    assert texts[0][-3:] == texts[1][:3]  # overlap is exactly 3 words
    assert texts[-1][-1] == "w24"  # nothing dropped at the end
    assert all(c[:2] == ("D", 4) for c in chunks)


def test_short_page_is_one_chunk_and_empty_page_is_none():
    assert len(list(chunk_page(Page("D", 1, words(5)), 10, 3))) == 1
    assert list(chunk_page(Page("D", 1, "   "), 10, 3)) == []


def test_exact_multiple_does_not_emit_a_duplicate_tail():
    # 10 words, size 10: one window only, not a second window of the last 3 words.
    assert len(list(chunk_page(Page("D", 1, words(10)), 10, 3))) == 1


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        list(chunk_page(Page("D", 1, words(5)), 10, 10))


def test_chunks_never_cross_pages_and_ids_are_sequential():
    pages = [Page("D", 1, words(12)), Page("D", 2, words(12))]
    chunks = chunk_pages(pages, size=10, overlap=2)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    page2_words = set(pages[1].text.split())
    for c in chunks:
        if c.page == 1:
            assert not (set(c.text.split()) - set(pages[0].text.split()))
        else:
            assert set(c.text.split()) <= page2_words


def test_extract_pages_is_one_based(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for text in ("first page", "second   page\nline two"):
        pdf.new_page().insert_text((72, 72), text)
    path = tmp_path / "TEST_2020_10K.pdf"
    pdf.save(path)
    pages = extract_pages(path)
    assert [(p.doc, p.page) for p in pages] == [("TEST_2020_10K", 1), ("TEST_2020_10K", 2)]
    assert pages[1].text == "second page line two"  # whitespace collapsed


def test_parse_question_converts_zero_based_evidence_pages():
    row = {
        "financebench_id": "q1",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question": "?",
        "answer": "1",
        "question_type": "metrics-generated",
        "evidence": [{"evidence_page_num": 59}, {"evidence_page_num": 59}, {"evidence_page_num": 3}],
    }
    q = parse_question(row)
    assert q.evidence_pages == (4, 60)
