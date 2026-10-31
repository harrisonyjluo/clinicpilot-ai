"""Sentence-aware chunking for the dense-vector index."""

from app.rag.chunking import build_chunks, chunk_text
from app.rag.index import KnowledgeDocument


def test_short_record_produces_one_chunk():
    chunks = chunk_text("A short clinic fact.", max_chars=200, overlap=40)

    assert chunks == ["A short clinic fact."]


def test_long_record_splits_into_bounded_overlapping_chunks():
    text = " ".join(
        f"Sentence {index} describes a clinic service."
        for index in range(80)
    )

    chunks = chunk_text(text, max_chars=400, overlap=80)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 400 for chunk in chunks)
    assert chunks[0][-40:] in chunks[1]
    assert chunks == chunk_text(text, max_chars=400, overlap=80)


def test_blank_record_produces_no_chunks():
    assert chunk_text("   \n\t ", max_chars=200, overlap=40) == []


def test_chunks_keep_parent_metadata_and_source_payload():
    document = KnowledgeDocument(
        kind="treatment",
        id="T999",
        title="Example Treatment",
        text="Example Treatment is a dermatology service. " * 20,
        payload={"treatment_id": "T999", "price_usd": 10.0},
        category="Dermatology",
    )

    chunks = build_chunks([document], max_chars=200, overlap=40)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"T999:{index}"
        assert chunk.parent_id == "T999"
        assert chunk.kind == "treatment"
        assert chunk.title == "Example Treatment"
        assert chunk.category == "Dermatology"
        assert chunk.chunk_index == index
        assert chunk.payload == document.payload
