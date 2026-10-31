"""Build and persist the dense-vector index.

Usage:
    python -m app.rag.build_index
    python -m app.rag.build_index --data-dir mock-data --index-dir data/rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.core.config import get_settings
from app.rag.embeddings import FastEmbedEmbeddingModel
from app.rag.index import DEFAULT_DATA_DIR, KnowledgeBase
from app.rag.runtime import build_store


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Build the index and return the summary printed to stdout."""

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build the ClinicPilot dense-vector knowledge index.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing clinics.csv, doctors.csv, treatments.csv.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(settings.rag_index_dir),
        help="Directory where the index artifacts are written.",
    )
    parser.add_argument(
        "--model",
        default=settings.rag_embedding_model,
        help="fastembed model name.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=settings.rag_embedding_dimension,
        help="Embedding dimension produced by the model.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(settings.rag_embedding_cache_dir),
        help="Local model cache directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.rag_chunk_max_chars,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.rag_chunk_overlap,
    )
    args = parser.parse_args(argv)

    knowledge = KnowledgeBase.load_from_directory(args.data_dir)
    embedder = FastEmbedEmbeddingModel(
        model_name=args.model,
        cache_dir=args.cache_dir,
        dimension=args.dimension,
    )
    store = build_store(
        knowledge,
        embedder=embedder,
        index_dir=args.index_dir,
        chunk_max_chars=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    summary: dict[str, object] = {
        "documents": len(knowledge.documents),
        "chunk_count": store.ntotal,
        "kinds": store.chunk_counts,
        "model": store.model_name,
        "dimension": store.dimension,
        "source_fingerprint": store.source_fingerprint,
        "index_dir": str(Path(args.index_dir)),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
