"""Golden-set evaluation for pure-vector retrieval.

The set mixes paraphrases, clinic questions, doctor lookup, and no-match
queries. It is intentionally small and curated: it catches ranking regressions
when the embedding model, chunking, or thresholds change.

Usage:
    python -m app.rag.evaluate
    python -m app.rag.evaluate --k 3 --fail-under-recall 0.9
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.core.config import get_settings
from app.rag.index import DEFAULT_DATA_DIR, KnowledgeBase
from app.rag.retriever import KnowledgeRetriever
from app.rag.runtime import create_configured_retriever


@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_id: str | None
    kind: str | None = None


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase("My face is breaking out", "T003", "treatment"),
    GoldenCase("I twisted my ankle playing soccer", "T005", "treatment"),
    GoldenCase("My little one needs their shots", "T002", "treatment"),
    GoldenCase("I need a flu jab", "T004", "treatment"),
    GoldenCase("I want my yearly physical", "T001", "treatment"),
    GoldenCase("What are your Saturday hours?", "C001", "clinic"),
    GoldenCase("What is the Sunset cancellation policy?", "C002", "clinic"),
    GoldenCase("Who is the dermatologist?", "D103", "doctor"),
    GoldenCase("What is the weather in Paris?", None),
    GoldenCase("Can you help me rent a car?", None),
    GoldenCase("I need tax advice", None),
)


def _ndcg_at_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_retriever(
    retriever: KnowledgeRetriever,
    *,
    k: int = 3,
    cases: Sequence[GoldenCase] = GOLDEN_CASES,
) -> dict[str, object]:
    """Return per-case ranks and aggregate retrieval metrics."""

    if k <= 0:
        raise ValueError("k must be positive")

    records: list[dict[str, object]] = []
    retrieval_cases = 0
    no_match_cases = 0
    recall_hits = 0
    reciprocal_rank = 0.0
    ndcg = 0.0
    no_match_hits = 0

    for case in cases:
        results = retriever.search(case.query, kind=case.kind, limit=k)
        returned = [result.id for result in results]
        rank = (
            returned.index(case.expected_id) + 1
            if case.expected_id is not None and case.expected_id in returned
            else None
        )

        if case.expected_id is None:
            no_match_cases += 1
            passed = not results
            if passed:
                no_match_hits += 1
        else:
            retrieval_cases += 1
            passed = rank is not None
            if passed:
                recall_hits += 1
                reciprocal_rank += 1.0 / rank
                ndcg += _ndcg_at_rank(rank)

        records.append(
            {
                "query": case.query,
                "kind": case.kind,
                "expected_id": case.expected_id,
                "returned": returned,
                "rank": rank,
                "passed": passed,
            }
        )

    recall_at_k = recall_hits / retrieval_cases if retrieval_cases else 1.0
    mean_reciprocal_rank = (
        reciprocal_rank / retrieval_cases if retrieval_cases else 1.0
    )
    mean_ndcg = ndcg / retrieval_cases if retrieval_cases else 1.0
    no_match_accuracy = (
        no_match_hits / no_match_cases if no_match_cases else 1.0
    )

    return {
        "k": k,
        "recall_at_k": recall_at_k,
        "mrr": mean_reciprocal_rank,
        "ndcg_at_k": mean_ndcg,
        "no_match_accuracy": no_match_accuracy,
        "passed": recall_at_k == 1.0 and no_match_accuracy == 1.0,
        "cases": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Evaluate the ClinicPilot retrieval golden set.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=Path(settings.rag_index_dir))
    parser.add_argument("--model", default=settings.rag_embedding_model)
    parser.add_argument(
        "--dimension",
        type=int,
        default=settings.rag_embedding_dimension,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(settings.rag_embedding_cache_dir),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fail-under-recall", type=float, default=0.0)
    parser.add_argument("--fail-under-no-match", type=float, default=0.0)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the cross-encoder relevance gate for this run.",
    )
    parser.add_argument(
        "--rerank-min-score",
        type=float,
        default=None,
        help="Override the cross-encoder relevance threshold.",
    )
    args = parser.parse_args(argv)

    updates: dict[str, object] = {
        "rag_index_dir": str(args.index_dir),
        "rag_embedding_model": args.model,
        "rag_embedding_dimension": args.dimension,
        "rag_embedding_cache_dir": str(args.cache_dir),
        "rag_rerank_enabled": not args.no_rerank,
    }
    if args.rerank_min_score is not None:
        updates["rag_rerank_min_score"] = args.rerank_min_score
    settings = settings.model_copy(update=updates)

    knowledge = KnowledgeBase.load_from_directory(args.data_dir)
    retriever = create_configured_retriever(
        settings,
        knowledge=knowledge,
    )
    report = evaluate_retriever(retriever, k=args.k)
    print(json.dumps(report, indent=2))

    if float(report["recall_at_k"]) < args.fail_under_recall:
        return 1
    if float(report["no_match_accuracy"]) < args.fail_under_no_match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
