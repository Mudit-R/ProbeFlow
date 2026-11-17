"""Small labelled-fixture evaluator for the job retrieval path."""
from __future__ import annotations

from statistics import mean

from .job_matching import EmbeddingProvider, JobSearch, search_jobs


def evaluate_job_search(path: str, cases: list[dict], embedder: EmbeddingProvider | None,
                        cutoff: int = 3) -> dict:
    if not cases or not 1 <= cutoff <= 20:
        raise ValueError("cases must be non-empty and cutoff must be 1..20")
    reciprocal_ranks = []
    hits = []
    details = []
    for case in cases:
        result = search_jobs(path, JobSearch(query=case["query"], profile_id=case.get("profile_id"), top_k=cutoff), embedder)
        ids = [item["id"] for item in result["results"]]
        expected = case["expected_job_id"]
        rank = ids.index(expected) + 1 if expected in ids else None
        hits.append(float(rank is not None))
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        details.append({"case_id": case["id"], "expected_job_id": expected, "rank": rank,
                        "returned_job_ids": ids, "embedding_search": result["embedding_search"]})
    return {"scope": "small synthetic labelled fixtures; not a measure of live-feed coverage, fairness, or hiring relevance",
            "examples": len(cases), "cutoff": cutoff, "recall_at_k": mean(hits),
            "mrr_at_k": mean(reciprocal_ranks), "details": details}
