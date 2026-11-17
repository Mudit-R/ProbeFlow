"""Reproducible retrieval metrics for explicitly labeled job-search fixtures."""
from __future__ import annotations

import math
from statistics import mean
from typing import Iterable

from .job_matching import EmbeddingProvider, JobSearch, RetrievalModeUnavailable, search_jobs


BASE_MODES: tuple[tuple[str, str, str | None], ...] = (
    ("BM25 only", "bm25_only", None),
    ("Embeddings only", "embeddings_only", None),
    ("Hybrid", "hybrid", None),
)
ABLATIONS: tuple[tuple[str, str], ...] = (
    ("Hybrid without embedding", "embedding"),
    ("Hybrid without BM25 rank", "bm25"),
    ("Hybrid without token overlap", "token_overlap"),
    ("Hybrid without skill overlap", "skill_overlap"),
    ("Hybrid without role alignment", "role_alignment"),
    ("Hybrid without location preference", "location_preference"),
)


def _judgments(case: dict) -> dict[str, int]:
    """Read graded labels, or adapt the original single-positive fixture shape."""
    labels = case.get("relevance")
    if isinstance(labels, dict):
        return {str(job_id): max(0, int(score)) for job_id, score in labels.items()}
    if "relevant_job_ids" in case:
        return {str(job_id): 1 for job_id in case["relevant_job_ids"]}
    if case.get("expected_job_id"):
        return {str(case["expected_job_id"]): 1}
    raise ValueError(f"case {case.get('id', '<unknown>')} has no relevance labels")


def _metrics(ranked_ids: list[str], relevance: dict[str, int], cutoff: int) -> dict[str, float]:
    ranked = ranked_ids[:cutoff]
    relevant_ids = {job_id for job_id, score in relevance.items() if score > 0}
    recall = (len(set(ranked) & relevant_ids) / len(relevant_ids)) if relevant_ids else 0.0
    first_rank = next((index for index, job_id in enumerate(ranked, start=1)
                       if relevance.get(job_id, 0) > 0), None)
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0

    def dcg(scores: Iterable[int]) -> float:
        return sum((2**score - 1) / math.log2(rank + 1)
                   for rank, score in enumerate(scores, start=1))

    actual_dcg = dcg(relevance.get(job_id, 0) for job_id in ranked)
    ideal_dcg = dcg(sorted(relevance.values(), reverse=True)[:cutoff])
    ndcg = actual_dcg / ideal_dcg if ideal_dcg else 0.0
    return {"recall_at_k": recall, "mrr_at_k": reciprocal_rank, "ndcg_at_k": ndcg}


def _selected_modes(ranking_modes: list[str] | None, include_ablations: bool) -> list[tuple[str, str, str | None]]:
    available = {name: (label, name, None) for label, name, _ in BASE_MODES}
    if ranking_modes is None:
        return [("Default (backward compatible)", "default", None)]
    selected = list(ranking_modes)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown ranking modes: {', '.join(sorted(unknown))}")
    modes = [available[name] for name in selected]
    if include_ablations and "hybrid" in selected:
        modes.extend((label, "hybrid", signal) for label, signal in ABLATIONS)
    return modes


def evaluate_job_search(path: str, cases: list[dict], embedder: EmbeddingProvider | None,
                        cutoff: int = 3, ranking_modes: list[str] | None = None,
                        include_ablations: bool = False) -> dict:
    """Evaluate ranking modes on explicit labels and report unavailable modes honestly.

    `ranking_modes=None` retains the previous default-mode evaluation while adding
    standard Recall@k, MRR@k and nDCG@k fields. Pass explicit modes to compare them.
    Pass explicit modes to produce controlled baseline and signal-ablation rows.
    """
    if not cases or not 1 <= cutoff <= 20:
        raise ValueError("cases must be non-empty and cutoff must be 1..20")
    modes = _selected_modes(ranking_modes, include_ablations)
    runs: list[dict] = []
    for label, mode, ablation in modes:
        details = []
        per_case = []
        unavailable_reason = None
        for case in cases:
            try:
                result = search_jobs(path, JobSearch(query=case["query"],
                    profile_id=case.get("profile_id"), top_k=cutoff,
                    ranking_mode=mode, ablate_signal=ablation), embedder)
            except RetrievalModeUnavailable as exc:
                unavailable_reason = str(exc)
                break
            ids = [item["id"] for item in result["results"]]
            labels = _judgments(case)
            metric = _metrics(ids, labels, cutoff)
            per_case.append(metric)
            details.append({"case_id": case["id"], "returned_job_ids": ids,
                            "relevance": labels, **metric})
        if unavailable_reason:
            runs.append({"label": label, "ranking_mode": mode, "ablation": ablation,
                         "status": "unavailable", "reason": unavailable_reason,
                         "recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None,
                         "details": []})
            continue
        runs.append({"label": label, "ranking_mode": mode, "ablation": ablation,
                     "status": "ok", "recall_at_k": mean(x["recall_at_k"] for x in per_case),
                     "mrr_at_k": mean(x["mrr_at_k"] for x in per_case),
                     "ndcg_at_k": mean(x["ndcg_at_k"] for x in per_case),
                     "details": details})

    table_lines = ["| Ranking variant | Status | Recall@k | MRR@k | nDCG@k |",
                   "|---|---:|---:|---:|---:|"]
    for run in runs:
        values = [f"{run[key]:.4f}" if run["status"] == "ok" else "—"
                  for key in ("recall_at_k", "mrr_at_k", "ndcg_at_k")]
        table_lines.append(f"| {run['label']} | {run['status']} | " + " | ".join(values) + " |")

    # Preserve the original top-level metrics for callers that evaluate a single mode.
    primary = next((run for run in runs if run["status"] == "ok"), runs[0])
    return {"scope": "labeled fixtures only; synthetic labels are not real hiring relevance",
            "examples": len(cases), "cutoff": cutoff,
            "recall_at_k": primary["recall_at_k"], "mrr_at_k": primary["mrr_at_k"],
            "ndcg_at_k": primary["ndcg_at_k"], "details": primary["details"],
            "runs": runs, "table": "\n".join(table_lines)}

