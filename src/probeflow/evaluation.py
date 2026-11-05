"""Retrieval metrics from actual HTTP responses and explicit synthetic gold labels."""
from __future__ import annotations

import asyncio
import json
import math
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .reference_api import Answer


def score_prediction(prediction: dict, gold: dict) -> dict:
    parsed = Answer.model_validate(prediction)
    ids = [c.document_id for c in parsed.citations]
    expected = gold.get("document_id")
    rank = ids.index(expected) + 1 if expected in ids else None
    valid_ids = set(gold["allowed_document_ids"])
    return {
        "id": gold["id"], "answerable": expected is not None,
        "retrieval_hit": bool(rank) if expected else None,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "abstention_correct": parsed.abstained == (expected is None),
        "citations_valid": all(i in valid_ids for i in ids),
        "expected_phrase_present": (gold.get("expected_phrase", "").lower() in parsed.answer.lower()) if expected else None,
        "prediction": prediction,
    }


def evaluate(base_url: str, api_key: str, dataset: Path) -> dict:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    predictions = []
    with httpx.Client(base_url=base_url, timeout=15, trust_env=False) as client:
        for gold in rows:
            response = client.post("/v1/answer", headers={"X-API-Key": api_key}, json={"question": gold["question"], "top_k": 3})
            response.raise_for_status()
            predictions.append(score_prediction(response.json(), gold))
    answerable = [p for p in predictions if p["answerable"]]
    mean = lambda values: sum(values) / len(values) if values else None
    return {"scope": "synthetic fixture set; no claim of general model quality", "examples": len(predictions),
            "answerable": len(answerable), "recall_at_3": mean([p["retrieval_hit"] for p in answerable]),
            "mrr_at_3": mean([p["reciprocal_rank"] for p in answerable]),
            "abstention_accuracy": mean([p["abstention_correct"] for p in predictions]),
            "citation_validity": mean([p["citations_valid"] for p in predictions]),
            "predictions": predictions}


async def benchmark(base_url: str, api_key: str, requests: int = 200, concurrency: int = 10) -> dict:
    if not 1 <= requests <= 100000 or not 1 <= concurrency <= 100:
        raise ValueError("requests must be 1..100000 and concurrency 1..100")
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    statuses: Counter = Counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=15, trust_env=False, limits=httpx.Limits(max_connections=concurrency)) as client:
        # Warmup is excluded from the recorded samples.
        warm = await client.post("/v1/answer", headers={"X-API-Key": api_key}, json={"question": "What are the backup recovery rules?"})
        warm.raise_for_status()
        async def one(index: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                status = "transport_error"
                try:
                    response = await client.post("/v1/answer", headers={"X-API-Key": api_key}, json={"question": "What are the backup recovery rules?"})
                    status = str(response.status_code)
                except httpx.HTTPError:
                    pass
                latencies.append({"request": index, "latency_ms": (time.perf_counter() - started) * 1000, "status": status})
                statuses[status] += 1
        started = time.perf_counter()
        await asyncio.gather(*(one(index) for index in range(requests)))
        elapsed = time.perf_counter() - started
    ordered = sorted(row["latency_ms"] for row in latencies)
    quantile = lambda q: ordered[max(0, math.ceil(len(ordered) * q) - 1)]
    return {"scope": "local synthetic extractive reference API, /v1/answer; includes HTTP and SQLite retrieval, excludes LLM inference", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(), "requests": requests, "concurrency": concurrency,
            "seconds": elapsed, "requests_per_second": requests / elapsed, "status_counts": dict(statuses),
            "p50_ms": quantile(.5), "p95_ms": quantile(.95), "p99_ms": quantile(.99), "raw_samples": latencies}

