"""Evaluate a fixed public-job snapshot using manually selected positive judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import httpx

from probeflow.job_evaluation import evaluate_job_search
from probeflow.job_matching import JobPosting, OllamaEmbedder, _job_text, initialize_jobs, upsert_job
from probeflow.storage import connect


class CachedEmbedder(OllamaEmbedder):
    def __init__(self, model: str, url: str):
        super().__init__(model, url, timeout=180)
        self.cache = {}

    def embed(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = super().embed(text)
        return self.cache[text]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/real_jobs_snapshot.json"))
    parser.add_argument("--labels", type=Path, default=Path("fixtures/real_job_queries.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_jobs_evaluation.json"))
    parser.add_argument("--model", default="nomic-embed-text")
    args = parser.parse_args()
    snapshot_bytes = args.snapshot.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    snapshot_sha = hashlib.sha256(snapshot_bytes).hexdigest()
    if labels["snapshot_sha256"] != snapshot_sha:
        raise SystemExit("Labels belong to a different snapshot. Do not apply judgments to changed postings.")
    if not snapshot["jobs"] or not labels["cases"]:
        raise SystemExit("Snapshot and judgments must not be empty")
    known = {item["id"] for item in snapshot["jobs"]}
    for case in labels["cases"]:
        if not case.get("relevance") or not set(case["relevance"]).issubset(known):
            raise SystemExit(f"Case {case['id']} has missing or unknown job IDs")
    url = os.getenv("PROBEFLOW_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    embedder = CachedEmbedder(args.model, url)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="probeflow-real-eval-") as tmp:
        path = str(Path(tmp) / "jobs.db")
        initialize_jobs(path)
        jobs = []
        for item in snapshot["jobs"]:
            job = JobPosting.model_validate({k: v for k, v in item.items() if k not in {"id", "source_url"}})
            upsert_job(path, item["id"], job, None)
            jobs.append((item["id"], job))
        with httpx.Client(timeout=180, trust_env=False) as client:
            for start in range(0, len(jobs), 16):
                batch = jobs[start:start+16]
                response = client.post(url + "/api/embed", json={"model": args.model,
                    "input": [_job_text(job)[:12000] for _, job in batch]})
                response.raise_for_status()
                vectors = response.json()["embeddings"]
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding response length does not match input")
                with connect(path) as db:
                    db.executemany("UPDATE job_postings SET embedding=? WHERE id=?",
                        [(json.dumps(vector), job_id) for (job_id, _), vector in zip(batch, vectors)])
                print(f"Embedded {min(start+16, len(jobs))}/{len(jobs)}", flush=True)
        report = evaluate_job_search(path, labels["cases"], embedder, cutoff=10,
            ranking_modes=["bm25_only", "embeddings_only", "hybrid"], include_ablations=True)
    report.update({"scope": "Author-selected positive labels on a fixed public-job snapshot. Unjudged jobs are treated as nonrelevant, so recall and nDCG are approximate and not marketplace quality.",
        "snapshot_sha256": snapshot_sha, "snapshot_collected_at": snapshot["collected_at_utc"],
        "jobs": len(snapshot["jobs"]), "labeled_pairs": sum(len(c["relevance"]) for c in labels["cases"]),
        "label_method": labels["method"], "embedding_model": args.model,
        "elapsed_seconds": round(time.perf_counter()-started, 2)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("jobs", "examples", "labeled_pairs", "elapsed_seconds", "snapshot_sha256")}, indent=2))
    print(report["table"])
    print(f"JSON report: {args.output}")


if __name__ == "__main__":
    main()

