"""Exercise an actual local Ollama planner over a fresh public Lever feed.

This is a private, temporary single-user demonstration. It uses no candidate
personal data and does not submit job applications.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from probeflow.career_agent import OllamaJobPlanner, run_job_search_agent
from probeflow.ingestion import LeverBoard, ingest_board
from probeflow.job_matching import CandidateProfile, JobPosting, OllamaEmbedder, initialize_jobs, upsert_job, upsert_profile
from probeflow.reference_api import initialize
from probeflow.storage import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--output", type=Path, default=Path("artifacts/live_career_verification.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    embedder = OllamaEmbedder(args.embed_model, timeout=120)
    with tempfile.TemporaryDirectory(prefix="probeflow-live-career-") as tmp:
        path = str(Path(tmp) / "jobs.db")
        initialize(path)
        initialize_jobs(path)
        result = ingest_board(path, LeverBoard("weekdayworks", "Weekday"), timeout=30, max_retries=1)
        if result.status != "success" or not result.jobs:
            raise RuntimeError("Live Lever feed could not be fetched: " + str(result.error))
        jobs = result.jobs
        print(f"Fetched {len(jobs)} live postings", flush=True)
        for i, item in enumerate(jobs, 1):
            job = JobPosting.model_validate(item.to_job_posting_data())
            upsert_job(path, item.job_id, job, embedder)
            if i % 10 == 0 or i == len(jobs):
                print(f"Indexed {i}/{len(jobs)} postings", flush=True)
        upsert_profile(path, "demo", CandidateProfile(summary="Final-year student looking for software internships",
            target_roles=["Software Engineer Intern"], skills=["Python", "SQL", "FastAPI"],
            preferred_locations=["India", "Bengaluru"], remote_ok=True))
        planner = OllamaJobPlanner(args.model, timeout=180)
        print("Running live model search...", flush=True)
        first = run_job_search_agent(path, "Find software engineering internships in India", "demo", planner, embedder, max_tool_calls=3)
        result_list = first.get("results", {}).get("results", [])
        if not result_list:
            raise RuntimeError("Live planner did not return search results")
        if not first["results"].get("embedding_search"):
            raise RuntimeError("Live search did not use embeddings")
        if not any(item.get("source_url") for item in result_list):
            raise RuntimeError("Live results lack source links")
        # Ask the model to resolve a specific record from the existing conversation.
        top_id = result_list[0]["id"]
        print("Running live model follow-up...", flush=True)
        second = run_job_search_agent(path, f"Get job details for {top_id}", "demo", planner,
                                      embedder, max_tool_calls=3, conversation_id=first["conversation_id"])
        if not any(item["tool"] == "get_job_details" and item["status"] == "ok"
                   and item["data"].get("id") == top_id for item in second["observations"]):
            raise RuntimeError("Live planner did not retrieve the requested job in the follow-up")
        with connect(path) as db:
            trace_count = db.execute("SELECT COUNT(*) FROM career_agent_tool_calls WHERE conversation_id=?",
                                     (first["conversation_id"],)).fetchone()[0]
            message_count = db.execute("SELECT COUNT(*) FROM career_agent_messages WHERE conversation_id=?",
                                       (first["conversation_id"],)).fetchone()[0]
        report = {"source": "live public Lever feed: weekdayworks", "indexed_postings": len(jobs),
                  "planner_model": args.model, "embedding_model": args.embed_model,
                  "first_turn": first, "second_turn": second, "tool_log_entries": trace_count,
                  "conversation_messages": message_count, "elapsed_seconds": round(time.perf_counter()-started, 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"indexed_postings": report["indexed_postings"], "first_tool_calls": first["tool_calls"],
                      "second_tools": [x["tool"] for x in second["observations"]],
                      "trace_entries": trace_count, "conversation_messages": message_count,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

