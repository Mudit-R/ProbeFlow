"""Load deterministic synthetic roles and emit a retrieval evaluation report."""
from __future__ import annotations

import json
import os
from pathlib import Path

from probeflow.job_evaluation import evaluate_job_search
from probeflow.job_matching import (CandidateProfile, JobPosting, configured_embedder,
                                    initialize_jobs, upsert_job, upsert_profile)
from probeflow.reference_api import initialize


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "job_search_eval.json"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    output_db = Path(os.getenv("PROBEFLOW_JOB_EVAL_DB", str(ROOT / "artifacts" / "job_eval.db")))
    output_db.parent.mkdir(parents=True, exist_ok=True)
    path = str(output_db)
    initialize(path)
    initialize_jobs(path)
    embedder = configured_embedder()
    for item in data["jobs"]:
        upsert_job(path, item["id"], JobPosting.model_validate({k: v for k, v in item.items() if k != "id"}), embedder)
    for profile_id, profile in data["profiles"].items():
        upsert_profile(path, profile_id, CandidateProfile.model_validate(profile))
    report = evaluate_job_search(path, data["cases"], embedder, cutoff=3)
    report["embedding_model"] = os.getenv("PROBEFLOW_EMBEDDING_MODEL") or None
    report["fixture_jobs"] = len(data["jobs"])
    report_path = ROOT / "artifacts" / "job_search_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
