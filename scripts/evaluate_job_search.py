"""Run reproducible job-search ranking comparisons on labeled fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from probeflow.job_evaluation import evaluate_job_search
from probeflow.job_matching import (CandidateProfile, JobPosting, configured_embedder,
                                    initialize_jobs, upsert_job, upsert_profile)
from probeflow.reference_api import initialize


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "job_search_eval.json"
MODES = ("bm25_only", "embeddings_only", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="ranking baseline to run; repeat to select several (default: all)")
    parser.add_argument("--cutoff", type=int, default=3, help="retrieval cutoff k (1..20)")
    parser.add_argument("--no-ablations", action="store_true",
                        help="skip leave-one-signal-out hybrid comparisons")
    parser.add_argument("--db", type=Path,
                        help="optional new/empty SQLite path; by default use a fresh temporary database")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "job_search_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.cutoff <= 20:
        raise SystemExit("--cutoff must be between 1 and 20")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    modes = args.mode or list(MODES)
    embedder = configured_embedder()

    temp_dir = None
    if args.db:
        output_db = args.db
        if output_db.exists():
            raise SystemExit(f"Refusing to reuse existing evaluation database: {output_db}")
        output_db.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="probeflow-job-eval-")
        output_db = Path(temp_dir.name) / "job_eval.db"

    path = str(output_db)
    initialize(path)
    initialize_jobs(path)
    for item in data["jobs"]:
        upsert_job(path, item["id"], JobPosting.model_validate(
            {key: value for key, value in item.items() if key != "id"}), embedder)
    for profile_id, profile in data["profiles"].items():
        upsert_profile(path, profile_id, CandidateProfile.model_validate(profile))

    report = evaluate_job_search(path, data["cases"], embedder, cutoff=args.cutoff,
                                 ranking_modes=modes, include_ablations=not args.no_ablations)
    report.update({"embedding_model": os.getenv("PROBEFLOW_EMBEDDING_MODEL") or None,
                   "fixture_jobs": len(data["jobs"]), "fixture_sha256": fixture_hash,
                   "fixture_file": "fixtures/job_search_eval.json",
                   "evaluation_database": "fresh temporary SQLite database" if temp_dir else str(output_db)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table_path = args.output.with_suffix(".md")
    table = ("# ProbeFlow job-search ranking evaluation\n\n"
             f"Fixture: `{report['fixture_file']}` ({report['examples']} labeled queries, "
             f"{report['fixture_jobs']} synthetic jobs, SHA-256 `{fixture_hash}`).\n\n"
             f"Cutoff: k={args.cutoff}. Labels are synthetic regression fixtures, not human-labeled "
             "or real-market hiring relevance. Unavailable embedding rows are shown as such, "
             "not scored as zero.\n\n"
             f"{report['table']}\n\n"
             f"Embedding model: `{report['embedding_model'] or 'not configured'}`.\n")
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"JSON report: {args.output}")
    print(f"Markdown table: {table_path}")
    if temp_dir:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()

