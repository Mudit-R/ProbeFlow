"""Fetch attributed public Lever postings into a dated, reproducible local snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from probeflow.ingestion import LeverBoard, LeverPostingsClient, normalize_lever_posting


BOARDS = [LeverBoard("zoox", "Zoox"), LeverBoard("palantir", "Palantir"),
          LeverBoard("weekdayworks", "Weekday"), LeverBoard("shopback-2", "ShopBack")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_jobs_snapshot.json"))
    parser.add_argument("--per-board", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.per_board <= 1000:
        raise SystemExit("--per-board must be between 1 and 1000")
    records = []
    provenance = []
    with LeverPostingsClient(timeout=30, max_retries=1, page_size=100) as client:
        for board in BOARDS:
            try:
                raw = client.list_published(board)
                jobs = sorted((normalize_lever_posting(item, board) for item in raw), key=lambda x: x.job_id)
                selected = jobs[:args.per_board]
                provenance.append({"board": board.board_key, "status": "ok", "fetched": len(jobs), "selected": len(selected)})
                for job in selected:
                    records.append({"id": job.job_id, "company": job.company, "title": job.title,
                        "description": job.description[:1500] or job.title, "location": job.location,
                        "work_mode": job.work_mode, "seniority": job.seniority,
                        "required_skills": [], "preferred_skills": [], "source_url": job.source_url})
            except Exception as exc:
                provenance.append({"board": board.board_key, "status": "failed", "error": type(exc).__name__})
    if not records:
        raise SystemExit("No boards returned a usable complete snapshot")
    data = {"collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "Lever public job postings API", "boards": provenance, "jobs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"collected_at_utc": data["collected_at_utc"], "jobs": len(records), "boards": provenance}, indent=2))
    print(f"Snapshot: {args.output}")


if __name__ == "__main__":
    main()

