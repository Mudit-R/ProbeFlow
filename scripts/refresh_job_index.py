"""Refresh configured public job boards once, or at a bounded interval.

Uses the same indexing path as the authenticated API. Run this process under a
service manager or scheduler if continuous refresh is needed. It is not a
durable scheduler and does not launch itself.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from probeflow.reference_api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="artifacts/reference.db")
    parser.add_argument("--interval-seconds", type=int, default=0,
                        help="repeat while running, minimum 300 seconds; omit for a single refresh")
    args = parser.parse_args()
    if args.interval_seconds and args.interval_seconds < 300:
        raise SystemExit("Refresh interval must be at least 300 seconds")
    app = create_app(args.db)
    refresh = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/v1/internal/ingest")
    while True:
        result = refresh()
        print(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result},
                         ensure_ascii=False), flush=True)
        if not args.interval_seconds:
            if not result["boards_configured"] or any(
                row["status"] != "success" or row["embedding_errors"] for row in result["boards"]
            ):
                raise SystemExit(1)
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

