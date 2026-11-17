from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx

from probeflow.ingestion import (
    LeverBoard,
    canonicalize_lever_url,
    get_board_health,
    ingest_board,
    list_active_jobs,
    normalize_lever_posting,
)
from probeflow.storage import connect


NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def posting(posting_id: str = "abc-123", **overrides) -> dict:
    value = {
        "id": posting_id,
        "text": "Software Engineer Intern",
        "categories": {
            "location": "Bengaluru, India",
            "allLocations": ["Bengaluru, India", "Remote"],
            "commitment": "Internship",
        },
        "workplaceType": "remote",
        "hostedUrl": f"https://jobs.lever.co/acme/{posting_id}?lever-source=board#details",
        "descriptionPlain": "Build reliable Python APIs and job search features.",
    }
    value.update(overrides)
    return value


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)


def test_normalizes_lever_posting_and_builds_job_matching_payload():
    board = LeverBoard("acme", "Acme, Inc.")
    normalized = normalize_lever_posting(
        posting(descriptionPlain="<p>Build APIs</p><script>bad()</script><p>Ship safely</p>"), board
    )
    assert normalized.job_id.startswith("lever_")
    assert normalized.job_id == normalize_lever_posting(posting(), board).job_id
    assert normalized.company == "Acme, Inc."
    assert normalized.location == "Bengaluru, India, Remote"
    assert normalized.work_mode == "Remote"
    assert "bad()" not in normalized.description
    assert normalized.source_url == "https://jobs.lever.co/acme/abc-123"
    assert normalized.canonical_url == normalized.source_url
    assert normalized.to_job_posting_data()["company"] == "Acme, Inc."
    assert normalized.to_job_posting_data()["required_skills"] == []


def test_untrusted_hosted_url_falls_back_to_official_posting_link():
    normalized = normalize_lever_posting(
        posting(hostedUrl="https://attacker.example/collect?candidate=private"),
        LeverBoard("acme", "Acme"),
    )
    assert normalized.source_url == "https://jobs.lever.co/acme/abc-123"


def test_url_canonicalization_rejects_untrusted_hosts_and_wrong_posting():
    assert canonicalize_lever_url(
        "https://jobs.lever.co/acme/abc-123/?utm_source=x#apply", site="acme", posting_id="abc-123"
    ) == "https://jobs.lever.co/acme/abc-123"
    for unsafe in (
        "http://jobs.lever.co/acme/abc-123",
        "https://evil.example/acme/abc-123",
        "https://jobs.lever.co/other/abc-123",
        "https://user@jobs.lever.co/acme/abc-123",
    ):
        try:
            canonicalize_lever_url(unsafe, site="acme", posting_id="abc-123")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe source URL accepted: {unsafe}")


def test_fetches_paginated_public_get_feed_and_persists_attribution(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.host == "api.lever.co"
        assert request.url.path == "/v0/postings/acme"
        assert "/apply" not in request.url.path
        params = parse_qs(request.url.query.decode())
        assert params["mode"] == ["json"]
        skip = int(params["skip"][0])
        if skip == 0:
            return httpx.Response(200, json=[posting("one"), posting("two")])
        if skip == 2:
            return httpx.Response(200, json=[posting("three")])
        raise AssertionError(f"Unexpected page offset: {skip}")

    client = mock_client(handler)
    try:
        result = ingest_board(str(tmp_path / "jobs.db"), LeverBoard("acme", "Acme"),
                              client=client, now=NOW, page_size=2, sleep=lambda _: None)
    finally:
        client.close()

    assert result.status == "success"
    assert result.fetched_count == 3
    assert result.upserted_count == 3
    assert [request.url.params.get("skip") for request in requests] == ["0", "2"]
    jobs = list_active_jobs(str(tmp_path / "jobs.db"))
    assert len(jobs) == 3
    assert all(row["source"] == "Lever" for row in jobs)
    assert all(row["source_url"].startswith("https://jobs.lever.co/acme/") for row in jobs)
    assert all(row["canonical_url"] == row["source_url"] for row in jobs)
    assert get_board_health(str(tmp_path / "jobs.db"), "lever:global:acme")["board"]["last_status"] == "success"


def test_retry_after_429_then_success(tmp_path):
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=[posting()])

    client = mock_client(handler)
    try:
        result = ingest_board(str(tmp_path / "retry.db"), LeverBoard("acme", "Acme"),
                              client=client, now=NOW, sleep=sleeps.append)
    finally:
        client.close()
    assert result.status == "success"
    assert len(calls) == 2
    assert sleeps == [0.0]


def test_only_successful_complete_runs_advance_staleness_and_expiry(tmp_path):
    db_path = str(tmp_path / "stale.db")
    board = LeverBoard("acme", "Acme")
    mode = {"response": [posting()]}

    def handler(request):
        if mode["response"] == "server_error":
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=mode["response"])

    client = mock_client(handler)
    try:
        first = ingest_board(db_path, board, client=client, now=NOW, sleep=lambda _: None)
        assert first.status == "success"

        mode["response"] = []
        second = ingest_board(db_path, board, client=client, now=NOW + timedelta(days=1),
                              sleep=lambda _: None)
        assert second.status == "success"
        with connect(db_path) as db:
            row = db.execute("SELECT missed_successes,is_stale FROM ingested_jobs").fetchone()
            assert tuple(row) == (1, 0)

        mode["response"] = "server_error"
        failed = ingest_board(db_path, board, client=client, now=NOW + timedelta(days=2),
                              sleep=lambda _: None)
        assert failed.status == "failed"
        with connect(db_path) as db:
            row = db.execute("SELECT missed_successes,is_stale FROM ingested_jobs").fetchone()
            assert tuple(row) == (1, 0)

        mode["response"] = []
        third = ingest_board(db_path, board, client=client, now=NOW + timedelta(days=3),
                             stale_retention_days=2, sleep=lambda _: None)
        assert third.stale_count == 1
        assert list_active_jobs(db_path) == []
        with connect(db_path) as db:
            row = db.execute("SELECT is_stale,is_expired,expires_at FROM ingested_jobs").fetchone()
            assert row["is_stale"] == 1
            assert row["is_expired"] == 0
            assert row["expires_at"] == (NOW + timedelta(days=5)).isoformat(timespec="seconds")

        # Expiry is applied only when a later complete successful snapshot runs.
        expired = ingest_board(db_path, board, client=client, now=NOW + timedelta(days=6),
                               stale_retention_days=2, sleep=lambda _: None)
        assert expired.status == "success"
        assert expired.expired_count == 1
        with connect(db_path) as db:
            assert db.execute("SELECT is_expired FROM ingested_jobs").fetchone()[0] == 1
    finally:
        client.close()


def test_invalid_or_partial_feed_does_not_mutate_existing_jobs(tmp_path):
    db_path = str(tmp_path / "partial.db")
    board = LeverBoard("acme", "Acme")
    mode = {"page": 0}

    def handler(request):
        mode["page"] += 1
        if mode["page"] == 1:
            return httpx.Response(200, json=[posting("kept"), posting("also-kept")])
        return httpx.Response(502, json={"error": "temporary"})

    client = mock_client(handler)
    try:
        first = ingest_board(db_path, board, client=client, now=NOW, page_size=2,
                              max_retries=0, sleep=lambda _: None)
        assert first.status == "failed"
        assert list_active_jobs(db_path) == []
        with connect(db_path) as db:
            assert db.execute("SELECT COUNT(*) FROM ingested_jobs").fetchone()[0] == 0
            assert db.execute("SELECT status FROM ingestion_runs").fetchone()[0] == "failed"
    finally:
        client.close()

