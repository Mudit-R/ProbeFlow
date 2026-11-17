"""Read-only ingestion for public Lever job boards.

Only the documented GET postings-list endpoint is called. The module stores
plain-text, source-attributed postings separately from the ranking tables so a
caller can explicitly adapt active jobs into ``job_matching.JobPosting``.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Callable, Literal
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx

from .storage import connect

LeverRegion = Literal["global", "eu"]
_SITE_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_JOB_HOSTS = {"jobs.lever.co", "jobs.eu.lever.co"}
_MAX_DESCRIPTION = 12_000
_DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=3.0)


@dataclass(frozen=True)
class LeverBoard:
    """A manually curated public Lever board. ``site`` is the Lever slug."""

    site: str
    company: str
    region: LeverRegion = "global"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not _SITE_RE.fullmatch(self.site):
            raise ValueError("Lever site must be a 1-80 character board slug")
        if not self.company.strip() or len(self.company.strip()) > 160:
            raise ValueError("Company attribution must be 1-160 characters")
        if self.region not in ("global", "eu"):
            raise ValueError("Lever region must be 'global' or 'eu'")

    @property
    def board_key(self) -> str:
        return f"lever:{self.region}:{self.site.casefold()}"

    @property
    def api_url(self) -> str:
        host = "api.eu.lever.co" if self.region == "eu" else "api.lever.co"
        return f"https://{host}/v0/postings/{quote(self.site, safe='')}"


@dataclass(frozen=True)
class LeverJob:
    """Normalized public posting with attribution and a safe source link."""

    job_id: str
    source_posting_id: str
    board_key: str
    company: str
    title: str
    description: str
    location: str
    work_mode: str
    seniority: str
    source_url: str
    canonical_url: str

    def to_job_posting_data(self) -> dict:
        """Return fields accepted by ``job_matching.JobPosting``."""
        return {
            "company": self.company,
            "title": self.title,
            "description": self.description or self.title,
            "location": self.location,
            "work_mode": self.work_mode,
            "seniority": self.seniority,
            "required_skills": [],
            "preferred_skills": [],
        }


@dataclass(frozen=True)
class BoardIngestionResult:
    board_key: str
    status: str
    fetched_count: int
    upserted_count: int
    stale_count: int
    expired_count: int
    error: str | None
    jobs: tuple[LeverJob, ...]


class LeverIngestionError(RuntimeError):
    """A feed failure that must not be treated as a complete snapshot."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden += 1
        elif tag.lower() in {"p", "br", "div", "li", "ul", "ol", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden:
            self._hidden -= 1
        elif tag.lower() in {"p", "div", "li", "ul", "ol", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.parts.append(data)


def _plain_text(value: object, max_chars: int = _MAX_DESCRIPTION) -> str:
    if not isinstance(value, str):
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(value)
        text = " ".join(parser.parts)
    except Exception:
        text = value
    text = html.unescape(text)
    text = "".join(ch if ch in "\n\t" or ord(ch) >= 32 else " " for ch in text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text).strip()
    return text[:max_chars].strip()


def canonicalize_lever_url(url: str, *, site: str, posting_id: str) -> str:
    """Validate and normalize a Lever-hosted detail URL.

    Query strings and fragments are discarded to remove tracking parameters.
    Non-Lever hosts, credentials, non-HTTPS URLs, and paths for other boards or
    postings are rejected to avoid unsafe links and accidental attribution.
    """
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").casefold()
    if (parts.scheme.casefold() != "https" or host not in _JOB_HOSTS
            or parts.username or parts.password or parts.port not in (None, 443)):
        raise ValueError("Posting URL is not a safe Lever-hosted HTTPS URL")
    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if len(segments) != 2 or segments[0].casefold() != site.casefold() or segments[1] != posting_id:
        raise ValueError("Posting URL does not match the configured Lever board and posting")
    path = "/" + "/".join(quote(segment, safe="-._~") for segment in segments)
    return urlunsplit(("https", host, path, "", ""))


def _official_job_url(board: LeverBoard, posting_id: str) -> str:
    host = "jobs.eu.lever.co" if board.region == "eu" else "jobs.lever.co"
    return f"https://{host}/{quote(board.site, safe='')}/{quote(posting_id, safe='')}"


def _category_text(categories: object, key: str) -> str:
    if not isinstance(categories, dict):
        return ""
    value = categories.get(key)
    if isinstance(value, str):
        return _plain_text(value, 500)
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, str):
                items.append(_plain_text(item, 300))
            elif isinstance(item, dict):
                for field in ("name", "location", "text"):
                    if isinstance(item.get(field), str):
                        items.append(_plain_text(item[field], 300))
                        break
        return ", ".join(x for x in items if x)[:500]
    return ""


def normalize_lever_posting(raw: object, board: LeverBoard) -> LeverJob:
    """Normalize one published Lever posting without retaining raw HTML."""
    if not isinstance(raw, dict):
        raise ValueError("Lever posting must be an object")
    posting_id = raw.get("id")
    if not isinstance(posting_id, str) or not posting_id.strip() or len(posting_id) > 200:
        raise ValueError("Lever posting is missing a valid stable id")
    posting_id = posting_id.strip()
    state = raw.get("state")
    if state is not None and state != "published":
        raise ValueError("Lever public feed contained a non-published posting")
    title = _plain_text(raw.get("text"), 200)
    if not title:
        raise ValueError("Lever posting is missing a title")

    source_url = raw.get("hostedUrl")
    if not isinstance(source_url, str):
        source_url = ""
    try:
        canonical_url = canonicalize_lever_url(source_url, site=board.site, posting_id=posting_id)
    except ValueError:
        source_url = _official_job_url(board, posting_id)
        canonical_url = canonicalize_lever_url(source_url, site=board.site, posting_id=posting_id)

    categories = raw.get("categories")
    location = _category_text(categories, "allLocations") or _category_text(categories, "location")
    workplace = raw.get("workplaceType")
    if not isinstance(workplace, str):
        workplace = ""
    workplace = workplace.strip().casefold()
    if workplace in {"remote", "hybrid", "on-site"}:
        work_mode = "On-site" if workplace == "on-site" else workplace.title()
    elif "remote" in location.casefold():
        work_mode = "Remote"
    else:
        work_mode = ""

    description = _plain_text(raw.get("descriptionPlain"))
    if not description:
        description = _plain_text(raw.get("description"))
    if not description:
        pieces = [_plain_text(raw.get("openingPlain")), _plain_text(raw.get("descriptionBodyPlain"))]
        lists = raw.get("lists")
        if isinstance(lists, list):
            for entry in lists:
                if isinstance(entry, dict):
                    pieces.extend((_plain_text(entry.get("text"), 500), _plain_text(entry.get("content"))))
        pieces.append(_plain_text(raw.get("additionalPlain")))
        description = "\n".join(part for part in pieces if part)[:_MAX_DESCRIPTION]
    if not description:
        description = title

    commitment = _category_text(categories, "commitment")[:80]
    identity = f"{board.region}\0{board.site.casefold()}\0{posting_id}".encode("utf-8")
    stable_id = "lever_" + hashlib.sha256(identity).hexdigest()
    return LeverJob(
        job_id=stable_id,
        source_posting_id=posting_id,
        board_key=board.board_key,
        company=_plain_text(board.company, 160),
        title=title,
        description=description,
        location=location[:160],
        work_mode=work_mode[:40],
        seniority=commitment,
        source_url=canonical_url,
        canonical_url=canonical_url,
    )


def initialize_ingestion_db(db_path: str) -> None:
    """Create source, normalized posting, run, and per-board health tables."""
    with connect(db_path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS ingestion_boards (
          board_key TEXT PRIMARY KEY,
          ats TEXT NOT NULL CHECK(ats='lever'),
          site TEXT NOT NULL,
          company TEXT NOT NULL,
          region TEXT NOT NULL CHECK(region IN ('global','eu')),
          enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
          last_attempt_at TEXT,
          last_success_at TEXT,
          last_status TEXT,
          last_fetched_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          board_key TEXT NOT NULL REFERENCES ingestion_boards(board_key),
          started_at TEXT NOT NULL,
          finished_at TEXT,
          status TEXT NOT NULL CHECK(status IN ('running','success','failed')),
          fetched_count INTEGER NOT NULL DEFAULT 0,
          upserted_count INTEGER NOT NULL DEFAULT 0,
          stale_count INTEGER NOT NULL DEFAULT 0,
          expired_count INTEGER NOT NULL DEFAULT 0,
          error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ingestion_runs_board_started
          ON ingestion_runs(board_key, started_at DESC);
        CREATE TABLE IF NOT EXISTS ingested_jobs (
          job_id TEXT PRIMARY KEY,
          board_key TEXT NOT NULL REFERENCES ingestion_boards(board_key),
          source_posting_id TEXT NOT NULL,
          source TEXT NOT NULL CHECK(source='Lever'),
          company TEXT NOT NULL,
          title TEXT NOT NULL,
          description TEXT NOT NULL,
          location TEXT NOT NULL,
          work_mode TEXT NOT NULL,
          seniority TEXT NOT NULL,
          source_url TEXT NOT NULL,
          canonical_url TEXT NOT NULL,
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          missed_successes INTEGER NOT NULL DEFAULT 0,
          is_stale INTEGER NOT NULL DEFAULT 0 CHECK(is_stale IN (0,1)),
          stale_at TEXT,
          expires_at TEXT,
          is_expired INTEGER NOT NULL DEFAULT 0 CHECK(is_expired IN (0,1)),
          UNIQUE(board_key, source_posting_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ingested_jobs_active
          ON ingested_jobs(board_key, is_stale, is_expired, expires_at);
        CREATE INDEX IF NOT EXISTS idx_ingested_jobs_canonical
          ON ingested_jobs(canonical_url);
        """)


def _utc(dt: datetime | None) -> datetime:
    value = dt or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _retry_after(response: httpx.Response, now: datetime) -> float | None:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return max(0.0, min(10.0, float(header)))
    except ValueError:
        try:
            when = parsedate_to_datetime(header)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, min(10.0, (when - now).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            return None


class LeverPostingsClient:
    """Small injectable HTTP client for Lever's documented public GET feed."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: httpx.Timeout | float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        page_size: int = 100,
        max_pages: int = 100,
    ) -> None:
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not 1 <= page_size <= 1000 or not 1 <= max_pages <= 100:
            raise ValueError("Pagination limits are outside safe bounds")
        self._owns_client = client is None
        self.client = client or httpx.Client(follow_redirects=False, trust_env=False)
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep = sleep
        self.page_size = page_size
        self.max_pages = max_pages

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "LeverPostingsClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_json(self, url: str, params: dict[str, int | str]) -> object:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self.sleep(min(4.0, 0.5 * (2 ** attempt)))
                continue
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                last_error = LeverIngestionError(f"Lever returned HTTP {response.status_code}")
                if attempt >= self.max_retries:
                    break
                delay = _retry_after(response, datetime.now(timezone.utc))
                self.sleep(delay if delay is not None else min(4.0, 0.5 * (2 ** attempt)))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise LeverIngestionError(f"Lever returned non-retryable HTTP {response.status_code}")
            try:
                return response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise LeverIngestionError("Lever returned invalid JSON") from exc
        raise LeverIngestionError("Lever request failed after bounded retries") from last_error

    def list_published(self, board: LeverBoard) -> list[object]:
        """Fetch a complete paginated public snapshot or fail without one."""
        all_items: list[object] = []
        for page in range(self.max_pages):
            skip = page * self.page_size
            data = self._get_json(board.api_url, {"mode": "json", "limit": self.page_size, "skip": skip})
            if not isinstance(data, list):
                raise LeverIngestionError("Lever postings response must be a JSON array")
            all_items.extend(data)
            if len(data) < self.page_size:
                return all_items
        raise LeverIngestionError("Lever feed exceeded the configured page limit; snapshot is incomplete")


def _register_board(db_path: str, board: LeverBoard, at: datetime) -> None:
    with connect(db_path) as db:
        db.execute("""INSERT INTO ingestion_boards(board_key,ats,site,company,region,enabled,last_attempt_at)
          VALUES(?,?,?,?,?,?,?) ON CONFLICT(board_key) DO UPDATE SET
          site=excluded.site,company=excluded.company,region=excluded.region,
          enabled=excluded.enabled,last_attempt_at=excluded.last_attempt_at""",
          (board.board_key, "lever", board.site, board.company.strip(), board.region,
           int(board.enabled), _iso(at)))


def _record_failure(db_path: str, board: LeverBoard, run_id: int, started: datetime, error: str) -> None:
    finished = datetime.now(timezone.utc)
    safe_error = _plain_text(error, 500) or "Ingestion failed"
    with connect(db_path) as db:
        db.execute("UPDATE ingestion_runs SET finished_at=?,status='failed',error=? WHERE run_id=?",
                   (_iso(finished), safe_error, run_id))
        db.execute("""UPDATE ingestion_boards SET last_attempt_at=?,last_status='failed',last_error=?
          WHERE board_key=?""", (_iso(started), safe_error, board.board_key))


def _persist_snapshot(
    db_path: str,
    board: LeverBoard,
    jobs: list[LeverJob],
    run_id: int,
    started: datetime,
    now: datetime,
    stale_after_successes: int,
    stale_retention_days: int,
) -> tuple[int, int, int]:
    now_text = _iso(now)
    expiry_text = _iso(now + timedelta(days=stale_retention_days))
    seen_ids = {job.source_posting_id for job in jobs}
    with connect(db_path) as db:
        for job in jobs:
            db.execute("""INSERT INTO ingested_jobs(
              job_id,board_key,source_posting_id,source,company,title,description,location,work_mode,
              seniority,source_url,canonical_url,first_seen_at,last_seen_at,missed_successes,
              is_stale,stale_at,expires_at,is_expired)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,NULL,NULL,0)
              ON CONFLICT(board_key,source_posting_id) DO UPDATE SET
              job_id=excluded.job_id,source='Lever',company=excluded.company,title=excluded.title,
              description=excluded.description,location=excluded.location,work_mode=excluded.work_mode,
              seniority=excluded.seniority,source_url=excluded.source_url,canonical_url=excluded.canonical_url,
              last_seen_at=excluded.last_seen_at,missed_successes=0,is_stale=0,stale_at=NULL,
              expires_at=NULL,is_expired=0""",
              (job.job_id, job.board_key, job.source_posting_id, "Lever", job.company, job.title,
               job.description, job.location, job.work_mode, job.seniority, job.source_url,
               job.canonical_url, now_text, now_text))

        stale_count = 0
        existing = db.execute(
            "SELECT source_posting_id,missed_successes,is_stale,stale_at,expires_at "
            "FROM ingested_jobs WHERE board_key=?",
            (board.board_key,),
        ).fetchall()
        # Avoid a giant NOT IN parameter list. Real boards can contain more
        # postings than SQLite's configured bind-variable limit.
        missing = [row for row in existing if row["source_posting_id"] not in seen_ids]
        for row in missing:
            missed = row["missed_successes"] + 1
            is_stale = missed >= stale_after_successes
            stale_at = row["stale_at"]
            expires_at = row["expires_at"]
            if is_stale and not row["is_stale"]:
                stale_at = now_text
                expires_at = expiry_text
                stale_count += 1
            db.execute("""UPDATE ingested_jobs SET missed_successes=?,is_stale=?,stale_at=?,expires_at=?
              WHERE board_key=? AND source_posting_id=?""",
              (missed, int(is_stale or bool(row["is_stale"])), stale_at, expires_at,
               board.board_key, row["source_posting_id"]))

        expired_count = db.execute(
            "UPDATE ingested_jobs SET is_expired=1 WHERE board_key=? AND is_stale=1 AND is_expired=0 AND expires_at<=?",
            (board.board_key, now_text),
        ).rowcount
        finished_text = _iso(datetime.now(timezone.utc))
        db.execute("""UPDATE ingestion_runs SET finished_at=?,status='success',fetched_count=?,
          upserted_count=?,stale_count=?,expired_count=?,error=NULL WHERE run_id=?""",
          (finished_text, len(jobs), len(jobs), stale_count, expired_count, run_id))
        db.execute("""UPDATE ingestion_boards SET last_attempt_at=?,last_success_at=?,last_status='success',
          last_fetched_count=?,last_error=NULL WHERE board_key=?""",
          (_iso(started), finished_text, len(jobs), board.board_key))
    return len(jobs), stale_count, expired_count


def ingest_board(
    db_path: str,
    board: LeverBoard,
    *,
    client: httpx.Client | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: httpx.Timeout | float = _DEFAULT_TIMEOUT,
    max_retries: int = 2,
    page_size: int = 100,
    max_pages: int = 100,
    stale_after_successes: int = 2,
    stale_retention_days: int = 30,
) -> BoardIngestionResult:
    """Fetch and persist a single board.

    Missing-job counters, stale flags, and expiry are updated only after every
    page has been fetched and parsed successfully. A failed or partial fetch
    only updates the run and board health records.
    """
    if stale_after_successes < 1 or stale_retention_days < 1:
        raise ValueError("Staleness threshold and retention must be positive")
    initialize_ingestion_db(db_path)
    started = _utc(now)
    _register_board(db_path, board, started)
    with connect(db_path) as db:
        cur = db.execute("INSERT INTO ingestion_runs(board_key,started_at,status) VALUES(?,?,'running')",
                         (board.board_key, _iso(started)))
        run_id = int(cur.lastrowid)
    if not board.enabled:
        error = "Board is disabled"
        _record_failure(db_path, board, run_id, started, error)
        return BoardIngestionResult(board.board_key, "failed", 0, 0, 0, 0, error, ())

    postings_client = LeverPostingsClient(client, timeout=timeout, max_retries=max_retries,
                                          sleep=sleep, page_size=page_size, max_pages=max_pages)
    try:
        raw_items = postings_client.list_published(board)
        normalized: dict[str, LeverJob] = {}
        for item in raw_items:
            job = normalize_lever_posting(item, board)
            # Keep the first posting if a board happens to expose aliases of
            # the same canonical URL. The source posting ID remains stable.
            normalized.setdefault(job.canonical_url, job)
        jobs = list(normalized.values())
        upserted, stale, expired = _persist_snapshot(
            db_path, board, jobs, run_id, started, _utc(now),
            stale_after_successes, stale_retention_days,
        )
        return BoardIngestionResult(board.board_key, "success", len(raw_items), upserted,
                                    stale, expired, None, tuple(jobs))
    except (httpx.HTTPError, LeverIngestionError, ValueError) as exc:
        error = str(exc)[:500] or "Ingestion failed"
        _record_failure(db_path, board, run_id, started, error)
        return BoardIngestionResult(board.board_key, "failed", 0, 0, 0, 0, error, ())
    finally:
        postings_client.close()


def list_active_jobs(db_path: str, *, board_key: str | None = None, now: datetime | None = None) -> list[dict]:
    """Return non-stale, non-expired rows for adaptation into job search."""
    initialize_ingestion_db(db_path)
    at = _iso(_utc(now))
    query = """SELECT job_id,board_key,source_posting_id,source,company,title,description,location,
      work_mode,seniority,source_url,canonical_url,first_seen_at,last_seen_at
      FROM ingested_jobs WHERE is_stale=0 AND is_expired=0 AND (expires_at IS NULL OR expires_at>?)"""
    params: tuple = (at,)
    if board_key:
        query += " AND board_key=?"
        params = (at, board_key)
    query += " ORDER BY company,title,job_id"
    with connect(db_path) as db:
        return [dict(row) for row in db.execute(query, params).fetchall()]


def get_board_health(db_path: str, board_key: str) -> dict | None:
    """Fetch latest board status and recent run history."""
    initialize_ingestion_db(db_path)
    with connect(db_path) as db:
        board = db.execute("SELECT * FROM ingestion_boards WHERE board_key=?", (board_key,)).fetchone()
        if board is None:
            return None
        runs = db.execute("""SELECT run_id,started_at,finished_at,status,fetched_count,upserted_count,
          stale_count,expired_count,error FROM ingestion_runs WHERE board_key=? ORDER BY run_id DESC LIMIT 10""",
          (board_key,)).fetchall()
    return {"board": dict(board), "runs": [dict(run) for run in runs]}


