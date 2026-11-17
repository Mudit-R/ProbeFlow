"""Synthetic reference system for real HTTP integration tests.

Answers are retrieved excerpts, not LLM generations. The workbench can instead
target another service that implements the documented answer contract.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import uuid
from pathlib import Path

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .career_agent import attach_agent_route, configured_planner
from .ingestion import LeverBoard, get_board_health, ingest_board, initialize_ingestion_db
from .job_matching import (CandidateProfile, EmbeddingProvider, JobPosting, JobSearch,
                           attach_job_routes, configured_embedder, search_jobs,
                           upsert_job, upsert_profile)
from .resume_profile import configured_resume_extractor, extract_resume_profile
from .storage import connect

DOCUMENTS = [
    ("retention", "Retention policy", "Clinical operations audit records are retained for seven years. Archived records remain read-only."),
    ("access", "Access review", "Access reviews occur every ninety days. A team lead approves access and the security team records the decision."),
    ("invoice", "Invoice approval", "Supplier invoices over 50000 rupees require finance approval. Invoice payment terms are thirty days."),
    ("incident", "Incident response", "Critical incidents must be escalated within fifteen minutes. The incident commander maintains the response timeline."),
    ("backup", "Backup recovery", "Database backups run daily. A recovery drill is performed monthly and the restore result is documented."),
    ("onboarding", "Vendor onboarding", "Vendor onboarding requires a risk assessment and a signed data processing agreement before activation."),
]
STOP = {"the", "a", "an", "of", "and", "or", "to", "are", "is", "in", "for", "how", "what", "when", "do", "does", "must", "be", "who", "with", "every", "within", "long", "often", "which", "can", "our", "we", "you", "they", "it", "their", "should", "many"}


class Document(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


class Citation(BaseModel):
    document_id: str
    title: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


class ResumeMatchRequest(BaseModel):
    resume_text: str = Field(min_length=40, max_length=30000)
    query: str = Field(default="software engineering internship", min_length=2, max_length=1000)
    profile_id: str | None = Field(default=None, max_length=80)
    top_k: int = Field(default=10, ge=1, le=20)


def _configured_lever_boards() -> list[LeverBoard]:
    try:
        values = json.loads(os.getenv("PROBEFLOW_LEVER_BOARDS", "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("PROBEFLOW_LEVER_BOARDS must be a JSON array") from exc
    if not isinstance(values, list) or len(values) > 100:
        raise ValueError("PROBEFLOW_LEVER_BOARDS must contain at most 100 boards")
    boards = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Each Lever board must be an object")
        boards.append(LeverBoard(site=value.get("site", ""), company=value.get("company", ""),
                                 region=value.get("region", "global"), enabled=value.get("enabled", True)))
    return [board for board in boards if board.enabled]


def initialize(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(id UNINDEXED, title, text);
        CREATE TABLE IF NOT EXISTS vendors(id TEXT PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS invoices(id TEXT PRIMARY KEY, vendor_id TEXT NOT NULL REFERENCES vendors(id), amount INTEGER NOT NULL CHECK(amount>=0), paid INTEGER NOT NULL CHECK(paid IN (0,1)));
        """)
        for item in DOCUMENTS:
            existing = db.execute("SELECT 1 FROM documents WHERE id=?", (item[0],)).fetchone()
            if not existing:
                db.execute("INSERT INTO documents VALUES (?,?,?)", item)
                db.execute("INSERT INTO document_fts VALUES (?,?,?)", item)
        db.executemany("INSERT OR IGNORE INTO vendors VALUES (?,?)", [("v1", "Synthetic North"), ("v2", "Synthetic South"), ("v3", "Synthetic Empty")])
        db.executemany("INSERT OR IGNORE INTO invoices VALUES (?,?,?,?)", [("i1", "v1", 10000, 1), ("i2", "v1", 25000, 0), ("i3", "v2", 8000, 0)])


def create_app(db_path: str | None = None, api_key: str | None = None,
               job_embedder: EmbeddingProvider | None = None) -> FastAPI:
    path = db_path or os.getenv("PROBEFLOW_DB", "artifacts/reference.db")
    key = api_key or os.environ.get("PROBEFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set PROBEFLOW_API_KEY; no default credential is accepted")
    initialize(path)
    app = FastAPI(title="ProbeFlow Synthetic Reference API", version="0.1.0")

    def authorize(x_api_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_api_key, key):
            raise HTTPException(401, "Invalid API key")

    @app.get("/healthz")
    def health() -> dict:
        with connect(path) as db:
            db.execute("SELECT 1").fetchone()
        return {"status": "ok", "mode": "synthetic-extractive-reference"}

    @app.put("/v1/documents/{document_id}", dependencies=[Depends(authorize)])
    def put_document(document_id: str, document: Document) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", document_id):
            raise HTTPException(422, "Invalid document id")
        with connect(path) as db:
            db.execute("INSERT INTO documents VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,text=excluded.text", (document_id, document.title, document.text))
            db.execute("DELETE FROM document_fts WHERE id=?", (document_id,))
            db.execute("INSERT INTO document_fts VALUES (?,?,?)", (document_id, document.title, document.text))
        return {"id": document_id, "stored": True}

    @app.put("/v1/documents-xml/{document_id}", dependencies=[Depends(authorize)])
    async def put_xml(document_id: str, request: Request) -> dict:
        body = await request.body()
        if len(body) > 50000:
            raise HTTPException(413, "XML body exceeds 50000 bytes")
        try:
            root = ElementTree.fromstring(body)
            if root.tag != "document":
                raise ValueError("Expected document root")
            document = Document(title=root.findtext("title") or "", text=root.findtext("text") or "")
        except (ElementTree.ParseError, DefusedXmlException, ValueError) as exc:
            raise HTTPException(422, "Invalid or unsafe XML") from exc
        return put_document(document_id, document)

    @app.post("/v1/answer", response_model=Answer, responses={400: {"description": "Malformed JSON request body"}}, dependencies=[Depends(authorize)], openapi_extra={"x-probeflow-safe": True})
    def answer(question: Question) -> Answer:
        tokens = [t for t in re.findall(r"[a-z0-9]+", question.question.lower()) if t not in STOP]
        if not tokens:
            return Answer(answer="Insufficient evidence in the indexed documents.", citations=[], abstained=True)
        match = " OR ".join('"' + token + '"' for token in sorted(set(tokens)))
        with connect(path) as db:
            rows = db.execute("SELECT id,title,text,bm25(document_fts) AS score FROM document_fts WHERE document_fts MATCH ? ORDER BY score, id LIMIT ?", (match, question.top_k)).fetchall()
        citations = [Citation(document_id=row["id"], title=row["title"], excerpt=row["text"]) for row in rows]
        return Answer(answer=rows[0]["text"] if rows else "Insufficient evidence in the indexed documents.", citations=citations, abstained=not rows)

    @app.get("/v1/vendors", dependencies=[Depends(authorize)])
    def vendors() -> dict:
        with connect(path) as db:
            rows = db.execute("""
              SELECT v.id, v.name, COUNT(i.id) AS invoice_count,
              COALESCE(SUM(i.amount),0) AS total_rupees,
              COALESCE(SUM(CASE WHEN i.paid=0 THEN i.amount ELSE 0 END),0) AS outstanding_rupees
              FROM vendors v LEFT JOIN invoices i ON i.vendor_id=v.id
              GROUP BY v.id,v.name ORDER BY v.id
            """).fetchall()
        return {"vendors": [dict(row) for row in rows]}

    embedder = job_embedder if job_embedder is not None else configured_embedder()
    initialize_ingestion_db(path)

    def authorize_ingestion(authorization: str = Header(default=""),
                            x_api_key: str = Header(default="")) -> None:
        cron_secret = os.getenv("CRON_SECRET", "")
        if cron_secret and secrets.compare_digest(authorization, "Bearer " + cron_secret):
            return
        if secrets.compare_digest(x_api_key, key):
            return
        raise HTTPException(401, "Invalid ingestion credential")

    @app.post("/v1/internal/ingest", dependencies=[Depends(authorize_ingestion)])
    def run_ingestion() -> dict:
        try:
            boards = _configured_lever_boards()
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        results = []
        for board in boards:
            result = ingest_board(path, board)
            indexed_count = 0
            embedding_count = 0
            reused_count = 0
            index_errors = 0
            if result.status == "success":
                for job in result.jobs:
                    try:
                        indexed = upsert_job(path, job.job_id,
                            JobPosting.model_validate(job.to_job_posting_data()), embedder)
                        indexed_count += 1
                        embedding_count += int(indexed["embedding_indexed"])
                        reused_count += int(indexed["embedding_reused"])
                    except RuntimeError:
                        # Keep the changed posting searchable without a vector, and
                        # report the failure so the next refresh can retry it.
                        upsert_job(path, job.job_id,
                            JobPosting.model_validate(job.to_job_posting_data()), None)
                        indexed_count += 1
                        index_errors += 1
            results.append({"board_key": result.board_key, "status": result.status,
                            "fetched_count": result.fetched_count, "indexed_count": indexed_count,
                            "embedding_indexed": embedding_count, "embedding_reused": reused_count,
                            "embedding_errors": index_errors,
                            "stale_count": result.stale_count, "expired_count": result.expired_count,
                            "error": result.error})
        return {"boards_configured": len(boards), "boards": results,
                "note": "Only configured public Lever boards were read. No postings are inferred when none are configured."}

    @app.get("/v1/ingestion/health", dependencies=[Depends(authorize)])
    def ingestion_health() -> dict:
        try:
            boards = _configured_lever_boards()
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        health_rows = [get_board_health(path, board.board_key) for board in boards]
        with connect(path) as db:
            active = db.execute("SELECT COUNT(*) FROM ingested_jobs WHERE is_stale=0 AND is_expired=0").fetchone()[0]
        return {"boards_configured": len(boards), "active_source_records": active,
                "boards": [row for row in health_rows if row is not None]}

    @app.post("/v1/resume/match", dependencies=[Depends(authorize)])
    def resume_match(request: ResumeMatchRequest) -> dict:
        extractor = configured_resume_extractor()
        if extractor is None:
            raise HTTPException(503, "Set PROBEFLOW_RESUME_MODEL and run a local Ollama model")
        try:
            extracted = extract_resume_profile(request.resume_text, extractor)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        profile_id = request.profile_id or "resume_" + uuid.uuid4().hex
        candidate = extracted.to_candidate_profile()
        upsert_profile(path, profile_id, candidate)
        result = search_jobs(path, JobSearch(query=request.query, profile_id=profile_id,
                                             top_k=request.top_k), embedder)
        return {"profile_id": profile_id, "extracted_profile": candidate.model_dump(),
                "experience": [item.model_dump() for item in extracted.experience],
                "results": result["results"], "ranking": result["ranking"],
                "resume_text_retained": False}

    attach_job_routes(app, path, authorize, embedder)
    attach_agent_route(app, path, authorize, embedder, configured_planner())
    return app


