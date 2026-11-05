"""Synthetic reference system for real HTTP integration tests.

Answers are retrieved excerpts, not LLM generations. The workbench can instead
target another service that implements the documented answer contract.
"""
from __future__ import annotations

import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

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


@contextmanager
def connect(path: str):
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with db:
            yield db
    finally:
        db.close()


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


def create_app(db_path: str | None = None, api_key: str | None = None) -> FastAPI:
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

    return app

