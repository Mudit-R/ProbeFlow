"""Preference-aware job retrieval with an optional embedding provider.

The API stores a small candidate profile as explicit, editable memory. Job
records are upserted by stable id, and the search response exposes the score
components so recommendations can be inspected instead of treated as opaque.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from .storage import connect


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Ollama /api/embed adapter. It is optional and never downloads a model."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: float = 20):
        if not model.strip():
            raise ValueError("An embedding model name is required")
        self.model = model.strip()
        self.url = url.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(self.url + "/api/embed", json={"model": self.model, "input": text[:12000]})
                response.raise_for_status()
                vector = response.json()["embeddings"][0]
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Embedding service unavailable or returned an invalid response") from exc
        if not isinstance(vector, list) or not vector or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise RuntimeError("Embedding service returned an invalid vector")
        return [float(x) for x in vector]


def configured_embedder() -> EmbeddingProvider | None:
    model = os.getenv("PROBEFLOW_EMBEDDING_MODEL", "").strip()
    if not model:
        return None
    return OllamaEmbedder(model, os.getenv("PROBEFLOW_OLLAMA_URL", "http://127.0.0.1:11434"))


class JobPosting(BaseModel):
    company: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=12000)
    location: str = Field(default="", max_length=160)
    work_mode: str = Field(default="", max_length=40)
    seniority: str = Field(default="", max_length=80)
    required_skills: list[str] = Field(default_factory=list, max_length=40)
    preferred_skills: list[str] = Field(default_factory=list, max_length=40)


class CandidateProfile(BaseModel):
    summary: str = Field(default="", max_length=2000)
    target_roles: list[str] = Field(default_factory=list, max_length=20)
    skills: list[str] = Field(default_factory=list, max_length=80)
    preferred_locations: list[str] = Field(default_factory=list, max_length=20)
    remote_ok: bool = True


class JobSearch(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    profile_id: str | None = Field(default=None, max_length=80)
    location: str | None = Field(default=None, max_length=160)
    work_mode: str | None = Field(default=None, max_length=40)
    top_k: int = Field(default=5, ge=1, le=20)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9+#.]+", text.casefold()) if len(t) > 1}


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def _job_text(job: JobPosting) -> str:
    return "\n".join([job.title, job.company, job.description, job.seniority, job.location,
                      " ".join(job.required_skills), " ".join(job.preferred_skills)])


def initialize_jobs(path: str) -> None:
    with connect(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS job_postings(
          id TEXT PRIMARY KEY, company TEXT NOT NULL, title TEXT NOT NULL,
          description TEXT NOT NULL, location TEXT NOT NULL, work_mode TEXT NOT NULL,
          seniority TEXT NOT NULL, required_skills TEXT NOT NULL,
          preferred_skills TEXT NOT NULL, embedding TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS job_postings_fts USING fts5(
          id UNINDEXED, company, title, description, location, seniority, skills
        );
        CREATE TABLE IF NOT EXISTS candidate_profiles(
          profile_id TEXT PRIMARY KEY, summary TEXT NOT NULL, target_roles TEXT NOT NULL,
          skills TEXT NOT NULL, preferred_locations TEXT NOT NULL,
          remote_ok INTEGER NOT NULL CHECK(remote_ok IN (0,1)), updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)


def _profile_row(db, profile_id: str | None) -> dict | None:
    if not profile_id:
        return None
    row = db.execute("SELECT * FROM candidate_profiles WHERE profile_id=?", (profile_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Candidate profile not found")
    return {"summary": row["summary"], "target_roles": json.loads(row["target_roles"]),
            "skills": json.loads(row["skills"]), "preferred_locations": json.loads(row["preferred_locations"]),
            "remote_ok": bool(row["remote_ok"])}


def upsert_job(path: str, job_id: str, job: JobPosting, embedder: EmbeddingProvider | None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
        raise HTTPException(422, "Invalid job id")
    vector = embedder.embed(_job_text(job)) if embedder else None
    skills = list(dict.fromkeys([*job.required_skills, *job.preferred_skills]))
    with connect(path) as db:
        db.execute("""INSERT INTO job_postings(id,company,title,description,location,work_mode,seniority,required_skills,preferred_skills,embedding,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
          ON CONFLICT(id) DO UPDATE SET company=excluded.company,title=excluded.title,description=excluded.description,
          location=excluded.location,work_mode=excluded.work_mode,seniority=excluded.seniority,
          required_skills=excluded.required_skills,preferred_skills=excluded.preferred_skills,
          embedding=excluded.embedding,updated_at=CURRENT_TIMESTAMP""",
          (job_id, job.company, job.title, job.description, job.location, job.work_mode, job.seniority,
           json.dumps(job.required_skills), json.dumps(job.preferred_skills), json.dumps(vector) if vector else None))
        db.execute("DELETE FROM job_postings_fts WHERE id=?", (job_id,))
        db.execute("INSERT INTO job_postings_fts VALUES(?,?,?,?,?,?,?)",
                   (job_id, job.company, job.title, job.description, job.location, job.seniority, " ".join(skills)))
    return {"id": job_id, "stored": True, "embedding_indexed": vector is not None}


def upsert_profile(path: str, profile_id: str, profile: CandidateProfile) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", profile_id):
        raise HTTPException(422, "Invalid profile id")
    with connect(path) as db:
        db.execute("""INSERT INTO candidate_profiles VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
          ON CONFLICT(profile_id) DO UPDATE SET summary=excluded.summary,target_roles=excluded.target_roles,
          skills=excluded.skills,preferred_locations=excluded.preferred_locations,remote_ok=excluded.remote_ok,
          updated_at=CURRENT_TIMESTAMP""",
          (profile_id, profile.summary, json.dumps(profile.target_roles), json.dumps(profile.skills),
           json.dumps(profile.preferred_locations), int(profile.remote_ok)))
    return {"profile_id": profile_id, "stored": True}


def search_jobs(path: str, request: JobSearch, embedder: EmbeddingProvider | None) -> dict:
    with connect(path) as db:
        profile = _profile_row(db, request.profile_id)
        rows = db.execute("SELECT * FROM job_postings ORDER BY id").fetchall()
        search_tokens = sorted(_tokens(request.query))
        fts_ranks: dict[str, int] = {}
        if search_tokens:
            expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in search_tokens)
            matches = db.execute("SELECT id FROM job_postings_fts WHERE job_postings_fts MATCH ? ORDER BY bm25(job_postings_fts), id", (expression,)).fetchall()
            fts_ranks = {row["id"]: rank for rank, row in enumerate(matches, start=1)}
    if not rows:
        return {"results": [], "embedding_search": embedder is not None, "profile_used": request.profile_id}

    profile_text = ""
    profile_skills: set[str] = set()
    profile_skill_labels: dict[str, str] = {}
    profile_roles: set[str] = set()
    preferred_locations: list[str] = []
    if profile:
        profile_text = " ".join([profile["summary"], *profile["target_roles"], *profile["skills"]])
        profile_skill_labels = {s.casefold(): s for s in profile["skills"]}
        profile_skills = set(profile_skill_labels)
        profile_roles = _tokens(" ".join(profile["target_roles"]))
        preferred_locations = [x.casefold() for x in profile["preferred_locations"]]
    query = request.query
    query_vector = embedder.embed(" ".join([query, profile_text])) if embedder else None
    has_vectors = any(row["embedding"] for row in rows)
    embedding_enabled = bool(query_vector and has_vectors)
    role_context = profile["target_roles"] if profile else []
    query_tokens = _tokens(" ".join([query, *role_context]))
    candidates = []
    for row in rows:
        if request.location and request.location.casefold() not in row["location"].casefold():
            continue
        if request.work_mode and request.work_mode.casefold() != row["work_mode"].casefold():
            continue
        job_skills_list = json.loads(row["required_skills"]) + json.loads(row["preferred_skills"])
        job_skill_keys = {s.casefold() for s in job_skills_list}
        job_tokens = _tokens(" ".join([row["title"], row["company"], row["description"], row["location"], row["seniority"], *job_skills_list]))
        token_overlap = len(query_tokens & job_tokens) / max(1, len(query_tokens))
        fts_rank = fts_ranks.get(row["id"])
        bm25_rank_score = 1.0 / fts_rank if fts_rank else 0.0
        lexical = 0.5 * token_overlap + 0.5 * bm25_rank_score
        skill_score = len(profile_skills & job_skill_keys) / max(1, len(profile_skills)) if profile_skills else 0.0
        role_tokens = _tokens(row["title"])
        role_score = len(profile_roles & role_tokens) / max(1, len(profile_roles)) if profile_roles else 0.0
        location_score = float(any(loc in row["location"].casefold() for loc in preferred_locations)) if preferred_locations else 0.0
        if profile and profile["remote_ok"] and row["work_mode"].casefold() == "remote":
            location_score = max(location_score, 1.0)
        stored_vector = json.loads(row["embedding"]) if row["embedding"] else None
        cosine = _cosine(query_vector, stored_vector) if query_vector and stored_vector else 0.0
        dense = (cosine + 1) / 2 if query_vector and stored_vector else 0.0
        if query_vector and stored_vector:
            total = 0.35 * dense + 0.25 * lexical + 0.25 * skill_score + 0.10 * role_score + 0.05 * location_score
        else:
            total = 0.40 * lexical + 0.40 * skill_score + 0.15 * role_score + 0.05 * location_score
        matched_skills = sorted(profile_skill_labels[key] for key in profile_skills & job_skill_keys)
        candidates.append({"id": row["id"], "company": row["company"], "title": row["title"],
            "location": row["location"], "work_mode": row["work_mode"], "seniority": row["seniority"],
            "description": row["description"], "required_skills": json.loads(row["required_skills"]),
            "preferred_skills": json.loads(row["preferred_skills"]), "score": round(total, 6),
            "score_breakdown": {"embedding": round(dense, 6), "lexical": round(lexical, 6),
                "skill_overlap": round(skill_score, 6), "role_alignment": round(role_score, 6),
                "location_preference": round(location_score, 6)},
            "matched_profile_skills": matched_skills})
    candidates.sort(key=lambda item: (-item["score"], item["id"]))
    return {"results": candidates[:request.top_k], "embedding_search": embedding_enabled,
            "profile_used": request.profile_id, "ranking": "hybrid_embedding_bm25_profile_v1" if embedding_enabled else "lexical_profile_v1"}


def attach_job_routes(app, path: str, api_key_dependency, embedder: EmbeddingProvider | None = None) -> None:
    """Mount write-limited job/profile ingestion and read-only search routes."""
    initialize_jobs(path)

    @app.put("/v1/jobs/{job_id}", dependencies=[Depends(api_key_dependency)])
    def put_job(job_id: str, job: JobPosting) -> dict:
        try:
            return upsert_job(path, job_id, job, embedder)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.put("/v1/profiles/{profile_id}", dependencies=[Depends(api_key_dependency)])
    def put_profile(profile_id: str, profile: CandidateProfile) -> dict:
        return upsert_profile(path, profile_id, profile)

    @app.post("/v1/jobs/search", dependencies=[Depends(api_key_dependency)])
    def post_job_search(search: JobSearch) -> dict:
        try:
            return search_jobs(path, search, embedder)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
