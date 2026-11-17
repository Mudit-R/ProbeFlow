from __future__ import annotations

from probeflow.career_agent import SearchPlan, run_job_search_agent
from probeflow.job_matching import CandidateProfile, JobPosting, JobSearch
from probeflow.career_agent import AgentSearchRequest
from probeflow.reference_api import create_app
from probeflow.storage import connect


class SemanticStub:
    """Deterministic vectors for retrieval plumbing tests, not model-quality evidence."""
    def embed(self, text: str) -> list[float]:
        lower = text.casefold()
        return [float(any(word in lower for word in ["backend", "server-side", "python", "api"])),
                float(any(word in lower for word in ["frontend", "react", "design system"])),
                float(any(word in lower for word in ["data", "analytics", "warehouse"]))]


class PlannerStub:
    name = "test:fixed-plan"

    def __init__(self):
        self.profile = None

    def plan(self, objective, profile):
        self.profile = profile
        return SearchPlan(action="search_jobs", query="server-side Python APIs", top_k=5)


def setup_app(tmp_path, embedder=None):
    path = str(tmp_path / "jobs.db")
    app = create_app(path, "test-key", job_embedder=embedder)
    return app, path


def route_endpoint(app, path, method="PUT"):
    route = next(r for r in app.routes if getattr(r, "path", None) == path and method in getattr(r, "methods", set()))
    return route.endpoint


def insert_fixtures(app):
    job_put = route_endpoint(app, "/v1/jobs/{job_id}")
    backend = JobPosting(company="Northstar", title="Backend Engineer Intern", description="Build Python services and REST APIs for an AI product.", location="India", work_mode="Remote", seniority="Intern", required_skills=["Python", "FastAPI"], preferred_skills=["SQL", "LLMs"])
    frontend = JobPosting(company="Studio", title="Frontend Engineer Intern", description="Build accessible React interfaces and design systems.", location="Bengaluru", work_mode="Hybrid", seniority="Intern", required_skills=["React", "TypeScript"], preferred_skills=["CSS"])
    assert job_put("backend-1", backend)["stored"]
    assert job_put("frontend-1", frontend)["stored"]
    return backend, frontend


def test_profile_memory_hybrid_rank_and_explanations(tmp_path):
    app, _ = setup_app(tmp_path)
    insert_fixtures(app)
    profile = CandidateProfile(summary="Final-year ICT student seeking an AI product engineering internship.", target_roles=["Backend Engineer Intern"], skills=["Python", "FastAPI", "SQL"], preferred_locations=["India"], remote_ok=True)
    assert route_endpoint(app, "/v1/profiles/{profile_id}")( "candidate-jobmatch", profile)["stored"]
    payload = route_endpoint(app, "/v1/jobs/search", "POST")(JobSearch(query="server-side APIs", profile_id="candidate-jobmatch", top_k=2))
    assert payload["embedding_search"] is False
    assert payload["results"][0]["id"] == "backend-1"
    assert payload["results"][0]["matched_profile_skills"] == ["FastAPI", "Python", "SQL"]
    assert set(payload["results"][0]["score_breakdown"]) == {"embedding", "lexical", "skill_overlap", "role_alignment", "location_preference"}


def test_upsert_updates_job_and_searchable_text_without_duplicates(tmp_path):
    app, db_path = setup_app(tmp_path)
    backend, _ = insert_fixtures(app)
    backend.title = "Data Platform Intern"
    backend.description = "Build analytics data warehouses and ETL pipelines."
    backend.required_skills = ["SQL", "Python"]
    assert route_endpoint(app, "/v1/jobs/{job_id}")("backend-1", backend)["stored"]
    with connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM job_postings WHERE id='backend-1'").fetchone()[0] == 1
        assert db.execute("SELECT title FROM job_postings WHERE id='backend-1'").fetchone()[0] == "Data Platform Intern"
        assert db.execute("SELECT COUNT(*) FROM job_postings_fts WHERE id='backend-1'").fetchone()[0] == 1


def test_filters_and_authentication_are_enforced(tmp_path):
    app, _ = setup_app(tmp_path)
    insert_fixtures(app)
    payload = route_endpoint(app, "/v1/jobs/search", "POST")(JobSearch(query="Python API", location="Bengaluru"))
    assert [row["id"] for row in payload["results"]] == ["frontend-1"]
    route = next(r for r in app.routes if getattr(r, "path", None) == "/v1/jobs/search")
    authorize = route.dependant.dependencies[0].call
    try:
        authorize("wrong")
    except Exception as exc:
        from fastapi import HTTPException
        assert isinstance(exc, HTTPException) and exc.status_code == 401
    else:
        raise AssertionError("invalid API key was accepted")


def test_embedding_provider_changes_ranking_and_upserts_refresh_vectors(tmp_path):
    app, db_path = setup_app(tmp_path, SemanticStub())
    job_put = route_endpoint(app, "/v1/jobs/{job_id}")
    profile_put = route_endpoint(app, "/v1/profiles/{profile_id}")
    job_put("backend", JobPosting(company="Northstar", title="Backend Intern", description="Python API service", required_skills=["Python"]))
    job_put("frontend", JobPosting(company="Studio", title="Frontend Intern", description="React design system", required_skills=["React"]))
    profile_put("student", CandidateProfile(target_roles=["Backend Intern"], skills=["Python"], remote_ok=True))
    result = route_endpoint(app, "/v1/jobs/search", "POST")(JobSearch(query="server-side API", profile_id="student"))
    assert result["embedding_search"] is True
    assert result["results"][0]["id"] == "backend"
    job_put("backend", JobPosting(company="Northstar", title="Analytics Intern", description="Data warehouse analytics", required_skills=["SQL"]))
    with connect(db_path) as db:
        row = db.execute("SELECT embedding FROM job_postings WHERE id='backend'").fetchone()
        assert row["embedding"] is not None


def test_agent_uses_saved_profile_and_one_allowlisted_search_tool(tmp_path):
    app, db_path = setup_app(tmp_path)
    insert_fixtures(app)
    route_endpoint(app, "/v1/profiles/{profile_id}")("candidate-agent", CandidateProfile(summary="ICT student", target_roles=["Backend Engineer Intern"], skills=["Python", "FastAPI"], preferred_locations=["India"], remote_ok=True))
    planner = PlannerStub()
    result = run_job_search_agent(db_path, "Find AI backend internships", "candidate-agent", planner, None)
    assert planner.profile["skills"] == ["Python", "FastAPI"]
    assert result["tool_calls"] == 1
    assert result["results"]["results"][0]["id"] == "backend-1"
    assert result["summary_is_deterministic"] is True


def test_agent_endpoint_requires_explicit_local_model(tmp_path):
    app, _ = setup_app(tmp_path)
    endpoint = route_endpoint(app, "/v1/jobs/agent-search", "POST")
    try:
        endpoint(AgentSearchRequest(objective="Find backend internships"))
    except Exception as exc:
        from fastapi import HTTPException
        assert isinstance(exc, HTTPException) and exc.status_code == 503
    else:
        raise AssertionError("agent unexpectedly ran without configured planner")
