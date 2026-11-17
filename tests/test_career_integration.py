import json

from probeflow.career_agent import AgentPlan, run_job_search_agent
from probeflow.job_matching import JobPosting, upsert_job
from probeflow.reference_api import create_app
from probeflow.storage import connect


class Planner:
    name = "fixture:planner"
    def __init__(self, actions):
        self.actions = iter(actions)
        self.histories = []
    def plan(self, objective, profile, history):
        self.histories.append(history)
        return AgentPlan.model_validate(next(self.actions))


def test_agent_tools_persist_and_recall_context(tmp_path):
    path = str(tmp_path / "app.db")
    create_app(path, "key")
    for job_id in ("a", "b"):
        upsert_job(path, job_id, JobPosting(company="Example", title="Python Intern", description="Python APIs"), None)
    planner = Planner([{"action":"search_jobs", "query":"Python"},
                       {"action":"get_job_details", "job_id":"a"},
                       {"action":"compare_jobs", "job_ids":["a", "b"]},
                       {"action":"finish"}])
    result = run_job_search_agent(path, "Find jobs", None, planner, None)
    assert result["tool_calls"] == 3
    assert [x["status"] for x in result["observations"]] == ["ok", "ok", "ok"]
    second = Planner([{"action":"finish"}])
    run_job_search_agent(path, "Continue", None, second, None, conversation_id=result["conversation_id"])
    assert any("Find jobs" in msg["content"] for msg in second.histories[0])
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM career_agent_tool_calls").fetchone()[0] == 3


def test_resume_match_and_feed_health_routes(tmp_path, monkeypatch):
    import probeflow.reference_api as api
    class Extractor:
        def extract(self, text, *, validation_feedback=None):
            return {"summary":"Backend intern", "target_roles":["Backend Intern"], "skills":["Python"],
                    "preferred_locations":[], "remote_ok":True, "experience":[]}
    monkeypatch.setattr(api, "configured_resume_extractor", lambda: Extractor())
    monkeypatch.setenv("PROBEFLOW_LEVER_BOARDS", json.dumps([]))
    app = create_app(str(tmp_path / "app.db"), "key")
    routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    routes["/v1/jobs/{job_id}"]("j1", JobPosting(company="Example", title="Backend Intern", description="Python API work"))
    response = routes["/v1/resume/match"](api.ResumeMatchRequest(resume_text="Python developer with internship experience building APIs."))
    assert response["results"][0]["id"] == "j1"
    assert response["results"][0]["matched_profile_skills"] == ["Python"]
    assert response["resume_text_retained"] is False
    assert routes["/v1/ingestion/health"]()["boards_configured"] == 0

