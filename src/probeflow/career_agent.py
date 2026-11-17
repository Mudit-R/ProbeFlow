"""Bounded natural-language job search using one allowlisted retrieval tool."""
from __future__ import annotations

import os
from typing import Literal, Protocol, TypedDict

import httpx
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .job_matching import EmbeddingProvider, JobSearch, _profile_row, search_jobs
from .storage import connect


class SearchPlan(BaseModel):
    action: Literal["search_jobs", "finish"]
    query: str = Field(default="", max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AgentSearchRequest(BaseModel):
    objective: str = Field(min_length=3, max_length=2000)
    profile_id: str | None = Field(default=None, max_length=80)


class Planner(Protocol):
    name: str

    def plan(self, objective: str, profile: dict | None) -> SearchPlan: ...


class OllamaJobPlanner:
    """Structured intent extraction; final recommendations remain API evidence."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: float = 60):
        if not model.strip():
            raise ValueError("A planner model is required")
        self.model = model.strip()
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.name = "ollama:" + self.model

    def plan(self, objective: str, profile: dict | None) -> SearchPlan:
        system = "Convert the user's request into one concise job-search query. Use the supplied candidate profile as context. Choose search_jobs when relevant. Do not invent job listings or claim a listing is eligible. Return only the required JSON object."
        payload = {"model": self.model, "stream": False,
                   "format": SearchPlan.model_json_schema(), "options": {"temperature": 0, "num_predict": 200},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": "Candidate profile data: " + str(profile or {}) + "\nSearch request: " + objective}]}
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(self.url + "/api/chat", json=payload)
                response.raise_for_status()
                content = response.json()["message"]["content"]
            return SearchPlan.model_validate_json(content)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Job-search planner failed to return a valid structured plan") from exc


def configured_planner() -> Planner | None:
    model = os.getenv("PROBEFLOW_AGENT_MODEL", "").strip()
    if not model:
        return None
    return OllamaJobPlanner(model, os.getenv("PROBEFLOW_OLLAMA_URL", "http://127.0.0.1:11434"))


class AgentState(TypedDict):
    objective: str
    profile_id: str | None
    plan: dict
    profile: dict | None
    results: dict
    errors: list[str]
    tool_calls: int
    summary: str


def run_job_search_agent(path: str, objective: str, profile_id: str | None,
                         planner: Planner, embedder: EmbeddingProvider | None,
                         max_tool_calls: int = 1) -> dict:
    """Plan once, call only `search_jobs`, then return traceable API results."""
    if max_tool_calls != 1:
        raise ValueError("This agent allows exactly one bounded search tool call")

    def load_profile(state: AgentState) -> dict:
        with connect(path) as db:
            profile = _profile_row(db, state["profile_id"])
        return {"profile": profile}

    def plan(state: AgentState) -> dict:
        decision = planner.plan(state["objective"], state["profile"])
        return {"plan": decision.model_dump()}

    def route(state: AgentState) -> str:
        return "search_jobs" if state["plan"].get("action") == "search_jobs" else "finish"

    def search_tool(state: AgentState) -> dict:
        plan_data = SearchPlan.model_validate(state["plan"])
        # Tool registry intentionally contains one read-only operation. The model
        # cannot supply a URL, SQL, code, or another tool name.
        result = search_jobs(path, JobSearch(query=plan_data.query or state["objective"],
                                             profile_id=state["profile_id"], top_k=plan_data.top_k), embedder)
        return {"results": result, "tool_calls": state["tool_calls"] + 1}

    def finish(state: AgentState) -> dict:
        count = len(state["results"].get("results", []))
        return {"summary": f"Job search returned {count} ranked result(s) from the indexed postings."}

    graph = StateGraph(AgentState)
    graph.add_node("load_profile", load_profile)
    graph.add_node("plan", plan)
    graph.add_node("search_jobs", search_tool)
    graph.add_node("finish", finish)
    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "plan")
    graph.add_conditional_edges("plan", route, {"search_jobs": "search_jobs", "finish": "finish"})
    graph.add_edge("search_jobs", "finish")
    graph.add_edge("finish", END)
    state = graph.compile().invoke({"objective": objective, "profile_id": profile_id, "plan": {},
        "profile": None, "results": {}, "errors": [], "tool_calls": 0, "summary": ""},
        {"recursion_limit": 8})
    return {"planner": planner.name, "plan": state["plan"], "profile_used": profile_id,
            "tool_calls": state["tool_calls"], "results": state["results"], "summary": state["summary"],
            "summary_is_deterministic": True}


def attach_agent_route(app: FastAPI, path: str, api_key_dependency,
                       embedder: EmbeddingProvider | None, planner: Planner | None) -> None:
    @app.post("/v1/jobs/agent-search", dependencies=[Depends(api_key_dependency)])
    def agent_search(request: AgentSearchRequest) -> dict:
        if planner is None:
            raise HTTPException(503, "Set PROBEFLOW_AGENT_MODEL to enable the local structured-output planner")
        try:
            return run_job_search_agent(path, request.objective, request.profile_id, planner, embedder)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
