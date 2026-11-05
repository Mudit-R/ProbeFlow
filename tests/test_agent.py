from probeflow.agent import Decision, investigate
from probeflow.runner import Case, Check


class FixtureModel:
    """Controlled decisions prove orchestration behavior, never LLM quality."""
    name = "fixture:controlled-decisions"

    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def decide(self, messages):
        value = next(self.decisions)
        if isinstance(value, Exception):
            raise value
        return Decision.model_validate(value)


def health_case():
    return Case(id="health", method="GET", path="/healthz", status=200, checks=[Check(path="status", equals="ok")])


def test_model_decision_runs_real_http_tool(service):
    model = FixtureModel([{"action": "run_check", "check_id": "health"}, {"action": "finish", "summary": "E1 returned HTTP 200 with status ok."}])
    result = investigate(service["url"], service["key"], [health_case()], model, "Check health")
    assert result["stop_reason"] == "model_finished"
    assert result["observations"][0]["passed"]
    assert result["observations"][0]["evidence_id"] == "E1"
    assert result["model"].startswith("fixture:")


def test_unknown_check_rejected(service):
    model = FixtureModel([{"action": "run_check", "check_id": "delete_everything"}, {"action": "finish", "summary": "Rejected"}])
    result = investigate(service["url"], service["key"], [health_case()], model, "Check health")
    assert result["observations"] == []
    assert "Rejected unapproved check" in result["errors"][0]


def test_step_budget_bounds_loop(service):
    model = FixtureModel([{"action": "run_check", "check_id": "health"}] * 10)
    result = investigate(service["url"], service["key"], [health_case()], model, "Check repeatedly", max_steps=3)
    assert result["steps"] == 3 and result["stop_reason"] == "step_limit"
    assert len(result["observations"]) == 2


def test_invalid_model_json_is_recorded_then_retried(service):
    model = FixtureModel([ValueError("malformed JSON"), {"action": "run_check", "check_id": "health"}, {"action": "finish", "summary": "E1 succeeded"}])
    result = investigate(service["url"], service["key"], [health_case()], model, "Check health")
    assert result["errors"] == ["ValueError: malformed JSON"]
    assert result["observations"][0]["passed"]


def test_write_operation_excluded_from_agent(service):
    write = Case(id="write", method="PUT", path="/v1/documents/x", status=200, read_only=False)
    model = FixtureModel([{"action": "run_check", "check_id": "write"}, {"action": "finish", "summary": "Rejected"}])
    result = investigate(service["url"], service["key"], [health_case(), write], model, "Write something")
    assert not result["observations"] and result["errors"]

