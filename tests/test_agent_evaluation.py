import json
from pathlib import Path

from probeflow.agent_evaluation import JudgeDecision, evaluate_agent_answers


class FixtureJudge:
    name = "fixture:hand-labeled"
    last_usage = {"prompt_tokens": 12, "completion_tokens": 4, "latency_ms": 2.0}

    def evaluate(self, question, evidence, answer):
        return JudgeDecision(grounded=True, quality_score=5, rationale="fixture response")


def test_judge_evaluator_runs_all_labeled_examples_and_summarizes_metrics():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "fixtures" / "agent_judge_eval.json").read_text(encoding="utf-8"))["cases"]
    report = evaluate_agent_answers(cases, FixtureJudge())
    assert report["examples"] == 20
    assert report["label_source"].startswith("author-curated synthetic")
    assert 0 <= report["groundedness_accuracy"] <= 1
    assert report["quality_mean_absolute_error"] >= 0
    assert report["usage"]["prompt_tokens"] == 240
    assert len(report["details"]) == 20

