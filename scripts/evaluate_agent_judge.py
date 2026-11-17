"""Run the local judge against labeled examples and emit a reproducible JSON report."""
from __future__ import annotations

import json
from pathlib import Path

from probeflow.agent_evaluation import configured_judge, evaluate_agent_answers

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    judge = configured_judge()
    if judge is None:
        raise SystemExit("Set PROBEFLOW_JUDGE_MODEL and start local Ollama before running this evaluation.")
    cases = json.loads((ROOT / "fixtures" / "agent_judge_eval.json").read_text(encoding="utf-8"))["cases"]
    report = evaluate_agent_answers(cases, judge)
    output = ROOT / "artifacts" / "agent_judge_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()

