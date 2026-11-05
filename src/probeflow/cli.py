from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .agent import OllamaModel, investigate
from .evaluation import benchmark, evaluate
from .runner import load_cases, suite
from .schema_testing import run_openapi


def main() -> None:
    parser = argparse.ArgumentParser(description="ProbeFlow real-HTTP quality workbench")
    parser.add_argument("command", choices=["suite", "evaluate", "benchmark", "investigate", "openapi"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-key", default=os.getenv("PROBEFLOW_API_KEY", ""))
    parser.add_argument("--cases", type=Path, default=Path("fixtures/contracts.json"))
    parser.add_argument("--dataset", type=Path, default=Path("fixtures/retrieval_gold.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--objective", default="Check the answer API and access controls, investigate any failure, and summarize observed evidence.")
    parser.add_argument("--schema", type=Path, default=Path("openapi/probeflow-readonly.json"))
    parser.add_argument("--target-url", default="http://127.0.0.1:8765")
    parser.add_argument("--header-env", action="append", default=[], metavar="NAME=ENV_NAME", help="Attach a header read from an environment variable, without passing its value on the command line")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-time", type=int, default=20)
    args = parser.parse_args()
    if args.command not in {"openapi"} and not args.api_key:
        parser.error("Set PROBEFLOW_API_KEY or --api-key")
    if args.command == "suite":
        rows = suite(args.base_url, args.api_key, load_cases(args.cases))
        result = {"cases": len(rows), "passed": sum(row["passed"] for row in rows), "results": rows}
    elif args.command == "evaluate":
        result = evaluate(args.base_url, args.api_key, args.dataset)
    elif args.command == "benchmark":
        result = asyncio.run(benchmark(args.base_url, args.api_key, args.requests, args.concurrency))
    elif args.command == "investigate":
        result = investigate(args.base_url, args.api_key, load_cases(args.cases), OllamaModel(args.model), args.objective)
    else:
        header_values, header_env_names = {}, {}
        for setting in args.header_env:
            if "=" not in setting:
                parser.error("--header-env must be NAME=ENV_NAME")
            name, env_name = setting.split("=", 1)
            if env_name not in os.environ:
                parser.error(f"Environment variable {env_name!r} is not set")
            header_values[name] = os.environ[env_name]
            header_env_names[name] = env_name
        result = run_openapi(args.schema, args.target_url, args.out.parent,
                             headers=header_values, header_env_names=header_env_names,
                             max_examples=args.max_examples, max_time=args.max_time)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"results", "predictions", "raw_samples", "observations"}}, indent=2))
    if args.command == "suite" and result["passed"] != result["cases"]:
        raise SystemExit(1)
    if args.command == "openapi" and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
