"""Safe OpenAPI subset + Schemathesis runner with local, reviewable reports."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

SAFE_DEFAULT_METHODS = {"get", "head", "options", "trace"}
HTTP_METHODS = {"get", "head", "options", "trace", "post", "put", "patch", "delete"}
EXTERNAL_REF = re.compile(r"^[^#]", re.MULTILINE)
HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")


def validate_target_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Target must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Target URL cannot embed credentials, a query or a fragment")
    if ".." in parsed.path.split("/"):
        raise ValueError("Target URL path cannot contain traversal segments")
    return value.rstrip("/")


def _check_refs(value: Any) -> None:
    if isinstance(value, dict):
        ref = value.get("$ref")
        if ref is not None and (not isinstance(ref, str) or not ref.startswith("#/")):
            raise ValueError("External OpenAPI references are disabled; bundle schema references first")
        for child in value.values():
            _check_refs(child)
    elif isinstance(value, list):
        for child in value:
            _check_refs(child)


def safe_schema(source: Any) -> dict:
    """Keep documented read-only methods; opt into other operations explicitly.

    Mutating methods run only when each OpenAPI operation includes
    `x-probeflow-safe: true`. The API owner is responsible for marking custom
    POST endpoints safe; generated checks still send fuzzed input values.
    """
    schema = copy.deepcopy(source)
    if not isinstance(schema, dict) or not str(schema.get("openapi", "")).startswith("3."):
        raise ValueError("Expected an OpenAPI 3.x document")
    _check_refs(schema)
    source_paths = schema.get("paths")
    if not isinstance(source_paths, dict):
        raise ValueError("OpenAPI document has no paths mapping")
    selected: dict[str, dict] = {}
    operation_count = 0
    for path, path_item in source_paths.items():
        if not isinstance(path_item, dict):
            raise ValueError(f"Invalid path item: {path}")
        kept = {key: copy.deepcopy(value) for key, value in path_item.items() if key not in HTTP_METHODS}
        kept.pop("servers", None)
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                raise ValueError(f"Invalid operation: {method.upper()} {path}")
            if method.lower() in SAFE_DEFAULT_METHODS or operation.get("x-probeflow-safe") is True:
                copy_of_operation = copy.deepcopy(operation)
                copy_of_operation.pop("callbacks", None)
                copy_of_operation.pop("servers", None)
                kept[method] = copy_of_operation
                operation_count += 1
        if any(key.lower() in HTTP_METHODS for key in kept):
            selected[path] = kept
    if not operation_count:
        raise ValueError("No safe operations found. Mark only verified, side-effect-free operations with x-probeflow-safe: true")
    schema["paths"] = selected
    # An OpenAPI spec may contain a credential in its server URL. The tested
    # origin comes from the explicit CLI target, so discard such overrides.
    schema.pop("servers", None)
    return schema


def load_safe_schema(source: str | Path) -> tuple[dict, int]:
    path = Path(source)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("OpenAPI schema exceeds the 10 MiB limit")
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw)
    safe = safe_schema(parsed)
    return safe, sum(len([m for m in item if m in HTTP_METHODS]) for item in safe["paths"].values())


def run_openapi(source: str | Path, base_url: str, output_root: str | Path,
                headers: dict[str, str] | None = None,
                header_env_names: dict[str, str] | None = None,
                max_examples: int = 20,
                max_time: int = 20, rate_limit: str = "5/s") -> dict:
    """Run bounded schema-derived checks; never logs auth headers to the report."""
    base_url = validate_target_url(base_url)
    if not 1 <= max_examples <= 1000 or not 1 <= max_time <= 300:
        raise ValueError("Use max_examples 1..1000 and max_time 1..300")
    if not re.fullmatch(r"\d{1,4}/(?:s|m)", rate_limit) or int(rate_limit.split("/")[0]) < 1:
        raise ValueError("rate_limit must look like 5/s or 100/m")
    schema, operation_count = load_safe_schema(source)
    auth_headers = dict(headers or {})
    for name, value in auth_headers.items():
        if not HEADER_NAME.fullmatch(name) or not isinstance(value, str) or not value or len(value) > 4096 or any(c in value for c in "\r\n\x00"):
            raise ValueError("Invalid authentication header")

    root = Path(output_root).resolve()
    run_dir = root / "openapi_runs" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    schema_path = run_dir / "safe-openapi.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    junit_path = run_dir / "junit.xml"
    json_path = run_dir / "result.json"
    html_path = run_dir / "html-report"
    html_path.mkdir()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    header_items = []
    variable_names = dict(header_env_names or {})
    for index, (name, value) in enumerate(auth_headers.items()):
        variable = variable_names.get(name, f"PROBEFLOW_RUN_HEADER_{index}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", variable):
            raise ValueError("Invalid environment variable name for header")
        env[variable] = value
        placeholder = "${" + variable + "}"
        header_items.append(f"{json.dumps(name)} = {json.dumps(placeholder)}")
    # Configuration supports environment substitution; credentials never go
    # into command-line arguments, saved schemas, or the report directory.
    config_path = run_dir / "schemathesis.toml"
    config_path.write_text("headers = { " + ", ".join(header_items) + " }\nrate-limit = " + json.dumps(rate_limit) + "\nmax-redirects = 0\n", encoding="utf-8")

    st = "st.exe" if os.name == "nt" else "st"
    executable = Path(sys.executable).parent / st
    if not executable.exists():
        executable = Path(sys.executable).parent / ("schemathesis.exe" if os.name == "nt" else "schemathesis")
    if not executable.exists():
        config_path.unlink(missing_ok=True)
        raise RuntimeError("Schemathesis is missing. Install the project with its dev dependencies.")
    command = [str(executable), "--config-file", str(config_path), "run", str(schema_path),
               "--url", base_url, "--workers", "1", "--max-time", str(max_time),
               "--max-examples", str(max_examples), "--max-failures", "10",
               "--request-timeout", "5", "--report-junit-path", str(junit_path),
               "--report-json-path", str(json_path), "--report-html-path", str(html_path),
               "--output-sanitize", "true", "--no-color"]
    process = subprocess.run(command, cwd=run_dir, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=max_time + 45, check=False)
    summary = {}
    if json_path.exists():
        try:
            summary = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}
    return {"id": run_dir.name, "status": "passed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode, "operation_count": operation_count,
            "run_directory": str(run_dir), "junit_report": str(junit_path) if junit_path.exists() else None,
            "html_report": str(html_path / "index.html") if (html_path / "index.html").exists() else None,
            "json_report": str(json_path) if json_path.exists() else None,
            "summary": summary, "output": process.stdout[-12000:], "stderr": process.stderr[-4000:]}

