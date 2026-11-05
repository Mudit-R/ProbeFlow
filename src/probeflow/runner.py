"""HTTP contract runner. Test cases contain data, never executable Python."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator


class Check(BaseModel):
    path: str
    equals: Any


class Case(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    method: str = Field(pattern=r"^(GET|POST|PUT)$")
    path: str = Field(pattern=r"^/[a-zA-Z0-9_/-]+$")
    status: int = Field(ge=100, le=599)
    json_body: dict | None = None
    xml_body: str | None = None
    authenticated: bool = True
    read_only: bool = True
    checks: list[Check] = []

    @field_validator("path")
    @classmethod
    def relative_service_path(cls, value: str) -> str:
        if "//" in value or ".." in value.split("/"):
            raise ValueError("Only service-relative paths are allowed")
        return value


@dataclass
class Result:
    case_id: str
    passed: bool
    status: int | None
    latency_ms: float
    failures: list[str]
    response: Any


def load_cases(path: str | Path) -> list[Case]:
    cases = [Case.model_validate(row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate case ids")
    return cases


def at_path(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def run_case(client: httpx.Client, case: Case, api_key: str) -> Result:
    headers = {"X-API-Key": api_key} if case.authenticated else {}
    if case.xml_body is not None:
        headers["Content-Type"] = "application/xml"
    started = time.perf_counter()
    failures: list[str] = []
    try:
        response = client.request(case.method, case.path, headers=headers, json=case.json_body, content=case.xml_body)
        elapsed = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text[:2000]}
        if response.status_code != case.status:
            failures.append(f"Expected HTTP {case.status}, observed {response.status_code}")
        for check in case.checks:
            try:
                actual = at_path(body, check.path)
                if type(actual) is not type(check.equals) or actual != check.equals:
                    failures.append(f"{check.path}: expected {check.equals!r}, observed {actual!r}")
            except (KeyError, IndexError, TypeError, ValueError):
                failures.append(f"Missing response path: {check.path}")
        return Result(case.id, not failures, response.status_code, round(elapsed, 3), failures, body)
    except httpx.HTTPError as exc:
        return Result(case.id, False, None, round((time.perf_counter() - started) * 1000, 3), [type(exc).__name__ + ": " + str(exc)], None)


def suite(base_url: str, api_key: str, cases: list[Case]) -> list[dict]:
    with httpx.Client(base_url=base_url, timeout=10, follow_redirects=False, trust_env=False) as client:
        return [asdict(run_case(client, case, api_key)) for case in cases]

