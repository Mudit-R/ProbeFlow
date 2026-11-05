import asyncio
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from probeflow.evaluation import benchmark, evaluate, score_prediction
from probeflow.runner import Case, Check, at_path, load_cases, run_case
from probeflow.schema_testing import load_safe_schema, run_openapi, safe_schema, validate_target_url


def test_failed_contract_records_defect(client, service):
    case = Case(id="intentional_failure", method="GET", path="/healthz", status=200, checks=[Check(path="status", equals="broken")])
    result = run_case(client, case, service["key"])
    assert not result.passed
    assert result.failures == ["status: expected 'broken', observed 'ok'"]


def test_missing_path_is_failure(client, service):
    case = Case(id="missing_path", method="GET", path="/healthz", status=200, checks=[Check(path="missing.field", equals=1)])
    result = run_case(client, case, service["key"])
    assert not result.passed and "Missing response path" in result.failures[0]


def test_contract_type_mismatch_is_failure(client, service):
    case = Case(id="bad_type", method="GET", path="/v1/vendors", status=200, checks=[Check(path="vendors.0.invoice_count", equals=True)])
    assert not run_case(client, case, service["key"]).passed


@pytest.mark.parametrize("path", ["https://example.org/", "//external/x", "/x?redirect=abc", "/../secrets"])
def test_cases_reject_arbitrary_urls(path):
    with pytest.raises(ValidationError):
        Case(id="unsafe", method="GET", path=path, status=200)


def test_duplicate_case_ids_rejected(tmp_path):
    source = tmp_path / "duplicates.json"
    case = {"id": "same", "method": "GET", "path": "/healthz", "status": 200}
    source.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(source)


def test_nested_list_path():
    assert at_path({"rows": [{"total": 17}]}, "rows.0.total") == 17


def test_retrieval_rank_and_citation_validity():
    gold = {"id": "rank", "document_id": "target", "allowed_document_ids": ["other", "target"], "expected_phrase": "evidence"}
    result = score_prediction({"answer": "Evidence", "citations": [{"document_id": i, "title": i, "excerpt": "x"} for i in ["other", "target"]], "abstained": False}, gold)
    assert result["reciprocal_rank"] == .5
    assert result["retrieval_hit"] and result["expected_phrase_present"] and result["citations_valid"]


def test_unknown_citation_is_not_valid():
    gold = {"id": "bad", "document_id": "target", "allowed_document_ids": ["target"]}
    result = score_prediction({"answer": "x", "citations": [{"document_id": "invented", "title": "x", "excerpt": "x"}], "abstained": False}, gold)
    assert not result["citations_valid"] and not result["retrieval_hit"]


def test_unanswerable_abstention():
    result = score_prediction({"answer": "Insufficient evidence", "citations": [], "abstained": True}, {"id": "unknown", "document_id": None, "allowed_document_ids": []})
    assert result["abstention_correct"] and result["retrieval_hit"] is None


def test_evaluation_keeps_raw_predictions(service, root):
    report = evaluate(service["url"], service["key"], root / "fixtures/retrieval_gold.json")
    assert report["examples"] == 20 and report["answerable"] == 15
    assert len(report["predictions"]) == 20
    assert 0 <= report["mrr_at_3"] <= 1
    assert all("prediction" in row for row in report["predictions"])


def test_benchmark_preserves_samples_and_counts(service):
    report = asyncio.run(benchmark(service["url"], service["key"], requests=12, concurrency=3))
    assert len(report["raw_samples"]) == 12
    assert report["status_counts"] == {"200": 12}
    assert report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"]
    assert math.isfinite(report["requests_per_second"])


@pytest.mark.parametrize("requests,concurrency", [(0, 1), (10, 0), (10, 101)])
def test_benchmark_limits(requests, concurrency):
    with pytest.raises(ValueError):
        asyncio.run(benchmark("http://127.0.0.1:1", "x", requests, concurrency))


def test_openapi_default_rejects_mutations_and_remote_refs():
    document = {"openapi": "3.1.0", "info": {"title": "demo", "version": "1"},
                "servers": [{"url": "https://user:pass@example.com?token=hidden"}],
                "paths": {"/safe": {"get": {"responses": {"200": {"description": "ok"}}},
                                      "post": {"responses": {"201": {"description": "created"}}}},
                          "/explicit": {"post": {"x-probeflow-safe": True, "responses": {"200": {"description": "read"}}}},
                          "/write": {"put": {"responses": {"200": {"description": "changed"}}}}}}
    filtered = safe_schema(document)
    assert set(filtered["paths"]) == {"/safe", "/explicit"}
    assert "get" in filtered["paths"]["/safe"]
    assert "post" not in filtered["paths"]["/safe"]
    assert "/write" not in filtered["paths"]
    assert "servers" not in filtered


def test_external_openapi_reference_rejected():
    document = {"openapi": "3.1.0", "paths": {"/x": {"get": {"$ref": "https://example.org/schema.json"}}}}
    with pytest.raises(ValueError, match="External OpenAPI"):
        safe_schema(document)


@pytest.mark.parametrize("url", ["ftp://example.org/api", "http://user:pass@example.org", "https://example.org/api?token=abc", "https://example.org/a/../b"])
def test_unsafe_target_url_rejected(url):
    with pytest.raises(ValueError):
        validate_target_url(url)


def test_openapi_spec_does_not_embed_target_api_key(service, root, tmp_path):
    document, operation_count = load_safe_schema(root / "openapi/probeflow-readonly.json")
    assert operation_count == 3
    assert set(document["paths"]) == {"/healthz", "/v1/answer", "/v1/vendors"}
    result = run_openapi(root / "openapi/probeflow-readonly.json", service["url"], tmp_path,
                         {"X-API-Key": service["key"]}, max_examples=20, max_time=25, rate_limit="20/s")
    assert result["status"] == "passed", result["output"] + result["stderr"]
    assert result["operation_count"] == 3
    assert Path(result["junit_report"]).is_file()
    assert Path(result["json_report"]).is_file()
    assert Path(result["html_report"]).is_file()
    saved_schema = Path(result["run_directory"], "safe-openapi.json").read_text(encoding="utf-8")
    assert service["key"] not in saved_schema
    config = Path(result["run_directory"], "schemathesis.toml")
    assert config.is_file()
    assert service["key"] not in config.read_text(encoding="utf-8")
    assert "PROBEFLOW_RUN_HEADER_0" in config.read_text(encoding="utf-8")

