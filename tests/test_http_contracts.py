import sqlite3
from pathlib import Path

import pytest

from probeflow.runner import load_cases, run_case
from probeflow.reference_api import connect

CASES = load_cases(Path(__file__).resolve().parents[1] / "fixtures/contracts.json")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_real_http_contract(client, service, case):
    result = run_case(client, case, service["key"])
    assert result.passed, result.failures


def test_sql_reconciliation_matches_database(client, service):
    response = client.get("/v1/vendors", headers={"X-API-Key": service["key"]})
    with sqlite3.connect(service["db"]) as db:
        expected = db.execute("SELECT vendor_id, SUM(amount) FROM invoices GROUP BY vendor_id ORDER BY vendor_id").fetchall()
    observed = [(v["id"], v["total_rupees"]) for v in response.json()["vendors"] if v["invoice_count"]]
    assert observed == expected


def test_xml_entity_expansion_rejected(client, service):
    payload = '<!DOCTYPE x [<!ENTITY steal SYSTEM "file:///private/secret">]><document><title>XXE</title><text>&steal;</text></document>'
    response = client.put("/v1/documents-xml/xxe", content=payload, headers={"X-API-Key": service["key"], "Content-Type": "application/xml"})
    assert response.status_code == 422


def test_xml_size_limit(client, service):
    response = client.put("/v1/documents-xml/large", content="x" * 50001, headers={"X-API-Key": service["key"]})
    assert response.status_code == 413


def test_wrong_api_key_rejected(client):
    assert client.get("/v1/vendors", headers={"X-API-Key": "wrong"}).status_code == 401


def test_malformed_json_has_documented_client_error(client, service):
    response = client.post("/v1/answer", headers={"X-API-Key": service["key"], "Content-Type": "application/json"}, content=b"{")
    spec = client.get("/openapi.json").json()
    assert str(response.status_code) in spec["paths"]["/v1/answer"]["post"]["responses"]
    assert "400" in spec["paths"]["/v1/answer"]["post"]["responses"]
    assert "422" in spec["paths"]["/v1/answer"]["post"]["responses"]


def test_document_upsert_keeps_fts_in_sync(client, service):
    headers = {"X-API-Key": service["key"]}
    for text in ["Cobaltquartz document before replacement", "Violetquartz document after replacement"]:
        assert client.put("/v1/documents/upsert_test", json={"title": "Unique fixture", "text": text}, headers=headers).status_code == 200
    before = client.post("/v1/answer", json={"question": "Cobaltquartz"}, headers=headers).json()
    after = client.post("/v1/answer", json={"question": "Violetquartz"}, headers=headers).json()
    assert before["abstained"]
    assert len(after["citations"]) == 1
    assert after["citations"][0]["document_id"] == "upsert_test"


def test_connection_is_closed_after_transaction(service):
    with connect(str(service["db"])) as db:
        assert db.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")

