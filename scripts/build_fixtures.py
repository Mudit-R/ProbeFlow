"""Generate transparent synthetic gold labels and a Postman collection.

There are no measured result constants here. Evaluation and benchmark commands
obtain results from a running API and write separate artifacts.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
allowed = ["retention", "access", "invoice", "incident", "backup", "onboarding"]
examples = [
    ("How long are audit records retained?", "retention", "seven years"),
    ("What is the retention policy for clinical operations?", "retention", "seven years"),
    ("Are archived records writable?", "retention", "read-only"),
    ("How often do access reviews occur?", "access", "ninety days"),
    ("Who approves access?", "access", "team lead"),
    ("Which invoices require finance approval?", "invoice", "50000"),
    ("What are invoice payment terms?", "invoice", "thirty days"),
    ("How soon are critical incidents escalated?", "incident", "fifteen minutes"),
    ("Who maintains the incident response timeline?", "incident", "incident commander"),
    ("How often are database backups run?", "backup", "daily"),
    ("How often is a recovery drill performed?", "backup", "monthly"),
    ("Where are restore results documented?", "backup", "documented"),
    ("What vendor onboarding requirements apply?", "onboarding", "risk assessment"),
    ("What agreement is signed before vendor activation?", "onboarding", "data processing agreement"),
    ("Which team records access decisions?", "access", "security team"),
    ("Who discovered Neptune?", None, ""),
    ("What is the tallest volcano?", None, ""),
    ("Explain photosynthesis chlorophyll", None, ""),
    ("Calculate orbital eccentricity", None, ""),
    ("What causes superconductivity?", None, ""),
]
gold = [{"id": f"q{i+1:02}", "question": question, "document_id": document, "expected_phrase": phrase, "allowed_document_ids": allowed} for i, (question, document, phrase) in enumerate(examples)]
(ROOT / "fixtures/retrieval_gold.json").write_text(json.dumps(gold, indent=2), encoding="utf-8")
items = []
for case in json.loads((ROOT / "fixtures/contracts.json").read_text(encoding="utf-8")):
    headers = [{"key": "X-API-Key", "value": "{{api_key}}"}] if case.get("authenticated", True) else []
    request = {"method": case["method"], "url": "{{base_url}}" + case["path"], "header": headers}
    if case.get("json_body") is not None:
        headers.append({"key": "Content-Type", "value": "application/json"})
        request["body"] = {"mode": "raw", "raw": json.dumps(case["json_body"]), "options": {"raw": {"language": "json"}}}
    if case.get("xml_body") is not None:
        headers.append({"key": "Content-Type", "value": "application/xml"})
        request["body"] = {"mode": "raw", "raw": case["xml_body"], "options": {"raw": {"language": "xml"}}}
    script = [f'pm.test("HTTP {case["status"]}", function () {{ pm.response.to.have.status({case["status"]}); }});']
    for check in case.get("checks", []):
        path = check["path"]
        script.append('pm.test(' + json.dumps(path) + ', function () { const observed = ' + json.dumps(path) + '.split(".").reduce((v,k) => v[k], pm.response.json()); pm.expect(observed).to.deep.equal(' + json.dumps(check["equals"]) + '); });')
    items.append({"name": case["id"], "request": request, "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": script}}]})
collection = {"info": {"name": "ProbeFlow HTTP Contracts", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json", "description": "Synthetic reference contracts. Set base_url and api_key; writes affect only explicit fixture documents."}, "item": items, "variable": [{"key": "base_url", "value": "http://127.0.0.1:8765"}, {"key": "api_key", "value": ""}]}
(ROOT / "postman").mkdir(exist_ok=True)
(ROOT / "postman/ProbeFlow.postman_collection.json").write_text(json.dumps(collection, indent=2), encoding="utf-8")
print(f"Generated {len(gold)} synthetic gold examples and {len(items)} Postman requests")

