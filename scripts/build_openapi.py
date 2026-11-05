"""Export only explicit safe operations from the synthetic reference service."""
import json
from pathlib import Path

from probeflow.reference_api import create_app
from probeflow.schema_testing import safe_schema

app = create_app(db_path=":memory:", api_key="schema-only-not-a-credential")
schema = safe_schema(app.openapi())
target = Path(__file__).resolve().parents[1] / "openapi" / "probeflow-readonly.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(schema, indent=2), encoding="utf-8")
print(f"Saved {sum(len([m for m in path if m in {'get', 'post', 'put', 'patch', 'delete'}]) for path in schema['paths'].values())} operations to {target}")
