"""Launch an isolated real server, run CLI checks, retain measured artifacts."""
from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="probeflow-") as temp:
        env = dict(os.environ, PROBEFLOW_API_KEY=secrets.token_hex(24), PROBEFLOW_DB=str(Path(temp) / "reference.db"),
                   PYTHONPATH=os.pathsep.join([str(ROOT / "src"), *[p for p in sys.path if p]]))
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        log = (artifacts / "reference-validation.log").open("w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        interpreter = getattr(sys, "_base_executable", sys.executable)
        process = subprocess.Popen([interpreter, "-m", "uvicorn", "probeflow.reference_api:create_app", "--factory", "--port", str(port), "--no-access-log"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        url = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(timeout=.5, trust_env=False) as client:
                for _ in range(200):
                    if process.poll() is not None:
                        raise RuntimeError("Server failed: " + (artifacts / "reference-validation.log").read_text())
                    try:
                        if client.get(url + "/healthz").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.05)
                else:
                    raise RuntimeError("Server startup timeout")
            for command, filename, extra in [("suite", "contracts.json", []), ("evaluate", "retrieval.json", []), ("benchmark", "benchmark.json", ["--requests", "300", "--concurrency", "10"])]:
                subprocess.run([sys.executable, "-m", "probeflow.cli", command, "--base-url", url, "--out", "artifacts/" + filename] + extra, env=env, cwd=ROOT, check=True)
            if "--schemathesis" in sys.argv:
                subprocess.run([sys.executable, "-m", "probeflow.cli", "openapi", "--schema", "openapi/probeflow-readonly.json", "--target-url", url, "--header-env", "X-API-Key=PROBEFLOW_API_KEY", "--max-examples", "20", "--max-time", "30", "--out", "artifacts/schemathesis-result.json"], env=env, cwd=ROOT, check=True)
            if "--live-model" in sys.argv:
                subprocess.run([sys.executable, "-m", "probeflow.cli", "investigate", "--base-url", url, "--out", "artifacts/live_model.json"], env=env, cwd=ROOT, check=True)
            if "--newman" in sys.argv:
                executable = shutil.which("npx.cmd" if os.name == "nt" else "npx")
                if not executable:
                    raise RuntimeError("Node/npx is needed for --newman")
                subprocess.run([executable, "--yes", "newman@6", "run", "postman/ProbeFlow.postman_collection.json", "--env-var", "base_url=" + url, "--env-var", "api_key=" + env["PROBEFLOW_API_KEY"], "--reporters", "cli,junit,json", "--reporter-junit-export", "artifacts/newman.xml", "--reporter-json-export", str(Path(temp) / "newman-full.json")], cwd=ROOT, check=True, creationflags=flags)
                # Newman exports request headers. Keep only aggregate counts to
                # avoid persisting even the temporary validation API credential.
                import json
                report = json.loads((Path(temp) / "newman-full.json").read_text(encoding="utf-8"))
                summary = {
                    "stats": report["run"]["stats"],
                    "timings": report["run"]["timings"],
                    "failures": [
                        {"name": f["error"].get("name"), "message": f["error"].get("message")}
                        for f in report["run"]["failures"]
                    ],
                }
                (artifacts / "newman-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    main()
