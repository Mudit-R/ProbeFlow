from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


@pytest.fixture(scope="session")
def service(tmp_path_factory):
    work = tmp_path_factory.mktemp("http_service")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    db = work / "reference.db"
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PROBEFLOW_DB=str(db), PROBEFLOW_API_KEY=key,
               PYTHONPATH=os.pathsep.join([str(source), *[p for p in sys.path if p]]))
    log = (work / "server.log").open("w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # Bypass the Windows venv redirector so terminate()/wait() own the actual
    # server process, rather than leaving its interpreter child running.
    interpreter = getattr(sys, "_base_executable", sys.executable)
    process = subprocess.Popen([interpreter, "-m", "uvicorn", "probeflow.reference_api:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"], env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(trust_env=False, timeout=.5) as client:
            for _ in range(200):
                if process.poll() is not None:
                    raise RuntimeError((work / "server.log").read_text())
                try:
                    if client.get(url + "/healthz").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.05)
            else:
                raise RuntimeError("Reference HTTP service did not start")
        yield {"url": url, "key": key, "db": db}
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


@pytest.fixture
def client(service):
    with httpx.Client(base_url=service["url"], timeout=5, trust_env=False) as client:
        yield client


@pytest.fixture(scope="session")
def root():
    return Path(__file__).resolve().parents[1]

