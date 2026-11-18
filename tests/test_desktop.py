"""Container deployment contracts, including negative controls for local file access."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("websockify")

from organoidtracker.desktop import checks, server  # noqa: E402


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    for name in ("static", "results", "runtime"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(server, "STATIC", tmp_path / "static")
    monkeypatch.setattr(server, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(server, "RUNTIME", tmp_path / "runtime")
    (server.STATIC / "index.html").write_text("<h1>Desktop</h1>")
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()

    def request(path, method="GET", headers=None):
        client = HTTPConnection("127.0.0.1", http.server_port, timeout=5)
        client.request(method, path, headers=headers or {})
        response = client.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        client.close()
        return result

    yield request
    http.shutdown()
    http.server_close()
    thread.join()


def test_results_download_confines_paths_and_escapes_listing(endpoint, tmp_path):
    (server.RESULTS / "report.pdf").write_bytes(b"%PDF-example")
    (server.RESULTS / "<script>.csv").write_text("x,y\n1,2\n")
    secret = tmp_path / "outside.txt"
    secret.write_text("must not be served")
    (server.RESULTS / "escape").symlink_to(secret)
    status, headers, body = endpoint("/reports/report.pdf")
    assert status == 200 and body == b"%PDF-example"
    assert headers["Content-Disposition"].startswith("attachment;")
    status, _, body = endpoint("/reports/")
    assert status == 200 and b"&lt;script&gt;.csv" in body and b"<script>" not in body
    assert b"escape" not in body
    for path in ("/reports/escape", "/reports/%2e%2e/outside.txt", "/%2e%2e/outside.txt"):
        assert endpoint(path)[0] == 404
    assert endpoint("/", headers={"Host": "attacker.example"})[0] == 403


def test_ready_state_and_stop_requires_same_origin(endpoint, monkeypatch):
    assert json.loads(endpoint("/api/status")[2]) == {"ready": False}
    (server.RUNTIME / "ready").write_text("123")
    assert json.loads(endpoint("/api/status")[2])["ready"]
    called = []
    monkeypatch.setenv("OT_SUPERVISOR_PID", "123")
    monkeypatch.setattr(server.os, "kill", lambda *args: called.append(args))
    for headers in ({}, {"Origin": "http://attacker.example", "X-OrganoidTracker": "local"}):
        assert endpoint("/api/stop", "POST", headers)[0] == 403
        assert not called
    headers = {"Host": "127.0.0.1:1234", "Origin": "http://127.0.0.1:1234", "X-OrganoidTracker": "local"}
    assert endpoint("/api/stop", "POST", headers)[0] == 200
    assert called == [(123, server.signal.SIGTERM)]


def test_gpu_absence_cannot_fall_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU mode cannot access CUDA"):
        checks.check_device("cuda")
    assert checks.check_device("cpu")["mode"] == "cpu"
    with pytest.raises(ValueError, match="explicitly"):
        checks.check_device("auto")


def test_cached_model_is_verified_offline_and_corruption_preserved(tmp_path, monkeypatch):
    from organoidtracker.desktop import models

    content = b"synthetic checkpoint fixture"
    import hashlib

    monkeypatch.setitem(models.MODELS, "tiny", (hashlib.sha256(content).hexdigest(), len(content)))
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    with pytest.raises(FileNotFoundError, match="not cached"):
        models.prepare(tmp_path, "tiny", offline=True)
    checkpoint.write_bytes(content)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("cached run accessed network"))
    assert models.prepare(tmp_path, "tiny", offline=True) == checkpoint
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        models.prepare(tmp_path, "tiny")
    assert checkpoint.read_bytes() == b"corrupted"
