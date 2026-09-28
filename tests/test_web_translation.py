"""Translation requests may access only verified sample fields in an allowed run."""

import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

from test_web import _run
from verge.web import DashboardHandler, TranslationUnavailable


def test_translation_is_explicit_and_only_uses_verified_sample_content(tmp_path, monkeypatch):
    directory, paths = _run(tmp_path)
    sent = []
    monkeypatch.setattr("verge.web._translate_apple", lambda text: sent.append(text) or "中文译文")
    handler = type("TranslationHandler", (DashboardHandler,), {"run_dir": directory})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = HTTPConnection("127.0.0.1", server.server_port)
        def translate(sample="a", field="task", **kwargs):
            body = json.dumps({"run": directory.name, "sample_id": sample,
                               "field": field, **kwargs}).encode()
            client.request("POST", "/api/translate", body,
                           headers={"Content-Type": "application/json"})
            response = client.getresponse()
            return response.status, json.loads(response.read())
        try:
            status, result = translate()
            assert (status, result) == (200, {"translation": "中文译文"})
            assert sent == ["<script>not code</script>"]
            status, _ = translate(field="chosen")
            assert status == 200 and sent[-1] == "好"
            status, _ = translate(sample="not-found")
            assert status == 404 and len(sent) == 2
            status, _ = translate(field="api_key")
            assert status == 404 and len(sent) == 2
            status, _ = translate(text="arbitrary", field="task")
            assert status == 400 and len(sent) == 2
            body = json.dumps({"run": directory.name, "sample_id": "a", "field": "task"})
            client.request("POST", "/api/translate", body,
                           headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
            response = client.getresponse()
            assert response.status == 403 and not response.read().startswith(b'{"translation"')
            assert len(sent) == 2
            # Original dataset changes must never be sent to the translator, even if the ID matches.
            from pathlib import Path
            Path(paths["train"]).write_text('"changed"\n', encoding="utf-8")
            status, _ = translate()
            assert status == 404 and len(sent) == 2
        finally:
            client.close()
            server.shutdown()
            worker.join()


def test_translation_unavailable_has_safe_message(tmp_path, monkeypatch):
    directory, _ = _run(tmp_path)
    def fail(_text):
        raise TranslationUnavailable("请安装语言包")
    monkeypatch.setattr("verge.web._translate_apple", fail)
    handler = type("TranslationErrorHandler", (DashboardHandler,), {"run_dir": directory})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port)
            client.request("POST", "/api/translate",
                           json.dumps({"run": directory.name, "sample_id": "a", "field": "task"}),
                           headers={"Content-Type": "application/json"})
            response = client.getresponse()
            assert response.status == 503
            assert json.loads(response.read()) == {"error": "请安装语言包"}
            client.close()
        finally:
            server.shutdown()
            worker.join()
