"""Read-only loopback viewer for the current history-learning checkpoint."""

from __future__ import annotations
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit
import argparse, json, platform, shutil, subprocess, sys, tempfile
from verge.models import task_identity
from verge.memory import ExperienceMemory
from verge.web_observation import load_observation

ASSETS = Path(__file__).with_name("web_assets")
_TRANSLATOR_SOURCE = Path(__file__).with_name("apple_translate.swift")
_translator_lock = Lock()
_translator_dir: tempfile.TemporaryDirectory[str] | None = None


class TranslationUnavailable(RuntimeError):
    """Displayable error from the on-device translator; never contains source text."""


def _translate_apple(text: str) -> str:
    """Compile the native bridge once, then translate via Apple without uploading text."""
    global _translator_dir
    version = platform.mac_ver()[0]
    if sys.platform != "darwin" or not version or int(version.split(".")[0]) < 26:
        raise TranslationUnavailable("苹果设备端翻译需要 macOS 26 或更新版本。")
    if not shutil.which("swiftc"):
        raise TranslationUnavailable(
            "未找到 Swift 编译器；请安装 Xcode Command Line Tools。"
        )
    with _translator_lock:
        if _translator_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="verge-apple-translation-")
            binary = Path(temporary.name) / "translate"
            try:
                result = subprocess.run(
                    [
                        "swiftc",
                        "-parse-as-library",
                        "-O",
                        "-o",
                        str(binary),
                        str(_TRANSLATOR_SOURCE),
                    ],
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode:
                    raise TranslationUnavailable(
                        "苹果翻译组件编译失败，请检查 Swift/macOS SDK 版本。"
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                temporary.cleanup()
                raise TranslationUnavailable("无法启动或编译苹果翻译组件。") from exc
            except TranslationUnavailable:
                temporary.cleanup()
                raise
            _translator_dir = temporary
        binary = Path(_translator_dir.name) / "translate"
    try:
        result = subprocess.run(
            [str(binary)],
            input=text,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranslationUnavailable("苹果翻译服务未响应，请稍后重试。") from exc
    if result.returncode:
        if "LANGUAGE_PACK_NOT_INSTALLED" in result.stderr:
            raise TranslationUnavailable(
                "未安装所需语言包：请在系统设置 → 通用 → 语言与地区 → 翻译语言中下载英语和简体中文。"
            )
        raise TranslationUnavailable(
            "苹果设备端翻译失败，请确认原文语言受支持并已安装相应语言包。"
        )
    return result.stdout.rstrip("\n")


def discover_runs(root):
    rows = []
    for path in sorted(
        Path(root).glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if (path.parent / "checkpoint.json").exists():
            try:
                manifest = json.loads(path.read_text())
                rows.append(
                    {
                        "name": path.parent.name,
                        "status": manifest.get("status", "unknown"),
                    }
                )
            except (ValueError, OSError):
                continue
    return rows


def load_run(directory):
    directory = Path(directory)
    checkpoint = directory / "checkpoint.json"
    saved = json.loads(checkpoint.read_text())
    schema = saved.get("schema")
    memory = ExperienceMemory.load(checkpoint)
    samples = [r.to_dict() for r in memory.trajectories.values()]
    experiences = [e.to_dict() for e in memory.experiences.values()]
    feedback = [
        {**asdict(r), "phase": "downstream", "credits": r.credits}
        for r in memory.ledger.records
    ]
    task_groups = memory.clusters.assignments("task")
    experience_groups = memory.clusters.assignments("experience")
    references = {
        tid: sorted(memory.clusters.references("task", tid)) for tid in task_groups
    }
    statistics = [
        {
            "task_group": g,
            "experience_id": eid,
            "mean_reward": value.mean_reward,
            "tasks": value.tasks,
        }
        for (g, eid), value in memory.ledger.by_task_group(task_groups).items()
    ]
    return {
        "name": directory.name,
        "schema": schema,
        "cluster_update": (
            asdict(memory.clusters.last_update) if memory.clusters.last_update else None
        ),
        "manifest": json.loads((directory / "manifest.json").read_text()),
        "samples": samples,
        "experiences": experiences,
        "feedback": feedback,
        "task_groups": task_groups,
        "experience_groups": experience_groups,
        "task_index": {
            task_identity(r["sample"]["task"]): r["sample"]["sample_id"]
            for r in samples
        },
        "references": references,
        "group_statistics": statistics,
    }


def load_role_io(directory, sample_id):
    path = Path(directory) / "role_io.jsonl"
    if not path.exists():
        return {"entries": [], "warnings": []}
    if path.stat().st_size > 1024**3:
        raise ValueError("调用记录过大，暂不扫描")
    entries = []
    warnings = []
    with path.open() as stream:
        while True:
            line = stream.readline(4 * 1024**2 + 1)
            if not line:
                break
            if len(line) > 4 * 1024**2:
                while line and not line.endswith("\n"):
                    line = stream.readline(4 * 1024**2)
                warnings.append("已跳过超大记录")
                continue
            try:
                row = json.loads(line)
            except ValueError:
                warnings.append("已跳过不完整日志行")
                continue
            if row.get("sample_id") == sample_id:
                entries.append(row)
                if len(entries) >= 512:
                    warnings.append("已达单题记录上限")
                    break
    return {"entries": entries, "warnings": warnings[:10]}


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def resolve_run(self, name):
        root = self.server.runs_root.resolve()
        directory = (root / name).resolve()
        if directory.parent != root or not (directory / "checkpoint.json").is_file():
            raise ValueError("运行不存在")
        return directory

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if parsed.path == "/api/runs":
                return self.send_json(
                    200, {"runs": discover_runs(self.server.runs_root)}
                )
            if parsed.path in ("/api/run", "/api/role-io", "/api/observation"):
                directory = self.resolve_run(query.get("run", [""])[0])
                if parsed.path == "/api/run":
                    result = load_run(directory)
                elif parsed.path == "/api/observation":
                    result = load_observation(directory)
                else:
                    result = load_role_io(directory, query.get("sample_id", [""])[0])
                return self.send_json(200, result)
            files = {
                "/": "index.html",
                "/index.html": "index.html",
                "/style.css": "style.css",
                "/app.js": "app.js",
                "/ui.mjs": "ui.mjs",
                "/metrics.mjs": "metrics.mjs",
                "/calls.mjs": "calls.mjs",
                "/trace.mjs": "trace.mjs",
                "/views.mjs": "views.mjs",
            }
            if parsed.path not in files:
                return self.send_json(404, {"error": "未找到页面"})
            path = ASSETS / files[parsed.path]
            body = path.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".js": "text/javascript; charset=utf-8",
                    ".mjs": "text/javascript; charset=utf-8",
                }[path.suffix],
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})

    def do_POST(self):
        if self.path != "/api/translate":
            return self.send_json(404, {"error": "未找到接口"})
        origin = self.headers.get("Origin")
        if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
            return self.send_json(403, {"error": "不允许跨域请求"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("请求大小无效")
            data = json.loads(self.rfile.read(size))
            text = data.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 12000:
                raise ValueError("翻译文本须为1至12000字符")
            self.send_json(200, {"translation": _translate_apple(text)})
        except TranslationUnavailable as exc:
            self.send_json(503, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    server.daemon_threads = True
    server.runs_root = Path(args.runs_root)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
