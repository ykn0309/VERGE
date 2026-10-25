"""Read-only, loopback-only dashboard for completed or in-progress VERGE runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlsplit

from verge.models import Experience, TrajectoryRecord
from verge.prompts import PROMPT_VERSION

ASSETS = Path(__file__).with_name("web_assets")
_MAX_ROLE_IO_BYTES = (
    1024 * 1024 * 1024
)  # Do not scan multi-gigabyte artifacts per HTTP request.
_MAX_ROLE_LINE_BYTES = 4 * 1024 * 1024
_MAX_ROLE_ENTRIES = 256
_MAX_ROLE_WARNINGS = 12
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
    root = Path(root)
    if not root.is_dir():
        return []
    names = []
    for p in root.iterdir():
        if p.is_dir() and (p / "run_manifest.json").is_file():
            try:
                data = json.loads((p / "run_manifest.json").read_text())
                if (
                    data.get("format_version") == 13
                    and data.get("method") == "semistructured_bayesian_v2"
                ):
                    names.append(p.name)
            except (OSError, ValueError):
                pass
    return sorted(names, reverse=True)


def _run_artifacts(directory):
    directory = Path(directory).resolve()
    m = json.loads((directory / "run_manifest.json").read_text())
    if m.get("format_version") != 13 or m.get("method") != "semistructured_bayesian_v2":
        raise ValueError("V2 run required")
    artifacts = (
        directory / m["config"]["store"]["config"].get("root_dir", "artifacts")
    ).resolve()
    if directory not in artifacts.parents:
        raise ValueError("store path escapes run")
    return m, artifacts


def load_role_io(run_dir, sample_id):
    m, root = _run_artifacts(run_dir)
    p = root / "role_io.jsonl"
    entries = []
    warnings = []
    if p.exists():
        if p.stat().st_size > _MAX_ROLE_IO_BYTES:
            raise ValueError("role log too large")
        with p.open() as f:
            for line in f:
                if len(line.encode()) > _MAX_ROLE_LINE_BYTES:
                    continue
                e = json.loads(line)
                if e.get("sample_id") == sample_id:
                    if len(entries) < _MAX_ROLE_ENTRIES:
                        entries.append(e)
                    elif not warnings:
                        warnings.append("Role log display limit reached")
    return {"entries": entries, "warnings": warnings}


def load_run(run_dir):
    from verge.memory import ExperienceMemory, TrajectoryLog

    m, root = _run_artifacts(run_dir)
    memory = ExperienceMemory(root, read_only=True)
    records = list(TrajectoryLog(root).all())
    samples = []
    for r in records:
        data = r.to_dict()
        if r.sample:
            data.update(r.sample.to_dict())
        samples.append(data)
    experiences = [e.to_dict() for e in memory.all()]
    return {
        "name": Path(run_dir).name,
        "manifest": m,
        "samples": samples,
        "experiences": experiences,
        "revisions": [r.to_dict() for r in memory.revisions],
        "warnings": [],
        "metrics": {
            "tasks": len(samples),
            "initial_correct": sum(
                bool(r.initial_judgment and r.initial_judgment.success) for r in records
            ),
            "active": sum(e.status == "active" for e in memory.all()),
            "retired": sum(e.status == "retired" for e in memory.all()),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    run_dir: Path
    runs_root: Path | None = None

    def do_POST(self) -> None:
        """Translate only a verified sample field; do not accept arbitrary text from the browser."""
        host = self.headers.get("Host", "").split(":")[0]
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost") or (
            origin and origin != f"http://{self.headers['Host']}"
        ):
            self.send_error(403, "invalid origin")
            return
        if urlsplit(self.path).path != "/api/translate" or urlsplit(self.path).query:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if (
            self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            != "application/json"
            or not 0 < size <= 4096
        ):
            self._json_error(400, "请求格式无效。")
            return
        try:
            request = json.loads(self.rfile.read(size))
        except (UnicodeError, ValueError):
            self._json_error(400, "请求 JSON 无效。")
            return
        if not isinstance(request, dict) or set(request) != {
            "run",
            "sample_id",
            "field",
        }:
            self._json_error(400, "请求字段无效。")
            return
        name, sample_id, field = (
            request.get(key) for key in ("run", "sample_id", "field")
        )
        root = self.runs_root or self.run_dir.parent
        if (
            not isinstance(name, str)
            or name not in discover_runs(root)
            or not isinstance(sample_id, str)
            or field not in ("task", "chosen", "rejected")
        ):
            self._json_error(404, "未找到运行或样本。")
            return
        try:
            sample = next(
                (
                    item
                    for item in load_run(root / name)["samples"]
                    if item["sample_id"] == sample_id
                ),
                None,
            )
            text = sample.get(field) if sample else None
            if not isinstance(text, str) or not text.strip():
                self._json_error(404, "该样本原文不可用或尚未验证。")
                return
            if len(text) > 20000:
                self._json_error(413, "原文超过单次翻译的 20000 字符上限。")
                return
            translated = _translate_apple(text)
        except TranslationUnavailable as exc:
            self._json_error(503, str(exc))
            return
        except (OSError, ValueError, KeyError, TypeError):
            self._json_error(500, "运行数据暂时无法读取。")
            return
        body = json.dumps({"translation": translated}, ensure_ascii=False).encode(
            "utf-8"
        )
        self._send(200, "application/json; charset=utf-8", body)

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:
        host = self.headers.get("Host", "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403, "invalid host")
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        root = self.runs_root or self.run_dir.parent
        if route == "/api/role-io":
            values = parse_qs(parsed.query, keep_blank_values=True)
            if set(values) != {"name", "sample_id"} or any(
                len(values[key]) != 1 for key in values
            ):
                self._json_error(400, "需要唯一的运行名称与样本 ID。")
                return
            name, sample_id = values["name"][0], values["sample_id"][0]
            if name not in discover_runs(root) or not sample_id or len(sample_id) > 512:
                self._json_error(404, "未找到运行或样本。")
                return
            try:
                directory = root / name
                if not any(
                    item["sample_id"] == sample_id
                    for item in load_run(directory)["samples"]
                ):
                    self._json_error(404, "未找到该样本。")
                    return
                result = load_role_io(directory, sample_id)
            except (OSError, ValueError, KeyError, TypeError):
                self._json_error(500, "运行数据暂时无法读取。")
                return
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            )
        elif route == "/api/runs":
            names = discover_runs(root)
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(
                    {"runs": names, "default": self.run_dir.name}, ensure_ascii=False
                ).encode("utf-8"),
            )
        elif route == "/api/run":
            values = parse_qs(parsed.query).get("name", [])
            if len(values) > 1:
                self.send_error(400, "only one run name allowed")
                return
            name = values[0] if values else self.run_dir.name
            if name not in discover_runs(root):
                self.send_error(404, "run not found")
                return
            try:
                data = json.dumps(
                    load_run(root / name), ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                data = json.dumps({"error": str(exc)}, ensure_ascii=False).encode(
                    "utf-8"
                )
                self._send(500, "application/json; charset=utf-8", data)
                return
            self._send(200, "application/json; charset=utf-8", data)
        elif route == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        elif route in ("/", "/index.html", "/app.js", "/style.css"):
            name = "index.html" if route == "/" else route[1:]
            content_type = (
                "text/html"
                if name.endswith(".html")
                else "text/javascript" if name.endswith(".js") else "text/css"
            )
            self._send(
                200, f"{content_type}; charset=utf-8", (ASSETS / name).read_bytes()
            )
        else:
            self.send_error(404)

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only VERGE run dashboard")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="initial run; also discover sibling runs")
    source.add_argument(
        "--runs-dir", help="directory containing multiple run directories"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = (
        Path(args.runs_dir).expanduser().resolve()
        if args.runs_dir
        else Path(args.run_dir).expanduser().resolve().parent
    )
    if not root.is_dir():
        parser.error(f"runs directory does not exist: {root}")
    names = discover_runs(root)
    directory = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else root / (names[0] if names else "__pending__")
    )
    # Fail early, before accepting requests; --run-dir must be a direct child.
    if args.run_dir and (directory.parent != root or directory.name not in names):
        parser.error(
            "--run-dir must contain a manifest and be an immediate child of the runs directory"
        )
    if names:
        load_run(directory)
    handler = type(
        "RunDashboardHandler",
        (DashboardHandler,),
        {"run_dir": directory, "runs_root": root},
    )
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"VERGE dashboard: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
