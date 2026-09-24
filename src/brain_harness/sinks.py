"""
HarnessPlatform · 内置输出端（Sink）
======================================
Sink: 消费 Result（日志/文件/HTTP/指标）。
"""
from __future__ import annotations
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List

from .context import TickContext, Result
from .registry import sink


@sink("log")
class LogSink:
    """控制台 + 日志文件（人类可读）"""

    def __init__(self, path: str = "", echo: bool = True, **kw):
        self.name = "log"
        self.echo = echo
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def handle(self, r: Result, ctx: TickContext) -> None:
        who = "自发" if r.extra.get("spontaneous") else r.source
        out = repr(r.output) if r.output is not None else "无(诚实)"
        line = (f"[{ctx.tick:5d}] {who:16s} | 输入={r.input[:24]:24s} "
                f"| 收敛={r.convergence} | 输出={out}")
        if r.error:
            line += f" | ⚠ {r.error}"
        if self.echo:
            print(line, flush=True)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def close(self) -> None:
        pass


@sink("jsonl")
class JSONLSink:
    """结构化结果（JSONL），用于下游分析"""

    def __init__(self, path: str = "run/outbox.jsonl", **kw):
        self.name = "jsonl"
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def handle(self, r: Result, ctx: TickContext) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        pass


@sink("http")
class HTTPSink:
    """把结果暴露为一个只读 HTTP 端点（NORM-2：输出端，非外挂 API）

    GET /latest   → 最近一条结果
    GET /results  → 最近 N 条
    GET /state    → 大脑状态
    """

    def __init__(self, port: int = 8700, keep: int = 200, **kw):
        self.name = f"http:{port}"
        self.port = port
        self.keep = keep
        self._results: List[Dict[str, Any]] = []
        self._state_fn = None
        self._server = None
        self._start()

    def bind_state(self, fn) -> None:
        self._state_fn = fn

    def _start(self):
        import threading
        results = self._results
        holder = self

        class H(BaseHTTPRequestHandler):
            def _send(self, obj):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/latest":
                    self._send(results[-1] if results else {})
                elif p == "/results":
                    self._send({"n": len(results), "items": results[-50:]})
                elif p == "/state":
                    try:
                        self._send(holder._state_fn() if holder._state_fn else {})
                    except Exception as e:
                        self._send({"error": str(e)})
                elif p == "/health":
                    self._send({"ok": True, "sink": "http", "port": holder.port})
                else:
                    self.send_error(404)

            def log_message(self, *a):
                pass

        self._server = HTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def handle(self, r: Result, ctx: TickContext) -> None:
        self._results.append(r.to_dict())
        if len(self._results) > self.keep:
            del self._results[:len(self._results) - self.keep]

    def close(self) -> None:
        try:
            if self._server:
                self._server.shutdown()
        except Exception:
            pass


@sink("memory")
class MemorySink:
    """把结果写进一个 JSONL 供记忆库外部使用（不直接碰大脑内部）"""

    def __init__(self, path: str = "run/memory.jsonl", **kw):
        self.name = "memory"
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def handle(self, r: Result, ctx: TickContext) -> None:
        rec = {"tick": ctx.tick, "text": r.input, "output": r.output,
               "convergence": r.convergence, "source": r.source}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        pass
