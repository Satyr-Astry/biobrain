"""
HarnessPlatform · 内置输入源（Source）
========================================
每个 Source 只需：name 属性 + poll() + close()

加新源只需写一个类 + @source("名字")，**不改平台核心**。
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

from .context import TickContext, Stimulus
from .registry import source


# ------------------------------------------------------------------
@source("file")
class FileSource:
    """从文件读一行（消费掉）—— 兼容旧 harness 的 inbox 模式"""

    def __init__(self, path: str = "run/inbox.txt", **kw):
        self.name = f"file:{os.path.basename(path)}"
        self.path = Path(path)

    def poll(self, ctx: TickContext) -> Optional[Stimulus]:
        if not self.path.is_file():
            return None
        try:
            txt = self.path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
        if not txt:
            return None
        try:
            self.path.write_text("", encoding="utf-8")
        except Exception:
            pass
        line = txt.splitlines()[0]
        return Stimulus(text=line, source=self.name, kind="text")

    def close(self) -> None:
        pass


# ------------------------------------------------------------------
@source("clock")
class ClockSource:
    """每隔 N tick 自发产一条刺激（思考流不等输入 —— NORM-1）"""

    _PROMPTS = [
        "我最近学到了什么",
        "有没有什么我还没搞懂的",
        "刚才那条经验说明了什么",
        "我擅长什么、不擅长什么",
        "接下来该做什么",
    ]

    def __init__(self, every: int = 5, prompts=None, seed: int = 0, **kw):
        self.name = "clock"
        self.every = max(1, int(every))
        self.prompts = list(prompts or self._PROMPTS)
        self._i = 0
        self._seed = seed

    def poll(self, ctx: TickContext) -> Optional[Stimulus]:
        if ctx.tick % self.every != 0:
            return None
        import numpy as np
        rng = np.random.default_rng(self._seed + ctx.tick)
        text = str(rng.choice(self.prompts))
        return Stimulus(text=text, source=self.name, kind="text",
                        weight=0.5, payload={"spontaneous": True})

    def close(self) -> None:
        pass


# ------------------------------------------------------------------
@source("static")
class StaticSource:
    """按固定列表依次产出（用于可复现实验/测试）"""

    def __init__(self, items=None, repeat: bool = False, **kw):
        self.name = "static"
        self.items = list(items or [])
        self.repeat = repeat
        self._i = 0

    def poll(self, ctx: TickContext) -> Optional[Stimulus]:
        if not self.items:
            return None
        if self._i >= len(self.items):
            if not self.repeat:
                return None
            self._i = 0
        t = self.items[self._i]
        self._i += 1
        return Stimulus(text=t, source=self.name, kind="text")

    def close(self) -> None:
        pass


# ------------------------------------------------------------------
@source("http")
class HTTPSource:
    """HTTP 输入：把 POST 的文本放进队列，poll 时取出

    依赖标准库 http.server，不引第三方（NORM-7）。
    """

    def __init__(self, port: int = 8642, path: str = "/inbox", **kw):
        self.name = f"http:{port}"
        self.port = port
        self.path = path
        self._queue = []
        self._server = None
        self._thread = None
        self._start()

    def _start(self):
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        q = self._queue

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != self.path_target:
                    self.send_error(404); return
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode("utf-8", "ignore")
                q.append(body.strip())
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *a):
                pass

        H.path_target = self.path
        self._server = HTTPServer(("127.0.0.1", self.port), H)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()

    def poll(self, ctx: TickContext) -> Optional[Stimulus]:
        if not self._queue:
            return None
        txt = self._queue.pop(0)
        return Stimulus(text=txt, source=self.name, kind="text", weight=2.0)

    def close(self) -> None:
        try:
            if self._server:
                self._server.shutdown()
        except Exception:
            pass
