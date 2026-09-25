"""
仿生大脑 · OpenAI 兼容服务（api_server.py）
==============================================
★ v3.11 P0-C：本文件已**降级为薄代理层**（治 P-ARCH-14）。

背景（实测）：
    8642（server.py）持 CogVec/NervousSystem（64 神经元，逐神经元）
    8700（本文件）原持 Conductor/TensorBrain（16384 神经元，矩阵版）
    → 两套引擎两个门面，路由重叠，**状态可能分裂**。
    且两条路的后端协议不兼容：
      · server.py has_generation_backend() 探 /health + /v1/chat/completions（OpenAI 兼容）
      · cortex.LLMCortex._probe()        探 Ollama /api/tags
    ⇒ 「导入路由」方案会造出**第二个 Conductor 实例**（正是要治的病），被证伪。

现行为（BIO_MERGE_HTTP=1，默认）：
    全部路由**原样转发**到 BIO_AUTHORITATIVE_URL（默认 http://127.0.0.1:8642），
    **不构造任何大脑实例** → 物理上不可能持有第二个大脑。

回退（BIO_MERGE_HTTP=0）：
    退回旧行为（自持 Conductor + TensorBrain + agent_memory.json）。
    保留用于对比与紧急回滚。

依赖：仅标准库（http.server + urllib），**不需要 flask**（实测本机未装 flask）。

用法：
    python api_server.py --port 8700        # 代理到 8642（需先起 8642）
    BIO_MERGE_HTTP=0 python api_server.py   # 回退：自持大脑
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MODEL_NAME = "bio-brain"
SERVER_VERSION = "0.2.0"

# ★v3.11 合并开关
MERGE_HTTP = os.environ.get("BIO_MERGE_HTTP", "1").strip() != "0"
AUTHORITATIVE_URL = os.environ.get(
    "BIO_AUTHORITATIVE_URL", "http://127.0.0.1:8642").rstrip("/")

# 全局大脑实例（★仅回退模式使用；合并模式下恒为 None）
_BRAIN = None
_BRAIN_LOCK = __import__("threading").Lock()


def get_brain():
    """懒加载大脑（★仅回退模式调用；合并模式永不构造）。

    注意：合并模式下本函数**不应被调用** —— 测试 C20 断言 `_BRAIN is None`。
    """
    global _BRAIN
    if _BRAIN is None:
        with _BRAIN_LOCK:
            if _BRAIN is None:
                try:
                    from conductor import Conductor
                    _BRAIN = Conductor(n_neurons=16384, n_tracts=1024, use_llm=True)
                    # ★载入已学记忆（否则每次重启都从零开始）
                    import json as _json
                    from pathlib import Path as _P
                    mem_file = _P(__file__).parent / "agent_memory.json"
                    if mem_file.exists():
                        try:
                            for m in _json.loads(mem_file.read_text(encoding="utf-8")):
                                _BRAIN.teach(m["text"], m["answer"],
                                             m.get("source", "oracle"))
                            print(f"[brain] 载入记忆 {len(_BRAIN.memory)} 条",
                                  file=sys.stderr)
                        except Exception as _e:
                            print(f"[brain] 记忆载入失败: {_e}", file=sys.stderr)
                except Exception as e:
                    print(f"[warn] 大脑加载失败: {e}", file=sys.stderr)
                    _BRAIN = None
    return _BRAIN


def proxy_available() -> bool:
    """权威服务是否可达（用于启动提示与 502 诊断）"""
    try:
        with urllib.request.urlopen(AUTHORITATIVE_URL + "/health", timeout=1.0):
            return True
    except Exception:
        return False


def forward(method: str, path: str, body: bytes, headers: dict):
    """把请求原样转发到权威服务，返回 (status, content_type, body)。

    ★关键：**不做任何本地大脑计算**，也**不缓存状态** —— 状态唯一来源是 8642。
    """
    req = urllib.request.Request(
        AUTHORITATIVE_URL + path, data=(body if method == "POST" else None),
        method=method,
        headers={"Content-Type": headers.get("Content-Type",
                                             "application/json")})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.headers.get("Content-Type",
                                                 "application/json"), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()
    except Exception as e:
        msg = json.dumps({
            "error": {
                "message": f"权威服务不可达 ({AUTHORITATIVE_URL}): {type(e).__name__}: {e}",
                "hint": "请先启动 python server.py --serve（端口 8642）",
            }
        }, ensure_ascii=False).encode()
        return 502, "application/json; charset=utf-8", msg


def sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---------------- 工具 ----------------
    def _send_json(self, obj: dict, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt, *args):
        pass   # 静默（自己记日志）

    def _relay(self, method: str):
        """★v3.11：合并模式 —— 原样转发到权威服务"""
        raw = b""
        n = int(self.headers.get("Content-Length", 0))
        if n:
            raw = self.rfile.read(n)
        status, ctype, body = forward(method, self.path, raw, dict(self.headers))
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Bio-Proxy-To", AUTHORITATIVE_URL)
        self.end_headers()
        self.wfile.write(body)

    # ---------------- GET ----------------
    def do_GET(self):
        if MERGE_HTTP:
            return self._relay("GET")
        if self.path.startswith("/v1/models"):
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME, "object": "model",
                    "created": int(time.time()), "owned_by": "bio-brain",
                }],
            })
        elif self.path.startswith("/health"):
            self._send_json({"status": "ok", "model": MODEL_NAME,
                             "version": SERVER_VERSION, "merged_http": False})
        elif self.path.startswith("/brain/state"):
            b = get_brain()
            if b is None:
                self._send_json({"error": "brain not loaded"}, 503)
            else:
                s = b.stats()
                s["memory"] = len(getattr(b, "memory", []))
                s["llm"] = {"model": getattr(b.llm, "model", None),
                            "available": bool(b.llm and b.llm.available)}
                self._send_json(s)
        else:
            self._send_json({"error": {"message": "not found"}}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        if MERGE_HTTP:
            return self._relay("POST")
        if self.path.startswith("/v1/chat/completions"):
            self._chat()
        elif self.path.startswith("/v1/embeddings"):
            self._embeddings()
        elif self.path.startswith("/brain/teach"):
            self._teach()
        else:
            self._send_json({"error": {"message": "not found"}}, 404)

    # ---------------- 回退模式：核心聊天补全 ----------------
    def _chat(self):
        req = self._read_body()
        messages = req.get("messages") or []
        stream = bool(req.get("stream"))

        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content") or ""
                break

        brain = get_brain()
        meta = {}
        t0 = time.time()

        if brain is not None:
            try:
                r = brain.respond(user_text, steps=4)
                content = r.get("output")
                meta = {
                    "confidence": r.get("confidence"),
                    "should_speak": r.get("should_speak"),
                    "recalled": [x["text"] for x in r.get("recalled", [])],
                    "convergence": r.get("convergence"),
                }
                if not content:
                    try:
                        content = brain.llm.generate(
                            user_text,
                            system="你不确定，请简短诚实地说明你不确定，不要编造。",
                            max_tokens=80) or "（暂时无法回答）"
                        meta["fallback"] = True
                    except Exception:
                        content = "（暂时无法回答）"
                        meta["abstained"] = True
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[brain-error] {type(e).__name__}: {e}\n{tb}",
                      file=sys.stderr, flush=True)
                content = f"[brain error: {type(e).__name__}: {e}]"
                meta["error"] = str(e)
                meta["traceback"] = tb[-800:]
        else:
            content = "[no brain loaded]"

        latency = round((time.time() - t0) * 1000, 1)
        cid = f"chatcmpl-{int(time.time()*1000)}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(sse({"id": cid, "object": "chat.completion.chunk",
                                  "created": int(time.time()), "model": MODEL_NAME,
                                  "choices": [{"index": 0,
                                               "delta": {"role": "assistant"}}]}))
            for i in range(0, len(content), 12):
                self.wfile.write(sse({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": content[i:i+12]}}]}))
            self.wfile.write(sse({"id": cid, "object": "chat.completion.chunk",
                                  "created": int(time.time()), "model": MODEL_NAME,
                                  "choices": [{"index": 0, "delta": {},
                                               "finish_reason": "stop"}]}))
            self.wfile.write(b"data: [DONE]\n\n")
            return

        self._send_json({
            "id": cid, "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(user_text),
                "completion_tokens": len(content),
                "total_tokens": len(user_text) + len(content),
            },
            "cog_vec": {**meta, "latency_ms": latency},
        })

    def _embeddings(self):
        req = self._read_body()
        inp = req.get("input")
        if isinstance(inp, str):
            inp = [inp]
        inp = inp or []
        try:
            from encoder import get_encoder
            enc = get_encoder()
            data = [{"object": "embedding", "index": i,
                     "embedding": enc.encode(t).tolist()}
                    for i, t in enumerate(inp)]
            self._send_json({
                "object": "list", "data": data, "model": "bio-brain-embed",
                "usage": {"prompt_tokens": sum(len(t) for t in inp),
                          "total_tokens": sum(len(t) for t in inp)},
            })
        except Exception as e:
            self._send_json({"error": {"message": str(e)}}, 500)

    def _teach(self):
        req = self._read_body()
        q = req.get("question") or ""
        a = req.get("answer") or ""
        src = req.get("source") or "oracle"
        b = get_brain()
        if b is None:
            self._send_json({"error": "brain not loaded"}, 503)
            return
        ok = b.teach(q, a, src)
        self._send_json({"ok": ok, "memory": len(b.memory)})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 OpenAI 兼容服务")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", action="store_true", help="启动时预载大脑（仅回退模式）")
    args = ap.parse_args()

    print("=" * 66)
    print("  仿生大脑 · OpenAI 兼容服务")
    print("=" * 66)

    if MERGE_HTTP:
        print("  模式:     ★合并（薄代理，不持大脑实例）")
        print(f"  转发目标: {AUTHORITATIVE_URL}  ← 唯一权威")
        print(f"  Base URL: http://{args.host}:{args.port}/v1")
        ok = proxy_available()
        print(f"  权威服务: {'✓ 可达' if ok else '✗ 不可达 —— 请先启动 python server.py --serve'}")
        print("  提示:     设 BIO_MERGE_HTTP=0 可退回旧的自持大脑模式")
    else:
        print("  模式:     回退（自持 Conductor/TensorBrain）")
        print(f"  Base URL: http://{args.host}:{args.port}/v1")
        if args.preload:
            print("  预载大脑…")
            t0 = time.time()
            b = get_brain()
            print(f"  完成 ({time.time()-t0:.1f}s) | {b.stats() if b else '失败'}")
    print()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"  服务已启动，Ctrl+C 停止")
    print("=" * 66)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  停止服务")
        srv.shutdown()


if __name__ == "__main__":
    main()
