"""
仿生大脑 · OpenAI 兼容服务 (api_server.py)
============================================
让它表现得像一个标准 LLM —— 任何 OpenAI 客户端都能直连。

接口：
    POST /v1/chat/completions     ← 聊天补全（含流式）
    GET  /v1/models               ← 模型列表
    POST /v1/embeddings           ← 嵌入（借用 encoder）
    GET  /health                  ← 健康检查
    GET  /brain/state             ← 大脑状态（私有扩展）

设计：
    · 神经认知层参与每轮对话（记忆召回 + 置信度调制）
    · 语言由本地 LLM 生成，但**内容受大脑指挥**
    · 纯标准库实现（http.server），零新依赖

用法：
    python api_server.py --port 8700
    # 然后任何 OpenAI 客户端可连 http://127.0.0.1:8700/v1
"""
from __future__ import annotations
import json
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MODEL_NAME = "bio-brain"
SERVER_VERSION = "0.1.0"

# 全局大脑实例（懒加载）
_BRAIN = None
_BRAIN_LOCK = threading.Lock()


def get_brain():
    """懒加载大脑（首次调用时构建）"""
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

    # ---------------- GET ----------------
    def do_GET(self):
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
                             "version": SERVER_VERSION})
        elif self.path.startswith("/brain/state"):
            b = get_brain()
            if b is None:
                self._send_json({"error": "brain not loaded"}, 503)
            else:
                s = b.stats()
                # ★补上记忆条数（Conductor.stats 不含 memory）
                s["memory"] = len(getattr(b, "memory", []))
                s["llm"] = {"model": getattr(b.llm, "model", None),
                            "available": bool(b.llm and b.llm.available)}
                self._send_json(s)
        else:
            self._send_json({"error": {"message": "not found"}}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        if self.path.startswith("/v1/chat/completions"):
            self._chat()
        elif self.path.startswith("/v1/embeddings"):
            self._embeddings()
        elif self.path.startswith("/brain/teach"):
            self._teach()
        else:
            self._send_json({"error": {"message": "not found"}}, 404)

    # ---------------- 核心：聊天补全 ----------------
    def _chat(self):
        req = self._read_body()
        messages = req.get("messages") or []
        stream = bool(req.get("stream"))

        # 取最后一条 user 消息
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
                    # ★修正：大脑"不说"≠"不会"。让语言区以最保守的方式回答，
                    #   而不是回一句死板的模板（否则用户以为坏了）
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
                # ★暴露完整堆栈（之前静默吞异常导致 19ms 假响应）
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
            # 首块含角色
            self.wfile.write(sse({"id": cid, "object": "chat.completion.chunk",
                                  "created": int(time.time()), "model": MODEL_NAME,
                                  "choices": [{"index": 0,
                                               "delta": {"role": "assistant"}}]}))
            # 分块输出
            for i in range(0, len(content), 12):
                self.wfile.write(sse({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": content[i:i+12]}}]}))
            # 结束
            self.wfile.write(sse({"id": cid, "object": "chat.completion.chunk",
                                  "created": int(time.time()), "model": MODEL_NAME,
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
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
            "bio_brain": {**meta, "latency_ms": latency},
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
    ap.add_argument("--preload", action="store_true", help="启动时预载大脑")
    args = ap.parse_args()

    print("=" * 66)
    print("  仿生大脑 · OpenAI 兼容服务")
    print("=" * 66)
    print(f"  Base URL: http://{args.host}:{args.port}/v1")
    print(f"  模型名:   {MODEL_NAME}")
    print(f"  端点:     /v1/chat/completions  /v1/models  /v1/embeddings")
    print()

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
