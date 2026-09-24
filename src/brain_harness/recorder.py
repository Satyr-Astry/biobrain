"""
HarnessPlatform · 统一记录器
==============================
所有组件的事件都进同一个 JSONL —— 治"各入口日志格式不同、无法对比实验"。

事件类型：
  tick      每 tick 一条（含来源标记）
  result    Processor 产出
  policy    Policy 执行
  snapshot  定期大脑状态快照
  error     异常
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .context import TickContext, Result


class Recorder:
    def __init__(self, path: Optional[str] = None, snapshot_every: int = 0,
                 echo: bool = False):
        self.path = path
        self.snapshot_every = snapshot_every
        self.echo = echo
        self.events = 0
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ---------- 写 ----------
    def _write(self, obj: Dict[str, Any]) -> None:
        obj.setdefault("ts", datetime.now().isoformat(timespec="milliseconds"))
        self.events += 1
        if self.echo:
            print(json.dumps(obj, ensure_ascii=False), flush=True)
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def tick(self, ctx: TickContext, n_sources: int, n_policies: int) -> None:
        self._write({"ev": "tick", "tick": ctx.tick,
                     "sources": n_sources, "policies": n_policies,
                     "ms": round(ctx.elapsed() * 1000, 3)})

    def result(self, r: Result) -> None:
        self._write({"ev": "result", **r.to_dict()})

    def policy(self, name: str, tick: int, summary: Dict[str, Any]) -> None:
        self._write({"ev": "policy", "name": name, "tick": tick,
                     "summary": summary})

    def snapshot(self, tick: int, state: Dict[str, Any]) -> None:
        self._write({"ev": "snapshot", "tick": tick, "state": state})

    def error(self, where: str, err: str, tick: int = -1) -> None:
        self._write({"ev": "error", "where": where, "err": err, "tick": tick})

    def maybe_snapshot(self, ctx: TickContext, state_fn) -> None:
        if self.snapshot_every and ctx.tick % self.snapshot_every == 0:
            try:
                self.snapshot(ctx.tick, state_fn())
            except Exception as e:
                self.error("snapshot", str(e), ctx.tick)

    # ---------- 读（复现/分析） ----------
    def replay(self):
        """逐条读回事件（供实验对比用）"""
        if not self.path or not os.path.isfile(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

    def convergence_series(self) -> list:
        """抽出收敛序列（等价性验收用）"""
        return [e.get("convergence") for e in self.replay()
                if e.get("ev") == "result"]
