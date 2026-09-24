"""
HarnessPlatform · Pipeline（组装 + 驱动）
==========================================
把四类插件按声明式配置组装起来，驱动主循环。

主循环（每 tick）：
  1. 询问所有 Source（按 weight 降序）→ 取最高优先级的一条刺激
  2. 若有刺激 → 交给第一个能处理它的 Processor → 得 Result
  3. 把 Result 给所有 Sink
  4. 询问所有 Policy（should_run）→ 执行并记 Recorder
  5. 定期 snapshot

遵守 NORM-1（主循环永不退出，除非显式 steps/stop）。
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from .brain_handle import BrainHandle
from .context import TickContext, Stimulus, Result
from .recorder import Recorder


class Pipeline:
    def __init__(self, handle: BrainHandle,
                 sources: List[Any] = None,
                 processors: List[Any] = None,
                 policies: List[Any] = None,
                 sinks: List[Any] = None,
                 recorder: Optional[Recorder] = None):
        self.handle = handle
        self.sources = sources or []
        self.processors = processors or []
        self.policies = policies or []
        self.sinks = sinks or []
        self.rec = recorder or Recorder()
        self.stats: Dict[str, int] = {
            "ticks": 0, "stimuli": 0, "results": 0,
            "policies_run": 0, "errors": 0,
            "by_source": {}, "by_policy": {},
        }

    # ---------------- 单 tick ----------------
    def tick(self, ctx: TickContext) -> Optional[Result]:
        ctx.begin_tick()                    # ★重置本轮计时（修 ms 语义 bug）
        ctx.tick += 1

        # 1. 收集刺激（多 Source，按 weight 取最高）
        stim: Optional[Stimulus] = None
        for src in self.sources:
            try:
                s = src.poll(ctx)
            except Exception as e:
                self.rec.error(f"source:{src.name}", str(e), ctx.tick)
                self.stats["errors"] += 1
                continue
            if s is not None and (stim is None or s.weight > stim.weight):
                stim = s

        result: Optional[Result] = None

        # 2. 处理刺激
        if stim is not None:
            self.stats["stimuli"] += 1
            self.stats["by_source"][stim.source] = \
                self.stats["by_source"].get(stim.source, 0) + 1
            for proc in self.processors:
                try:
                    # ★传 handle：保住单锁串行化 + tick 计数
                    result = proc.process(stim, self.handle, ctx)
                    result.tick = ctx.tick
                    break
                except Exception as e:
                    self.rec.error(f"processor:{proc.name}", str(e), ctx.tick)
                    self.stats["errors"] += 1
                    result = Result(tick=ctx.tick, source=stim.source,
                                    processor=proc.name, input=stim.text,
                                    error=str(e))
                    break
            if result is not None:
                self.stats["results"] += 1
                self.rec.result(result)

        # 3. 输出到 Sink
        if result is not None:
            for sk in self.sinks:
                try:
                    sk.handle(result, ctx)
                except Exception as e:
                    self.rec.error(f"sink:{sk.name}", str(e), ctx.tick)
                    self.stats["errors"] += 1

        # 4. 旁路策略
        for pol in self.policies:
            try:
                if pol.should_run(ctx):
                    summary = pol.run(self.handle, ctx)
                    self.stats["policies_run"] += 1
                    self.stats["by_policy"][pol.name] = \
                        self.stats["by_policy"].get(pol.name, 0) + 1
                    self.rec.policy(pol.name, ctx.tick, summary or {})
            except Exception as e:
                self.rec.error(f"policy:{pol.name}", str(e), ctx.tick)
                self.stats["errors"] += 1

        # 5. 快照
        self.rec.maybe_snapshot(ctx, self.handle.state)

        # 6. ★本 tick 结束 → 落 tick 事件（此时 ms 才是真实单 tick 耗时）
        self.rec.tick(ctx, len(self.sources), len(self.policies))

        self.stats["ticks"] += 1
        return result

    # ---------------- 主循环 ----------------
    def run(self, steps: int = 0, paced: float = 0.0,
            ctx: Optional[TickContext] = None) -> TickContext:
        ctx = ctx or TickContext()
        t0 = time.time()
        n = 0
        while ctx.running:
            self.tick(ctx)
            n += 1
            if steps and n >= steps:
                break
            if paced:
                time.sleep(paced)
        ctx.meta["elapsed"] = ctx.total_elapsed()
        return ctx

    def stop(self) -> None:
        for c in list(self.sources) + list(self.sinks):
            try:
                c.close()
            except Exception:
                pass

    # ---------------- 内省 ----------------
    def describe(self) -> Dict[str, Any]:
        return {
            "sources": [s.name for s in self.sources],
            "processors": [p.name for p in self.processors],
            "policies": [p.name for p in self.policies],
            "sinks": [s.name for s in self.sinks],
            "stats": self.stats,
        }
