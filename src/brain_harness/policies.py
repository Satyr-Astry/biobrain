"""
HarnessPlatform · 内置旁路策略（Policy）
==========================================
Policy: 不产刺激，但可影响大脑（睡眠/进化/蒸馏）。
每个 Policy 用 should_run(ctx) 自判频率 —— 不写死在主循环里。
"""
from __future__ import annotations
import json
from typing import Any, Dict

from .context import TickContext
from .registry import policy


@policy("sleep")
class SleepPolicy:
    """定期睡眠巩固（治"睡眠从不自动触发"）"""

    def __init__(self, every: int = 8, min_buffer: int = 2,
                 has_running_task: bool = False, **kw):
        self.name = "sleep"
        self.every = max(1, int(every))
        self.min_buffer = min_buffer
        self.has_running_task = has_running_task
        self.count = 0

    def should_run(self, ctx: TickContext) -> bool:
        return ctx.tick % self.every == 0

    def run(self, brain: Any, ctx: TickContext) -> Dict[str, Any]:
        # ★兼容：handle 有 .brain，裸 brain 直接用
        b = getattr(brain, "brain", brain)
        try:
            buf = b.buffer.stats()
            size = int(buf.get("size", 0))
        except Exception:
            size = 0
        if size < self.min_buffer:
            return {"skipped": True, "reason": "buffer 太小", "size": size}
        r = brain.sleep(has_pending_task=self.has_running_task)  # handle 已加锁
        self.count += 1
        return {"slept": True, "n": self.count,
                "sleep": r.get("sleep", r) if isinstance(r, dict) else str(r)}


@policy("evolve")
class EvolvePolicy:
    """定期结构演化（包一层 auto_evolve，不删它）"""

    def __init__(self, every: int = 50, **kw):
        self.name = "evolve"
        self.every = max(1, int(every))
        self.count = 0
        self._mod = None

    def should_run(self, ctx: TickContext) -> bool:
        return ctx.tick % self.every == 0

    def run(self, brain: Any, ctx: TickContext) -> Dict[str, Any]:
        try:
            if self._mod is None:
                import sys, os
                code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if code_dir not in sys.path:
                    sys.path.insert(0, code_dir)
                import auto_evolve as _m
                self._mod = _m
            # auto_evolve 若是脚本式入口，尝试找可调函数
            fn = (getattr(self._mod, "evolve_once", None)
                  or getattr(self._mod, "run_once", None))
            if fn is None:
                return {"skipped": True, "reason": "auto_evolve 无可调函数"}
            r = fn(brain)
            self.count += 1
            return {"evolved": True, "n": self.count, "detail": str(r)[:200]}
        except Exception as e:
            return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}


@policy("metrics")
class MetricsPolicy:
    """定期采集指标（不改变大脑，只观测）"""

    def __init__(self, every: int = 10, **kw):
        self.name = "metrics"
        self.every = max(1, int(every))
        self.history = []

    def should_run(self, ctx: TickContext) -> bool:
        return ctx.tick % self.every == 0

    def run(self, brain: Any, ctx: TickContext) -> Dict[str, Any]:
        b = getattr(brain, "brain", brain)
        st = b.state()
        m = {
            "tick": ctx.tick,
            "active": st.get("active"),
            "neurons": st.get("neurons"),
            "buffer": (st.get("experience_buffer") or {}).get("size"),
            "experiences": (st.get("self_model") or {}).get("经历数"),
        }
        self.history.append(m)
        return m


@policy("workmem_probe")
class WorkMemProbePolicy:
    """历史区分度探针（ALG-18 的验收指标在线监控）"""

    _INPUTS = ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]

    def __init__(self, every: int = 25, **kw):
        self.name = "workmem_probe"
        self.every = max(1, int(every))
        self.history = []

    def should_run(self, ctx: TickContext) -> bool:
        return ctx.tick % self.every == 0 and ctx.tick > 0

    def run(self, brain: Any, ctx: TickContext) -> Dict[str, Any]:
        import numpy as np
        # ★修：brain 可能是 BrainHandle（不接受 seed），必须用 BioBrain 类
        import sys, os
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from server import BioBrain

        def probe(hist):
            b = BioBrain(seed=1)
            for h in hist:
                b.think(h)
            b.think("猫是哺乳动物")
            return np.array([n.activity for n in b.ns.neurons.values()])
        ps = {i: probe([i]) for i in self._INPUTS}
        ds = []
        ks = list(ps)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ds.append(float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()))
        out = {"dmin": min(ds), "dmean": sum(ds) / len(ds),
               "all": [round(d, 6) for d in ds]}
        self.history.append(out)
        return out
