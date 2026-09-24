"""
HarnessPlatform · 平台外壳（组装入口）
========================================
从配置（dict 或 YAML）组装 Pipeline —— 这是唯一的组装入口。

★ 可扩展性验收点：
   第三方加新插件只需：
     1. @source("my_src") class MySrc: ...
     2. 在配置里写 type: my_src
   **无需改本文件、无需改平台核心。**
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from . import registry
from .brain_handle import BrainHandle
from .pipeline import Pipeline
from .recorder import Recorder


# ------------------------- 默认配置 -------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "brain": {"seed": 1},
    "pipeline": {
        "sources": [{"type": "clock", "every": 5}],
        "processors": [{"type": "think", "task_type": "general"}],
        "policies": [
            {"type": "metrics", "every": 10},
            {"type": "sleep", "every": 8, "min_buffer": 2},
        ],
        "sinks": [{"type": "log", "echo": True}],
    },
    "recorder": {"path": "", "snapshot_every": 0, "echo": False},
}


def load_config(path: str) -> Dict[str, Any]:
    """读 YAML（无 pyyaml 时回退到极简解析）"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _mini_yaml(path)
    except FileNotFoundError:
        raise


def _mini_yaml(path: str) -> Dict[str, Any]:
    """极简 YAML 子集解析（无 pyyaml 时用）：只支持本平台的配置形状"""
    import re
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip())
            line = raw.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if line.startswith("- "):
                item = line[2:].strip()
                if isinstance(parent, list):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        d = {k.strip(): _scalar(v.strip())}
                        parent.append(d)
                        stack.append((indent, d))
                    else:
                        parent.append(_scalar(item))
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v == "":
                    container: Any = {}
                    parent[k] = container
                    stack.append((indent, container))
                else:
                    parent[k] = _scalar(v)
    return root


def _scalar(v: str):
    if v in ("", "null", "~"):
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_scalar(x.strip()) for x in inner.split(",")] if inner else []
    return v


# ------------------------- 组装 -------------------------
class HarnessPlatform:
    """平台外壳：一个进程、一个大脑、一条 Pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        registry.load_builtins()
        self.cfg = _merge(DEFAULT_CONFIG, config or {})
        self.brain: Optional[BrainHandle] = None
        self.pipeline: Optional[Pipeline] = None
        self._inject_src = None

    # ---- 构造大脑 ----
    def _build_brain(self) -> BrainHandle:
        bcfg = dict(self.cfg.get("brain", {}))
        seed = bcfg.pop("seed", 1)
        state_path = bcfg.pop("state_path", None)
        return BrainHandle.create(seed=seed, state_path=state_path, **bcfg)

    # ---- 构造 Pipeline ----
    def build(self) -> Pipeline:
        self.brain = self._build_brain()

        pcfg = self.cfg.get("pipeline", {})
        sources = [registry.build("source", s) for s in pcfg.get("sources", [])]
        procs = [registry.build("processor", p) for p in pcfg.get("processors", [])]
        policies = [registry.build("policy", p) for p in pcfg.get("policies", [])]
        sinks = [registry.build("sink", s) for s in pcfg.get("sinks", [])]
        if not procs:
            procs = [registry.build("processor", {"type": "think"})]

        rcfg = dict(self.cfg.get("recorder", {}))
        rec = Recorder(path=rcfg.get("path") or None,
                       snapshot_every=int(rcfg.get("snapshot_every", 0) or 0),
                       echo=bool(rcfg.get("echo", False)))

        # 若某个 Sink 需要大脑状态（HTTPSink/WebUISink），绑定
        metrics_pol = next((q for q in policies if getattr(q, "name", "") == "metrics"), None)
        for sk in sinks:
            if hasattr(sk, "bind_state"):
                sk.bind_state(self.brain.state)
            if hasattr(sk, "bind_extra"):
                sk.bind_extra(
                    stats_fn=lambda: self.pipeline.stats if self.pipeline else {},
                    metrics_fn=(lambda: metrics_pol.history) if metrics_pol else None,
                    inject_fn=self.inject,
                )

        self.pipeline = Pipeline(self.brain, sources, procs, policies, sinks, rec)
        return self.pipeline

    # ---- 外部注入（Web UI / API 用） ----
    def inject(self, text: str) -> None:
        """把一条文本注入平台（下一 tick 被 Source 取走）

        走平台自己的 StaticSource 队列 —— 不绕过 tick 边界（NORM-4）。
        """
        from .sources import StaticSource
        if self._inject_src is None:
            self._inject_src = StaticSource(items=[])
            self._inject_src.name = "inject"
            if self.pipeline:
                self.pipeline.sources.insert(0, self._inject_src)
        self._inject_src.items.append(text)

    # ---- 运行 ----
    def run(self, steps: int = 0, paced: float = 0.0) -> Dict[str, Any]:
        pl = self.pipeline or self.build()
        try:
            ctx = pl.run(steps=steps, paced=paced)
        finally:
            pl.stop()
            try:
                if self.brain:
                    self.brain.save()
            except Exception:
                pass
        return {
            "ticks": pl.stats["ticks"],
            "stimuli": pl.stats["stimuli"],
            "results": pl.stats["results"],
            "policies_run": pl.stats["policies_run"],
            "errors": pl.stats["errors"],
            "by_source": pl.stats["by_source"],
            "by_policy": pl.stats["by_policy"],
            "elapsed": round(ctx.meta.get("elapsed", 0.0), 3),
            "brain_ticks": self.brain.tick_count if self.brain else 0,
        }


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
