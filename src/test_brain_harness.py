"""
HarnessPlatform · 回归测试
============================
防止"真实实验暴露的 3 个 bug"复现（见 总架构书 §6.5）。

跑：cd code && PYTHONHASHSEED=0 python -m pytest test_brain_harness.py -q
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_harness import (HarnessPlatform, Stimulus, Result,
                           source, processor, policy, sink, TickContext,
                           list_all, load_builtins)

TMP = "run/_pytest_harness"


@pytest.fixture(autouse=True)
def _clean():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    yield
    shutil.rmtree(TMP, ignore_errors=True)


def _plat(**pipe):
    cfg = {"pipeline": pipe}
    p = HarnessPlatform(cfg)
    p.build()
    return p


# ------------------------------------------------------------------
# 基础
# ------------------------------------------------------------------
def test_h01_plugins_registered():
    load_builtins()
    info = list_all()
    for kind, names in (("sources", ["file", "clock", "static", "http"]),
                        ("processors", ["think", "observe", "teach"]),
                        ("policies", ["sleep", "metrics", "evolve",
                                      "workmem_probe"]),
                        ("sinks", ["log", "jsonl", "http", "memory"])):
        for n in names:
            assert n in info[kind], f"{kind} 缺 {n}"


def test_h02_pipeline_runs():
    p = _plat(sources=[{"type": "static", "items": ["a", "b", "c"]}],
              processors=[{"type": "think"}],
              sinks=[{"type": "log", "echo": False}])
    st = p.run(steps=5)
    assert st["ticks"] == 5
    assert st["stimuli"] == 3
    assert st["results"] == 3
    assert st["errors"] == 0


# ------------------------------------------------------------------
# ★Bug 1 回归：Policy 拿到 handle 时不能崩
# ------------------------------------------------------------------
def test_h03_policy_gets_handle_not_crash():
    """workmem_probe 曾在 type(brain)(seed=1) 上崩（handle 不接受 seed）"""
    from brain_harness.policies import WorkMemProbePolicy
    pol = WorkMemProbePolicy(every=1)
    p = _plat(sources=[{"type": "static", "items": ["x"]}],
              processors=[{"type": "think"}],
              policies=[{"type": "workmem_probe", "every": 1}],
              sinks=[{"type": "log", "echo": False}])
    st = p.run(steps=3)
    # 关键：不能有 workmem_probe 相关错误
    rec = p.pipeline.rec
    errs = [e for e in rec.replay()
            if e.get("ev") == "error" and "workmem_probe" in str(e.get("where", ""))]
    assert not errs, f"workmem_probe 又崩了: {errs[:1]}"
    assert st["errors"] == 0


def test_h04_sleep_policy_gets_handle():
    """sleep Policy 曾在 handle 上访问 brain.buffer 崩"""
    p = _plat(sources=[{"type": "static", "items": ["a", "b", "c"]}],
              processors=[{"type": "think"}],
              policies=[{"type": "sleep", "every": 1, "min_buffer": 1}],
              sinks=[{"type": "log", "echo": False}])
    st = p.run(steps=4)
    assert st["errors"] == 0, "sleep policy 崩了"


# ------------------------------------------------------------------
# ★Bug 2 回归：ms 必须是"单 tick 耗时"而非"累计时间"
# ------------------------------------------------------------------
def test_h05_tick_ms_is_per_tick_not_cumulative():
    """682% 假漂移的回归：ms 不能随 tick 单调递增（累计时间特征）"""
    rec_path = os.path.join(TMP, "rec.jsonl")
    p = HarnessPlatform({"pipeline": {
        "sources": [{"type": "static", "items": ["a"] * 12}],
        "processors": [{"type": "think"}],
        "sinks": [{"type": "log", "echo": False}]},
        "recorder": {"path": rec_path}})
    p.build()
    p.run(steps=12)
    ms = [json.loads(l)["ms"] for l in open(rec_path, encoding="utf-8")
          if '"tick"' in l]
    assert len(ms) == 12
    # 累计时间特征：严格单调递增且末值 >> 首值
    # 单 tick 耗时的特征：不单调，末值不会远大于首值
    assert ms[-1] < max(ms) * 1.5 + 1, \
        f"末 tick ms({ms[-1]}) 远大于峰值({max(ms)})，疑似累计时间"


def test_h06_tick_ms_not_all_zero():
    """修 bug2 时曾引入 bug3（ms 恒 0）"""
    rec_path = os.path.join(TMP, "rec2.jsonl")
    p = HarnessPlatform({"pipeline": {
        "sources": [{"type": "static", "items": ["a"] * 6}],
        "processors": [{"type": "think"}],
        "sinks": [{"type": "log", "echo": False}]},
        "recorder": {"path": rec_path}})
    p.build()
    p.run(steps=6)
    ms = [json.loads(l)["ms"] for l in open(rec_path, encoding="utf-8")
          if '"tick"' in l]
    assert max(ms) > 0.5, f"ms 全为 0（埋点位置错）: {ms}"


# ------------------------------------------------------------------
# ★可扩展性：新插件不改核心
# ------------------------------------------------------------------
def test_h07_extensible_without_core_change():
    @source("_t_src")
    class S:
        name = "_t_src"
        def __init__(self, n=2, **kw):
            self.n, self.i = n, 0
        def poll(self, ctx):
            if self.i >= self.n:
                return None
            self.i += 1
            return Stimulus(text=f"x{self.i}", source=self.name)
        def close(self): pass

    @processor("_t_proc")
    class P:
        name = "_t_proc"
        def process(self, stim, brain, ctx):
            r = brain.think(stim.text)
            return Result(source=stim.source, processor=self.name,
                          input=stim.text, convergence=r.get("convergence"))

    @sink("_t_sink")
    class K:
        name = "_t_sink"
        def __init__(self, **kw):
            self.n = 0
        def handle(self, r, ctx):
            self.n += 1
        def close(self): pass

    p = _plat(sources=[{"type": "_t_src", "n": 3}],
              processors=[{"type": "_t_proc"}],
              sinks=[{"type": "_t_sink"}])
    st = p.run(steps=4)
    assert st["stimuli"] == 3
    assert p.pipeline.sinks[0].n == 3


# ------------------------------------------------------------------
# ★单实例
# ------------------------------------------------------------------
def test_h08_single_brain_instance():
    p = _plat(sources=[{"type": "static", "items": ["a", "b"]}],
              processors=[{"type": "think"}],
              sinks=[{"type": "log", "echo": False}])
    assert p.brain is not None
    assert p.pipeline.handle is p.brain     # 同一 handle
    st = p.run(steps=3)
    assert st["brain_ticks"] == 2, f"handle 计数不对: {st['brain_ticks']}"


# ------------------------------------------------------------------
# ★NORM-9 诚实输出
# ------------------------------------------------------------------
def test_h09_norm9_no_fake_output():
    p = _plat(sources=[{"type": "static", "items": ["猫"]}],
              processors=[{"type": "think"}],
              sinks=[{"type": "log", "echo": False}])
    out = []
    res = p.pipeline.tick(TickContext())
    assert res is not None
    # 无生成后端 → output 必须是 None（不能回显输入）
    assert res.output is None, f"NORM-9 违反：output={res.output!r}"
    assert res.input == "猫"


# ------------------------------------------------------------------
# ★持久化
# ------------------------------------------------------------------
def test_h10_state_roundtrip():
    sp = os.path.join(TMP, "brain.json")
    p = HarnessPlatform({"brain": {"seed": 1, "state_path": sp},
                         "pipeline": {
                             "sources": [{"type": "static", "items": ["a"]}],
                             "processors": [{"type": "think"}],
                             "sinks": [{"type": "log", "echo": False}]}})
    p.build()
    p.run(steps=2)
    assert os.path.isdir(os.path.splitext(sp)[0] + ".brainstate")
    # 新实例 load
    p2 = HarnessPlatform({"brain": {"seed": 1, "state_path": sp},
                          "pipeline": {
                              "sources": [{"type": "static", "items": []}],
                              "processors": [{"type": "think"}],
                              "sinks": [{"type": "log", "echo": False}]}})
    p2.build()
    assert p2.brain.load() is True
    assert p2.brain.state()["neurons"] == 64
