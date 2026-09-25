"""
HarnessPlatform · 核心数据类型
================================
平台内部流转的三种对象：TickContext / Stimulus / Result。

设计原则：
  · 全部是**纯数据**（dataclass），无行为 → 易序列化、易测试
  · 不依赖 CogVec（平台内核与大脑解耦）
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time


@dataclass
class TickContext:
    """一个 tick 的上下文（在所有组件间传递）"""
    tick: int = 0
    started_at: float = field(default_factory=time.time)   # 本轮 tick 起始时刻
    t0: float = field(default_factory=time.time)           # 整个 run 起始时刻
    running: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def elapsed(self) -> float:
        """★本轮 tick 已耗时（秒）—— 必须在每 tick 开始时重置 started_at"""
        return time.time() - self.started_at

    def total_elapsed(self) -> float:
        """整个 run 的累计耗时"""
        return time.time() - self.t0

    def begin_tick(self) -> None:
        """★每 tick 开头调用，重置计时"""
        self.started_at = time.time()


@dataclass
class Stimulus:
    """一条刺激（待大脑处理）"""
    text: str
    source: str = "unknown"          # 来源标识（哪个 Source 产的）
    kind: str = "text"               # text | image | teach | observe
    payload: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0              # 优先级（多 Source 时用）
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Result:
    """一条处理结果（待 Sink 消费）"""
    tick: int = 0
    source: str = "unknown"          # 刺激来源
    processor: str = "unknown"       # 哪个 Processor 处理的
    input: str = ""
    output: Optional[str] = None     # NORM-9：无生成后端时必须是 None
    dont_know: bool = False
    convergence: Optional[float] = None
    activation: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
