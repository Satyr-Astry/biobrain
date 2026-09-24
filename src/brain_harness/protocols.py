"""
HarnessPlatform · 插件协议（4 个扩展点）
==========================================
★ 关键设计：全部用 typing.Protocol（结构化子类型）
   → 插件**无需继承基类**，只要方法签名匹配即可
   → 零耦合扩展（第三方插件不必 import 本模块）

四个扩展点：
  Source     输入源   —— 产出 Stimulus
  Processor  处理器   —— Stimulus → Result
  Policy     旁路策略 —— 不产刺激，但可影响大脑（睡眠/进化/蒸馏）
  Sink       输出端   —— 消费 Result
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .context import TickContext, Stimulus, Result


@runtime_checkable
class Source(Protocol):
    """输入源：每 tick 被询问一次，可返回一条刺激或 None"""
    name: str

    def poll(self, ctx: TickContext) -> Optional[Stimulus]:
        """产出刺激。返回 None 表示本 tick 无输入。"""
        ...

    def close(self) -> None:
        """释放资源（可选实现）"""
        ...


@runtime_checkable
class Processor(Protocol):
    """处理器：把刺激交给大脑，产出结果"""
    name: str

    def process(self, stim: Stimulus, brain: Any, ctx: TickContext) -> Result:
        ...


@runtime_checkable
class Policy(Protocol):
    """旁路策略：判断是否该执行 + 执行"""
    name: str

    def should_run(self, ctx: TickContext) -> bool:
        ...

    def run(self, brain: Any, ctx: TickContext) -> Dict[str, Any]:
        """返回执行摘要（进 Recorder）"""
        ...


@runtime_checkable
class Sink(Protocol):
    """输出端：消费结果"""
    name: str

    def handle(self, result: Result, ctx: TickContext) -> None:
        ...

    def close(self) -> None:
        ...
