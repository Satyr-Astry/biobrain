"""
HarnessPlatform · 内置处理器（Processor）
==========================================
Processor: Stimulus → Result（调用大脑）
"""
from __future__ import annotations
from typing import Any

from .context import TickContext, Stimulus, Result
from .registry import processor


def _brain_of(obj: Any) -> Any:
    """兼容：既接受 BrainHandle（有 .think）也接受裸 BioBrain"""
    return obj



@processor("think")
class ThinkProcessor:
    """把文本交给 brain.think()（最常用）"""

    def __init__(self, task_type: str = "general", **kw):
        self.name = "think"
        self.task_type = task_type

    def process(self, stim: Stimulus, brain: Any, ctx: TickContext) -> Result:
        r = brain.think(stim.text, task_type=self.task_type)
        return Result(
            source=stim.source,
            processor=self.name,
            input=stim.text,
            # ★NORM-9：无生成后端时 brain.think() 返回 output=None
            #   这里如实透传，绝不回显输入冒充生成
            output=r.get("output"),
            dont_know=bool(r.get("should_say_dont_know")),
            convergence=r.get("convergence"),
            activation=r.get("activation"),
            extra={"spontaneous": stim.payload.get("spontaneous", False)},
        )


@processor("observe")
class ObserveProcessor:
    """多模态观察（image_vec 由 payload 提供）"""

    def __init__(self, **kw):
        self.name = "observe"

    def process(self, stim: Stimulus, brain: Any, ctx: TickContext) -> Result:
        img = stim.payload.get("image_vec")
        r = brain.observe(image_vec=img, text=stim.text or None)
        return Result(source=stim.source, processor=self.name,
                      input=stim.text, output=r.get("output"),
                      convergence=r.get("convergence"),
                      activation=r.get("activation"))


@processor("teach")
class TeachProcessor:
    """外部知识注入（NORM-6：source 必须是可靠来源）"""

    def __init__(self, source_name: str = "oracle", **kw):
        self.name = "teach"
        self.source_name = source_name

    def process(self, stim: Stimulus, brain: Any, ctx: TickContext) -> Result:
        payload = stim.payload.get("teach") or {"text": stim.text}
        r = brain.teach(payload, source=self.source_name)
        return Result(source=stim.source, processor=self.name,
                      input=stim.text, output=None,
                      extra={"taught": r})
