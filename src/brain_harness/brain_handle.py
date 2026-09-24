"""
HarnessPlatform · 大脑句柄（单实例 + 线程安全）
================================================
★ 核心不变量：**整个平台进程里只有一个 BioBrain 实例**。
   治 P-ARCH-1「两套引擎/多实例状态分裂」。

所有访问都过 `self._lock`（单大锁）—— BioBrain 不是线程安全的，
而平台的 tick 频率远低于锁竞争代价，串行化是最简单且正确的选择。
"""
from __future__ import annotations
import threading
from typing import Any, Callable, Optional


class BrainHandle:
    """BioBrain 的单例包装 + 串行化访问"""

    def __init__(self, brain: Any):
        self._brain = brain
        self._lock = threading.RLock()
        self._tick_count = 0

    # ---------- 访问 ----------
    @property
    def brain(self) -> Any:
        """★ 注意：直接访问绕过了锁，仅限单线程初始化路径使用"""
        return self._brain

    def with_lock(self, fn: Callable[[Any], Any]) -> Any:
        """在锁内执行 fn(brain)"""
        with self._lock:
            return fn(self._brain)

    # ---------- 便捷方法（都加锁） ----------
    def think(self, text: str, **kw) -> dict:
        with self._lock:
            self._tick_count += 1
            return self._brain.think(text, **kw)

    def observe(self, **kw) -> dict:
        with self._lock:
            return self._brain.observe(**kw)

    def teach(self, payload: dict, **kw) -> dict:
        with self._lock:
            return self._brain.teach(payload, **kw)

    def sleep(self, **kw) -> dict:
        with self._lock:
            return self._brain.sleep(**kw)

    def state(self) -> dict:
        with self._lock:
            return self._brain.state()

    def save(self):
        with self._lock:
            return self._brain.save()

    def load(self) -> bool:
        with self._lock:
            return self._brain.load()

    def run_ticks(self, n: int = 1) -> dict:
        with self._lock:
            self._tick_count += n
            return self._brain.run_ticks(n)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, seed: int = 1, state_path: Optional[str] = None,
               **kwargs) -> "BrainHandle":
        """在 code/ 目录上下文里构造 BioBrain"""
        import sys, os
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from server import BioBrain
        kw = {"seed": seed}
        if state_path:
            kw["state_path"] = state_path
        kw.update(kwargs)
        return cls(BioBrain(**kw))
