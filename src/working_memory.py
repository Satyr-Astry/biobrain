"""ALG-18 实施方案：WorkingMemory（中途再注入版）——可直接接线进 server.think()。

设计（实测依据见 ALG18_REPORT.md 第 2–5 节）：
  · 存最近 WM_SIZE 轮的**编码向量**（不是 activity）
  · 当前轮：先跑 WM_INJECT_TICK 个 tick（冲刷掉当前输入的前锋），
    再用 slerp 把「历史加权编码」以 WM_INJECT_T 的比例融入当前编码，重新点火，
    跑完剩余 tick。
  · 读出 = 全神经元 activity（不改既有口径）

实测（12 tick 轮）：
  · 历史间 Δ 由 0.001102 → min 0.014115（t=10/12），21× 于基线
  · 但「注入后剩余 tick 数」必须 ≤4 才能过线（见报告 §5），属**瞬态**机制

★回退：BIO_WORKMEM=0 → 逐位回到基线
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

WM_SIZE = 4
WM_DECAY = 0.5          # 第 i 近的历史权重 = 0.5**i
WM_ALPHA = 1.0          # 当前输入权重
WM_BETA = 0.5           # 历史混合权重
WM_INJECT_TICK = 11     # 在轮内第几个 tick 再注入（12 tick 轮的实测最优：t=11 → 13.7×）
WM_INJECT_T = 0.5       # slerp 比例 t = β/(α+β)
WM_ROUND_TICKS = 12     # think() 一轮的 tick 数（须与 server.think 一致）
WM_DEFAULT = "1"


def _enabled() -> bool:
    return os.environ.get("BIO_WORKMEM", WM_DEFAULT) != "0"


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """球面插值：t=0 → a，t=1 → b。保 |a|（不改变输入能量）。"""
    a = np.asarray(a, float).flatten()
    b = np.asarray(b, float).flatten()
    n = min(len(a), len(b))
    if n == 0:
        return a
    aa, bb = a[:n], b[:n]
    na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
    if na < 1e-12 or nb < 1e-12:
        return a
    cu, au = aa / na, bb / nb
    d = float(np.clip(np.dot(cu, au), -1.0, 1.0))
    om = np.arccos(d)
    if om < 1e-6:
        return a
    z = (np.sin((1 - t) * om) * cu + np.sin(t * om) * au) / np.sin(om)
    out = z * na
    if len(a) > n:
        out = np.concatenate([out, a[n:]])
    return out


class WorkingMemory:
    """最近 N 轮【编码】的环形缓冲 + 轮内再注入。"""

    def __init__(self, size: int = WM_SIZE, decay: float = WM_DECAY,
                 alpha: float = WM_ALPHA, beta: float = WM_BETA,
                 inject_tick: int = WM_INJECT_TICK, enabled: Optional[bool] = None):
        self.size = max(1, int(size))
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.inject_tick = int(inject_tick)
        self._enabled = _enabled() if enabled is None else bool(enabled)
        self._items: List[np.ndarray] = []      # 旧 → 新
        self.last_injected = False              # 观测用

    @property
    def enabled(self) -> bool:
        return self._enabled and _enabled()

    def clear(self) -> None:
        self._items = []
        self.last_injected = False

    def __len__(self) -> int:
        return len(self._items)

    # ---------- 读：历史加权和（NORM-4：纯读，不改状态）----------
    def history_vector(self) -> Optional[np.ndarray]:
        if not self.enabled or not self._items:
            return None
        ref = self._items[-1]
        acc = np.zeros_like(ref)
        wsum = 0.0
        for k, v in enumerate(reversed(self._items)):
            w = self.decay ** k
            n = min(len(v), len(acc))
            acc[:n] += w * np.asarray(v, float).flatten()[:n]
            wsum += w
        return acc / wsum if wsum > 0 else None

    def inject(self, current: np.ndarray) -> Optional[np.ndarray]:
        """返回「当前编码 ⊕ 历史」的混合编码；无历史/禁用 → None（调用方保持原路径）。"""
        hv = self.history_vector()
        if hv is None:
            self.last_injected = False
            return None
        t = self.beta / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.0
        self.last_injected = True
        return slerp(current, hv, t)

    # ---------- 写：轮末压栈 ----------
    def push(self, vec: np.ndarray) -> None:
        if not self.enabled:
            return
        self._items.append(np.asarray(vec, float).flatten().copy())
        if len(self._items) > self.size:
            self._items = self._items[-self.size:]

    def stats(self) -> dict:
        return {"enabled": self.enabled, "n": len(self._items), "size": self.size,
                "inject_tick": self.inject_tick, "t": round(self.beta / (self.alpha + self.beta), 3),
                "decay": self.decay, "last_injected": self.last_injected}
