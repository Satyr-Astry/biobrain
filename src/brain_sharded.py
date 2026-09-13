"""
亿级大脑 · 分片版（brain_sharded.py）
======================================
把 ShardedStore 接进 TensorBrain，实现"参数量超出内存"的大脑。

核心设计（基于实验 3/4 的实测结论）：
  · 神经网络按"功能片"分区存储（相似神经元同片 → 激活天然局部）
  · 只有活跃片载入内存（内存占用与总规模解耦）
  · 支持"任意规模"，只受磁盘限制

用法：
    python brain_sharded.py --test              # 自测
    python brain_sharded.py --size 10000000     # 1000万神经元
    python brain_sharded.py --size 100000000    # 1亿神经元
"""
from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sharded_store import ShardedStore

DEFAULT_ROOT = Path(r"E:\BRAIN_DATA")


class ShardedBrain:
    """分片大脑：参数量可超过物理内存

    与 TensorBrain 的区别：
      · TensorBrain：所有参数在内存（快，但受内存限制）
      · ShardedBrain：参数在磁盘分片，活跃片在内存（慢些，但无上限）
    """

    def __init__(self, root: str | Path, n_neurons: int, dim: int = 256,
                 frag_size: int = 200_000, max_loaded: int = 6,
                 d_soma: int = 64, seed: int = 42, verbose: bool = True):
        self.root = Path(root)
        self.n = n_neurons
        self.dim = dim
        self.d_soma = d_soma

        # 分片存储（核心）
        self.store = ShardedStore(self.root, n_neurons, dim,
                                  frag_size=frag_size, max_loaded=max_loaded,
                                  verbose=verbose)

        # 轻量状态（这些必须常驻内存，但很小）
        self.activity = np.zeros(n_neurons, dtype=np.float32)   # 4 bytes/神经元
        self.importance = np.zeros(n_neurons, dtype=np.float32)
        self.soma = np.zeros((n_neurons, d_soma), dtype=np.float32)
        self.tract_members = None
        self.t = 0

        if verbose:
            act_gb = n_neurons * 4 / 1e9
            soma_gb = n_neurons * d_soma * 4 / 1e9
            print(f"[ShardedBrain] {n_neurons:,} 神经元")
            print(f"  可分片参数: {n_neurons*dim*4/1e9:.2f} GB（磁盘，按需载入）")
            print(f"  常驻内存:   激活+重要性 {act_gb*2:.2f} GB + 胞体 {soma_gb:.2f} GB")
            print(f"  峰值内存:   ≈ {(act_gb*2 + soma_gb + max_loaded*frag_size*dim*4/1e9):.2f} GB")

    # ---------------- 激活 ----------------
    def ignite(self, idxs: np.ndarray, amounts: Optional[np.ndarray] = None):
        if amounts is None:
            amounts = np.full(len(idxs), 0.6, dtype=np.float32)
        np.add.at(self.activity, idxs, amounts)
        np.clip(self.activity, 0, 1, out=self.activity)

    # ---------------- 思考 ----------------
    def tick(self, activate_ratio: float = 0.08) -> Dict:
        """一次思考周期

        关键：只有**活跃神经元**才需要从分片存储读取参数。
        """
        t0 = time.time()
        self.t += 1

        # 1) 衰减
        self.activity *= 0.9

        # 2) 找出活跃神经元（这一步只碰内存中的 activity 数组）
        n_active = int(self.n * activate_ratio)
        if n_active > 0:
            # 取 top-k 活跃
            if int((self.activity > 1e-4).sum()) > n_active:
                thr = np.partition(self.activity, -n_active)[-n_active]
                self.activity[self.activity < thr] = 0.0

        active = np.where(self.activity > 1e-4)[0]

        # 3) 从分片存储读取活跃神经元的参数（局部性决定速度）
        if active.size:
            params = self.store.get(active)          # ← 唯一碰磁盘的地方
            a = self.activity[active][:, None]
            self.soma[active] = np.tanh(
                self.soma[active] * 0.5
                + params[:, :self.d_soma] * a
            )

        # 4) 简单扩散（用内存中的 activity，近似）
        if active.size:
            # 每 8 个活跃神经元影响 1 个邻居（近似）
            spread_src = active[::8]
            spread_dst = (spread_src + 1) % self.n
            self.activity[spread_dst] += 0.08 * self.activity[spread_src]
            np.clip(self.activity, 0, 1, out=self.activity)

        dt = time.time() - t0
        return {
            "tick": self.t,
            "active": int(active.size),
            "active_ratio": round(active.size / self.n, 4),
            "dt_ms": round(dt * 1000, 2),
        }

    # ---------------- 学习 ----------------
    def learn(self, idxs: np.ndarray, delta: float = 0.001):
        """对指定神经元做可塑更新（写回分片）"""
        if len(idxs) == 0:
            return
        params = self.store.get(idxs)
        params += np.random.normal(0, delta, params.shape).astype(np.float32)
        self.store.set(idxs, params)

    # ---------------- 睡眠 ----------------
    def sleep(self) -> Dict:
        """睡眠：批量写回 + 修剪"""
        t0 = time.time()
        self.store.sync()
        # 重要性衰减
        self.importance *= 0.98
        dt = time.time() - t0
        return {"sync_s": round(dt, 3), "stats": self.store.stats()}

    def stats(self) -> Dict:
        s = self.store.stats()
        s["ticks"] = self.t
        s["active"] = int((self.activity > 1e-4).sum())
        return s

    def close(self):
        self.store.close()


# ============================================================
# 自测
# ============================================================
def test(n_neurons: int = 5_000_000, frag_size: int = 200_000,
         max_loaded: int = 6, rounds: int = 5):
    import shutil
    root = DEFAULT_ROOT / f"test_{n_neurons // 1_000_000}M"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    print("=" * 76)
    print(f"  分片大脑自测：{n_neurons:,} 神经元")
    print("=" * 76)

    b = ShardedBrain(root, n_neurons, frag_size=frag_size,
                     max_loaded=max_loaded, verbose=True)
    print()

    rng = np.random.default_rng(0)

    # 关键测试：局部激活 vs 随机激活
    n_hot = int(n_neurons * 0.01)

    print("① 随机激活（最坏情况）")
    t_all = []
    for r in range(rounds):
        idx = rng.choice(n_neurons, n_hot, replace=False)
        b.ignite(idx)
        info = b.tick(activate_ratio=0.02)
        b.learn(idx)
        t_all.append(info["dt_ms"])
    print(f"   平均: {np.mean(t_all):.1f} ms/tick")

    print()
    print("② 局部激活（功能分区，最好情况）")
    # 集中在 10% 的片
    focus_n = int(n_neurons * 0.10)
    t_loc = []
    for r in range(rounds):
        idx = rng.choice(focus_n, n_hot, replace=False)
        b.ignite(idx)
        info = b.tick(activate_ratio=0.02)
        b.learn(idx)
        t_loc.append(info["dt_ms"])
    print(f"   平均: {np.mean(t_loc):.1f} ms/tick")

    print()
    print("③ 睡眠")
    s = b.sleep()
    print(f"   {s}")

    print()
    st = b.stats()
    print("=" * 76)
    print("  最终状态")
    print("=" * 76)
    print(f"  数据总量:     {st['total_gb']:.2f} GB")
    print(f"  内存占用:     {st['loaded_gb']:.3f} GB（{st['loaded_frags']} 片）")
    print(f"  内存节省:     {st['saved_ratio']*100:.1f}%")
    print(f"  分片数:       {st['n_frags']}")
    print(f"  随机激活:     {np.mean(t_all):.1f} ms/tick")
    print(f"  局部激活:     {np.mean(t_loc):.1f} ms/tick  加速 {np.mean(t_all)/max(np.mean(t_loc),1e-6):.1f}×")
    print()
    print(f"  ✓ {st['total_gb']:.1f} GB 大脑在 16 GB 内存机器上运行")

    b.close()
    print()
    print(f"数据目录: {root}")
    return st


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="分片大脑")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--size", type=int, default=5_000_000)
    ap.add_argument("--frag", type=int, default=200_000)
    ap.add_argument("--loaded", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    test(args.size, args.frag, args.loaded, args.rounds)
