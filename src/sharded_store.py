"""
仿生大脑 · 分片存储层 (sharded_store.py)
==========================================
让大脑突破内存限制 —— 参数量可以大于 RAM。

设计（基于实测：实验1/2）：
  · 神经元按"脑片(fragment)"分组，每片独立文件
  · 只有**活跃片**被载入内存（稠密数组，快）
  · 沉睡片只保留 memmap 句柄（不占内存）
  · 操作系统页缓存自动做 L0/L1/L2 管理

实测依据：
  · 10.24GB 矩阵 memmap 打开仅 0.001s，不占内存
  · 随机读 1% = 164ms（冷）/ 47ms（热，页缓存）
  · 全量顺序扫描 2.44 GB/s

核心 API：
    store = ShardedStore(root, n_neurons, dim, frag_size)
    store.get(idx)              # 读神经元参数（自动载入所在片）
    store.set(idx, values)      # 写
    store.sync()                # 写回磁盘
    store.stats()               # 内存/磁盘/载入片统计
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class Fragment:
    """一个脑片：一段连续的神经元"""

    __slots__ = ("fid", "start", "end", "path", "_mm", "loaded", "last_used", "hits")

    def __init__(self, fid: int, start: int, end: int, path: Path):
        self.fid = fid
        self.start = start
        self.end = end
        self.path = path
        self._mm = None
        self.loaded = False
        self.last_used = 0.0
        self.hits = 0

    @property
    def size(self) -> int:
        return self.end - self.start

    def open(self, dim: int, mode: str = "r+") -> np.memmap:
        """打开 memmap（不加载数据到内存）"""
        if self._mm is None:
            if not self.path.exists():
                # 创建空文件
                with open(self.path, "wb") as f:
                    f.truncate(self.size * dim * 4)
            self._mm = np.memmap(self.path, dtype=np.float32, mode=mode,
                                 shape=(self.size, dim))
        return self._mm

    def close(self):
        if self._mm is not None:
            try:
                self._mm.flush()
            except Exception:
                pass
            self._mm = None

    def touch(self):
        self.last_used = time.time()
        self.hits += 1


class ShardedStore:
    """分片存储：让参数量突破内存限制

    内存占用 ∝ 活跃片数 × 片大小（而非总神经元数）
    """

    def __init__(self, root: str | Path, n_neurons: int, dim: int,
                 frag_size: int = 100_000, max_loaded: int = 8,
                 verbose: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = n_neurons
        self.dim = dim
        self.frag_size = frag_size
        self.max_loaded = max_loaded        # 内存中最多同时载入的片数
        self.verbose = verbose

        # 建分片索引
        self.n_frags = (n_neurons + frag_size - 1) // frag_size
        self.frags: List[Fragment] = []
        meta_path = self.root / "manifest.json"
        for i in range(self.n_frags):
            s = i * frag_size
            e = min(s + frag_size, n_neurons)
            fp = self.root / f"frag_{i:06d}.dat"
            self.frags.append(Fragment(i, s, e, fp))

        self.meta_path = meta_path
        self.t_load = 0.0
        self.n_loads = 0
        self.n_evicts = 0

        if verbose:
            total_gb = n_neurons * dim * 4 / 1e9
            print(f"[ShardedStore] {n_neurons:,} 神经元 × {dim} 维 = {total_gb:.2f} GB")
            print(f"  分片: {self.n_frags} 片 × {frag_size:,} 神经元"
                  f"（每片 {frag_size*dim*4/1e6:.1f} MB）")
            print(f"  最大载入: {max_loaded} 片 = "
                  f"{max_loaded*frag_size*dim*4/1e6:.0f} MB 内存")

    # ---------------- 片定位 ----------------
    def frag_of(self, idx: int) -> int:
        return min(idx // self.frag_size, self.n_frags - 1)

    def frags_of(self, idxs: np.ndarray) -> np.ndarray:
        return np.minimum(idxs // self.frag_size, self.n_frags - 1)

    # ---------------- 载入 / 卸载 ----------------
    def _ensure_loaded(self, fid: int) -> Fragment:
        """确保片已打开（memmap 不算"载入内存"）

        ★修正：先 touch（更新使用时间），再 evict——否则刚打开的片
        可能因为 last_used 太旧而被立刻卸载，导致 _mm 变 None。
        且 evict 必须保护当前片。
        """
        fr = self.frags[fid]
        fr.touch()                     # ★先标记使用时间
        if not fr.loaded:
            fr.open(self.dim)
            fr.loaded = True
            self.n_loads += 1
        self._maybe_evict(protect=fid) # ★保护当前片
        return fr

    def _maybe_evict(self, protect: Optional[int] = None):
        """超出上限时，卸载最久未用的片（不卸载 protect 指定的片）"""
        loaded = [f for f in self.frags
                  if f.loaded and f._mm is not None and f.fid != protect]
        if len(loaded) + 1 > self.max_loaded:
            loaded.sort(key=lambda f: f.last_used)
            # 需要卸载的数量（+1 是当前片）
            n_evict = len(loaded) + 1 - self.max_loaded
            for f in loaded[:n_evict]:
                f.close()
                f.loaded = False
                self.n_evicts += 1

    # ---------------- 读写 ----------------
    def get(self, idxs: np.ndarray) -> np.ndarray:
        """读神经元参数（按片分组，减少 memmap 访问次数）"""
        idxs = np.asarray(idxs).ravel()
        out = np.zeros((len(idxs), self.dim), dtype=np.float32)
        # 按片分组
        fids = self.frags_of(idxs)
        order = np.argsort(fids)
        sorted_idxs = idxs[order]
        sorted_fids = fids[order]
        bounds = np.searchsorted(sorted_fids, np.arange(self.n_frags + 1))

        pos = 0
        for fid in range(self.n_frags):
            lo, hi = bounds[fid], bounds[fid + 1]
            if lo >= hi:
                continue
            fr = self._ensure_loaded(fid)
            mm = fr._mm
            local = sorted_idxs[lo:hi] - fr.start
            out[order[lo:hi]] = mm[local]
            pos = hi
        return out

    def set(self, idxs: np.ndarray, values: np.ndarray):
        """写神经元参数"""
        idxs = np.asarray(idxs).ravel()
        fids = self.frags_of(idxs)
        order = np.argsort(fids)
        sorted_idxs = idxs[order]
        sorted_vals = values[order]
        sorted_fids = fids[order]
        bounds = np.searchsorted(sorted_fids, np.arange(self.n_frags + 1))

        for fid in range(self.n_frags):
            lo, hi = bounds[fid], bounds[fid + 1]
            if lo >= hi:
                continue
            fr = self._ensure_loaded(fid)
            local = sorted_idxs[lo:hi] - fr.start
            fr._mm[local] = sorted_vals[lo:hi]

    def sync(self, frags: Optional[List[int]] = None):
        """写回磁盘"""
        targets = self.frags if frags is None else [self.frags[i] for i in frags]
        t0 = time.time()
        for fr in targets:
            if fr._mm is not None:
                fr._mm.flush()
        self.t_load += time.time() - t0

    def close(self):
        for fr in self.frags:
            fr.close()

    # ---------------- 统计 ----------------
    def stats(self) -> Dict:
        loaded = [f for f in self.frags if f.loaded and f._mm is not None]
        total_gb = self.n * self.dim * 4 / 1e9
        loaded_gb = sum(f.size for f in loaded) * self.dim * 4 / 1e9
        return {
            "n_neurons": self.n,
            "n_frags": self.n_frags,
            "frag_size": self.frag_size,
            "total_gb": round(total_gb, 2),
            "loaded_frags": len(loaded),
            "loaded_gb": round(loaded_gb, 3),
            "n_loads": self.n_loads,
            "n_evicts": self.n_evicts,
            "saved_ratio": round(1 - loaded_gb / total_gb, 4) if total_gb else 0,
        }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import tempfile, shutil

    print("=" * 74)
    print("  分片存储层 · 自测")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="shard_test_"))
    try:
        # 100 万神经元，每片 10 万 → 10 片
        N, D, FS = 1_000_000, 256, 100_000
        store = ShardedStore(tmp, N, D, frag_size=FS, max_loaded=3, verbose=True)
        print()

        rng = np.random.default_rng(0)

        # 1) 写入热数据（1%）
        n_hot = N // 100
        hot = rng.choice(N, n_hot, replace=False)
        vals = rng.normal(0, 0.02, (n_hot, D)).astype(np.float32)
        t0 = time.time()
        store.set(hot, vals)
        store.sync()
        t_write = time.time() - t0
        print(f"① 写入 {n_hot:,} 行（1%）: {t_write:.2f}s")

        # 2) 读取验证（正确性）
        t0 = time.time()
        back = store.get(hot[:1000])
        t_read = time.time() - t0
        err = float(np.abs(back - vals[:1000]).max())
        print(f"② 读回 1000 行: {t_read*1000:.1f}ms | 最大误差={err:.2e} "
              f"{'✓ 正确' if err < 1e-6 else '✗ 错误!'}")

        # 3) 内存占用
        st = store.stats()
        print(f"③ 内存占用:")
        print(f"     总数据: {st['total_gb']} GB")
        print(f"     实际载入: {st['loaded_gb']} GB ({st['loaded_frags']} 片)")
        print(f"     节省: {st['saved_ratio']*100:.1f}%")
        print(f"     载入次数={st['n_loads']} 卸载次数={st['n_evicts']}")

        # 4) 随机访问性能
        idx = rng.choice(N, 50_000, replace=False)
        t0 = time.time()
        _ = store.get(idx)
        t_rand = time.time() - t0
        print(f"④ 随机读 5 万行: {t_rand*1000:.1f}ms "
              f"({50_000/t_rand/1000:.0f}K 行/s)")

        # 5) 二次访问（页缓存）
        t0 = time.time()
        _ = store.get(idx)
        t_rand2 = time.time() - t0
        print(f"⑤ 二次读同样数据: {t_rand2*1000:.1f}ms "
              f"（页缓存加速 {t_rand/max(t_rand2,1e-6):.1f}×）")

        print()
        print("=" * 74)
        print("  分片存储可行 ✓")
        print("=" * 74)
        print(f"  核心意义: {st['total_gb']}GB 数据，内存只用 {st['loaded_gb']}GB")
        print(f"  扩展性: 神经元数量不再受内存限制，只受磁盘限制")

        store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
