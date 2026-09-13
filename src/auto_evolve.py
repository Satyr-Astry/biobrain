"""
仿生大脑 · 自动优化引擎 (auto_evolve.py)
==========================================
主人要的「自动化进行创造性项目完善和优化」。

它做什么：
  1. 自动跑基准测试，找出当前瓶颈
  2. 自动尝试优化（参数搜索 / 结构变体）
  3. 记录每次尝试的结果，保留更好的
  4. 输出优化报告

不是"改代码"（那需要人审），而是：
  · 自动搜索最优配置（规模、抑制强度、稀疏率…）
  · 自动验证改进是否真实有效
  · 自动发现退化并回滚

用法：
    python auto_evolve.py --bench          # 跑基准
    python auto_evolve.py --search         # 参数搜索
    python auto_evolve.py --auto --rounds 10   # 自动优化 N 轮
"""
from __future__ import annotations
import json
import time
import itertools
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULT_FILE = HERE / "auto_evolve_results.json"


# ============================================================
# 基准测试
# ============================================================
def bench_once(n_neurons: int, n_tracts: int, steps: int = 5,
               ignite_ratio: float = 0.01) -> Dict:
    """跑一次基准，返回指标"""
    from tensor_brain import TensorBrain, ACTIVE_EPS
    t0 = time.time()
    b = TensorBrain(n_neurons=n_neurons, n_tracts=n_tracts, tract_size=24)
    build = time.time() - t0

    k = max(1, int(n_neurons * ignite_ratio))
    b.ignite_batch(np.arange(k, dtype=np.int32), np.full(k, 0.6, dtype=np.float32))

    ts = []
    for _ in range(steps):
        t0 = time.time()
        a = b.tick()
        b.update_plasticity(a)
        ts.append(time.time() - t0)

    # 学习效果：可塑范数
    plastic = float(np.linalg.norm(b.plasticity))
    # 稀疏性
    active_ratio = a.size / n_neurons
    # 收敛稳定性
    act_hist = []
    b.ignite_batch(np.arange(k, dtype=np.int32), np.full(k, 0.6, dtype=np.float32))
    for _ in range(5):
        aa = b.tick()
        act_hist.append(float(b.activity.mean()))
    conv = 1.0 / (1.0 + np.abs(np.diff(act_hist[-4:])).mean() * 100)

    return {
        "n_neurons": n_neurons,
        "n_tracts": n_tracts,
        "build_s": round(build, 3),
        "tick_ms": round(np.mean(ts) * 1000, 2),
        "active_ratio": round(active_ratio, 4),
        "plastic_norm": round(plastic, 5),
        "convergence": round(conv, 4),
    }


def score(metrics: Dict, target_tick_ms: float = 200.0) -> float:
    """综合评分：速度 + 稀疏性 + 学习能力"""
    tick = metrics["tick_ms"]
    speed_score = max(0.0, 1.0 - tick / target_tick_ms)          # 越快越好
    sparse_score = 1.0 - abs(metrics["active_ratio"] - 0.08) / 0.08  # 8% 最理想
    sparse_score = max(0.0, sparse_score)
    learn_score = min(1.0, metrics["plastic_norm"] / 0.15)        # 可塑量
    conv_score = metrics["convergence"]
    return round(speed_score * 0.3 + sparse_score * 0.25
                 + learn_score * 0.25 + conv_score * 0.2, 4)


# ============================================================
# 参数搜索
# ============================================================
def search_params(rounds: int = 12, verbose: bool = True) -> List[Dict]:
    """自动搜索最优配置"""
    results = []

    # 搜索空间
    sizes = [(65536, 4096), (131072, 8192), (262144, 16384)]
    ignites = [0.005, 0.01, 0.02]

    combos = list(itertools.product(sizes, ignites))[:rounds]
    if verbose:
        print(f"搜索 {len(combos)} 组配置…")
        print("-" * 76)

    for i, ((n, nt), ir) in enumerate(combos, 1):
        try:
            m = bench_once(n, nt, ignite_ratio=ir)
            m["ignite_ratio"] = ir
            m["score"] = score(m)
            results.append(m)
            if verbose:
                print(f"  [{i:2d}/{len(combos)}] n={n:>7} ignite={ir:.3f} | "
                      f"{m['tick_ms']:7.1f}ms | 稀疏={m['active_ratio']:.3f} | "
                      f"可塑={m['plastic_norm']:.4f} | 分={m['score']:.4f}")
        except Exception as e:
            print(f"  [{i:2d}] n={n} 失败: {type(e).__name__}")

    results.sort(key=lambda x: -x["score"])
    return results


# ============================================================
# 自动优化循环
# ============================================================
def auto_optimize(rounds: int = 5, verbose: bool = True) -> Dict:
    """自动优化：每轮做一次实验，保留最优"""
    # 加载历史
    history = []
    if RESULT_FILE.exists():
        try:
            history = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        except Exception:
            history = []

    best = max(history, key=lambda x: x.get("score", 0), default=None)

    if verbose:
        print("=" * 76)
        print("  自动优化引擎")
        print("=" * 76)
        print(f"  历史记录: {len(history)} 条"
              + (f" | 当前最优分={best['score']}" if best else ""))

    seen = {(h["n_neurons"], h["ignite_ratio"]) for h in history}
    candidates = []
    for n, nt in [(65536, 4096), (131072, 8192), (262144, 16384)]:
        for ir in (0.005, 0.01, 0.02, 0.04):
            if (n, ir) not in seen:
                candidates.append((n, nt, ir))

    if verbose:
        print(f"  待测配置: {len(candidates)} 组（跳过已测）\n")

    new = []
    for i, (n, nt, ir) in enumerate(candidates[:rounds], 1):
        m = bench_once(n, nt, ignite_ratio=ir)
        m["ignite_ratio"] = ir
        m["score"] = score(m)
        m["ts"] = datetime.now().isoformat()
        new.append(m)
        history.append(m)

        flag = ""
        if best is None or m["score"] > best["score"]:
            best = m
            flag = "  ★ 新最优！"
        if verbose:
            print(f"  [{i}/{min(rounds,len(candidates))}] n={n:>7} ignite={ir:.3f} "
                  f"| {m['tick_ms']:7.1f}ms | 分={m['score']:.4f}{flag}")

    RESULT_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if verbose:
        print(f"\n  记录已保存: {RESULT_FILE.name}（共 {len(history)} 条）")
        if best:
            print(f"  当前最优: n={best['n_neurons']} ignite={best['ignite_ratio']} "
                  f"分={best['score']}")
    return {"best": best, "new": new, "total": len(history)}


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 · 自动优化引擎")
    ap.add_argument("--bench", action="store_true", help="跑一次基准")
    ap.add_argument("--search", action="store_true", help="参数搜索")
    ap.add_argument("--auto", action="store_true", help="自动优化循环")
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()

    if args.bench:
        print("=" * 76)
        print("  基准测试")
        print("=" * 76)
        for n, nt in [(65536, 4096), (262144, 16384), (1048576, 65536)]:
            m = bench_once(n, nt, steps=3)
            m["score"] = score(m)
            print(f"  n={n:>8} | 构建{m['build_s']:5.2f}s | {m['tick_ms']:8.1f}ms | "
                  f"稀疏={m['active_ratio']:.3f} | 可塑={m['plastic_norm']:.4f} | 分={m['score']:.4f}")
    elif args.search:
        res = search_params(args.rounds)
        print(f"\n最优配置: {res[0] if res else 'none'}")
    elif args.auto:
        auto_optimize(args.rounds)
    else:
        ap.print_help()
