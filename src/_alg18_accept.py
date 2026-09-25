"""ALG-18 验收：工作记忆「历史区分度」测试

复现原病灶：喂不同历史 → 探测同一输入 → 输出差异应随历史不同而不同。
原实测：Δ = 0.0001（几乎为零）；无历史 vs 有历史 Δ≈0.13（只记得"第几次"）。
目标：Δ > 0.01（100×）。

用法:
    PYTHONHASHSEED=0 python _alg18_accept.py
    BIO_WORKMEM=0 PYTHONHASHSEED=0 python _alg18_accept.py    # 对照组
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import CogVec

PROBE = "猫"
HISTS = {
    "H1": ["猫", "猫", "猫"],                    # 同域历史
    "H2": ["狗", "狗", "狗"],                    # 近邻域历史
    "H3": ["量子纠缠", "薛定谔方程", "波函数"],     # 远域历史
    "H4": [],                                    # 无历史（对照）
}
N_PROBE = 3          # 探测轮数（取最后一轮读数）
TICKS = 12


def probe_vector(brain: CogVec) -> np.ndarray:
    """探测读数：符号端 + 运动端 + 全部神经元的活动快照。

    必须是【可观测量】而非内部状态 —— 用神经元 activity（这正是下游
    read/emit 会读到的东西）。
    """
    ids = sorted(set(brain.symbol_ids) | set(brain.motor_ids))
    return np.array([brain.ns.neurons[i].activity for i in ids], dtype=float)


def run_history(hist, seed=42, use_memory_registry=None):
    """喂一段历史，然后探测同一输入，返回可观测量。"""
    brain = CogVec(seed=seed)
    for t in hist:
        brain.think(t)
    reads = []
    for _ in range(N_PROBE):
        brain.think(PROBE)
        reads.append(probe_vector(brain))
    return reads[-1], brain


def main():
    mode = os.environ.get("BIO_WORKMEM", "1")
    print("=" * 70)
    print(f"  ALG-18 工作记忆 · 历史区分度验收   (BIO_WORKMEM={mode})")
    print("=" * 70)

    reads = {}
    brains = {}
    for name, hist in HISTS.items():
        v, b = run_history(hist)
        reads[name] = v
        brains[name] = b
        print(f"  {name}  历史={str(hist):38s} 注入栈深={b.workmem.depth} "
              f"混合次数={b.workmem.blends}")

    print("-" * 70)
    print("  探测输入 = %r ；读数 = 符号端+运动端 activity" % PROBE)
    print("-" * 70)

    # ★核心指标：不同历史之间的两两差异（同一次探测）
    pairs = []
    names = [n for n in HISTS if HISTS[n]]      # 只比较"有历史"的，与原来一致
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]
            d = float(np.max(np.abs(reads[a] - reads[b_])))
            pairs.append((a, b_, d))
            print(f"  Δ({a} 历史, {b_} 历史) = {d:.6f}")

    deltas = [p[2] for p in pairs]
    mean_d = float(np.mean(deltas)); max_d = float(np.max(deltas))

    # 同历史重复稳定性（防止"随机抖动冒充区分度"）
    r1, _ = run_history(HISTS["H1"])
    r2, _ = run_history(HISTS["H1"])
    repeat_d = float(np.max(np.abs(r1 - r2)))

    # 无历史 vs 有历史（原 Δ≈0.13，说明只记得"第几次"）
    novhist = float(np.max(np.abs(reads["H1"] - reads["H4"])))

    print("-" * 70)
    print(f"  ★历史区分度 平均 Δ = {mean_d:.6f}   最大 Δ = {max_d:.6f}")
    print(f"    同历史重复性 Δ = {repeat_d:.6f}   （应≈0：确定性，非随机抖动）")
    print(f"    无历史 vs 有历史 Δ = {novhist:.6f}  （原病灶：只区分第几次）")

    # 非零断言：soma / axon 必须非零（防"没在算"）
    ns = brains["H1"].ns
    soma_nz = sum(1 for n_ in ns.neurons.values() if float(np.max(np.abs(n_.soma))) > 0)
    axon_nz = sum(1 for n_ in ns.neurons.values() if float(np.max(np.abs(n_.axon))) > 0)
    print(f"    soma 非零神经元 = {soma_nz}/{len(ns.neurons)}   "
          f"axon 非零神经元 = {axon_nz}/{len(ns.neurons)}")

    BASE = 0.0001
    print("=" * 70)
    ok = True
    if mode == "0":
        print(f"  【对照组 BIO_WORKMEM=0】历史区分度 = {mean_d:.6f}")
        print(f"    回退是否生效（应≈基线 0.0001）: "
              f"{'✅ 是' if mean_d < 3 * BASE else '❌ 否'}")
        ok = mean_d < 3 * BASE
    else:
        gain = mean_d / BASE if BASE else float("inf")
        print(f"  基线（旧版实测）= {BASE:.6f}")
        print(f"  现值            = {mean_d:.6f}")
        print(f"  提升倍数        = {gain:.1f}×   （目标 ≥ 100×）")
        print(f"  阈值 > 0.01     : {'✅ 通过' if mean_d > 0.01 else '❌ 未通过'}")
        print(f"  soma/axon 非零  : "
              f"{'✅ 通过' if (soma_nz > 0 and axon_nz > 0) else '❌ 未通过'}")
        print(f"  确定性（重复性≈0）: {'✅ 通过' if repeat_d < BASE else '❌ 未通过'}")
        ok = (mean_d > 0.01) and soma_nz > 0 and axon_nz > 0 and repeat_d < BASE
    print("=" * 70)
    print("  结论: " + ("✅ 验收通过" if ok else "❌ 验收失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
