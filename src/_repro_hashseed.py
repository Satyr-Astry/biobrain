"""最小复现：PYTHONHASHSEED 非确定性暗雷（Bug A / Bug B）

用法：
    PYTHONHASHSEED=1 python _repro_hashseed.py
    PYTHONHASHSEED=2 python _repro_hashseed.py

输出一个稳定的指纹（fingerprint）。修复前不同 hashseed 下指纹不同；修复后必须相同。
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cog_vec as bb  # noqa: E402
from cog_vec import NervousSystem, SensoryPort  # noqa: E402


def fp(arr) -> str:
    """数组 → 稳定指纹（与内存地址/哈希随机化无关）"""
    a = np.asarray(arr, dtype=np.float64)
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def make_brain(seed=42):
    ns = NervousSystem(seed=seed)
    visual = [ns.add_neuron("sensory") for _ in range(4)]
    symbol = [ns.add_neuron("sensory") for _ in range(4)]
    inter = [ns.add_neuron("inter") for _ in range(8)]
    motor = [ns.add_neuron("motor") for _ in range(3)]
    t1 = ns.add_tract(visual + inter[:2])
    t2 = ns.add_tract(symbol + inter[:2])
    t3 = ns.add_tract(inter[2:6] + motor)
    t4 = ns.add_tract(inter[:2] + inter[2:6])
    g1 = ns.add_group([t1, t2], "percept")
    g2 = ns.add_group([t3, t4], "action")
    ns.add_library([g1])
    ns.add_library([g2])
    return ns, {"visual": visual, "symbol": symbol, "inter": inter,
                "motor": motor}


def main():
    seed_env = os.environ.get("PYTHONHASHSEED", "<unset>")
    print(f"PYTHONHASHSEED = {seed_env}")

    # ---------- Bug A 证据：投影矩阵受 hash() 影响 ----------
    print("\n[Bug A] abs(hash(modality)) 投影种子")
    for mod in ("vision", "symbol"):
        print(f"  abs(hash({mod!r})) % 2**32 = {abs(hash(mod)) % (2 ** 32)}")
    vp = SensoryPort("vision", list(range(4)), 8)
    sp = SensoryPort("symbol", list(range(4)), 8)
    print(f"  proj(vision) 指纹 = {fp(vp.proj)}")
    print(f"  proj(symbol) 指纹 = {fp(sp.proj)}")

    # 同一进程内重复构造必须稳定（这条修复前后都成立）
    vp2 = SensoryPort("vision", list(range(4)), 8)
    print(f"  同进程重建 proj 指纹 = {fp(vp2.proj)}  "
          f"({'一致' if fp(vp.proj) == fp(vp2.proj) else '不一致'})")

    # ---------- Bug B 证据：端到端 rollout 指纹 ----------
    print("\n[Bug B] 端到端 tick 演化指纹（字符串键迭代顺序）")
    ns, ids = make_brain()
    vp3 = SensoryPort("vision", ids["visual"], 8)
    sp3 = SensoryPort("symbol", ids["symbol"], 8)
    active = set(vp3.ignite(np.array([1., .5, .3, 0, 0, 0, 0, 0]), ns)["ignited"]) | \
        set(sp3.ignite(np.array([.8, .6, 0, 0, 0, 0, 0, 0]), ns)["ignited"])
    for _ in range(15):
        ns.apply_pending(active)

    acts = np.array([ns.neurons[i].activity for i in sorted(ns.neurons)])
    print(f"  15 tick 后全部神经元激活指纹 = {fp(acts)}")
    print(f"  motor 峰值 = {max(ns.neurons[m].activity for m in ids['motor']):.10f}")
    print(f"  总激活 = {float(acts.sum()):.10f}")

    # 角色推断（_infer_tract_tier 依赖角色集合）也要稳定
    tiers = sorted(ns._infer_tract_tier(list(t.neurons if hasattr(t, 'neurons')
                                             else t.members))
                   if False else "n/a")
    tier_map = {tid: ns._infer_tract_tier(list(tr.members))
                for tid, tr in ns.tracts.items()}
    print(f"  束层级映射 = {sorted(tier_map.items())}")

    # ---------- 综合数字签名 ----------
    sig = hashlib.md5(
        (fp(vp.proj) + fp(sp.proj) + fp(acts) + str(sorted(tier_map.items()))).encode()
    ).hexdigest()
    print(f"\n=== 综合签名 SIGNATURE = {sig} ===")
    return sig


if __name__ == "__main__":
    main()
