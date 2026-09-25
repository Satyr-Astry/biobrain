"""ALG-18 探针：不同历史间的区分度（基线 / 改后对比）。

口径与主脑一致：读过历史后再读 target，比较全部神经元 activity 的 MAE。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from server import CogVec


def act_vec(b):
    return np.array([n.activity for n in b.ns.neurons.values()], dtype=float)


def probe(history, target, extra_ticks=0):
    b = CogVec(seed=1)
    for h in history:
        b.think(h)
    r = b.think(target)
    if extra_ticks:
        b.run_ticks(extra_ticks)
    return act_vec(b), b


HISTS = ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]
TARGET = "猫是哺乳动物"


def report(tag):
    ps, brains = {}, {}
    for h in HISTS:
        v, b = probe([h], TARGET)
        ps[h] = v
        brains[h] = b
    ds = []
    ks = list(ps)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = float(np.abs(ps[ks[i]] - ps[ks[j]]).mean())
            ds.append((f"{ks[i][:4]} vs {ks[j][:4]}", d))
    # 有历史 vs 无历史
    base_v, base_b = probe([], TARGET)
    for h in HISTS:
        ds.append((f"{h[:4]} vs 无历史", float(np.abs(ps[h] - base_v).mean())))
    print(f"--- {tag} ---")
    for name, d in ds:
        print(f"  {name:22s} Δ={d:.6f}")
    hd = [d for n, d in ds if "无历史" not in n]
    print(f"  >>> 历史间 Δ 均值={np.mean(hd):.6f}  最小={min(hd):.6f}  最大={max(hd):.6f}")
    return ps, ds, base_v


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "baseline")
