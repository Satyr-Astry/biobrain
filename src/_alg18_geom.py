"""ALG-18 诊断2：分离「点火端饱和」与「tick 冲刷」两个损失源，并试不同注入几何。

损失源分离（hist=2 手写路径，只跑 target_ticks）：
  - 测 t=1 的 Δ  = 点火端能给出多少（未冲刷）
  - 测 t=12 的 Δ = 冲刷后剩多少
注入几何对比：
  A) 加法   mix = α·cur + β̂·hist,  β̂ 归一化到 |hist| = β·|cur|     ← 当前实现
  B) 等能量 mix = αβ 加权平均后**重归一化到 |cur|**（球面插值）
  C) 球面插值 slerp(cur, hist, t)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from server import CogVec

H = ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]
TARGET = "猫是哺乳动物"
NAMES = [h[:1] for h in H]   # 猫/狗/量子


def _unit(v):
    v = np.asarray(v, float).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def build(hist_words, geom, p1, p2, tticks):
    """hist_words: 历史词列表; geom: 'add'|'renorm'|'slerp'; p1=α p2=β"""
    b = CogVec(seed=1)
    hists = []
    for x in hist_words:
        v = b._encode_text(x)
        hists.append(v)
        ign = b.sensory_symbol.ignite(v, b.ns)
        b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(12)
    cur = b._encode_text(TARGET)
    if hists:
        acc = np.zeros_like(cur); wsum = 0.0
        for k, hv in enumerate(reversed(hists)):
            w = 0.5 ** k
            n = min(len(hv), len(cur)); acc[:n] += w * hv[:n]; wsum += w
        acc = acc / wsum
        cu, au = _unit(cur), _unit(acc)
        if geom == "add":                       # 能量随 β 增长（会撞饱和）
            inj = p1 * cur + p2 * au * np.linalg.norm(cur)
        elif geom == "renorm":                  # 等能量球面加权
            z = _unit(p1 * cu + p2 * au)
            inj = z * np.linalg.norm(cur)
        elif geom == "slerp":                   # 球面插值，t=p2/(p1+p2)
            t = p2 / (p1 + p2)
            d = float(np.clip(np.dot(cu, au), -1, 1)); om = np.arccos(d)
            if om < 1e-6:
                z = cu
            else:
                z = (np.sin((1 - t) * om) * cu + np.sin(t * om) * au) / np.sin(om)
            inj = z * np.linalg.norm(cur)
        else:
            raise ValueError(geom)
    else:
        inj = cur
    ign = b.sensory_symbol.ignite(inj, b.ns)
    b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(tticks)
    return np.array([n.activity for n in b.ns.neurons.values()], dtype=float)


def delta(geom, p1, p2, tticks=12):
    ps = {w: build([w], geom, p1, p2, tticks) for w in NAMES}
    ks = list(ps)
    hd = [float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()) for i in range(3) for j in range(i + 1, 3)]
    none_v = build([], geom, p1, p2, tticks)
    v0 = float(np.mean([np.abs(ps[k] - none_v).mean() for k in ks]))
    return float(np.mean(hd)), float(min(hd)), v0


if __name__ == "__main__":
    print(f"{'geom':>7} {'p1':>5} {'p2':>5} {'ticks':>6} | {'历史间Δ':>10} {'min':>10} {'有vs无':>10}")
    print("基线(无历史) t=12:", [round(x, 6) for x in delta("add", 1.0, 0.0, 12)])
    for geom in ("add", "renorm", "slerp"):
        for p2 in (0.25, 0.5, 1.0):
            for tt in (1, 12):
                d = delta(geom, 1.0, p2, tt)
                print(f"{geom:>7} {1.0:>5.2f} {p2:>5.2f} {tt:>6d} | {d[0]:>10.6f} {d[1]:>10.6f} {d[2]:>10.6f}")
