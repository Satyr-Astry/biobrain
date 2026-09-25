"""ALG-18 参数扫描：α/β/decay 对「历史间区分度」的影响（think 完整路径）。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from server import CogVec
from working_memory import WorkingMemory

H = ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]
TARGET = "猫是哺乳动物"


def run(alpha, beta, decay, size=4, target_ticks=12, hist=None):
    """复刻 think()，但注入口径由参数控制。返回 (Δ历史间, Δ有vs无)"""
    hist = hist if hist is not None else [h[:1] for h in H]
    ps = {}
    for h in hist:
        b = CogVec(seed=1)
        wm = WorkingMemory(size=size, decay=decay, alpha=alpha, beta=beta)
        for x in [h]:
            v = b._encode_text(x)
            ign = b.sensory_symbol.ignite(wm.mix(v), b.ns)
            b.active |= set(ign["ignited"])
            b.tick_count += 1
            b.run_ticks(12)
            wm.push(v)
        v = b._encode_text(TARGET)
        ign = b.sensory_symbol.ignite(wm.mix(v), b.ns)
        b.active |= set(ign["ignited"])
        b.tick_count += 1
        b.run_ticks(target_ticks)
        ps[h] = np.array([n.activity for n in b.ns.neurons.values()], dtype=float)
    # 无历史
    b = CogVec(seed=1)
    wm = WorkingMemory(size=size, decay=decay, alpha=alpha, beta=beta)
    v = b._encode_text(TARGET)
    ign = b.sensory_symbol.ignite(wm.mix(v), b.ns)
    b.active |= set(ign["ignited"])
    b.tick_count += 1
    b.run_ticks(target_ticks)
    none_v = np.array([n.activity for n in b.ns.neurons.values()], dtype=float)

    ks = list(ps)
    hd = [float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()) for i in range(len(ks)) for j in range(i + 1, len(ks))]
    vs0 = [float(np.abs(ps[k] - none_v).mean()) for k in ks]
    return float(np.mean(hd)), float(min(hd)), float(np.mean(vs0))


if __name__ == "__main__":
    print("基线 β=0（=禁用历史）：")
    m, mn, v0 = run(1.0, 0.0, 0.5)
    print(f"  历史间Δ={m:.6f} min={mn:.6f} 有vs无={v0:.6f}\n")
    print(f"{'alpha':>6} {'beta':>6} {'decay':>6} | {'历史间Δ':>10} {'min':>10} {'有vs无':>10}")
    for beta in (0.25, 0.5, 1.0, 2.0):
        for alpha in (1.0,):
            m, mn, v0 = run(alpha, beta, 0.5)
            print(f"{alpha:>6.2f} {beta:>6.2f} {0.5:>6.2f} | {m:>10.6f} {mn:>10.6f} {v0:>10.6f}")
    print()
    for decay in (0.3, 0.5, 0.7, 1.0):
        m, mn, v0 = run(1.0, 0.5, decay)
        print(f"{1.0:>6.2f} {0.5:>6.2f} {decay:>6.2f} | {m:>10.6f} {mn:>10.6f} {v0:>10.6f}")
    print("\n多轮历史（3 轮 [猫,猫,狗] vs [猫,猫,量子] vs [猫,猫,猫]）：")
    seqs = [["猫", "猫", "狗"], ["猫", "猫", "量子"], ["猫", "猫", "猫"]]
    for beta in (0.5, 1.0):
        ps = {}
        for s in seqs:
            b = CogVec(seed=1)
            wm = WorkingMemory(size=4, decay=0.5, alpha=1.0, beta=beta)
            for x in s:
                v = b._encode_text(x)
                ign = b.sensory_symbol.ignite(wm.mix(v), b.ns)
                b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(12); wm.push(v)
            v = b._encode_text(TARGET)
            ign = b.sensory_symbol.ignite(wm.mix(v), b.ns)
            b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(12)
            ps["-".join(s)] = np.array([n.activity for n in b.ns.neurons.values()], dtype=float)
        ks = list(ps)
        hd = [float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()) for i in range(3) for j in range(i + 1, 3)]
        print(f"  β={beta}: Δ={np.mean(hd):.6f} {[round(x,6) for x in hd]}")
