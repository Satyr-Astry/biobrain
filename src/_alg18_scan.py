"""ALG-18 诊断3：中途再注入的稳健性扫描（历史长度 / 注入时刻 / 强度 / 序列）。"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from server import CogVec

H3 = ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]
W3 = ["猫", "狗", "量子"]
TARGET = "猫是哺乳动物"


def slerp(a, b, t):
    cu = a / (np.linalg.norm(a) + 1e-12)
    au = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.clip(np.dot(cu, au), -1, 1))
    om = np.arccos(d)
    if om < 1e-6:
        return a
    z = (np.sin((1 - t) * om) * cu + np.sin(t * om) * au) / np.sin(om)
    return z * np.linalg.norm(a)


def run(hist_words, t_inject, p2=1.0, total=12, hist_ticks=12, beta_hist=0.5, inject=True):
    b = CogVec(seed=1)
    # 历史轮（各自跑满，纯为了累积状态）
    for x in hist_words:
        v = b._encode_text(x)
        ign = b.sensory_symbol.ignite(v, b.ns)
        b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(hist_ticks)
    # 当前轮
    cur = b._encode_text(TARGET)
    ign = b.sensory_symbol.ignite(cur, b.ns)
    b.active |= set(ign["ignited"]); b.tick_count += 1
    if inject and hist_words and t_inject > 0:
        b.run_ticks(t_inject)
        acc = np.zeros_like(cur); wsum = 0.0
        for k, x in enumerate(reversed(hist_words)):
            w = beta_hist ** k
            hv = b._encode_text(x)
            n = min(len(hv), len(cur)); acc[:n] += w * hv[:n]; wsum += w
        if wsum > 0:
            inj = slerp(cur, acc / wsum, p2 / (1.0 + p2))
            ign = b.sensory_symbol.ignite(inj, b.ns)
            b.active |= set(ign["ignited"])
        b.run_ticks(total - t_inject)
    else:
        b.run_ticks(total)
    return np.array([n.activity for n in b.ns.neurons.values()], dtype=float)


def metric(**kw):
    ps = {w: run([w], **kw) for w in W3}
    ks = list(ps)
    hd = [float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()) for i in range(3) for j in range(i + 1, 3)]
    nv = run([], inject=False, t_inject=0,
             **{k: v for k, v in kw.items() if k != "t_inject"})
    v0 = float(np.mean([np.abs(ps[k] - nv).mean() for k in ks]))
    return float(np.mean(hd)), float(min(hd)), v0


if __name__ == "__main__":
    print("基线（t_inject=0 → 关）：")
    m, mn, v0 = metric(t_inject=0)
    print(f"  Δ={m:.6f} min={mn:.6f} 有vs无={v0:.6f}\n")
    print("注入时刻扫描 (p2=1.0)：")
    for t in (0, 2, 4, 6, 8, 10):
        m, mn, v0 = metric(t_inject=t)
        print(f"  t={t:>2} | Δ={m:>9.6f} min={mn:>9.6f} {'  ✅>10×' if mn > 0.01102 else ''}")
    print("\n强度扫描 (t=6)：")
    for p2 in (0.25, 0.5, 1.0, 2.0, 4.0):
        m, mn, v0 = metric(t_inject=6, p2=p2)
        print(f"  p2={p2:>4} | Δ={m:>9.6f} min={mn:>9.6f} 有vs无={v0:.6f}")
    print("\n历史长度扫描 (t=6, p2=1.0)：")
    for hl in ([], ["猫"], ["猫", "狗"], ["猫", "狗", "量子"], ["猫", "狗", "量子", "猫"]):
        m, mn, v0 = metric(hist_words=hl, t_inject=6)
        print(f"  len={len(hl)} | Δ={m:>9.6f} min={mn:>9.6f} 有vs无={v0:.6f}")
