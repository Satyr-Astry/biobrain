"""ALG-18 诊断4：加长 target 轮，检验中途注入是否**稳态**效应（而非热身瞬态）。

决定性检验：
  在 tick T 注入，最后 12 tick 读出。
  · 若「早注入被冲刷、晚注入保住」是稳态性质 → 只要留够冲刷时间，Δ 应大致恒定，
    且随注入后剩余 tick 数单调下降。
  · 若是热身瞬态 → 只有 t≈10 这种贴着读出口的时刻才高。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from server import CogVec
from _alg18_scan import slerp, W3

TARGET = "猫是哺乳动物"


def run_t(hist_words, t_inject, total, p2=1.0, hist_ticks=12):
    b = CogVec(seed=1)
    for x in hist_words:
        v = b._encode_text(x)
        ign = b.sensory_symbol.ignite(v, b.ns)
        b.active |= set(ign["ignited"]); b.tick_count += 1; b.run_ticks(hist_ticks)
    cur = b._encode_text(TARGET)
    ign = b.sensory_symbol.ignite(cur, b.ns)
    b.active |= set(ign["ignited"]); b.tick_count += 1
    b.run_ticks(t_inject)
    if hist_words:
        acc = np.zeros_like(cur); wsum = 0.0
        for k, x in enumerate(reversed(hist_words)):
            w = 0.5 ** k
            hv = b._encode_text(x)
            n = min(len(hv), len(cur)); acc[:n] += w * hv[:n]; wsum += w
        inj = slerp(cur, acc / wsum, p2 / (1.0 + p2))
        ign = b.sensory_symbol.ignite(inj, b.ns)
        b.active |= set(ign["ignited"])
    b.run_ticks(total - t_inject)
    return np.array([n.activity for n in b.ns.neurons.values()], dtype=float)


def metric3(hist_words, t_inject, total, p2=1.0):
    ps = {w: run_t([w] if hist_words else [], t_inject, total, p2) for w in W3}
    ks = list(ps)
    hd = [float(np.abs(ps[ks[i]] - ps[ks[j]]).mean()) for i in range(3) for j in range(i + 1, 3)]
    nv = run_t([], t_inject, total, p2)
    v0 = float(np.mean([np.abs(ps[k] - nv).mean() for k in ks]))
    return float(np.mean(hd)), float(min(hd)), v0


if __name__ == "__main__":
    TOTAL = 24
    print(f"=== 总 tick={TOTAL}（注入后剩余时间从 24 递减到 0）===")
    print(f"{'注入t':>6} {'剩余':>5} | {'历史间Δ':>10} {'min':>10} {'倍数':>7}")
    base = None
    for t in (0, 4, 8, 12, 16, 20, 22, 24):
        m, mn, v0 = metric3(["猫"], t, TOTAL)
        if t == 0:
            base = mn
        print(f"{t:>6} {TOTAL-t:>5} | {m:>10.6f} {mn:>10.6f} {mn/base:>6.2f}×")
    print("\n=== 注入后「剩余 tick 数」决定性：固定 total=12，只变注入时刻 ===")
    print(f"{'注入t':>6} {'剩余':>5} | {'历史间Δ':>10} {'min':>10}")
    for t in (0, 2, 4, 6, 8, 10, 11, 12):
        m, mn, v0 = metric3(["猫"], t, 12)
        print(f"{t:>6} {12-t:>5} | {m:>10.6f} {mn:>10.6f}")
    print("\n=== 历史长度（t=10, total=12, p2=1.0）===")
    for hl in ([], ["猫"], ["猫", "狗"], ["猫", "狗", "量子"]):
        ps = {}
        from itertools import product
        seqs = [hl] if hl else [[]]
        print(f"  （下面按「每条历史各自跑同一序列」统计，n={len(hl)}）")
        break
