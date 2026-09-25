"""C 算法专家 · 计算热点与优化验证台  v2
关键点：每次测量前完整重置神经元状态（activity/soma/axon），保留结构权重。
"""
from __future__ import annotations
import time, sys, cProfile, pstats, io as _io
import numpy as np

sys.path.insert(0, ".")
from cog_vec import (NervousSystem, SensoryPort, D_SOMA, N_HEADS, ROUTE_K,
                       TIER_BIAS, ATTN_GAIN, ACTIVE_EPS, MSG_GAIN, TIER_ORDER)


def build(NTRACTS=5, K=13):
    """复刻 server.py 的拓扑构造风格：重叠束 + 组 + 库。"""
    ns = NervousSystem(seed=7)
    sensory = [ns.add_neuron("sensory") for _ in range(K)]
    inter = [ns.add_neuron("inter") for _ in range(2 * K)]
    motor = [ns.add_neuron("motor") for _ in range(K)]
    tids = []
    tids.append(ns.add_tract(sensory + inter[:2]))
    tids.append(ns.add_tract(sensory[:K // 2] + inter[:2]))
    tids.append(ns.add_tract(inter[:K] + motor[:K // 2]))
    tids.append(ns.add_tract(inter[K:] + motor[K // 2:]))
    tids.append(ns.add_tract(inter[:2] + inter[K:2 + K]))
    g1 = ns.add_group(tids[:2], "percept")
    g2 = ns.add_group(tids[2:4], "action")
    ns.add_library([g1]); ns.add_library([g2])
    return ns, sensory, inter, motor, tids


def reset_state(ns):
    """★完整重置神经元状态 —— 不重置结构权重。
    教训：只清 active 不清 activity → 假 dmin=0。"""
    for n in ns.neurons.values():
        n.activity = 0.0
        n.soma = np.zeros(D_SOMA)
        n.axon = np.zeros(D_SOMA)
        n.plasticity_pending = None
    for tr in ns.tracts.values():
        tr.activation = 0.0
    ns.pending_ignitions.clear()
    ns.active_concepts = getattr(ns, "active_concepts", set())


def prime(ns, sensory, n_ticks=8, seed=0):
    reset_state(ns)
    rng = np.random.default_rng(seed)
    shp = SensoryPort("s", sensory, D_SOMA)
    active = set()
    for t in range(n_ticks):
        r = shp.ignite(rng.normal(0, 1, D_SOMA), ns)
        active |= set(r["ignited"])
        active = ns.tick_phased(active, n_spread=1)
    return active


def health(ns, active):
    soma_nz = sum(1 for i in active if np.linalg.norm(ns.neurons[i].soma) > 1e-8)
    axon_nz = sum(1 for i in active if np.linalg.norm(ns.neurons[i].axon) > 1e-8)
    s_max = max((float(np.linalg.norm(ns.neurons[i].soma)) for i in active), default=0.0)
    a_max = max((float(np.linalg.norm(ns.neurons[i].axon)) for i in active), default=0.0)
    return soma_nz, axon_nz, s_max, a_max


def tm(fn, reps=200):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


if __name__ == "__main__":
    ns, sensory, inter, motor, tids = build()
    sizes = [len(ns.tracts[t].members) for t in sorted(ns.tracts)]
    n_neu = len(ns.neurons)
    edges = sum(s * s for s in sizes)
    print("=" * 70)
    print("PART 0 · 结构 / 健康")
    print("=" * 70)
    print(f"神经元={n_neu} 束={len(ns.tracts)} 束规模={sizes}")
    print(f"束内边(含自环) Σ K² = {edges}")
    print(f"全局稀疏度 = {1 - edges/(n_neu*(n_neu-1)):.4f}")
    active = prime(ns, sensory)
    snz, anz, smax, amax = health(ns, active)
    print(f"活跃集={len(active)}  活跃率={len(active)/n_neu:.1%}")
    print(f"★防线 soma非零={snz}/{len(active)}  max||soma||={smax:.4f}")
    print(f"★防线 axon非零={anz}/{len(active)}  max||axon||={amax:.4f}")
    assert snz > 0 and anz > 0, "✗ 致命：compute 空转！"

    print("\n" + "=" * 70)
    print("PART 1 · cProfile 热点归因（20 tick）")
    print("=" * 70)
    prime(ns, sensory)
    rng = np.random.default_rng(1)
    shp = SensoryPort("s", sensory, D_SOMA)
    act = set()
    pr = cProfile.Profile()
    pr.enable()
    for t in range(20):
        r = shp.ignite(rng.normal(0, 1, D_SOMA), ns)
        act |= set(r["ignited"])
        act = ns.tick_phased(act, n_spread=1)
    pr.disable()
    s = _io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(16)
    txt = s.getvalue()
    tot = None
    for ln in txt.splitlines():
        if "function calls" in ln and "seconds" in ln:
            tot = ln
        if ln.strip() and (ln.strip()[0].isdigit() or "{" in ln or "built-in" in ln):
            print(ln[:160])
