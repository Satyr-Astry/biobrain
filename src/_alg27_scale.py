"""ALG-27 规模验收：束数 > 池容量时，两阶段路由的打分降幅与一致率。

为什么需要这个脚本：
  本项目当前拓扑只有 **5 个束**，而 CAND_BLOCKS×CAND_BLOCK_SIZE = 32 →
  候选池把全部束都装进去 → 两阶段**不可能省成本**（已由 _resolve_search_domain 的
  「束数 ≤ 池容量 → 直接全量」守卫诚实处理，避免净增成本）。
  DeepSeek 论文的前提是 **位置数 ≫ 池容量**（上下文几万 token vs 池 32 块）。

  所以「降 ≥30%」这条必须在**束数 > 池容量**的口径下验收，否则测的是守卫分支。

用法：PYTHONHASHSEED=0 python _alg27_scale.py [n_tracts]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cog_vec import (NervousSystem, SensoryPort, ROUTE_K, N_HEADS, D_SOMA,
                       TIER_BIAS, TIER_ORDER, norm, CAND_BLOCKS, CAND_BLOCK_SIZE)


def build_brain(n_tracts, members_per_tract=8, n_neurons=None):
    ns = NervousSystem(seed=42)
    n_neurons = n_neurons or max(n_tracts * members_per_tract // 2, 64)
    ids = [ns.add_neuron("inter") for _ in range(n_neurons)]
    for k in range(2):
        ids[k] = ns.add_neuron("sensory")
    for k in range(n_neurons - 2, n_neurons):
        ns.neurons[ids[k]].role = "motor"
    rng = np.random.default_rng(7)
    for t in range(n_tracts):
        start = (t * members_per_tract) % n_neurons
        mem = [ids[(start + j) % n_neurons] for j in range(members_per_tract)]
        mem = list(dict.fromkeys(mem))
        if len(mem) >= 2:
            ns.add_tract(mem)
    return ns, ids


def score_matrix(ns, active, tract_ids):
    d_head = max(1, D_SOMA // N_HEADS)
    d_proj = N_HEADS * d_head
    idx_valid = [i for i in active
                 if i in ns.neurons and norm(ns.neurons[i].soma) >= 1e-8]
    if not idx_valid or not tract_ids:
        return None, None
    Q = np.stack([ns.neurons[i].soma[:d_proj] for i in idx_valid])
    K = np.stack([ns.tracts[t].key[:d_proj] for t in tract_ids])
    Qh = Q.reshape(len(idx_valid), N_HEADS, d_head)
    Kh = K.reshape(len(tract_ids), N_HEADS, d_head)
    qn = np.linalg.norm(Qh, axis=2)
    kn = np.linalg.norm(Kh, axis=2)
    S = np.einsum('nhd,thd->nht', Qh, Kh) / (qn[:, :, None] * kn.T[None, :, :] + 1e-8)
    src_t = np.array([TIER_ORDER.get(ns.neurons[i].role, 1) for i in idx_valid])
    dst_t = np.array([TIER_ORDER.get(ns.tracts[t].tier, 1) for t in tract_ids])
    S = S + np.where(dst_t[None, None, :] >= src_t[:, None, None], TIER_BIAS, -TIER_BIAS)
    return S, idx_valid


def topk_lexsort(S, k):
    neg_idx = np.broadcast_to(-np.arange(S.shape[2]), S.shape)
    return np.lexsort((neg_idx, -S), axis=2)[:, :, :min(k, S.shape[2])]


def run(n_tracts, n_ticks, hier):
    os.environ["BIO_HIER_ROUTE"] = "1" if hier else "0"
    ns, ids = build_brain(n_tracts)
    port = SensoryPort("text", ids[:8], dim=8)
    seq = [np.array([1., .5, .3, 0, 0, 0, 0, 0]),
           np.array([0., .8, .6, .2, 0, 0, 0, 0])]
    active = set()
    for s in seq:
        active |= set(port.ignite(s, ns)["ignited"].keys())
    traj = []
    for step in range(n_ticks):
        ns.apply_pending(active)
        active = ns.tick_activation(active)
        for tid in ns.tracts:
            ns.tracts[tid].activation = ns.tract_activation(tid)
        ns.compute(active)
        ns.t += 1
        all_ids = sorted(ns.tracts.keys())
        S, idx_valid = score_matrix(ns, list(active), all_ids)
        if S is None:
            traj.append({})
            continue
        order = topk_lexsort(S, ROUTE_K)
        rec = {}
        for a_, i in enumerate(idx_valid):
            for h in range(N_HEADS):
                rec[(i, h)] = frozenset(all_ids[j] for j in order[a_, h])
        traj.append(rec)
    return {"traj": traj, "calls": ns._score_calls_total,
            "full_scan": ns._full_scan_calls, "refreshes": ns._cand_refreshes,
            "fallbacks": ns._cand_fallbacks, "pool": list(ns._cand_pool),
            "n_tracts": len(ns.tracts)}


def main():
    n_tracts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pool_cap = CAND_BLOCKS * CAND_BLOCK_SIZE
    print("=" * 70)
    print(f"ALG-27 规模验收（束数={n_tracts}，池容量={pool_cap}，"
          f"块={CAND_BLOCK_SIZE}×{CAND_BLOCKS}）")
    print("=" * 70)
    full = run(n_tracts, 12, False)
    hier = run(n_tracts, 12, True)
    total = same = 0
    for rf, rh in zip(full["traj"], hier["traj"]):
        for k, v in rf.items():
            if k in rh:
                total += 1
                same += (rh[k] == v)
    rate = same / total * 100 if total else 0.0
    drop = (1 - hier["calls"] / max(1, full["calls"])) * 100
    print(f"实际束数              : {full['n_tracts']}")
    print(f"路由一致率            : {same}/{total} = {rate:.2f}%")
    print(f"打分次数（全量）      : {full['calls']}")
    print(f"打分次数（两阶段）    : {hier['calls']}（其中粗筛 {hier['full_scan']}）")
    print(f"打分降幅              : {drop:.2f}%")
    print(f"粗筛刷新/fallback     : {hier['refreshes']} / {hier['fallbacks']}")
    print(f"候选池                : {len(hier['pool'])} 束 → {hier['pool'][:12]}...")
    print("-" * 70)
    print(f"一致率 >90% : {'PASS' if rate > 90 else 'FAIL'} ({rate:.2f}%)")
    print(f"降幅 ≥30%   : {'PASS' if drop >= 30 else 'FAIL'} ({drop:.2f}%)")
    print("=" * 70)
    return 0 if (rate > 90 and drop >= 30) else 1


if __name__ == "__main__":
    sys.exit(main())
