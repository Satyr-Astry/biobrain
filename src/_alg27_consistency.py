"""ALG-27 两阶段路由验收：路由一致率 + 打分次数 + soma/axon 非零断言

口径（必须写清，避免误判）：
  · 基准（真值） = 全量路由（BIO_HIER_ROUTE=0）下，每个活跃神经元×每个注意力头
    在 **全部束** 上打分后的 top-ROUTE_K 束集合。
  · 实测（两阶段） = 默认（BIO_HIER_ROUTE=1）下，候选池内精排得到的 top-ROUTE_K 束集合。
  · 一致率 = 两次选择 **集合完全相同** 的 (神经元,头) 格子数 / 总格子数。

  ★与「2.4% 陷阱」的区别：那次比的是「硬性限制在自身束内」vs 全量；
    本次比的是「全范围粗筛候选池 + 池内精排」vs 全量 ——
    候选池来自全范围扫描，保留跨束可能。

用法：PYTHONHASHSEED=0 python _alg27_consistency.py
"""
import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cog_vec import (NervousSystem, SensoryPort, ROUTE_K, N_HEADS, D_SOMA,
                       TIER_BIAS, TIER_ORDER, norm)


def build_brain():
    """复刻 server.py 的拓扑规模（64 神经元 / 5 束），保证与项目实测同口径。"""
    ns = NervousSystem(seed=42)
    sensory = [ns.add_neuron("sensory") for _ in range(16)]
    inter = [ns.add_neuron("inter") for _ in range(42)]
    motor = [ns.add_neuron("motor") for _ in range(6)]
    self_sensory = []
    ns.add_tract(sensory[:2] + self_sensory)
    ns.add_tract(sensory[2:22] if len(sensory) >= 22 else sensory[2:] + inter[:2])
    ns.add_tract(inter[:22])
    ns.add_tract(inter[22:])
    ns.add_tract(motor + inter[-2:])
    return ns, sensory, inter, motor


def score_matrix(ns, active, tract_ids):
    """复刻 _phase_bind 的评分公式（cos + tier 偏置），返回 (Na,H,T)。"""
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
    """★必须 np.lexsort(...)：主键 −得分，次键 −索引（复刻 sorted(reverse=True)）。"""
    neg_idx = np.broadcast_to(-np.arange(S.shape[2]), S.shape)
    return np.lexsort((neg_idx, -S), axis=2)[:, :, :min(k, S.shape[2])]


def run_pipeline(seed_state, n_ticks=12, hier=True):
    """跑同一套输入序列，分别在 hier=True/False 下收集每 tick 的 (神经元,头)→路由集合。"""
    os.environ["BIO_HIER_ROUTE"] = "1" if hier else "0"
    ns, sensory, inter, motor = build_brain()
    port = SensoryPort("text", sensory, dim=8)
    seq = [np.array([1., .5, .3, 0, 0, 0, 0, 0]),
           np.array([0., .8, .6, .2, 0, 0, 0, 0]),
           np.array([.2, 0., .9, .4, .1, 0, 0, 0])]
    active = set()
    for s in seq:
        active |= set(port.ignite(s, ns)["ignited"].keys())
    traj = []
    soma_max = 0.0
    axon_max = 0.0
    for step in range(n_ticks):
        ns.apply_pending(active)
        active = ns.tick_activation(active)
        for tid in ns.tracts:
            ns.tracts[tid].activation = ns.tract_activation(tid)
        ns.compute(active)
        ns.update_plasticity(active, {})
        ns.tick_scheduler()
        ns.t += 1
        # 收集本 tick 的路由（用与 _phase_bind 相同的域与 tie-break）
        all_ids = sorted(ns.tracts.keys())
        S_full, idx_valid = score_matrix(ns, list(active), all_ids)
        if S_full is not None:
            order = topk_lexsort(S_full, ROUTE_K)
            rec = {}
            for a_, i in enumerate(idx_valid):
                for h in range(N_HEADS):
                    rec[(i, h)] = frozenset(all_ids[j] for j in order[a_, h])
            traj.append(rec)
        else:
            traj.append({})
        for n in ns.neurons.values():
            soma_max = max(soma_max, norm(n.soma))
            axon_max = max(axon_max, norm(n.axon))
    return {"traj": traj, "soma_max": soma_max, "axon_max": axon_max,
            "score_calls_total": ns._score_calls_total,
            "full_scan_calls": ns._full_scan_calls,
            "refreshes": ns._cand_refreshes,
            "fallbacks": ns._cand_fallbacks,
            "pool": list(ns._cand_pool)}


def main():
    # ---- 注意：必须在同一进程内分别跑（env 在进程启动时读，故直接在 run 里设）----
    full = run_pipeline(None, 12, hier=False)
    hier = run_pipeline(None, 12, hier=True)

    total = 0
    same = 0
    per_tick = []
    for t, (rf, rh) in enumerate(zip(full["traj"], hier["traj"])):
        tt = ss = 0
        for key, sel_f in rf.items():
            if key not in rh:
                continue
            tt += 1
            if rh[key] == sel_f:
                ss += 1
        total += tt
        same += ss
        per_tick.append((ss, tt))
    rate = (same / total * 100.0) if total else 0.0

    print("=" * 66)
    print("ALG-27 两阶段路由验收")
    print("=" * 66)
    print(f"路由一致率（两阶段 vs 全量，集合完全相同）：{same}/{total} = {rate:.2f}%")
    print(f"  逐 tick：{', '.join(f'{s}/{t}' for s, t in per_tick)}")
    print("-" * 66)
    print(f"打分次数（全量路由）  : {full['score_calls_total']}")
    print(f"打分次数（两阶段路由）: {hier['score_calls_total']}"
          f"  （其中粗筛全范围 {hier['full_scan_calls']}）")
    drop = (1 - hier["score_calls_total"] / max(1, full["score_calls_total"])) * 100
    print(f"打分次数降幅          : {drop:.2f}%")
    print(f"粗筛刷新次数          : {hier['refreshes']}   fallback 次数: {hier['fallbacks']}")
    print(f"候选池                : {hier['pool']}（{len(hier['pool'])} 束）")
    print("-" * 66)
    print(f"soma 非零断言 : max|soma| = {hier['soma_max']:.6f}  → "
          f"{'PASS' if hier['soma_max'] > 1e-8 else 'FAIL'}")
    print(f"axon 非零断言 : max|axon| = {hier['axon_max']:.6f}  → "
          f"{'PASS' if hier['axon_max'] > 1e-8 else 'FAIL'}")
    print("-" * 66)
    print(f"一致率 >90%  : {'PASS' if rate > 90 else 'FAIL'}  ({rate:.2f}%)")
    print(f"降幅 ≥30%    : {'PASS' if drop >= 30 else 'FAIL'}  ({drop:.2f}%)")
    print("=" * 66)
    return 0 if (rate > 90 and drop >= 30) else 1


if __name__ == "__main__":
    sys.exit(main())
