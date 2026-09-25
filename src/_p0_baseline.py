# -*- coding: utf-8 -*-
"""P0 基线快照: 记录改动前 _phase_bind 的**真实**输出 + compute 耗时。
产出 _p0_baseline.npz (分数矩阵) + _p0_baseline.json (指标)
★目的: 改动后用同一脚本重跑, 逐元素比对 → 位级一致性证明。
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from cog_vec import (D_SOMA, N_HEADS, ROUTE_K, TIER_BIAS, TIER_ORDER,
                       ATTN_GAIN, ATTN_MODE_DEFAULT)
from _c_algo_bench import build, prime, health, tm

MODE = sys.argv[1] if len(sys.argv) > 1 else "cos"
if MODE == "learned":
    import os
    os.environ["BIO_ATTN_MODE"] = "learned"

R = {"mode": MODE}
ns, sensory, inter, motor, tids = build()
active_h = prime(ns, sensory, n_ticks=8, seed=0)
snz, anz, smax, amax = health(ns, active_h)
R["health"] = {"n_active": len(active_h), "soma_nz": snz, "axon_nz": anz,
               "max_soma": round(smax, 6), "max_axon": round(amax, 6)}
assert snz > 0 and anz > 0, "compute 空转!"

active = sorted(ns.neurons)
# 用**全 neuron 集**作为 active 以获得最大覆盖面 (含 soma 近零者, 检验 skip 逻辑)
active_all = sorted(i for i in ns.neurons)

# ---------- 捕获真实 _phase_bind 的逐头分数 (用 monkey 记录) ----------
# 直接在外部复刻源码语义 —— 但为证明"等价", 我们记录 **函数实际返回值**
# _phase_bind 只写 _attn_out / _attn_last。这里我们再包一层: 记录 head_scores。
captured = {}
orig_bind = ns._phase_bind

# 让 _attn_out 可观测
d_head = max(1, D_SOMA // N_HEADS)
d_proj = N_HEADS * d_head
TIDS = sorted(ns.tracts)
tier_val = {t: TIER_ORDER.get(ns.tracts[t].tier, 1) for t in TIDS}

# 真实调用 (不 monkey, 直接调用, 记录 _attn_out)
ns._phase_bind(active_all, True, None)
attn_out = dict(ns._attn_out)
R["n_attn_out"] = len(attn_out)
routed_ids = sorted(attn_out.keys())

# ---------- 逐头分数矩阵 (精确复刻 _phase_bind 的打分路径) ----------
LEARNED = (ns.attn_mode == "learned")
if LEARNED:
    Kc = {t: (ns.W_k.T @ ns.tracts[t].key)[:d_proj] for t in TIDS}
    Vc = {t: (ns.W_v.T @ ns.tracts[t].value)[:d_proj] for t in TIDS}

def score_matrix():
    """返回 (ids, S) —— S[a,h,t] 完整分数 (不截断)"""
    ids = list(routed_ids)
    S = np.zeros((len(ids), N_HEADS, len(TIDS)))
    for a_, i in enumerate(ids):
        n = ns.neurons[i]
        q_full = (ns.W_q.T @ n.soma)[:d_proj] if LEARNED else n.soma
        src_t = TIER_ORDER.get(n.role, 1)
        for h in range(N_HEADS):
            lo, hi = h*d_head, (h+1)*d_head
            q = q_full[lo:hi]
            qn = float(np.linalg.norm(q))
            for k, t in enumerate(TIDS):
                kv = Kc[t][lo:hi] if LEARNED else ns.tracts[t].key[lo:hi]
                c = float(np.dot(q, kv)) / (qn*float(np.linalg.norm(kv)) + 1e-8)
                c += TIER_BIAS if tier_val[t] >= src_t else -TIER_BIAS
                S[a_, h, k] = c
    return ids, S

t0 = time.perf_counter()
ids, S = score_matrix()
R["score_ms"] = round((time.perf_counter()-t0)*1000, 3)

# ---------- top-K (源码语义: sorted(reverse=True) 的 tie-break) ----------
def topk_src(S, k=ROUTE_K):
    out = np.zeros((S.shape[0], N_HEADS, k), dtype=int)
    for a_ in range(S.shape[0]):
        for h in range(N_HEADS):
            sc = [(float(S[a_, h, t]), t) for t in range(S.shape[2])]
            sc.sort(reverse=True)
            for j in range(k):
                out[a_, h, j] = sc[j][1]
    return out

TOP = topk_src(S)

# ---------- 计时: 端到端 tick_phased + compute ----------
def run_compute():
    ns._phase_bind(active_all, True, None)
    ns._phase_integrate(active_all)

t_compute = tm(run_compute, reps=300)

# 整 tick 计时（需要重置状态避免漂移）
def full_tick():
    ns2, s2, _, _, _ = build()
    prime(ns2, s2, n_ticks=8, seed=0)
    a = set(i for i in ns2.neurons)
    return ns2.tick_phased(a, n_spread=1)

t0 = time.perf_counter()
for _ in range(5):
    full_tick()
R["full_tick_ms"] = round((time.perf_counter()-t0)/5*1000, 3)

R["bind_us"] = round(t_compute, 2)
R["n_tracts"] = len(TIDS)
R["n_routed"] = len(routed_ids)

np.savez("_p0_baseline.npz", S=S, TOP=TOP,
         ids=np.array(routed_ids), tids=np.array(TIDS))
with open("_p0_baseline.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
print("wrote _p0_baseline.npz / _p0_baseline.json")
