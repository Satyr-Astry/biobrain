# -*- coding: utf-8 -*-
"""learned 模式为何非位级? 定位 A/B 差异来源 (候选: 批量 matmul 结合顺序)"""
import sys, os, json
os.environ["BIO_ATTN_MODE"] = "learned"
sys.path.insert(0, ".")
import numpy as np
from cog_vec import D_SOMA, N_HEADS, ROUTE_K, TIER_BIAS, TIER_ORDER, norm
from _c_algo_bench import build, prime

R = {}
d_head = max(1, D_SOMA // N_HEADS); d_proj = N_HEADS * d_head
R.update(D_SOMA=D_SOMA, N_HEADS=N_HEADS, d_head=d_head, d_proj=d_proj)

ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)
R["attn_mode"] = ns.attn_mode
active = sorted(ns.neurons)
TIDS = sorted(ns.tracts)

# 1. Q 投影: 逐神经元 vs 批量 stack
i0 = active[0]
one = (ns.W_q.T @ ns.neurons[i0].soma)[:d_proj]
stk = np.stack([(ns.W_q.T @ ns.neurons[i].soma)[:d_proj] for i in active])
R["q_proj_bitwise_equal"] = bool(np.array_equal(one, stk[0]))

# 2. 逐头点积: einsum 批量 vs 源码逐头 np.dot
tr0 = ns.tracts[TIDS[0]]
kv0 = (ns.W_k.T @ tr0.key)[:d_proj].reshape(N_HEADS, d_head)
Qh = one.reshape(N_HEADS, d_head)
batch_h = np.einsum('hd,hd->h', Qh, kv0)
per_h = np.array([float(np.dot(Qh[h], kv0[h])) for h in range(N_HEADS)])
R["head_dot_max_abs_diff"] = float(np.max(np.abs(batch_h - per_h)))
R["head_dot_bitwise_equal"] = bool(np.array_equal(batch_h, per_h))

# 3. cos 分母: norm 批量 vs 逐头
nb = np.linalg.norm(kv0, axis=1)
nr = np.array([float(np.linalg.norm(kv0[h])) for h in range(N_HEADS)])
R["norm_batch_max_abs_diff"] = float(np.max(np.abs(nb - nr)))
R["norm_batch_bitwise_equal"] = bool(np.array_equal(nb, nr))

# 4. softmax 加权融合: 批量 (Wt*Vg).sum  vs 源码 for 累加 w*vv
Vh = np.stack([(ns.W_v.T @ ns.tracts[t].value)[:d_proj].reshape(N_HEADS, d_head) for t in TIDS])
rng = np.random.default_rng(0)
order = np.stack([rng.permutation(len(TIDS))[:ROUTE_K] for _ in range(N_HEADS)])   # H × K
Wt = np.abs(rng.normal(size=(N_HEADS, ROUTE_K))); Wt = Wt / Wt.sum(axis=1, keepdims=True)
hh = np.arange(N_HEADS)[:, None]
Vg = Vh[order, hh, :]                                  # H × K × d
batch_f = (Wt[:, :, None] * Vg).sum(axis=1)            # H × d
src_f = np.zeros((N_HEADS, d_head))
for h in range(N_HEADS):
    for k in range(ROUTE_K):
        src_f[h] += Wt[h, k] * Vh[order[h, k], h, :]
R["fuse_max_abs_diff"] = float(np.max(np.abs(batch_f - src_f)))
R["fuse_bitwise_equal"] = bool(np.array_equal(batch_f, src_f))

# 5. W_o 投影: acc @ W_o  vs  W_o.T @ acc
acc = np.random.default_rng(1).normal(0, 1, d_proj)
a1 = (acc @ ns.W_o)[:D_SOMA]
a2 = (ns.W_o.T @ acc)[:D_SOMA]
R["Wo_proj_max_abs_diff"] = float(np.max(np.abs(a1 - a2)))
R["Wo_proj_bitwise_equal"] = bool(np.array_equal(a1, a2))

# 6. 逐头权重归一: 源码 exp 是 np.exp(scalar) 标量, 批量是 ndarray → 比对
z = 1.0 + 0.0
s_batch = np.array([0.3, -0.7, 1.1, 0.0])
e_batch = np.exp(s_batch - s_batch.max())
e_src = np.array([np.exp(float(v - s_batch.max())) for v in s_batch])
R["exp_scalar_vs_vec_bitwise"] = bool(np.array_equal(e_batch, e_src))

print(json.dumps(R, indent=1, ensure_ascii=False))
