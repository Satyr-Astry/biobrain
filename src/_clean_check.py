# -*- coding: utf-8 -*-
"""清理前后: 候选 lexsort key 写法位级等价性 + 输出快照"""
import sys, os, json
sys.path.insert(0, ".")
import numpy as np
from cog_vec import D_SOMA, N_HEADS, ROUTE_K
from _c_algo_bench import build, prime

ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)

# ---- A. lexsort key 三种写法位级等价 ----
K = ROUTE_K
rng = np.random.default_rng(0)
# 含大量精确平局: 构造 Na×H×T 矩阵, 一半是整数平局
S = rng.integers(0, 3, size=(7, N_HEADS, 5)).astype(float)
S2 = rng.normal(0, 1, size=(52, N_HEADS, 5))
res = {}
for name, Sx in [("tie_heavy", S), ("random", S2)]:
    na, nh, t = Sx.shape
    kk = min(K, t)
    # 旧写法
    old = np.lexsort((-np.arange(t)[None, None, :].repeat(na, 0).repeat(nh, 1), -Sx),
                     axis=2)[:, :, :kk]
    # 新写法 A: 直接一维索引广播 (单键形状 (T,), 依赖 last-axis broadcast)
    newA = np.lexsort((-np.arange(t), -Sx), axis=2)[:, :, :kk]
    # 新写法 B: 显式 (1,1,T) 广播
    newB = np.lexsort((-np.arange(t)[None, None, :], -Sx), axis=2)[:, :, :kk]
    res[name] = {"old_eq_newA": bool(np.array_equal(old, newA)),
                 "old_eq_newB": bool(np.array_equal(old, newB))}
    # 与源码 sorted(reverse=True) 语义比对
    src = np.zeros((na, nh, kk), dtype=int)
    for a in range(na):
        for h in range(nh):
            sc = [(float(Sx[a, h, j]), j) for j in range(t)]
            sc.sort(reverse=True)
            for j in range(kk):
                src[a, h, j] = sc[j][1]
    res[name]["old_eq_src"] = bool(np.array_equal(old, src))
    res[name]["newA_eq_src"] = bool(np.array_equal(newA, src))
    res[name]["tie_frac"] = float((np.diff(np.sort(Sx, axis=2), axis=2) == 0).mean())

# ---- B. 输出快照 (供清理后位级比对) ----
act = sorted(ns.neurons)
ns._phase_bind(act, True, None)
snap = {str(i): ns._attn_out[i] for i in ns._attn_out}
np.savez("_pre_clean_snap.npz", **snap)
res["n_snap"] = len(snap)
res["snap_hash"] = float(sum(float(v.sum()) for v in snap.values()))
print(json.dumps(res, indent=1, ensure_ascii=False))
