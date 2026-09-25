# -*- coding: utf-8 -*-
"""清理后 vs 清理前: 12-tick soma 轨迹 + attn_out 位级比对"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import cog_vec
assert "_preclean" not in cog_vec.__file__
from cog_vec import D_SOMA, N_HEADS, ROUTE_K
from _c_algo_bench import build, prime

R = {}
ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)
act = sorted(ns.neurons)
traj = []
for _ in range(12):
    ns.tick_phased(set(act), n_spread=1)
    traj.append({i: ns.neurons[i].soma.copy() for i in ns.neurons})
pre = np.load("_preclean_run/_pre_traj.npz")
mx_t = 0.0; bad = []
for k in range(12):
    for i in ns.neurons:
        a, b = pre[f"t{k}_{i}"], traj[k][i]
        d = float(np.max(np.abs(a - b)))
        mx_t = max(mx_t, d)
        if d > 0: bad.append((k, i, d))
R["traj12_max_abs_err"] = mx_t
R["traj12_bitwise_equal"] = bool(not bad)
R["traj12_mismatch_count"] = len(bad)
R["traj12_mismatch_sample"] = bad[:3]

ns._phase_bind(act, True, None)
pre_s = np.load("_preclean_run/_pre_snap.npz")
mx = max(float(np.max(np.abs(pre_s[k] - ns._attn_out[int(k)]))) for k in pre_s.files)
R["attn_out_max_abs_err"] = mx
R["attn_out_bitwise_equal"] = bool(all(np.array_equal(pre_s[k], ns._attn_out[int(k)])
                                       for k in pre_s.files))

# tie-heavy 反例: 新 lexsort key 写法 vs 源码 sorted(reverse=True) 语义
rng = np.random.default_rng(0)
det = []
for label, S in [("tie_heavy_int", rng.integers(0, 3, (7, N_HEADS, 5)).astype(float)),
                 ("all_equal", np.zeros((5, N_HEADS, 5))),
                 ("random", rng.normal(0, 1, (52, N_HEADS, 5)))]:
    na, nh, t = S.shape; k = min(ROUTE_K, t)
    got = np.lexsort((np.broadcast_to(-np.arange(t), S.shape), -S), axis=2)[:, :, :k]
    want = np.zeros((na, nh, k), dtype=int)
    for a in range(na):
        for h in range(nh):
            sc = [(float(S[a, h, j]), j) for j in range(t)]
            sc.sort(reverse=True)
            for j in range(k):
                want[a, h, j] = sc[j][1]
    det.append({"case": label, "eq": bool(np.array_equal(got, want)),
                "tie_frac": float((np.diff(np.sort(S, axis=2), axis=2) == 0).mean())})
R["lexsort_semantics"] = det
R["lexsort_all_eq"] = all(d["eq"] for d in det)
print(json.dumps(R, indent=1, ensure_ascii=False))
