# -*- coding: utf-8 -*-
"""用清理前的 cog_vec.py 生成快照 (物理隔离: _preclean_pkg/)"""
import sys, os, json, shutil
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "_preclean_pkg"))   # ★优先命中旧版
sys.path.insert(0, CODE)
import numpy as np
import cog_vec
assert "preclean" in cog_vec.__file__, cog_vec.__file__
from cog_vec import D_SOMA, N_HEADS, ROUTE_K
from _c_algo_bench import build, prime
ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)
act = sorted(ns.neurons)
traj = []
for _ in range(12):
    ns.tick_phased(set(act), n_spread=1)
    traj.append({i: ns.neurons[i].soma.copy() for i in ns.neurons})
np.savez("_pre_clean_traj.npz", **{f"t{k}_{i}": v for k, d in enumerate(traj) for i, v in d.items()})
ns._phase_bind(act, True, None)
np.savez("_pre_clean_snap.npz", **{str(i): ns._attn_out[i] for i in ns._attn_out})
print(json.dumps({"src": cog_vec.__file__, "n_snap": len(ns._attn_out),
                  "quick_hash": float(sum(float(v.sum()) for v in ns._attn_out.values()))}))
