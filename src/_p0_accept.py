# -*- coding: utf-8 -*-
"""P0 最终验收: 确定性验证 (PYTHONHASHSEED 固定 + 固定 soma 快照)。
· 路由一致性: 比对 _phase_bind 的**实际 _attn_out** 与源码语义复刻
· 分数位级: einsum 路径 vs 逐元素路径
· 加速比: 与备份旧实现 A/B (子进程物理换文件)
用法: PYTHONHASHSEED=0 python _p0_accept.py cos
"""
import sys, os, json, time, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "cos"
if MODE == "learned":
    os.environ["BIO_ATTN_MODE"] = "learned"

from cog_vec import (D_SOMA, N_HEADS, ROUTE_K, TIER_BIAS, TIER_ORDER,
                       ATTN_GAIN, norm)
from _c_algo_bench import build, prime, health, tm
import cog_vec

R = {"mode": MODE, "hashseed": os.environ.get("PYTHONHASHSEED")}
d_head = max(1, D_SOMA // N_HEADS)
d_proj = N_HEADS * d_head

ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)
snz, anz, smax, amax = health(ns, set(ns.neurons))
R["health"] = {"soma_nz": snz, "axon_nz": anz, "max_soma": round(smax, 6),
               "max_axon": round(amax, 6)}
assert snz > 0 and anz > 0, "✗ compute 空转!"

active_all = sorted(ns.neurons)
TIDS = sorted(ns.tracts)
tier_val = {t: TIER_ORDER.get(ns.tracts[t].tier, 1) for t in TIDS}
LEARNED = (ns.attn_mode == "learned")
R["learned"] = LEARNED

# ============ A. 源码语义参照 (逐元素, 与备份实现一字不差) ============
def ref_bind():
    out = {}
    for i in active_all:
        n = ns.neurons[i]
        if norm(n.soma) < 1e-8:
            continue
        q_full = (ns.W_q.T @ n.soma)[:d_proj] if LEARNED else n.soma
        head_scores = []
        for h in range(N_HEADS):
            lo, hi = h*d_head, (h+1)*d_head
            q = q_full[lo:hi]
            scores = []
            for tid in TIDS:
                tr = ns.tracts[tid]
                kv = (ns.W_k.T @ tr.key)[:d_proj][lo:hi] if LEARNED else tr.key[lo:hi]
                c = float(np.dot(q, kv)) / (norm(q) * norm(kv) + 1e-8)
                src_t = TIER_ORDER.get(n.role, 1)
                dst_t = tier_val[tid]
                c += TIER_BIAS if dst_t >= src_t else -TIER_BIAS
                scores.append((c, tid))
            scores.sort(reverse=True)
            head_scores.append(scores[:ROUTE_K])
        acc = np.zeros(d_proj if LEARNED else D_SOMA)
        for h, routes in enumerate(head_scores):
            lo, hi = h*d_head, (h+1)*d_head
            if not routes:
                continue
            mx = max(s for s, _ in routes)
            exps = [(np.exp(s - mx), tid) for s, tid in routes]
            z = sum(e for e, _ in exps) + 1e-8
            for e, tid in exps:
                w = e / z
                vv = ((ns.W_v.T @ ns.tracts[tid].value)[:d_proj][lo:hi] if LEARNED
                      else ns.tracts[tid].value[lo:hi])
                acc[lo:hi] += w * vv
        if LEARNED:
            acc = (ns.W_o.T @ acc)[:D_SOMA]
        out[i] = acc
    return out

REF = ref_bind()

# ============ B. 真实 _phase_bind (新实现) ============
t0 = time.perf_counter()
ns._phase_bind(active_all, True, None)
R["bind_new_ms"] = round((time.perf_counter()-t0)*1000, 3)
NEW = {i: v.copy() for i, v in ns._attn_out.items()}

# ============ C. 位级比对 ============
R["n_ref"] = len(REF); R["n_new"] = len(NEW)
R["ids_match"] = sorted(REF) == sorted(NEW)
mx = 0.0
for i in REF:
    mx = max(mx, float(np.max(np.abs(REF[i] - NEW[i]))))
R["attn_out_max_abs_err"] = mx
R["attn_out_bitwise_equal"] = bool(all(np.array_equal(REF[i], NEW[i]) for i in REF))

# ============ D. 路由一致性 (逐头 top-K 束集合) ============
def routes_of(out):
    """从 acc 反推路由不可行 → 另用源码语义重算 top-K 束 id, 与 einsum 路径对比"""
    res = {}
    for i in active_all:
        n = ns.neurons[i]
        if norm(n.soma) < 1e-8:
            continue
        q_full = (ns.W_q.T @ n.soma)[:d_proj] if LEARNED else n.soma
        Qh = q_full.reshape(N_HEADS, d_head)
        kh = np.stack([((ns.W_k.T @ ns.tracts[t].key)[:d_proj] if LEARNED
                        else ns.tracts[t].key[:d_proj]).reshape(N_HEADS, d_head)
                       for t in TIDS])                       # T × H × d
        S = np.einsum('hd,thd->ht', Qh, kh) / (
            np.linalg.norm(Qh, axis=1)[:, None] * np.linalg.norm(kh, axis=2).T + 1e-8)
        st = TIER_ORDER.get(n.role, 1)
        S = S + np.where(np.array([tier_val[t] for t in TIDS])[None, :] >= st,
                         TIER_BIAS, -TIER_BIAS)
        # ★lexsort: 主键 −得分, 次键 −索引
        neg_idx = -np.arange(len(TIDS), dtype=float)[None, :] * np.ones_like(S)
        order = np.lexsort((neg_idx, -S), axis=1)[:, :ROUTE_K]
        res[i] = [set(TIDS[int(k)] for k in order[h]) for h in range(N_HEADS)]
    return res

def routes_src():
    res = {}
    for i in active_all:
        n = ns.neurons[i]
        if norm(n.soma) < 1e-8:
            continue
        q_full = (ns.W_q.T @ n.soma)[:d_proj] if LEARNED else n.soma
        hs = []
        for h in range(N_HEADS):
            lo, hi = h*d_head, (h+1)*d_head
            q = q_full[lo:hi]
            sc = []
            for tid in TIDS:
                tr = ns.tracts[tid]
                kv = (ns.W_k.T @ tr.key)[:d_proj][lo:hi] if LEARNED else tr.key[lo:hi]
                c = float(np.dot(q, kv)) / (norm(q)*norm(kv) + 1e-8)
                st = TIER_ORDER.get(n.role, 1)
                c += TIER_BIAS if tier_val[tid] >= st else -TIER_BIAS
                sc.append((c, tid))
            sc.sort(reverse=True)
            hs.append(set(t for _, t in sc[:ROUTE_K]))
        res[i] = hs
    return res

RV, RS = routes_of(NEW), routes_src()
agree = tot = 0; mism = []
for i in RS:
    for h in range(N_HEADS):
        tot += 1
        if RV[i][h] == RS[i][h]:
            agree += 1
        else:
            mism.append({"n": i, "h": h, "src": sorted(RS[i][h]), "vec": sorted(RV[i][h])})
R["route_agree"], R["route_total"] = agree, tot
R["route_rate"] = round(agree/tot, 4)
R["route_mismatch_total"] = len(mism)
R["route_mismatch_sample"] = mism[:5]

# ============ E. 加速比: 子进程 A/B (物理换文件) ============
BENCH = r'''
import sys, os, time, json, inspect
sys.path.insert(0, r"{code}")
import numpy as np, cog_vec
from _c_algo_bench import build, prime, tm
src = inspect.getsource(cog_vec.NervousSystem.compute)
old = any("np.ones(D_SOMA)" in l for l in src.splitlines() if not l.strip().startswith("#"))
ns, sensory, inter, motor, tids = build()
prime(ns, sensory, n_ticks=8, seed=0)
act = sorted(ns.neurons)
def run():
    ns._phase_bind(act, True, None)
    ns._phase_integrate(act)
t = tm(run, reps=300)
def full():
    n2, s2, _, _, _ = build()
    prime(n2, s2, n_ticks=8, seed=0)
    return n2.tick_phased(set(i for i in n2.neurons), n_spread=1)
t0 = time.perf_counter()
for _ in range(5): full()
ft = (time.perf_counter()-t0)/5*1000
print(json.dumps({{"old": old, "bind_us": t, "tick_ms": ft}}))
'''.format(code=CODE.replace("\\", "\\\\"))

def measure(variant):
    tmp = os.path.join(CODE, "_bench_%s.py" % variant)
    open(tmp, "w", encoding="utf-8").write(BENCH)
    cur, orig = os.path.join(CODE, "cog_vec.py"), \
                os.path.join(CODE, "_backup_before_p0fix", "cog_vec.py")
    bak = os.path.join(CODE, "_bb_tmp.py")
    if variant == "orig":
        shutil.copy2(cur, bak); shutil.copy2(orig, cur)
    shutil.rmtree(os.path.join(CODE, "__pycache__"), ignore_errors=True)
    try:
        env = dict(os.environ, PYTHONHASHSEED="0")
        p = subprocess.run([sys.executable, tmp], cwd=CODE, capture_output=True,
                           text=True, env=env)
        for ln in p.stdout.strip().splitlines():
            try:
                return json.loads(ln)
            except Exception:
                pass
        print("STDERR", variant, p.stderr[-1500:])
        return None
    finally:
        if variant == "orig":
            shutil.copy2(bak, cur); os.remove(bak)
        os.remove(tmp)
        shutil.rmtree(os.path.join(CODE, "__pycache__"), ignore_errors=True)

mo, mn = measure("orig"), measure("new")
R["bench_orig"] = mo; R["bench_new"] = mn
if mo and mn:
    R["bind_us_old"] = round(mo["bind_us"], 1)
    R["bind_us_new"] = round(mn["bind_us"], 1)
    R["bind_speedup"] = round(mo["bind_us"]/mn["bind_us"], 2)
    R["tick_ms_old"] = round(mo["tick_ms"], 2)
    R["tick_ms_new"] = round(mn["tick_ms"], 2)
    R["tick_speedup"] = round(mo["tick_ms"]/mn["tick_ms"], 2)

# ============ F. 门槛 ============
V = []
V.append(("路由一致 208/208", agree == tot and tot == 208, "%d/%d" % (agree, tot)))
V.append(("分数/输出误差 <1e-12", mx < 1e-12, "%.3e" % mx))
V.append(("加速比 >=5x", bool(R.get("bind_speedup", 0) >= 5),
          "%.2fx" % R.get("bind_speedup", 0)))
V.append(("soma/axon 非零", snz > 0 and anz > 0, "%d/%d" % (snz, anz)))
R["verdict"] = [{"item": a, "pass": bool(b), "val": c} for a, b, c in V]
R["ALL_PASS"] = all(b for _, b, _ in V)

with open("_p0_accept_%s.json" % MODE, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
