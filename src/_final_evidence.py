# -*- coding: utf-8 -*-
"""交付证据汇总: cos/learned 双模式 + 清理位级等价 + lexsort 语义"""
import sys, os, json, subprocess
CODE = os.path.dirname(os.path.abspath(__file__))
out = {}

# 1. pytest
p = subprocess.run([sys.executable, "-m", "pytest", "test_cog_vec.py", "-q"],
                   cwd=CODE, capture_output=True, text=True,
                   env=dict(os.environ, PYTHONHASHSEED="0"))
out["pytest"] = p.stdout.strip().splitlines()[-1]

# 2. 双模式 accept json
for mode in ("cos", "learned"):
    f = os.path.join(CODE, "_p0_accept_%s.json" % mode)
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        out[mode] = {k: d.get(k) for k in
                     ("mode", "hashseed", "learned", "route_agree", "route_total",
                      "route_rate", "attn_out_max_abs_err", "attn_out_bitwise_equal",
                      "bind_us_old", "bind_us_new", "bind_speedup", "tick_ms_old",
                      "tick_ms_new", "tick_speedup", "ALL_PASS", "health")}
print(json.dumps(out, indent=1, ensure_ascii=False))
