
import os, sys
sys.path.insert(0, "E:/BIONIC_AI/code")
import numpy as np
from server import CogVec

PROBE="猫"
HISTS={"H1":["猫","猫","猫"],"H2":["狗","狗","狗"],
       "H3":["量子纠缠","薛定谔方程","波函数"]}

def read(brain):
    ids=sorted(set(brain.symbol_ids)|set(brain.motor_ids))
    return np.array([brain.ns.neurons[i].activity for i in ids])

res={}
for beta in [1.5, 3.0, 5.0, 8.0]:
    os.environ["_BETA"]=str(beta)
    import importlib, cog_vec
    importlib.reload(cog_vec)
    cog_vec.WORKMEM_BETA=beta
    reads={}
    for n,h in HISTS.items():
        b=CogVec(seed=42)
        # 手动注入 beta（绕过 import 顺序）
        b.workmem.beta=beta
        for t in h: b.think(t)
        b.think(PROBE); reads[n]=read(b)
    ds=[float(np.max(np.abs(reads[a]-reads[c]))) for a,c in
        [("H1","H2"),("H1","H3"),("H2","H3")]]
    res[beta]=(float(np.mean(ds)), float(np.max(ds)))
    print(f"beta={beta:4.1f}  mean_d={res[beta][0]:.6f}  max_d={res[beta][1]:.6f}  ratio_vs_base={res[beta][0]/0.000491:.1f}x")
