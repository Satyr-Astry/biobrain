
import os, sys
sys.path.insert(0, "E:/BIONIC_AI/code")
import numpy as np
from server import CogVec
import cog_vec as bb

PROBE="猫"; N_PROBE=3
HISTS={"H1":["猫","猫","猫"],"H2":["狗","狗","狗"],
       "H3":["量子纠缠","薛定谔方程","波函数"]}

def read(brain):
    ids=sorted(set(brain.symbol_ids)|set(brain.motor_ids))
    return np.array([brain.ns.neurons[i].activity for i in ids])

def run(h,beta,push_mode,nprobe):
    b=CogVec(seed=42); b.workmem.beta=beta; b.workmem.push_mode=push_mode
    for t in h: b.think(t)
    for _ in range(nprobe): b.think(PROBE)
    return read(b)

for push_mode in ("raw","blend"):
  print(f"=== push_mode={push_mode} ===")
  for beta in [1.5, 3.0, 5.0]:
    reads={n:run(h,beta,push_mode,3) for n,h in HISTS.items()}
    ds=[float(np.max(np.abs(reads[a]-reads[c]))) for a,c in
        [("H1","H2"),("H1","H3"),("H2","H3")]]
    print(f"  beta={beta:4.1f} N_PROBE=3  mean_d={np.mean(ds):.6f}  max_d={np.max(ds):.6f}")
  for beta in [3.0]:
    reads={n:run(h,beta,push_mode,1) for n,h in HISTS.items()}
    ds=[float(np.max(np.abs(reads[a]-reads[c]))) for a,c in
        [("H1","H2"),("H1","H3"),("H2","H3")]]
    print(f"  beta={beta:4.1f} N_PROBE=1  mean_d={np.mean(ds):.6f}  max_d={np.max(ds):.6f}")
