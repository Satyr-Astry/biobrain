
import os, sys
sys.path.insert(0,"E:/BIONIC_AI/code")
import numpy as np
from cog_vec import WorkingMemory, WORKMEM_BETA

# 1) blend 输出范数必须是 1（严格单位向量）—— 上游 W_align 路径假设单位范数
wm = WorkingMemory()
probe = np.random.default_rng(0).normal(0,1,16)
norms=[]
for i in range(8):
    v = np.random.default_rng(i).normal(0,1,16)
    out = wm.blend(v)
    norms.append(float(np.linalg.norm(out)))
    wm.push(v)
print("blend 输出范数:", [round(x,9) for x in norms])
print("全部为 1.0 :", all(abs(x-1.0)<1e-9 for x in norms))

# 2) 退化输入（零向量 / 反向量抵消）
wm2 = WorkingMemory()
print("零向量输入 blend 范数:", float(np.linalg.norm(wm2.blend(np.zeros(16)))))
wm3 = WorkingMemory(); wm3.push(probe); wm3.push(-probe)
out = wm3.blend(probe)
print("历史互相抵消后 blend 范数:", round(float(np.linalg.norm(out)),9), "  (应=1，退化为纯当前)")

# 3) 关闭态必须【位级一致】
wm4 = WorkingMemory(enabled=False)
v = np.random.default_rng(7).normal(0,1,16)
print("BIO_WORKMEM=0 位级一致:", np.array_equal(wm4.blend(v), v))

# 4) 栈容量上限
wm5 = WorkingMemory()
for i in range(10): wm5.push(np.random.default_rng(i).normal(0,1,16))
print("栈深上限:", wm5.depth, "(应=4)")
