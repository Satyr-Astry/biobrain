"""最小复现 B：真实 CogVec 管线跨进程指纹（PYTHONHASHSEED 暗雷）

用法：
    PYTHONHASHSEED=1 python _repro_hashseed_pipe.py
    PYTHONHASHSEED=2 python _repro_hashseed_pipe.py

跑一段真实观察+巩固，输出指纹。不同 hashseed 必须一致。
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import CogVec  # noqa: E402


def fp(arr) -> str:
    a = np.asarray(arr, dtype=np.float64)
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def brain_fp(b: CogVec) -> str:
    """全部神经元 soma 指纹（决定分数的一切可观测量）"""
    vecs = [np.asarray(b.ns.neurons[i].soma, dtype=np.float64)
            for i in sorted(b.ns.neurons)]
    return fp(np.concatenate(vecs)) if vecs else "empty"


def main():
    print(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED', '<unset>')}")
    b = CogVec(seed=42)
    print("  初始 soma 指纹 =", brain_fp(b))
    print("  proj(vision)   指纹 =", fp(b.sensory_vision.proj))
    print("  proj(symbol)   指纹 =", fp(b.sensory_symbol.proj))

    # 两轮真实观察（图像+文本 → 共激活接地）
    rng = np.random.default_rng(7)
    for r in range(2):
        img = rng.normal(0, 1, 32)
        out = b.observe(image_vec=img, text=f"测试样本{r}")
        print(f"  观察{r}: {out}")
    print("  观察后 soma 指纹 =", brain_fp(b))

    # 学习 + 巩固 + 存盘序列化
    for i in range(3):
        b.teach({"pattern": f"p{i}", "value": i}, source="oracle")
    st = b.state()
    print("  teach/state OK, active =", st["active"], "ticks =", st["ticks"])

    # 自我模型（字符串 key 排序，验证确定性）
    who = b.self_model.who_am_i()
    print("  self_model =", who)

    sig = hashlib.md5((brain_fp(b) + str(sorted(st.get("tiers", {}).items()))
                       + repr(who)).encode()).hexdigest()
    print(f"\n=== 管线签名 SIGNATURE = {sig} ===")
    return sig


if __name__ == "__main__":
    main()
