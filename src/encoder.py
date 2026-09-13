"""
仿生大脑 · 语义编码器（可插拔 + 智能降级）
=============================================
把文字变成"感觉神经元的激活模式"。

三档实现，自动选择最优可用的一档：
  1. SentenceTransformer —— 真语义嵌入（若 sentence-transformers 可用）
  2. Transformers       —— 真语义嵌入（若 transformers + 本地模型可用）
  3. NGramHash          —— 增强字符 n-gram 哈希（无依赖兜底，比单字符哈希强很多）

设计要点：
  · 编码器必须把任意文本映射到固定维度 (D_SOMA)
  · 语义相近的文本 → 激活模式相近（1/2 档天生满足；3 档靠 n-gram 重叠）
  · 输出归一化到 [-1,1]，可直接当点火强度用

用法：
    enc = get_encoder()          # 自动选最优
    vec = enc.encode("你好")      # → np.ndarray(D_SOMA)
    print(enc.kind, enc.info)    # 看用的哪一档
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np

D_SOMA = 128         # ★v2：原16维太挤（丢掉语义分辨率），提到128；bge-small 原生512维
_CACHE = {}


# ==================================================================
# 档 3：增强 n-gram 哈希（无依赖兜底）
# ==================================================================
class NGramHashEncoder:
    """字符 n-gram + 哈希投影。

    比单字符哈希强在：
      · 捕捉字符组合（"猫" 和 "猫咪" 共享 "猫" 这个 unigram）
      · 多粒度（1/2/3-gram）→ 更稳的相似度
      · 用 blake2b 哈希替代 ord 取模，分布更均匀、抗碰撞
    """
    kind = "ngram-hash"

    def __init__(self, dim: int = D_SOMA):
        self.dim = dim
        self.info = f"n-gram(1-3) + blake2b → {dim}d"

    def _ngrams(self, text: str):
        t = text.strip()
        for n in (1, 2, 3):
            for i in range(max(0, len(t) - n + 1)):
                yield t[i:i + n]

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        if not text:
            return vec
        for g in self._ngrams(text):
            h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            # 短 gram 权重低（区分度小），长 gram 权重高
            w = len(g) / 3.0
            vec[idx] += sign * w
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


# ==================================================================
# 档 1/2：真语义嵌入
# ==================================================================
class SemanticEncoder:
    """SentenceTransformer 或 Transformers 的语义嵌入。

    语义相近 → 向量相近（余弦相似度高），这是哈希档做不到的。
    """
    kind = "semantic"

    # 候选小模型（按体积从小到大；中文/多语言优先）
    MODELS = [
        "BAAI/bge-small-zh-v1.5",          # 中文优化，~95MB
        "sentence-transformers/all-MiniLM-L6-v2",  # 多语言通用，~90MB
        "shibing624/text2vec-base-chinese",
    ]

    def __init__(self, dim: int = D_SOMA):
        self.dim = dim
        self.backend = None
        self.model = None
        self.info = "未加载"
        self._try_load()

    def _try_load(self):
        # 走国内镜像（HF 直连被墙）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # --- 尝试 sentence-transformers ---
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            for m in self.MODELS:
                try:
                    self.model = SentenceTransformer(m, cache_folder=str(_CACHE_DIR))
                    self.backend = "sentence-transformers"
                    self.info = f"ST/{m}"
                    return
                except Exception:
                    continue
        except ImportError:
            pass

        # --- 尝试 transformers ---
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            for m in self.MODELS:
                try:
                    tok = AutoTokenizer.from_pretrained(m, cache_dir=str(_CACHE_DIR))
                    mod = AutoModel.from_pretrained(m, cache_dir=str(_CACHE_DIR))
                    mod.eval()
                    self.model = (tok, mod)
                    self.backend = "transformers"
                    self.info = f"HF/{m}"
                    return
                except Exception:
                    continue
        except ImportError:
            pass

        self.backend = None

    def available(self) -> bool:
        return self.model is not None

    def encode(self, text: str) -> np.ndarray:
        if not text or self.model is None:
            return np.zeros(self.dim)
        try:
            if self.backend == "sentence-transformers":
                v = self.model.encode(text, normalize_embeddings=True)
                v = np.asarray(v, dtype=float)
            else:
                tok, mod = self.model
                import torch
                with torch.no_grad():
                    b = tok(text, return_tensors="pt", truncation=True, max_length=128)
                    out = mod(**b)
                    # mean pooling
                    mask = b["attention_mask"].unsqueeze(-1).float()
                    v = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                    v = v[0].cpu().numpy().astype(float)
            return self._project(v)
        except Exception:
            return np.zeros(self.dim)

    def _project(self, v: np.ndarray) -> np.ndarray:
        """把嵌入投影到感觉神经元维度（保持语义距离）"""
        if v.shape[0] == self.dim:
            out = v
        else:
            # 用固定随机投影（Johnson-Lindenstrauss）
            rng = np.random.default_rng(20260913)
            proj = _CACHE.setdefault(
                "proj", rng.normal(0, 1.0 / np.sqrt(v.shape[0]), (v.shape[0], self.dim))
            )
            out = v @ proj
        n = np.linalg.norm(out)
        return out / n if n > 0 else out


_CACHE_DIR = None


def _init_cache_dir():
    global _CACHE_DIR
    if _CACHE_DIR is None:
        from pathlib import Path
        _CACHE_DIR = Path(__file__).resolve().parent / "models"
        _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR


# ==================================================================
# 工厂：自动选最优
# ==================================================================
def get_encoder(force: Optional[str] = None, dim: int = D_SOMA):
    """返回最优可用编码器。force: 'semantic' | 'ngram' | None"""
    _init_cache_dir()
    if force == "ngram":
        return NGramHashEncoder(dim)
    if force == "semantic":
        enc = SemanticEncoder(dim)
        return enc if enc.available() else NGramHashEncoder(dim)

    key = f"enc-{dim}"
    if key in _CACHE:
        return _CACHE[key]

    sem = SemanticEncoder(dim)
    enc = sem if sem.available() else NGramHashEncoder(dim)
    _CACHE[key] = enc
    return enc


if __name__ == "__main__":
    print("=== 编码器自测 ===")
    enc = get_encoder()
    print(f"选用: {enc.kind}  |  {enc.info}")

    tests = ["猫", "猫咪", "狗", "hello", "你好", "你好啊"]
    vs = {t: enc.encode(t) for t in tests}

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print("\n余弦相似度（越接近1越相似）:")
    print(f"  猫 ↔ 猫咪 : {cos(vs['猫'], vs['猫咪']):.3f}   ← 语义相近应较高")
    print(f"  猫 ↔ 狗   : {cos(vs['猫'], vs['狗']):.3f}")
    print(f"  你 ↔ 你好 : {cos(vs['你好'], vs['你好啊']):.3f}   ← 应较高")
    print(f"  猫 ↔ hello: {cos(vs['猫'], vs['hello']):.3f}   ← 无关应较低")
