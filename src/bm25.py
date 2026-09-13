"""
BM25 检索器 (bm25.py)
======================
标准全文检索算法（搜索引擎核心组件之一）。

为什么 BM25 能解决当前问题：
  · 当前问题：语义检索分不清"基本概念/应用特性/存储优化"（都是"KV缓存"开头）
  · BM25 的优势：用 IDF 给稀有词高权重 + 文档长度归一化
    → "基本概念"这种词在标题里出现，权重会被显著放大

公式：
  score(q,d) = Σ IDF(qi) · (f(qi,d)·(k1+1)) / (f(qi,d) + k1·(1-b+b·|d|/avgdl))
  IDF(qi) = ln(1 + (N - n(qi) + 0.5)/(n(qi) + 0.5))

特性：
  · 零依赖（纯 Python + 标准库）
  · 中文用 2-gram 分词（无需 jieba）
  · 索引构建 O(n)，查询 O(log n)
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional


def tokenize(text: str) -> List[str]:
    """中英文混合分词（英文按词，中文按 2-gram）"""
    if not text:
        return []
    text = text.lower()
    toks: List[str] = []
    # 英文/数字词
    for w in re.findall(r"[a-z][a-z0-9_+#.]*[a-z0-9]|[a-z]|\d+", text):
        toks.append(w)
    # 中文 2-gram（无需分词库）
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            toks.append(seg)
        else:
            for i in range(len(seg) - 1):
                toks.append(seg[i:i + 2])
            # 也加入整段（长词匹配）
            if len(seg) <= 6:
                toks.append(seg)
    return toks


class BM25:
    """BM25 检索索引"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Any] = []          # 原始文档
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: List[Counter] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.df: Dict[str, int] = {}       # 词 → 出现的文档数
        self.N: int = 0

    def add(self, doc: Any, text: str):
        """添加文档"""
        toks = tokenize(text)
        self.docs.append(doc)
        self.doc_tokens.append(toks)
        self.doc_freq.append(Counter(toks))
        self.doc_len.append(len(toks))
        for w in set(toks):
            self.df[w] = self.df.get(w, 0) + 1
        self.N = len(self.docs)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

    def add_batch(self, items: List[Tuple[Any, str]]):
        for doc, text in items:
            self.add(doc, text)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, idx: int) -> float:
        """单个文档的 BM25 分"""
        q_terms = tokenize(query)
        if not q_terms or self.avgdl == 0:
            return 0.0
        freq = self.doc_freq[idx]
        dl = self.doc_len[idx]
        s = 0.0
        for t in set(q_terms):
            f = freq.get(t, 0)
            if f == 0:
                continue
            idf = self._idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Any]]:
        """检索 top-k（返回 (分数, 文档)）"""
        if not self.docs:
            return []
        scored = [(self.score(query, i), i) for i in range(self.N)]
        scored.sort(key=lambda x: -x[0])
        return [(s, self.docs[i]) for s, i in scored[:top_k] if s > 0]

    def stats(self) -> Dict:
        return {"docs": self.N, "vocab": len(self.df), "avgdl": round(self.avgdl, 1)}


class HybridRetriever:
    """混合检索：语义向量 + BM25

    两者互补：
      · 语义：能匹配"换个说法的意思"（同义、改写）
      · BM25：能精确匹配关键词（术语、专有名词）

    融合方式：加权求和（可调），并做归一化。
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.6):
        """alpha = 语义权重, beta = BM25 权重"""
        self.alpha = alpha
        self.beta = beta
        self.bm25 = BM25()
        self.encoder = None

    def set_encoder(self, encoder):
        self.encoder = encoder

    def add(self, doc: dict, text: str):
        self.bm25.add(doc, text)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        if self.bm25.N == 0:
            return []

        # BM25 分
        bm_scores = {id(d): s for s, d in self.bm25.search(query, top_k=self.bm25.N)}
        bm_max = max(bm_scores.values()) if bm_scores else 1.0
        bm_max = bm_max or 1.0

        # 语义分
        sem_scores = {}
        if self.encoder is not None:
            q_emb = self.encoder.encode(query)
            for d in self.bm25.docs:
                emb = d.get("emb")
                if emb is None:
                    emb = self.encoder.encode(d.get("text", ""))
                    d["emb"] = emb
                sem_scores[id(d)] = float(q_emb @ emb)
        else:
            for d in self.bm25.docs:
                sem_scores[id(d)] = 0.0

        # 融合
        out = []
        for d in self.bm25.docs:
            bm = bm_scores.get(id(d), 0.0) / bm_max
            sm = sem_scores.get(id(d), 0.0)
            score = self.alpha * sm + self.beta * bm
            out.append((score, d))
        out.sort(key=lambda x: -x[0])
        return out[:top_k]

    def stats(self) -> Dict:
        return {**self.bm25.stats(),
                "alpha_semantic": self.alpha, "beta_bm25": self.beta}


# ---------------- 自测 ----------------
if __name__ == "__main__":
    print("=" * 72)
    print("  BM25 检索器自测")
    print("=" * 72)

    docs = [
        {"text": "KV缓存基本概念", "answer": "KV缓存是键值存储系统"},
        {"text": "KV缓存应用特性", "answer": "减少磁盘访问提升查询性能"},
        {"text": "KV缓存存储优化", "answer": "哈希分桶和动态调整"},
        {"text": "KV缓存核心机制", "answer": "通过哈希表实现键值对存储"},
        {"text": "Transformer位置编码", "answer": "表示序列中词的位置关系"},
        {"text": "MoE混合专家模型", "answer": "将参数分成多个专家网络"},
    ]
    b = BM25()
    for d in docs:
        b.add(d, d["text"] + " " + d["answer"])
    print(f"\n索引: {b.stats()}")

    print("\n查询测试：")
    for q in ["KV缓存是什么", "基本概念", "位置编码", "MoE"]:
        r = b.search(q, top_k=3)
        print(f"\n  问: {q}")
        for s, d in r:
            print(f"    {s:6.2f} | {d['text']}")

    # 对比：分词效果
    print("\n分词演示：")
    for t in ["KV缓存是什么", "Transformer位置编码"]:
        print(f"  {t!r} → {tokenize(t)[:12]}")
