"""
ALG-19 记忆召回（recall.py）
============================
治 P-ARCH-16：**记忆库只写不读**。

背景（实测）
------------
  · `ExperienceBuffer` 确实在存经验（`server.think` 每次思考 store 一条）
  · 但 `think()` 里**没有任何检索逻辑** → 存进去的经验永远不被使用
  · 已有的 `HybridRetriever`（BM25+语义）只在 `conductor.recall` 里用，
    而 conductor **不在主线路**（CogVec）上

两条检索通路（各自独立，互不依赖）
----------------------------------
① **符号通路 `_sym_trace`（主）** —— ALG-19 的硬核心
   经验里存的 `thought_trace` 是**活跃神经元快照（神经元 id）**。
   把当前活跃集与经验轨迹按**神经元 id 重合度**比对：
       sim_sym = |A ∩ T| / min(|A|, |T|)
   这是**集合论上的客观匹配**，不经过任何编码器、不看任何自评标量 →
   必然满足 NORM-6（自评禁用）。
   ★这是"想起那条经验"的物理含义：想起 = 当时的神经元群被重新激活。

② **语义通路 `_sem`（辅）** —— 让"换个说法"也能召回
   用 `encoder.encode` 把输入文本与经验轨迹文本都编码成向量比余弦。
   ★实测：`encoder.encode("n7 n11 ...")` 命中 n-gram 兜底 → 任意两串神经元 id
     的余弦恒为 1.0（实测 t1@t2 = 0.971~1.0，随机 query@t1 = 0.44~0.51）→
     该通路**没有区分度**，故它只能"加分"，不能单独决定命中。

③ **BM25/关键词通路 `_lex`（辅）**
   用 `bm25.tokenize` 对"输入文本 ↔ 轨迹 token 串"做词面匹配。
   ★实测：输入是"猫"时，轨迹串里没有"猫"这个词 → 0 分，符合预期
     （神经元 id 与中文词没有字面关系）。故它同样只作加分项。

判据（★这就是"有没有相关记忆"的操作化定义）
--------------------------------------------
    relevant  ⇔  sim_sym ≥ SYM_EPS      （符号命中，硬条件）
               或 sim_sem ≥ SEM_EPS     （语义命中，软条件）
每 tick 召回一次，把命中经验的轨迹重新点火。
*未命中*时**不注入**（见 `inject`）→ 有无相关记忆两条分支必然可分。

常量（架构书 §4.1 ALG-19）
--------------------------
  RECALL_K      = 3      每次召回条数
  RECALL_WEIGHT = 0.3    召回注入强度（活动度）
  SYM_EPS       = 0.30   神经元集合重合度阈值
  SEM_EPS       = 0.60   语义余弦阈值
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from bm25 import tokenize as _bm25_tokenize
except Exception:                                    # pragma: no cover
    _bm25_tokenize = None

# ---------------- 常量 ----------------
RECALL_K = 3                 # 每次召回条数
RECALL_WEIGHT = 0.3          # 召回注入强度
SYM_EPS = 0.30               # 神经元集合重合度阈值（符号命中）
SEM_EPS = 0.60               # 语义余弦阈值（软命中，见模块 docstring ②）


def recall_enabled() -> bool:
    """回退开关：`BIO_RECALL=0` → 关闭召回（退回 ALG-19 之前的行为）"""
    return os.environ.get("BIO_RECALL", "1") != "0"


class RecallEngine:
    """经验召回引擎（神经元集合重合 + 语义 + 词面）→ 轨迹再点火。

    ★它不自己存经验：唯一事实源永远是 `ExperienceBuffer`。
    """

    def __init__(self, buffer, encoder, enabled: Optional[bool] = None,
                 k: int = RECALL_K, weight: float = RECALL_WEIGHT):
        self.buffer = buffer
        self.encoder = encoder
        self.enabled = recall_enabled() if enabled is None else bool(enabled)
        self.k = int(k)
        self.weight = float(weight)
        self.last: Dict[str, Any] = {}      # 可观测性：最近一次召回
        self.n_calls = 0
        self.n_hits = 0

    # ================= 检索 =================
    @staticmethod
    def _sym_trace(active, trace) -> float:
        """符号相似度：当前活跃集与经验轨迹的**神经元 id 重合度**。

        sim = |A ∩ T| / min(|A|, |T|) ∈ [0,1]

        ★为什么用 min 归一化而不是并集归一化：`think` 存的是
          `list(self.active)`（set → list，无序），经验轨迹可能被截断到 8 个，
          故只保证"**小集合被覆盖**"这一条可判据。
        ★NORM-6：这里只用神经元编号这一定性事实，不用任何自评标量。
        """
        T = {int(i) for i in (trace or [])}
        if not T:
            return 0.0
        A = {int(i) for i in active}
        inter = A & T
        return len(inter) / float(min(len(A), len(T))) if A else 0.0

    @staticmethod
    def _lex(text: str, trace) -> float:
        """词面相似度：输入文本 token 与轨迹 token 串的 Jaccard。"""
        if _bm25_tokenize is None or not text:
            return 0.0
        a = set(_bm25_tokenize(text))
        b = set(_bm25_tokenize(" ".join(f"n{i}" for i in (trace or []))))
        if not a or not b:
            return 0.0
        return len(a & b) / float(len(a | b))

    def _sem(self, q_emb: Optional[np.ndarray], trace, cache: Dict) -> float:
        """语义相似度：输入向量 ↔ 轨迹文本向量（余弦，已做 L2 归一化）。

        ★注意：`encoder.encode("n7 n11 ...")` 会命中 n-gram 兜底且**恒等于 1.0**
          （实测），故该分值需与 `SEM_EPS=0.60` 的高阈值配合，
          否则会变成"永远命中"。它是加分项，不是判据。
        """
        if q_emb is None or not trace:
            return 0.0
        key = tuple(int(i) for i in trace)
        v = cache.get(key)
        if v is None:
            v = self.encoder.encode(" ".join(f"n{i}" for i in key))
            n = float(np.linalg.norm(v))
            v = (v / n) if n > 1e-8 else None
            cache[key] = v
        if v is None:
            return 0.0
        return float(np.clip(q_emb @ v, 0.0, 1.0))

    def score(self, text: str, active, e, q_emb, cache) -> Dict[str, float]:
        """对单条经验打分（三个通道各自的客观分）。"""
        sym = self._sym_trace(active, e.thought_trace)
        return {"sym": sym,
                "sem": self._sem(q_emb, e.thought_trace, cache),
                "lex": self._lex(text, e.thought_trace)}

    def is_relevant(self, s: Dict[str, float]) -> bool:
        """相关判据：符号硬命中 或 语义软命中。"""
        return s["sym"] >= SYM_EPS or s["sem"] >= SEM_EPS

    def recall(self, text: str, active, top_k: Optional[int] = None
               ) -> List[Tuple[Dict[str, float], Any]]:
        """召回与当前状态相关的经验（按综合分降序，只返回相关者）。"""
        if not self.enabled:
            return []
        items = list(getattr(self.buffer, "items", []) or [])
        if not items:
            self.last = {"n": 0, "reasons": [], "reason": "empty_buffer"}
            return []
        q_emb = None
        if text:
            v = self.encoder.encode(text)
            n = float(np.linalg.norm(v))
            q_emb = (v / n) if n > 1e-8 else None
        cache: Dict = {}
        out: List[Tuple[Dict[str, float], Any]] = []
        for e in items:
            s = self.score(text, active, e, q_emb, cache)
            if self.is_relevant(s):
                # 综合分：符号 0.5 + 语义 0.2 + 词面 0.3（三者均为客观量）
                s = dict(s)
                s["total"] = 0.5 * s["sym"] + 0.2 * s["sem"] + 0.3 * s["lex"]
                out.append((s, e))
        out.sort(key=lambda x: (-x[0]["total"], -float(getattr(x[1], "surprise", 0.0))))
        out = out[:(int(top_k) if top_k else self.k)]
        self.last = {"n": len(out), "top": out[0][0]["total"] if out else 0.0,
                     "ids": [e.id for _, e in out],
                     "reasons": sorted({k for s, _ in out for k in ("sym", "sem", "lex")
                                        if s[k] >= (SYM_EPS if k == "sym" else SEM_EPS)})}
        return out

    # ================= 注入 =================
    def inject(self, ns, recalled, baseline_trace=None) -> Dict[str, Any]:
        """把召回经验的轨迹**重新点火**（写 activity + pending_ignitions）。

        ★这是"召回影响激活"的**唯一落点**。调用方必须在 `run_ticks` 之前调用，
          注入才会进入后续 tick 的扩散 / 注意力 / 读出。

        ★关键设计：**没有相关记忆时不注入**。
          若两条分支都注入等量活性，差异只会被后续扩散抹平成噪声
          （这就是"召回只写不读"的翻版：注入了但不改变输出）。
          所以：
            · 有相关记忆 → 注入轨迹神经元（每 tick 一次，新异优先）
            · 无相关记忆 → 不注入（返回 n_ignited=0）
          这样"有无相关记忆"直接决定了一大批神经元的活性来源，
          输出差异 Δ 才是**因果产生**的，不是调参凑出来的。
        """
        info: Dict[str, Any] = {"n_recalled": len(recalled or []), "n_ignited": 0,
                                "traces": [], "mass": 0.0, "mode": "none"}
        if not self.enabled:
            info["disabled"] = True
            return info
        if not recalled:
            self.last = {**self.last, "inject": info}
            return info

        neurons = ns.neurons
        dose = self.weight * float(len(recalled))      # ★剂量与召回条数挂钩
        touched: List[int] = []
        seen = set()
        for _, e in recalled:
            tr = [int(i) for i in (e.thought_trace or [])[:8]]
            info["traces"].append(tr)
            for i in tr:
                if i in neurons and i not in seen:
                    seen.add(i)
                    touched.append(i)
        for i in touched:
            n = neurons[i]
            # ★新异优先门控：已在活跃集的神经元不再被"炒热"，
            #   否则注入会被活跃集本身吸收，有/无记忆差异趋零。
            if float(n.activity) >= SYM_ACTIVE_GATE:
                continue
            info["mass"] += self._add(ns, i, dose)
        info["n_ignited"] = len(touched)
        info["mode"] = "recall"
        self.last = {**self.last, "inject": info}
        return info

    @staticmethod
    def _add(ns, nid: int, amount: float) -> float:
        """给单个神经元注入活性（activity + pending_ignitions 各一份）。"""
        n = ns.neurons.get(nid)
        if n is None:
            return 0.0
        amt = float(amount)
        n.activity = min(1.0, max(0.0, float(n.activity) + amt))
        ns.pending_ignitions.append((nid, amt))        # ★经 ALG-3 提交路径继续传播
        return amt


# 活跃集门控：activity ≥ 此值的神经元视为"已在想"，不再被召回炒热
SYM_ACTIVE_GATE = 0.85
