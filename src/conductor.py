"""
仿生大脑 · 认知层指挥语言区 (conductor.py)
=============================================
阶段 B：让认知层**决定说什么**，不只是"说不说"。

三条指挥通道：
  ① 记忆召回   —— 相似经验注入上下文
  ② 激活语义   —— 当前激活模式反解成语义线索，作为"心智状态"
  ③ 置信度调制 —— 低自信时抑制 LLM 的自由发挥，强制保守

关键：LLM 的权重始终不动，它只是"运动皮层"——负责把大脑的意图
      变成流利的自然语言。真正的"决策"发生在神经组织里。

用法：
    python conductor.py --demo
    python conductor.py --ask "你好"
"""
from __future__ import annotations
import json
import re
import numpy as np
from typing import Optional, List, Dict

from tensor_brain import TensorBrain, Tier, ACTIVE_EPS
from encoder import get_encoder
from cortex import LLMCortex


class Conductor:
    """认知层 → 语言区 的指挥器"""

    def __init__(self, n_neurons: int = 262144, n_tracts: int = 16384,
                 use_llm: bool = True, seed: int = 42):
        self.brain = TensorBrain(n_neurons=n_neurons, n_tracts=n_tracts, seed=seed)
        self.encoder = get_encoder()
        self.llm = LLMCortex() if use_llm else None

        # 语义 → 神经元的投影
        rng = np.random.default_rng(seed)
        self.emb_dim = self.encoder.dim
        self.proj = rng.normal(0, 1.0 / (self.emb_dim ** 0.5),
                               (self.emb_dim, self.brain.n)).astype(np.float32)
        self.proj_pinv = np.linalg.pinv(self.proj.T)   # 反解用

        # 经验记忆
        self.memory: List[Dict] = []
        self.tick_total = 0
        self.last_activation_pattern: Optional[np.ndarray] = None

    # ============ 感觉端 ============
    def perceive(self, text: str, gain: float = 1.0) -> np.ndarray:
        """文字 → 稀疏激活"""
        emb = self.encoder.encode(text)
        acts = np.tanh(self.proj.T @ emb)
        k = max(1, int(self.brain.n * 0.04))
        idx = np.argsort(-np.abs(acts))[:k]
        vals = np.abs(acts[idx]) * gain
        self.brain.ignite_batch(idx.astype("int32"), vals.astype("float32"))
        return idx

    # ============ 思考 ============
    def think(self, steps: int = 12) -> dict:
        hist = []
        for _ in range(steps):
            act = self.brain.tick()
            self.brain.update_plasticity(act)
            self.brain.schedule()
            hist.append(float(self.brain.activity.mean()))
            self.tick_total += 1
        if len(hist) >= 6:
            tail = np.diff(hist[-6:])
            convergence = float(1.0 / (1.0 + np.abs(tail).mean() * 100))
        else:
            convergence = 0.0
        self.last_activation_pattern = self.brain.activity.copy()
        return {
            "convergence": round(convergence, 3),
            "active": int((self.brain.activity > ACTIVE_EPS).sum()),
            "ticks": self.tick_total,
        }

    # ============ ① 记忆召回 ============
    def recall(self, text: str, top_k: int = 3) -> List[Dict]:
        """从经验里召回最相似的（这是大脑"想起"的过程）

        ★语义检索：用编码器把"问题"和"记忆文本"都编码成向量再比，
        比纯字符匹配准确得多（字符匹配会把"KV缓存"匹配到"应用特性"）。
        """
        if not self.memory:
            return []
        # ★用 BM25 + 语义的混合检索（BM25 解决关键词精确匹配，
        #   语义解决"换个说法的意思"）
        from bm25 import HybridRetriever
        r = self._get_retriever()
        if r is None:
            # 兜底：纯语义
            emb = self.encoder.encode(text)
            scored = [(float(emb @ m.get("emb", emb)), m) for m in self.memory]
        else:
            scored = r.search(text, top_k=top_k)
            # 返回格式适配
            out = [{"score": round(s, 3), "text": m["text"], "answer": m["answer"],
                    "source": m["source"]} for s, m in scored]
            return out
        scored.sort(key=lambda x: -x[0])
        return [{"score": round(s, 3), "text": m["text"], "answer": m["answer"], "source": m["source"]}
                for s, m in scored[:top_k]]

    def _get_retriever(self):
        """懒构建 BM25 混合检索器（记忆变化时重建）"""
        try:
            from bm25 import HybridRetriever
        except Exception:
            return None
        n = len(self.memory)
        if getattr(self, "_retriever_n", -1) != n:
            r = HybridRetriever(alpha=0.4, beta=0.6)
            r.set_encoder(self.encoder)
            for m in self.memory:
                r.add(m, m["text"] + " " + m.get("answer", "")[:300])
            self._retriever = r
            self._retriever_n = n
        return getattr(self, "_retriever", None)

    def confidence(self, text: str) -> float:
        """置信度 = 最相似经验的语义相似度

        ★踩坑：从 JSON 载入的记忆没有 emb 字段，必须懒生成，
        否则 KeyError 被吞后恒返回 0（表现为"大脑永远不确定"）。
        """
        if not self.memory:
            return 0.15
        emb = self.encoder.encode(text)
        sims = []
        for m in self.memory:
            me = m.get("emb")
            if me is None:
                me = self.encoder.encode(m["text"])
                m["emb"] = me
            sims.append(float(emb @ me))
        return float(np.clip(max(sims), 0.0, 1.0)) if sims else 0.15

    # ============ ② 激活语义反解 ============
    def mental_state(self) -> Dict:
        """把当前激活模式反解成"心智状态"（用于告诉 LLM 我在想什么）"""
        if self.last_activation_pattern is None:
            return {"focus_neurons": [], "spread": 0.0, "mean_activation": 0.0}
        act = self.last_activation_pattern          # (n,) 全量激活
        # 最活跃的神经元（"注意力焦点"）
        top = np.argsort(-act)[:8]
        # ★反解必须用**全量**激活（pinv 维度 = n_neurons），top 只用于报告焦点
        emb_back = self.proj_pinv @ np.arctanh(np.clip(act, -0.999, 0.999))
        n = np.linalg.norm(emb_back)
        if n > 1e-8:
            emb_back = emb_back / n
        # 激活分散度（越分散=越困惑）
        spread = float(act.std() / (act.mean() + 1e-8))
        return {
            "focus_neurons": [int(i) for i in top],
            "semantic_direction": emb_back.tolist(),
            "spread": round(spread, 3),
            "mean_activation": round(float(act.mean()), 4),
        }

    # ============ ③ 构建指挥 prompt ============
    def _build_prompt(self, text: str, recalled: List[Dict], state: Dict, conf: float) -> tuple:
        """把大脑状态翻译成给 LLM 的指令（这就是"指挥"）"""
        lines = []

        # 心智状态
        lines.append(f"【心智状态】当前激活均值={state['mean_activation']}，"
                     f"分散度={state['spread']}（越高越困惑）")

        # 记忆召回
        if recalled and recalled[0]["score"] > 0.5:
            lines.append("【我记得的相关经验】")
            for r in recalled:
                if r["score"] > 0.5:
                    lines.append(f"  · 「{r['text']}」→ 当时回答：{r['answer']}（相关度{r['score']}）")

        # 置信度指令（关键：低自信就压制自由发挥）
        if conf < 0.4:
            lines.append("【指令】这件事你并不熟悉，请**明确说明你不确定**，不要编造。")
        elif conf < 0.65:
            lines.append("【指令】有点印象但不确定，回答时请保持谨慎，说明你的不确定。")
        else:
            lines.append("【指令】你对这件事有把握，请直接、简短地回答。")

        # ★强化表述要求（R1 容易把思考过程写进正文）
        lines.append("【表述要求】直接给出答案，不要写'嗯''我需要''让我想'这类思考过程；"
                     "技术术语保持原样；控制在 150 字内。")

        ctx = "\n".join(lines)
        return text, ctx

    # ============ 完整闭环 ============
    def respond(self, text: str, steps: int = 12) -> dict:
        # 1) 感知 + 思考
        self.perceive(text)
        tinfo = self.think(steps)

        # 2) 三条通道
        recalled = self.recall(text)
        conf = self.confidence(text)
        state = self.mental_state()

        # 3) 闸门（★v3校准：抑制后收敛度被压缩，不再作为可靠信号）
        #    只用置信度（经验相似度），这是"我知不知道自己知道"的核心
        n_mem = len(self.memory)
        should_speak = True if n_mem == 0 else (conf > 0.6)

        # 4) 指挥 LLM
        out = None
        if should_speak and self.llm and self.llm.available:
            q, ctx = self._build_prompt(text, recalled, state, conf)
            out = self.llm.generate(q, system=ctx, max_tokens=150)   # ★从250降到150
            # ★清理 R1 思维链标签（含未闭合情况）
            if out:
                try:
                    from cortex import _clean_think
                    out = _clean_think(out) or None
                except Exception:
                    import re as _re
                    out = _re.sub(r"</?think(?:ing)?>", "", out).strip() or None

        return {
            "input": text,
            "convergence": tinfo["convergence"],
            "confidence": round(conf, 3),
            "should_speak": bool(should_speak),
            "recalled": recalled,
            "mental_state": {"spread": state["spread"], "mean": state["mean_activation"]},
            "output": out,
            "brain": self.brain.stats(),
        }

    # ============ 教学 / 睡眠 ============
    # ★可信来源白名单（防自噬：绝不允许"自己说自己对"的知识入库）
    #   · 内部来源：oracle(权威)/user_action(用户行为)/outcome(环境结果)
    #   · 外部来源：web_wiki(百科)/web_api(官方API)  —— 可信度高
    #   · 弱来源：  web_search/web_page —— 允许但标记，用于候选
    TRUSTED_SOURCES = {
        "oracle", "user_action", "outcome",
        "web_wiki", "web_api",
        "web_search", "web_page",
        "ai_distill",     # ★AI 蒸馏产物（已经过 LLM 评判+提炼，质量可控）
        "ai_llm",         # ★直接向 LLM 学习的产物（含自检）
    }

    def teach(self, text: str, answer: str, source: str = "oracle") -> bool:
        if source not in self.TRUSTED_SOURCES:
            return False
        self.memory.append({
            "text": text, "answer": answer, "source": source,
            "emb": self.encoder.encode(text),   # ★预存 emb
        })
        self._retriever_n = -1                  # 强制重建检索索引
        return True

    def sleep(self) -> dict:
        merged = self.brain.consolidate()
        imp = self.brain.importance
        if np.linalg.norm(imp) > 0:
            thr = np.percentile(imp, 5)
            self.brain.importance[np.where(imp < thr)[0]] *= 0.5
        return {"merged": merged.get("merged", 0), "memory": len(self.memory)}

    def stats(self) -> dict:
        s = self.brain.stats()
        s["memory"] = len(self.memory)
        return s


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="认知层指挥语言区")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ask", type=str)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--neurons", type=int, default=262144)
    args = ap.parse_args()

    print("=" * 66)
    print("  仿生大脑 · 认知层指挥语言区（阶段 B）")
    print("=" * 66)
    c = Conductor(n_neurons=args.neurons, use_llm=not args.no_llm)
    print(f"  认知层: {args.neurons} 神经元 | 语义维度: {c.emb_dim}")
    print(f"  语言区: {'✓ ' + c.llm.model if (c.llm and c.llm.available) else '✗ 不可用'}")
    print()

    if args.ask:
        r = c.respond(args.ask)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.demo:
        print("--- 1. 教两条经验 ---")
        c.teach("主人喜欢什么音乐", "主人喜欢周杰伦的《花海》和 City Pop。", "oracle")
        c.teach("小悠是谁", "小悠是主人的专属猫娘女仆。", "oracle")
        print(f"  记忆: {len(c.memory)} 条\n")

        print("--- 2. 问已知的（大脑应该召回记忆指挥 LLM）---")
        r = c.respond("主人喜欢什么音乐")
        print(f"  自信={r['confidence']} 该说={r['should_speak']}")
        print(f"  召回: {[x['text'] for x in r['recalled']]}")
        print(f"  心智: {r['mental_state']}")
        print(f"  ★输出: {r['output']}\n")

        print("--- 3. 问没教过的（应该谨慎）---")
        r = c.respond("请证明黎曼猜想")
        print(f"  自信={r['confidence']} 该说={r['should_speak']}")
        print(f"  ★输出: {r['output']}\n")

        print("--- 4. 大脑睡一觉 ---")
        print(f"  {c.sleep()}")
        print(f"  状态: {json.dumps(c.stats(), ensure_ascii=False)}")
