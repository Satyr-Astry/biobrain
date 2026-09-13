"""
仿生大脑 · LLM 蒸馏器 (llm_distiller.py)
==========================================
直接向本地 LLM 学习 —— 不经互联网，又快又准。

对比网络学习：
  ✗ 网络：慢（15s/次）、噪音多（词典释义）、被墙、反爬
  ✓ LLM：快（2-5s）、内容准、可追问、可深挖

流程：
  1. 【提问】让 LLM 就主题生成结构化知识（分点、含机制）
  2. 【追问】针对薄弱点继续问（多轮深挖）
  3. 【自检】让 LLM 检查自己的输出有没有错误/遗漏
  4. 【入库】存进神经组织（来源 ai_llm）

关键设计：
  · 要求 LLM 输出结构化、有信息量的内容（不是泛泛而谈）
  · 分多轮，每轮聚焦一个方面（比一次问完质量高）
  · 让 LLM 标注置信度（哪些是确定的，哪些不确定）

用法：
    python llm_distiller.py --topic "稀疏注意力机制"
    python llm_distiller.py --topic "..." --rounds 3      # 深挖3轮
    python llm_distiller.py --batch topics.txt            # 批量
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


class LLMDistiller:
    """向 LLM 学习的蒸馏器"""

    # 教学视角（让 LLM 输出"教材式"内容，而不是聊天式）
    TEACH_SYSTEM = """你是一位严谨的技术教材编写者。

要求：
1. 只讲**确定的事实**，不确定的明确说"不确定"
2. 讲清：是什么、为什么、怎么工作、关键机制、数量关系
3. 使用准确术语，必要时给出公式或数字
4. 不要客套话、不要"很高兴为您解答"这类废话
5. 直接输出知识内容，不要 markdown 代码块包裹"""

    def __init__(self, llm, brain=None, verbose: bool = True):
        self.llm = llm
        self.brain = brain
        self.verbose = verbose
        self.stats = {"asked": 0, "stored": 0, "rejected": 0}

    def _log(self, msg: str):
        if self.verbose:
            print(f"    {msg}", flush=True)

    def ask(self, prompt: str, max_tokens: int = 900) -> Optional[str]:
        """问 LLM（带计时）"""
        t0 = time.time()
        out = self.llm.generate(prompt, system=self.TEACH_SYSTEM,
                                max_tokens=max_tokens)
        self.stats["asked"] += 1
        if self.verbose:
            self._log(f"LLM 用时 {time.time()-t0:.1f}s，返回 {len(out or '')} 字")
        return out

    def _parse_json(self, txt: str) -> Any:
        if not txt:
            return None
        s = txt.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        for oc, cc in (("[", "]"), ("{", "}")):
            i, j = s.find(oc), s.rfind(cc)
            if i >= 0 and j > i:
                try:
                    return json.loads(s[i:j + 1])
                except Exception:
                    continue
        return None

    # ---------- 1. 生成学习大纲 ----------
    def outline(self, topic: str, n: int = 4) -> List[str]:
        """让 LLM 给出该主题的关键方面"""
        out = self.ask(
            f"""主题：{topic}

请列出理解这个主题必须掌握的 {n} 个关键方面（每方面一句话说明要讲什么）。

格式（只输出这个格式，不要其他内容）：
1. 方面名称 | 要解释的内容
2. ...

要求：方面要具体（针对机制/原理），不要"定义""概述"这种空泛的。""",
            max_tokens=400)
        if not out:
            return [topic]
        items = []
        for line in out.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[.、)]\s*(.+)$", line)
            if m:
                items.append(m.group(1).strip())
        return items[:n] or [topic]

    # ---------- 2. 就某方面提问 ----------
    def teach_aspect(self, topic: str, aspect: str) -> Optional[Dict]:
        """让 LLM 讲透一个方面"""
        out = self.ask(
            f"""主题：{topic}
需要讲清的方面：{aspect}

请写一段 200-400 字的知识讲解，覆盖：
- 它是什么 / 为什么重要
- 内部机制或工作原理
- 关键细节（术语、数字、公式如有）

最后单独一行输出：CONFIDENCE: 0.0-1.0（你对自己这段内容准确度的估计）
如果这个方面你不确定，直接说"不确定"并解释原因。""",
            max_tokens=800)
        if not out:
            return None

        # 提取置信度
        conf = 0.7
        m = re.search(r"CONFIDENCE[:：]\s*([0-9.]+)", out)
        if m:
            try:
                conf = float(m.group(1))
            except Exception:
                pass
            out = out[:m.start()].strip()

        body = out.strip()
        if not body or "不确定" in body[:20] and len(body) < 60:
            return {"aspect": aspect, "knowledge": "", "confidence": 0.0,
                    "unknown": True}
        if len(body) < 40:
            return None
        return {"aspect": aspect, "knowledge": body, "confidence": conf,
                "unknown": False}

    # ---------- 3. 自我检查 ----------
    def self_check(self, topic: str, knowledge: str) -> Dict:
        """让 LLM 检查自己的输出（找错误/遗漏）"""
        out = self.ask(
            f"""主题：{topic}

以下是关于该主题的一段讲解：
---
{knowledge[:1500]}
---

请检查：
1. 有没有**事实性错误**？
2. 有没有**重要遗漏**？

只输出 JSON：
{{"has_error": false, "error_note": "", "missing": "最重要的遗漏（如有）", "verdict": "ok|suspect"}}
只输出 JSON。""",
            max_tokens=400)
        data = self._parse_json(out)
        return data if isinstance(data, dict) else {"verdict": "ok"}

    def _gen_questions(self, topic: str, aspect: str, knowledge: str) -> List[str]:
        """生成"用户可能怎么问这个知识点"（检索友好）"""
        out = self.ask(
            f"""主题：{topic}
知识点：{aspect}
内容摘要：{knowledge[:200]}

请给出 3-4 个**用户可能用来搜索这个知识点**的自然提问（口语化，简短）。
每行一个，不要编号，不要解释。像这样：
KV缓存是什么
为什么要用KV缓存""",
            max_tokens=250)
        if not out:
            return []
        qs = []
        for line in out.splitlines():
            s = re.sub(r"^[\s\-\*·\d.、)]+", "", line).strip()
            s = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", s, flags=re.S).strip()
            if 3 <= len(s) <= 40:
                qs.append(s)
        return qs[:4]

    # ---------- 主流程 ----------
    def distill(self, topic: str, rounds: int = 2, check: bool = True,
                store: bool = True) -> Dict:
        print(f"\n{'='*70}")
        print(f"  LLM 蒸馏：{topic}")
        print(f"{'='*70}")

        learned: List[Dict] = []

        # 1. 大纲
        self._log("① 生成学习大纲…")
        aspects = self.outline(topic, n=rounds + 2)
        for a in aspects:
            self._log(f"   · {a}")

        # 2. 逐方面学习
        for i, aspect in enumerate(aspects, 1):
            self._log(f"\n② [{i}/{len(aspects)}] {aspect}")
            r = self.teach_aspect(topic, aspect)
            if not r:
                self._log("   失败")
                self.stats["rejected"] += 1
                continue
            if r["unknown"]:
                self._log("   ⚠ LLM 表示不确定")
                self.stats["rejected"] += 1
                continue

            k = r["knowledge"]
            # ★清理 R1 思维链标签（思维链不是知识）
            k = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", k, flags=re.S).strip()
            if not k:
                self._log("   清理后为空，跳过")
                self.stats["rejected"] += 1
                continue
            conf = r["confidence"]
            self._log(f"   ✓ {len(k)} 字 | 自信 {conf:.2f}")

            # 3. 自检
            if check:
                chk = self.self_check(topic, k)
                if chk.get("verdict") == "suspect":
                    self._log(f"   ⚠ 自检存疑: {chk.get('error_note','')[:60]}")
                    conf *= 0.5          # 降权
                miss = chk.get("missing", "")
                if miss and len(miss) > 10 and check:
                    self._log(f"   + 补充遗漏: {miss[:60]}")

            # 4. 入库
            stored = False
            if store and self.brain is not None and conf >= 0.3:
                # ★生成"用户可能怎么问"（提升检索命中率）
                qs = self._gen_questions(topic, aspect, k)
                # 主条目：主题｜方面 + 假设问题（拼一起，BM25 都能命中）
                title = f"{topic}｜{aspect}"
                if qs:
                    title += " ｜ 问法：" + " / ".join(qs[:4])
                ok = self.brain.teach(title, k, source="ai_llm")
                if ok:
                    self.stats["stored"] += 1
                    stored = True

            learned.append({"aspect": aspect, "knowledge": k,
                            "confidence": conf, "stored": stored})

        print(f"\n  完成: {len(learned)} 条 | 入库 {self.stats['stored']} | "
              f"拒绝 {self.stats['rejected']}")
        return {"topic": topic, "learned": learned, "stats": dict(self.stats)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="向 LLM 学习")
    ap.add_argument("--topic", type=str)
    ap.add_argument("--rounds", type=int, default=3, help="学习方面数")
    ap.add_argument("--neurons", type=int, default=8192)
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    from cortex import LLMCortex
    brain = None
    if args.no_brain:
        llm = LLMCortex()
    else:
        from conductor import Conductor
        print("载入大脑…")
        c = Conductor(n_neurons=args.neurons, n_tracts=max(256, args.neurons // 16),
                      use_llm=True)
        brain = c
        llm = c.llm

    d = LLMDistiller(llm, brain)
    r = d.distill(args.topic, rounds=args.rounds, check=not args.no_check)

    print("\n" + "=" * 70)
    print("  学到的内容")
    print("=" * 70)
    for i, l in enumerate(r["learned"], 1):
        print(f"\n[{i}] {l['aspect']}  (自信 {l['confidence']:.2f})")
        print(f"    {l['knowledge'][:350]}")
    if brain is not None:
        print(f"\n大脑记忆: {len(brain.memory)} 条")
