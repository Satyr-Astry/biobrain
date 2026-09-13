"""
仿生大脑 · AI 蒸馏器 (ai_distiller.py)
========================================
让 AI 模型**全程主导**知识获取与提炼（而非人工规则）。

对比：
  ✗ 旧方式：规则搜关键词 → 拿摘要 → 直接存（浅、噪声多）
  ✓ 新方式：AI 决定搜什么 → AI 判断质量 → AI 提炼合成 → 存进大脑

流程（AI 蒸馏循环）：
  1. 【提问】AI 把主题拆成 3-5 个具体的子问题
  2. 【检索】对每个子问题搜索（多角度）
  3. 【评判】AI 判断资料是否相关/可信（打分，过滤噪声）
  4. 【提炼】AI 把资料压缩成准确的知识条目（保留关键概念）
  5. 【查漏】AI 判断"还缺什么"，决定是否继续深挖
  6. 【入库】存进神经组织（标记来源与质量分）

关键：**AI 不只是搬运工，而是理解者**。

用法：
    python ai_distiller.py --topic "稀疏注意力机制"
    python ai_distiller.py --topic "..." --deep      # 多轮深挖
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


class AIDistiller:
    """AI 主导的知识蒸馏器"""

    def __init__(self, llm, tools_call, brain=None, verbose: bool = True):
        self.llm = llm
        self.call = tools_call
        self.brain = brain
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"    {msg}", flush=True)

    def _ask_json(self, prompt: str, system: str, max_tokens: int = 800) -> Any:
        """让 LLM 返回 JSON（容错解析）

        ★踩坑记录：LLM 常把 JSON 包在 ```json ... ``` 里，
        贪婪正则 \{.*\} 会把代码块标记也吃进去导致解析失败。
        """
        txt = self.llm.generate(prompt, system=system, max_tokens=max_tokens)
        if not txt:
            return None
        return self._parse_json(txt)

    @staticmethod
    def _parse_json(txt: str) -> Any:
        """从容错提取 JSON（处理代码块、前后缀文字）"""
        if not txt:
            return None
        s = txt.strip()

        # 1) 剥掉 markdown 代码块
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

        # 2) 直接试
        try:
            return json.loads(s)
        except Exception:
            pass

        # 3) 退而求其次：找最外层的 [] 或 {}
        for open_c, close_c in (("[", "]"), ("{", "}")):
            i = s.find(open_c)
            j = s.rfind(close_c)
            if i >= 0 and j > i:
                frag = s[i:j + 1]
                try:
                    return json.loads(frag)
                except Exception:
                    # 4) 最后尝试：修常见小毛病（尾逗号）
                    frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
                    try:
                        return json.loads(frag2)
                    except Exception:
                        continue
        return None

    # ---------- 1. 拆解子问题 ----------
    def decompose(self, topic: str) -> List[str]:
        """AI 把主题拆成具体子问题（比关键词搜索精准得多）"""
        data = self._ask_json(
            f"""主题：{topic}

请把这个主题拆解成 3-5 个**具体的、能搜索到好资料**的子问题。
好的子问题应该：
- 针对核心机制/原理，而不是泛泛的定义
- 用搜索引擎能查到技术文章的说法

输出 JSON 数组：["子问题1", "子问题2", ...]
只输出 JSON。""",
            system="你是研究助手，擅长把大问题拆成可检索的小问题。",
            max_tokens=400)
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)][:5]
        return [topic]

    # ---------- 2. 检索 ----------
    def retrieve(self, question: str, n: int = 5) -> List[Dict]:
        """搜索 + 抓正文"""
        out = []
        r = self.call("web_search", query=question, n=n)
        if not r.get("ok"):
            return out
        for item in r.get("results", []):
            out.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })
        # 尝试抓取技术站点正文
        TECH = ("zhihu.com", "csdn.net", "cnblogs.com", "jianshu.com",
                "segmentfault.com", "infoq.cn", "51cto.com", "github.com")
        for item in out[:3]:
            if any(x in item["url"] for x in TECH):
                b = self.call("web_fetch", url=item["url"], limit=3000)
                if b.get("ok") and len(b.get("text", "")) > 300:
                    item["body"] = b["text"][:2000]
                    break
        return out

    # ---------- 3. 评判质量 ----------
    def judge(self, topic: str, question: str, docs: List[Dict]) -> List[int]:
        """AI 判断哪些资料相关（返回保留的索引）"""
        if not docs:
            return []
        listing = "\n".join(
            f"[{i}] {d['title'][:60]}\n    {d.get('snippet','')[:150]}"
            for i, d in enumerate(docs))
        data = self._ask_json(
            f"""主题：{topic}
子问题：{question}

以下是搜到的资料：
{listing}

请判断哪些资料**真正相关且有用**（不是广告、不是词典释义、不是无关内容）。
输出 JSON：{{"keep": [保留的索引], "reason": "简短理由"}}
只输出 JSON。""",
            system="你是严格的质量评审，只保留真正有用的资料。",
            max_tokens=300)
        if isinstance(data, dict) and isinstance(data.get("keep"), list):
            return [i for i in data["keep"] if isinstance(i, int) and 0 <= i < len(docs)]
        return list(range(min(len(docs), 3)))

    # ---------- 4. 提炼知识 ----------
    def extract(self, topic: str, question: str, docs: List[Dict]) -> Optional[Dict]:
        """AI 把资料提炼成准确的知识条目"""
        if not docs:
            return None
        material = "\n\n".join(
            f"【{d['title'][:50]}】\n{d.get('body') or d.get('snippet','')}"
            for d in docs)[:4000]

        data = self._ask_json(
            f"""主题：{topic}
子问题：{question}

原始资料：
{material}

请把资料提炼成一条**准确、有信息量**的知识条目：
- **必须写满 150-400 字**（这是硬要求，太短等于没提炼）
- 说明：是什么、怎么工作、关键机制、为什么重要
- 只保留事实性内容，不要"我认为""据了解"这类模糊表述
- 保留关键术语、机制、数字、公式（如有）
- 如果资料互相矛盾或明显不可靠，confidence 标低
- **只有资料完全无法回答子问题时**，才把 say_unknown 设为 true

输出 JSON：
{{"knowledge": "提炼后的知识（150-400字）", "confidence": 0.0-1.0, "say_unknown": false, "key_terms": ["术语1","术语2"]}}
只输出 JSON。""",
            system="你是知识提炼专家，只输出准确的事实。",
            max_tokens=900)
        return data if isinstance(data, dict) else None

    # ---------- 5. 查漏 ----------
    def find_gaps(self, topic: str, learned: List[str]) -> List[str]:
        """AI 判断还缺什么"""
        if not learned:
            return []
        data = self._ask_json(
            f"""主题：{topic}

已经掌握的内容：
{chr(10).join('· ' + l[:120] for l in learned[:6])}

请判断：为了真正理解这个主题，还缺哪些**关键方面**？
输出 JSON 数组（最多 2 条，如果已经足够则输出空数组）：["还缺的问题1", ...]
只输出 JSON。""",
            system="你是严谨的研究者，善于发现知识盲区。",
            max_tokens=300)
        if isinstance(data, list):
            return [str(x) for x in data][:2]
        return []

    # ---------- 主流程 ----------
    def distill(self, topic: str, deep: bool = False) -> Dict:
        """完整蒸馏流程"""
        print(f"\n{'='*70}")
        print(f"  AI 蒸馏：{topic}")
        print(f"{'='*70}")

        learned: List[Dict] = []
        stats = {"queries": 0, "docs": 0, "kept": 0, "stored": 0, "unknown": 0}

        # 1. 拆解
        self._log("① 拆解子问题…")
        questions = self.decompose(topic)
        for q in questions:
            self._log(f"   · {q}")

        # 2. 逐个子问题处理
        for qi, q in enumerate(questions, 1):
            self._log(f"\n② [{qi}/{len(questions)}] {q}")
            docs = self.retrieve(q, n=5)
            stats["queries"] += 1
            stats["docs"] += len(docs)
            if not docs:
                self._log("   无资料")
                continue

            keep = self.judge(topic, q, docs)
            kept = [docs[i] for i in keep]
            stats["kept"] += len(kept)
            self._log(f"   检索 {len(docs)} 篇 → 保留 {len(kept)} 篇")

            k = self.extract(topic, q, kept)
            if not k:
                self._log("   提炼失败")
                continue

            if k.get("say_unknown"):
                stats["unknown"] += 1
                self._log(f"   ⚠ 资料不足，AI 说不知道")
                continue

            knowledge = (k.get("knowledge") or "").strip()
            conf = float(k.get("confidence") or 0)
            # ★校准：LLM 自评普遍偏高，按内容量打折（字数越多越可信）
            content_len = len((k.get("knowledge") or ""))
            len_factor = min(1.0, content_len / 300.0)      # 300字满分
            conf = conf * (0.5 + 0.5 * len_factor)           # 最多打 5 折
            if len(knowledge) < 30:
                self._log(f"   内容太短({len(knowledge)}字)，丢弃")
                continue

            self._log(f"   ✓ 提炼 {len(knowledge)} 字 | 自信 {conf:.2f} | "
                      f"术语 {k.get('key_terms', [])[:3]}")

            # 3. 入库
            if self.brain is not None and conf >= 0.4:
                ok = self.brain.teach(f"{topic}：{q}", knowledge,
                                      source="ai_distill")
                if ok:
                    stats["stored"] += 1

            learned.append({"question": q, "knowledge": knowledge,
                            "confidence": conf, "terms": k.get("key_terms", [])})

        # 4. 深挖（可选）
        if deep and learned:
            self._log("\n③ 查漏补缺…")
            gaps = self.find_gaps(topic, [l["knowledge"] for l in learned])
            for g in gaps:
                self._log(f"   补充: {g}")
                docs = self.retrieve(g, n=4)
                keep = self.judge(topic, g, docs)
                k = self.extract(topic, g, [docs[i] for i in keep])
                if k and not k.get("say_unknown") and k.get("knowledge"):
                    kn = k["knowledge"].strip()
                    if len(kn) >= 30:
                        learned.append({"question": g, "knowledge": kn,
                                        "confidence": float(k.get("confidence") or 0),
                                        "terms": k.get("key_terms", [])})
                        if self.brain is not None:
                            if self.brain.teach(f"{topic}：{g}", kn, source="ai_distill"):
                                stats["stored"] += 1

        print(f"\n  蒸馏完成: {len(learned)} 条知识 | 入库 {stats['stored']} | "
              f"资料 {stats['docs']}→{stats['kept']}")
        return {"topic": topic, "learned": learned, "stats": stats}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI 蒸馏器")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--neurons", type=int, default=8192)
    ap.add_argument("--no-brain", action="store_true", help="只蒸馏不入库")
    args = ap.parse_args()

    from conductor import Conductor
    from tools_layer import call_tool

    brain = None
    llm = None
    if not args.no_brain:
        print("载入大脑…")
        c = Conductor(n_neurons=args.neurons, n_tracts=max(256, args.neurons // 16),
                      use_llm=True)
        brain = c
        llm = c.llm
    else:
        from cortex import LLMCortex
        llm = LLMCortex()

    d = AIDistiller(llm, call_tool, brain)
    r = d.distill(args.topic, deep=args.deep)

    print("\n" + "=" * 70)
    print("  蒸馏结果")
    print("=" * 70)
    for i, l in enumerate(r["learned"], 1):
        print(f"\n[{i}] {l['question']}")
        print(f"    自信 {l['confidence']:.2f} | {l['knowledge'][:250]}")
    if brain is not None:
        print(f"\n大脑记忆: {len(brain.memory)} 条")
