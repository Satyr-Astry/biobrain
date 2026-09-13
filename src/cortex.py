"""
仿生大脑 · LLM 语言区接口 (cortex.py)
========================================
路线 B：LLM 作为"语言区"脑区，神经组织作为"认知/记忆层"。

架构：
    ┌──────────────┐   感觉端(text)   ┌─────────────────┐   运动端    ┌──────────┐
    │  外部输入    │ ───────────────► │  仿生神经组织    │ ─────────► │  LLM     │
    │ (用户/工具)  │                  │  (记忆/联想/     │            │ 语言区   │
    └──────────────┘ ◄─────────────── │   学习/决策)     │ ◄───────── └──────────┘
                      语义编码(embed)  └─────────────────┘   语言输出
                                            │
                                            ▼
                                     思考流 harness

关键点：
  1. LLM **不改权重**（不破坏语言能力）
  2. 神经组织负责：记住经验、学到关联、决定"该说什么/该不该说"
  3. LLM 只负责：把神经组织的意图 → 变成人话
  4. 大脑的"自信校准"能决定是否调用 LLM（不懂就不编）

用法：
    python cortex.py --demo              # 演示完整闭环
    python cortex.py --ask "你好"        # 问一句
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import Optional, List, Dict

from tensor_brain import TensorBrain, Tier, ACTIVE_EPS
from encoder import get_encoder

# ---------------- LLM 后端（Ollama）----------------
def _clean_think(txt: str) -> str:
    """清理 R1 思维链（★处理未闭合标签的情况）

    踩坑：R1 在 max_tokens 限制下可能只输出 <think> 而**没有闭合标签**，
    此时正则 <think>.*?</think> 匹配不到，需要特判。
    """
    import re as _re
    if not txt:
        return ""
    # 1) 完整标签对
    txt = _re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", txt, flags=_re.S)
    # 2) 未闭合的 <think>：把它之后的内容全去掉？不行——那会丢掉正文。
    #    更好：去掉 <think> 标记本身，但保留后面的内容作为候选答案，
    #    因为 R1 常常"思考完直接给答案"，中间没有闭合标签。
    txt = _re.sub(r"</?think(?:ing)?>", "", txt)

    # 2) 去掉开头的"思考口癖"（R1 常见：嗯/好的/让我/我需要/用户问的是...）
    txt = txt.strip()
    # 反复剥离句首的口癖短语（最多剥 3 次）
    PAT = (r"^(?:嗯|哦|啊|好的|好|明白了?|让我|我来|我需要|用户问的?是|"
           r"根据我的知识|根据资料|根据之前|This is|Okay|Let me)[，,。:：\s]*")
    for _ in range(3):
        new = _re.sub(PAT, "", txt)
        if new == txt:
            break
        txt = new
    # 去掉"我"开头的自述句（只删第一句，若它明显是思考）
    m = _re.match(r"^[^。！？\n]{0,40}(?:思考|想一下|回忆|分析|梳理)[^。！？\n]{0,20}[。！？]\s*", txt)
    if m:
        txt = txt[m.end():]
    return txt.strip()


OLLAMA_URL = os.environ.get("BRAIN_LLM_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("BRAIN_LLM_MODEL", "huihui_ai/deepseek-r1-abliterated:14b")


class LLMCortex:
    """语言区（LLM 封装）。可选；没有 LLM 时自动降级为"模板输出"。"""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: float = 60.0):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.url}/api/tags")
            with urllib.request.urlopen(req, timeout=15) as r:   # ★首次可能冷启动
                data = json.loads(r.read().decode())
            names = [m.get("name", "") for m in data.get("models", [])]
            return any(self.model.split(":")[0] in n for n in names)
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "", max_tokens: int = 200) -> Optional[str]:
        """调 LLM 生成（失败返回 None，绝不抛异常中断大脑）"""
        if not self.available:
            return None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7},
        }
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.url}/api/generate", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode())
            txt = (out.get("response") or "").strip()
            # ★清理 R1 的思维链标签（思维链不是给用户看的输出）
            import re
            txt = _clean_think(txt)
            return txt or None
        except Exception:
            return None


# ---------------- 认知层（神经组织）----------------
class CognitiveLayer:
    """认知层：用仿生神经组织做记忆/联想/置信度校准。"""

    def __init__(self, n_neurons: int = 4096, n_tracts: int = 256, seed: int = 42):
        self.brain = TensorBrain(n_neurons=n_neurons, n_tracts=n_tracts, seed=seed)
        self.encoder = get_encoder()
        # 语义 → 神经元的投影（把一个 embedding 映射成激活模式）
        import numpy as np
        rng = np.random.default_rng(seed)
        self.emb_dim = self.encoder.dim
        self.proj = rng.normal(0, 1.0 / (self.emb_dim ** 0.5),
                               (self.emb_dim, self.brain.n)).astype("float32")
        # 经验记忆：语义 → 结果
        self.memory: List[Dict] = []
        self.tick_count = 0

    # ---------- 感觉端 ----------
    def perceive(self, text: str, gain: float = 1.0):
        """文字 → 激活模式（稀疏点火）"""
        import numpy as np
        emb = self.encoder.encode(text)
        acts = np.tanh(self.proj.T @ emb)
        # 稀疏：只点亮最强的 4% 神经元
        k = max(1, int(self.brain.n * 0.04))
        idx = np.argsort(-np.abs(acts))[:k]
        vals = np.abs(acts[idx]) * gain
        self.brain.ignite_batch(idx.astype("int32"), vals.astype("float32"))
        return idx

    # ---------- 思考 ----------
    def think(self, steps: int = 12) -> dict:
        """跑若干 tick，返回收敛情况"""
        import numpy as np
        hist = []
        for _ in range(steps):
            act = self.brain.tick()
            self.brain.update_plasticity(act)
            self.brain.schedule()
            hist.append(float(self.brain.activity.mean()))
            self.tick_count += 1
        # 收敛度 = 后段均值变化率的稳定度
        if len(hist) >= 6:
            tail = np.diff(hist[-6:])
            convergence = float(1.0 / (1.0 + np.abs(tail).mean() * 100))
        else:
            convergence = 0.0
        return {
            "convergence": round(convergence, 3),
            "active": int((self.brain.activity > ACTIVE_EPS).sum()),
            "ticks": self.tick_count,
        }

    # ---------- 置信度（防幻觉）----------
    def confidence(self, text: str) -> float:
        """基于经验的置信度：见过类似的 → 高；没见过 → 低"""
        if not self.memory:
            return 0.1
        emb = self.encoder.encode(text)
        import numpy as np
        sims = [float(emb @ m["emb"]) for m in self.memory]
        return float(np.clip(max(sims), 0.0, 1.0))

    def remember(self, text: str, answer: str, source: str = "oracle"):
        """记住一次经验（★来源必须是可信的，防自噬）"""
        if source not in ("oracle", "user_action", "outcome"):
            return False          # 禁止自评信号入缓冲
        self.memory.append({
            "text": text, "answer": answer, "source": source,
            "emb": self.encoder.encode(text),
        })
        return True

    # ---------- 睡眠 ----------
    def sleep(self) -> dict:
        import numpy as np
        # NREM：合并可塑量
        merged = self.brain.consolidate()
        # 重要性修剪：忘掉最不重要的 5%
        imp = self.brain.importance
        if np.linalg.norm(imp) > 0:
            thr = np.percentile(imp, 5)
            weak = np.where(imp < thr)[0]
            self.brain.importance[weak] *= 0.5
        return {"merged": merged.get("merged", 0), "memory_size": len(self.memory)}

    def stats(self) -> dict:
        s = self.brain.stats()
        s["memory"] = len(self.memory)
        return s


# ---------------- 总装：大脑 + 语言区 ----------------
class BioCortex:
    """完整系统：认知层（神经组织）+ 语言区（LLM）"""

    def __init__(self, use_llm: bool = True, n_neurons: int = 4096, n_tracts: int = 256):
        self.cog = CognitiveLayer(n_neurons=n_neurons, n_tracts=n_tracts)
        self.llm = LLMCortex() if use_llm else None

    def think(self, text: str, steps: int = 12) -> dict:
        """完整思考：感知 → 神经处理 → 决定是否输出"""
        # 1) 感知 + 思考
        self.cog.perceive(text)
        tinfo = self.cog.think(steps)

        # 2) 置信度（防幻觉闸门）
        conf = self.cog.confidence(text)
        # ★闸门逻辑（校准版）：
        #   - 无经验（冷启动）→ 允许说，否则系统永远开不了口
        #   - 有经验 → 置信度低就不说（防幻觉）
        n_mem = len(self.cog.memory)
        if n_mem == 0:
            should_speak = True
        else:
            should_speak = conf > 0.6 or tinfo["convergence"] > 0.55

        # 3) 决定输出
        out = None
        if should_speak and self.llm and self.llm.available:
            sys_prompt = ("你是一个仿生大脑的语言区。只说你确定的，不确定就直说不知道。"
                          "回答要简短。")
            out = self.llm.generate(text, system=sys_prompt, max_tokens=200)

        return {
            "input": text,
            "convergence": tinfo["convergence"],
            "confidence": round(conf, 3),
            "should_speak": bool(should_speak),
            "output": out,
            "brain": self.cog.stats(),
        }

    def teach(self, text: str, answer: str, source: str = "oracle") -> bool:
        return self.cog.remember(text, answer, source)

    def sleep(self) -> dict:
        return self.cog.sleep()


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 · LLM 语言区")
    ap.add_argument("--demo", action="store_true", help="演示完整闭环")
    ap.add_argument("--ask", type=str, help="问一句")
    ap.add_argument("--no-llm", action="store_true", help="不接 LLM（纯神经组织）")
    ap.add_argument("--neurons", type=int, default=4096)
    args = ap.parse_args()

    print("=" * 62)
    print("  仿生大脑 · 认知层 + 语言区")
    print("=" * 62)

    bc = BioCortex(use_llm=not args.no_llm, n_neurons=args.neurons)
    print(f"  认知层: {args.neurons} 神经元")
    print(f"  语言区: {'✓ ' + bc.llm.model if (bc.llm and bc.llm.available) else '✗ 不可用（降级为纯神经）'}")
    print()

    if args.ask:
        r = bc.think(args.ask)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.demo:
        print("--- 1. 先教它一条经验 ---")
        ok = bc.teach("你好", "你好呀！我是仿生大脑。", source="oracle")
        print(f"  teach('你好') → {ok}  记忆: {bc.cog.stats()['memory']} 条\n")

        print("--- 2. 问熟悉的（应该会答）---")
        r = bc.think("你好")
        print(f"  收敛={r['convergence']} 自信={r['confidence']} 该说={r['should_speak']}")
        print(f"  输出: {r['output']}\n")

        print("--- 3. 问陌生的（应该诚实说不知道）---")
        r = bc.think("请证明黎曼猜想")
        print(f"  收敛={r['convergence']} 自信={r['confidence']} 该说={r['should_speak']}")
        print(f"  输出: {r['output']}\n")

        print("--- 4. 睡眠巩固 ---")
        print(f"  {bc.sleep()}")
        print(f"  状态: {json.dumps(bc.cog.stats(), ensure_ascii=False)}")
