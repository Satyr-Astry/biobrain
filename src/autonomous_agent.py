"""
仿生大脑 · 自主迭代智能体 (autonomous_agent.py)
=================================================
P3 + P4：让它自主规划、上网学习、迭代改进。

自主循环（Observe → Plan → Act → Learn → Evaluate）：
  1. 接收目标（用户给的，或自己提出的）
  2. 规划：把目标拆成步骤（用 LLM 规划，计划存进神经组织）
  3. 执行：调用工具（搜索/抓取/文件）
  4. 学习：把结果"教"给神经组织（可信来源才入库）
  5. 评估：这一步成功了吗？失败就重规划
  6. 记录：写入迭代日志（断电可恢复）

安全护栏：
  · 最大轮数限制（防死循环）
  · 时间预算限制
  · 工具调用频率限制
  · Ctrl+C 优雅停止 + 存档
  · 危险操作白名单（只读文件、只 GET 请求）

用法：
    python autonomous_agent.py --goal "学习 Python asyncio 并总结"
    python autonomous_agent.py --self                     # 自己找事做
    python autonomous_agent.py --goal "..." --max-steps 10
"""
from __future__ import annotations
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

STATE_FILE = HERE / "agent_state.json"
LOG_FILE = HERE / "agent_log.jsonl"
MEM_FILE = HERE / "agent_memory.json"

RUNNING = True
_MAX_STEPS = 12
_TIME_BUDGET = 600.0        # 秒


def _log(msg: str, also_file: bool = True):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if also_file:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": datetime.now().isoformat(),
                                    "msg": msg}, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _stop(sig, frame):
    global RUNNING
    RUNNING = False
    _log("⚠ 收到停止信号，正在保存…")


# ============================================================
# 规划器
# ============================================================
class Planner:
    """任务规划：把目标拆成步骤（用 LLM，但计划存进神经组织）"""

    def __init__(self, llm, brain):
        self.llm = llm
        self.brain = brain

    def plan(self, goal: str, context: str = "") -> List[Dict[str, str]]:
        """生成步骤列表"""
        prompt = f"""把下面的目标拆解成 3-5 个可执行的步骤。
每一步必须能用这些工具之一完成：
- web_search: 搜索互联网（参数 query）
- wiki_lookup: 查百科（参数 term，只需词条名，不要写 URL）
- web_fetch: 抓取**具体网址**（参数 url，必须是真实存在的 http 地址）
- fs_read: 读本地文件（参数 path，必须是真实存在的路径）
- think: 自己思考总结（参数 content）

★重要约束：
1. 不要输出 wikipedia.org 的链接（本机无法访问）
2. 不要臆造本地文件路径（除非确定存在）
3. 优先用 web_search 和 wiki_lookup（这两个一定能用）
4. 最后一步用 think 总结

目标：{goal}
{context}

只输出 JSON 数组，格式：
[{{"step":1,"action":"web_search","arg":"搜索词","why":"为什么"}}]
不要输出其他内容。"""
        try:
            txt = self.llm.generate(prompt, system="你是任务规划器，只输出 JSON。",
                                    max_tokens=600)
            if not txt:
                return []
            m = re.search(r"\[.*\]", txt, re.S)
            if not m:
                return []
            steps = json.loads(m.group(0))
            return [s for s in steps if isinstance(s, dict) and s.get("action")]
        except Exception as e:
            _log(f"规划失败: {e}")
            return []


# ============================================================
# 执行器
# ============================================================
class Executor:
    """执行单个步骤"""

    def __init__(self, brain, tools_call, tools_list):
        self.brain = brain
        self.call = tools_call
        self.tools = tools_list

    def run(self, step: Dict[str, str]) -> Dict[str, Any]:
        action = step.get("action", "")
        arg = step.get("arg", "") or step.get("query", "") or step.get("url", "")

        if action == "think":
            # 纯思考：让神经组织处理
            self.brain.perceive(arg)
            self.brain.think(steps=5)
            return {"ok": True, "action": "think", "result": "已思考"}

        if action == "wiki_lookup":
            return self.call("wiki_lookup", term=arg)
        if action == "web_search":
            return self.call("web_search", query=arg, n=3)
        if action == "web_fetch":
            return self.call("web_fetch", url=arg, limit=3000)
        if action == "fs_read":
            return self.call("fs_read", path=arg, limit=3000)

        return {"ok": False, "error": f"unknown action: {action}"}


# ============================================================
# 评估器
# ============================================================
class Evaluator:
    """判断步骤是否成功"""

    def evaluate(self, step: Dict, result: Dict) -> Dict[str, Any]:
        ok = bool(result.get("ok"))
        detail = ""
        if ok:
            if "results" in result:
                detail = f"找到 {len(result['results'])} 条结果"
            elif "extract" in result:
                detail = f"获得 {len(result['extract'])} 字"
            elif "text" in result:
                detail = f"抓取 {len(result.get('text',''))} 字"
            else:
                detail = "完成"
        else:
            detail = result.get("error", "未知错误")
        return {"ok": ok, "detail": detail}


# ============================================================
# 自主智能体
# ============================================================
class AutonomousAgent:

    def __init__(self, brain, llm, tools_call, tools_list, use_sharded: bool = False):
        self.brain = brain
        self.llm = llm
        self.planner = Planner(llm, brain)
        self.executor = Executor(brain, tools_call, tools_list)
        self.evaluator = Evaluator()
        self.history: List[Dict] = []
        self.start_time = time.time()

    # ---------- 学习 ----------
    def learn(self, step: Dict, result: Dict) -> int:
        """把结果教给神经组织（可信来源才入库）"""
        if not result.get("ok"):
            return 0
        n = 0
        topic = step.get("arg", "")[:80]

        if "results" in result:            # 搜索结果
            for r in result["results"][:2]:
                if self.brain.teach(f"搜索:{topic}|{r['title'][:40]}",
                                    r.get("snippet", "")[:300],
                                    source="web_search"):
                    n += 1
        elif "extract" in result:          # 百科/摘要
            if self.brain.teach(topic, result["extract"][:1500], source="web_wiki"):
                n += 1
        elif "text" in result:             # 网页正文
            if self.brain.teach(topic, result["text"][:400], source="web_page"):
                n += 1
        return n

    # ---------- 单步 ----------
    def run_step(self, step: Dict, idx: int) -> Dict:
        act = step.get("action", "?")
        arg = (step.get("arg") or "")[:60]
        _log(f"  步骤{idx}: [{act}] {arg}")

        t0 = time.time()
        result = self.executor.run(step)
        ev = self.evaluator.evaluate(step, result)
        learned = self.learn(step, result)
        dt = time.time() - t0

        icon = "✓" if ev["ok"] else "✗"
        _log(f"    {icon} {ev['detail'][:60]} | 学习{learned}条 | {dt:.1f}s")

        rec = {
            "step": idx, "action": act, "arg": arg,
            "ok": ev["ok"], "detail": ev["detail"],
            "learned": learned, "elapsed_s": round(dt, 2),
            "t": datetime.now().isoformat(),
        }
        self.history.append(rec)
        return rec

    # ---------- 总结 ----------
    def summarize(self, goal: str) -> str:
        """让大脑+LLM 总结这次迭代"""
        # 把学到的内容整理给 LLM
        learned = [h for h in self.history if h.get("learned")]
        facts = []
        for m in self.brain.memory[-8:]:
            facts.append(f"· {m['text']}: {m['answer'][:150]}")
        ctx = "\n".join(facts) if facts else "（无）"

        prompt = f"""目标：{goal}

本次迭代学到的事实：
{ctx}

请用 3-5 句话总结：这次做到了什么、学到了什么、还有什么没搞懂。"""
        out = self.llm.generate(prompt, system="你是总结器，简短诚实。", max_tokens=400)
        return out or "（总结失败）"

    # ---------- 主循环 ----------
    def run(self, goal: str, max_steps: int = _MAX_STEPS,
            time_budget: float = _TIME_BUDGET) -> Dict:
        global RUNNING
        _log("=" * 70)
        _log(f"🎯 目标: {goal}")
        _log("=" * 70)

        # 1) 规划
        _log("📋 规划中…")
        steps = self.planner.plan(goal)
        if not steps:
            _log("⚠ 规划失败，用兜底计划")
            steps = [
                {"step": 1, "action": "web_search", "arg": goal[:50], "why": "先搜索"},
                {"step": 2, "action": "think", "arg": goal, "why": "消化信息"},
            ]
        _log(f"📋 计划 {len(steps)} 步:")
        for s in steps:
            _log(f"    {s.get('step','?')}. [{s.get('action')}] "
                 f"{(s.get('arg') or '')[:50]}  ← {s.get('why','')}")

        # 2) 执行
        for i, step in enumerate(steps[:max_steps], 1):
            if not RUNNING:
                _log("⏹ 用户中断")
                break
            if time.time() - self.start_time > time_budget:
                _log(f"⏹ 时间预算用尽（{time_budget}s）")
                break
            self.run_step(step, i)
            self.save()

        # 3) 睡眠（巩固）
        _log("😴 睡眠巩固…")
        try:
            s = self.brain.sleep() if hasattr(self.brain, "sleep") else {}
            _log(f"    {s}")
        except Exception as e:
            _log(f"    睡眠失败: {e}")

        # 4) 总结
        _log("📝 总结中…")
        summary = self.summarize(goal)
        _log(f"📝 {summary[:300]}")

        # 5) 报告
        report = {
            "goal": goal,
            "steps_planned": len(steps),
            "steps_done": len(self.history),
            "learned_total": sum(h.get("learned", 0) for h in self.history),
            "memory_size": len(getattr(self.brain, "memory", [])),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "summary": summary,
            "history": self.history,
        }
        self.save(report)
        return report

    # ---------- 持久化 ----------
    def save(self, report: Optional[Dict] = None):
        try:
            data = {
                "last_goal": getattr(self, "_goal", ""),
                "history": self.history,
                "ts": datetime.now().isoformat(),
            }
            if report:
                data["last_report"] = report
            STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            # 记忆单独存（便于复用）
            mem = getattr(self.brain, "memory", [])
            if mem:
                MEM_FILE.write_text(json.dumps(
                    [{"text": m["text"], "answer": m["answer"], "source": m["source"]}
                     for m in mem], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log(f"存档失败: {e}")


# ============================================================
# 入口
# ============================================================
def build_agent(neurons: int = 16384):
    """构建智能体（载入大脑 + 工具）"""
    from conductor import Conductor
    from tools_layer import call_tool, list_tools

    _log("载入神经组织…")
    c = Conductor(n_neurons=neurons, n_tracts=max(256, neurons // 16), use_llm=True)
    _log(f"  神经元={c.brain.n:,} | 语言区={'✓' if (c.llm and c.llm.available) else '✗'}")

    # 恢复历史记忆
    if MEM_FILE.exists():
        try:
            mem = json.loads(MEM_FILE.read_text(encoding="utf-8"))
            for m in mem:
                c.teach(m["text"], m["answer"], m.get("source", "oracle"))
            _log(f"  恢复记忆 {len(mem)} 条")
        except Exception as e:
            _log(f"  记忆恢复失败: {e}")

    return AutonomousAgent(c, c.llm, call_tool, list_tools())


def main():
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 · 自主迭代智能体")
    ap.add_argument("--goal", type=str, help="目标")
    ap.add_argument("--self", action="store_true", help="自己找事做")
    ap.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    ap.add_argument("--budget", type=float, default=_TIME_BUDGET, help="时间预算(秒)")
    ap.add_argument("--neurons", type=int, default=16384)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    agent = build_agent(args.neurons)

    if args.self:
        goal = ("自主学习一个对仿生大脑有用的知识点，"
                "比如「稀疏注意力机制」或「记忆巩固的神经机制」")
    elif args.goal:
        goal = args.goal
    else:
        ap.print_help()
        return

    report = agent.run(goal, args.max_steps, args.budget)

    print()
    print("=" * 70)
    print("  迭代报告")
    print("=" * 70)
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
