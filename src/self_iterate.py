"""
仿生大脑 · 持续自主迭代守护进程 (self_iterate.py)
===================================================
让智能体**持续自主迭代**，直到：
  · 遇到无法解决的问题（连续失败达阈值）
  · 被主人打断（Ctrl+C / 停止文件）

行为：
  1. 从"待学主题队列"取目标（没有就自己生成）
  2. 执行自主迭代（规划→执行→学习→总结）
  3. 记录成果；如果某主题反复失败 → 换下一个
  4. 每轮之间睡眠巩固
  5. 全部状态持久化（重启可续）

用法：
    python self_iterate.py                    # 持续跑
    python self_iterate.py --topics "主题1" "主题2"
    python self_iterate.py --max-rounds 5     # 跑5轮就停
    python self_iterate.py --stop-at 0.5      # 成功率低于此值就停
"""
from __future__ import annotations
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

PROGRESS = HERE / "iterate_progress.json"
STOP_FILE = HERE / "STOP_ITERATE"      # 放这个文件就停
RUNNING = True


def log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)


def stop(sig, frame):
    global RUNNING
    RUNNING = False
    log("⚠ 收到停止信号")


# 默认待学主题（可被 --topics 覆盖）
DEFAULT_TOPICS = [
    "稀疏注意力机制的原理",
    "记忆巩固的神经科学基础",
    "向量数据库与相似度检索",
    "Transformer 的 KV 缓存优化",
    "对比学习与表征学习",
    "图神经网络的基本原理",
    "强化学习中的信用分配问题",
    "知识蒸馏技术",
    "模型量化与推理加速",
    "神经科学中的预测编码理论",
]


class SelfIterator:

    def __init__(self, topics, neurons: int = 16384, max_rounds: int = 0,
                 stop_ratio: float = 0.0, budget_per_round: float = 240.0):
        self.topics = list(topics)
        self.neurons = neurons
        self.max_rounds = max_rounds
        self.stop_ratio = stop_ratio
        self.budget = budget_per_round
        self.progress = self._load()
        self.agent = None

    # ---------- 进度 ----------
    def _load(self):
        if PROGRESS.exists():
            try:
                return json.loads(PROGRESS.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"done": [], "failed": [], "rounds": 0,
                "learned_total": 0, "started": datetime.now().isoformat()}

    def _save(self):
        try:
            PROGRESS.write_text(json.dumps(self.progress, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        except Exception as e:
            log(f"进度保存失败: {e}")

    # ---------- 主循环 ----------
    def run(self):
        from autonomous_agent import build_agent

        log("=" * 72)
        log("  仿生大脑 · 持续自主迭代")
        log("=" * 72)
        log(f"  待学主题: {len(self.topics)} 个")
        log(f"  已完成:   {len(self.progress['done'])} 个")
        log(f"  已失败:   {len(self.progress['failed'])} 个")
        log(f"  累计学习: {self.progress['learned_total']} 条")
        log("")

        log("构建智能体…")
        t0 = time.time()
        self.agent = build_agent(self.neurons)
        log(f"完成 ({time.time()-t0:.1f}s)")

        consecutive_fail = 0

        while RUNNING:
            # 检查停止条件
            if STOP_FILE.exists():
                log(f"⏹ 检测到停止文件 {STOP_FILE.name}")
                break
            if self.max_rounds and self.progress["rounds"] >= self.max_rounds:
                log(f"⏹ 达到最大轮数 {self.max_rounds}")
                break

            # 取下一个未做的主题
            pending = [t for t in self.topics
                       if t not in self.progress["done"]
                       and t not in self.progress["failed"]]
            if not pending:
                log("🎉 所有主题已完成！")
                # 自己生成新主题
                new = self._propose_topics()
                if not new:
                    log("⏹ 无法生成新主题，停止")
                    break
                self.topics.extend(new)
                pending = new

            topic = pending[0]
            self.progress["rounds"] += 1
            rnd = self.progress["rounds"]
            log("")
            log("─" * 72)
            log(f"第 {rnd} 轮 | 主题: {topic}")
            log("─" * 72)

            try:
                self.agent.start_time = time.time()
                self.agent.history = []
                report = self.agent.run(topic, max_steps=5,
                                        time_budget=self.budget)
                learned = report.get("learned_total", 0)

                if learned > 0:
                    self.progress["done"].append(topic)
                    self.progress["learned_total"] += learned
                    consecutive_fail = 0
                    log(f"✓ 本轮成功 | 学到 {learned} 条")
                else:
                    self.progress["failed"].append(topic)
                    consecutive_fail += 1
                    log(f"✗ 本轮无收获 | 连续失败 {consecutive_fail}")
            except Exception as e:
                self.progress["failed"].append(topic)
                consecutive_fail += 1
                log(f"✗ 异常: {type(e).__name__}: {e}")

            self._save()

            # 停止条件：连续失败太多
            if consecutive_fail >= 3:
                log(f"⏹ 连续 {consecutive_fail} 轮失败，判定为无法解决的问题，停止")
                break

            # 每轮后睡眠
            try:
                if hasattr(self.agent.brain, "sleep"):
                    s = self.agent.brain.sleep()
                    log(f"😴 睡眠: {s}")
            except Exception:
                pass

        # 收尾
        self._save()
        log("")
        log("=" * 72)
        log("  迭代停止")
        log("=" * 72)
        log(f"  总轮数:   {self.progress['rounds']}")
        log(f"  完成主题: {len(self.progress['done'])}")
        log(f"  失败主题: {len(self.progress['failed'])}")
        log(f"  累计学习: {self.progress['learned_total']} 条")

    def _propose_topics(self, n: int = 3):
        """用 LLM 自己提出新学习主题（自主性）"""
        try:
            already = self.progress["done"] + self.progress["failed"]
            prompt = (f"已经学过的主题：{already}\n"
                      f"请提出 {n} 个**新的、不同的**、对构建通用智能有用的学习主题。"
                      f"每行一个，不要编号，不要解释。")
            txt = self.agent.llm.generate(prompt, max_tokens=300)
            if not txt:
                return []
            out = [l.strip(" -·*0123456789.、") for l in txt.splitlines()]
            return [t for t in out if 4 < len(t) < 40][:n]
        except Exception:
            return []


def main():
    import argparse
    ap = argparse.ArgumentParser(description="持续自主迭代")
    ap.add_argument("--topics", nargs="*", help="待学主题")
    ap.add_argument("--neurons", type=int, default=16384)
    ap.add_argument("--max-rounds", type=int, default=0, help="0=无限")
    ap.add_argument("--stop-ratio", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=240.0, help="每轮时间预算(秒)")
    ap.add_argument("--reset", action="store_true", help="重置进度")
    args = ap.parse_args()

    if args.reset and PROGRESS.exists():
        PROGRESS.unlink()
        log("进度已重置（★注意：这不影响记忆库，只重置进度）")
    # ★警告：绝不要在此处删除 agent_memory.json —— 那会丢掉所有学到的知识

    signal.signal(signal.SIGINT, stop)
    try:
        signal.signal(signal.SIGTERM, stop)
    except Exception:
        pass

    it = SelfIterator(args.topics or DEFAULT_TOPICS,
                      neurons=args.neurons,
                      max_rounds=args.max_rounds,
                      stop_ratio=args.stop_ratio,
                      budget_per_round=args.budget)
    it.run()


if __name__ == "__main__":
    main()
