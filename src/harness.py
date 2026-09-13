"""
仿生大脑 · Harness（思考流驱动器）
====================================
按《工程规范》NORM-1「无待机、思考永不停止」实现一个真正的驱动外壳。

它做什么：
  · 让大脑**持续运行**（思考流主循环，永不退出）
  · 定期注入自发刺激（记忆回放/联想/自发提问）
  · 在低能耗时段自动触发睡眠巩固
  · 接收外部输入（文件队列 / HTTP / 键盘）
  · 把大脑的输出写到日志 + 可选外发

三种运行模式：
  --daemon     纯后台持续思考（默认）
  --paced N    每 N 秒一个 tick（可观察）
  --once       跑固定轮数后退出（用于测试）

外部输入：
  往 brain_inbox.txt 写一行文本 → harness 会拾取并让大脑处理
  大脑的输出写到 brain_outbox.jsonl

依赖：仅标准库 + numpy（复用 code/ 下的实现）
运行：
  cd code
  python harness.py --paced 1          # 每秒一次，看它思考
  python harness.py --daemon           # 后台持续思考
  python harness.py --once --steps 30  # 跑30轮退出
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 复用已有的仿生大脑实现
from server import BioBrain
from advanced import EnergyScheduler

# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
INBOX = HERE / "brain_inbox.txt"
OUTBOX = HERE / "brain_outbox.jsonl"
LOG = HERE / "brain_harness.log"

RUNNING = True


def _log(msg: str, echo: bool = True):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    if echo:
        print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _emit(obj: dict):
    """把大脑的输出写到 outbox（JSONL）"""
    try:
        with open(OUTBOX, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------
# 刺激源：思考流不只是等外部输入
# ------------------------------------------------------------------
class Stimulus:
    """刺激来源：外部输入 + 自发（记忆/联想/提问）"""

    def __init__(self, brain: BioBrain, rng=None):
        import numpy as np
        self.brain = brain
        self.rng = rng or np.random.default_rng()
        self._self_prompts = [
            "我最近学到了什么",
            "有没有什么我还没搞懂的",
            "刚才那条经验说明了什么",
            "我擅长什么、不擅长什么",
            "接下来该做什么",
        ]

    def poll_external(self) -> Optional[str]:
        """从 inbox 文件读一行（外部输入）"""
        if not INBOX.exists():
            return None
        try:
            txt = INBOX.read_text(encoding="utf-8").strip()
            if txt:
                INBOX.write_text("", encoding="utf-8")   # 消费掉
                return txt.splitlines()[0]
        except Exception:
            pass
        return None

    def spontaneous(self) -> Optional[str]:
        """★自发提问（无外部输入时也会发生）"""
        p = self.brain.state().get("self_model", {})
        # 有一定概率自发想起点什么
        if self.rng.random() < 0.25:
            base = str(self.rng.choice(self._self_prompts))
            # 根据自我认知微调
            weak = p.get("不擅长") or []
            if weak and self.rng.random() < 0.5:
                base = f"关于{weak[0]}，我是不是该再想想"
            return base
        return None


# ------------------------------------------------------------------
# Harness 主循环
# ------------------------------------------------------------------
class BrainHarness:
    def __init__(self, brain: Optional[BioBrain] = None):
        self.brain = brain or BioBrain()
        self.brain.load()
        self.stim = Stimulus(self.brain)
        self.energy = EnergyScheduler()
        self.cycle = 0
        self.slept = 0
        self.external_count = 0
        self.spontaneous_count = 0

    def inject(self, text: str, source: str = "external") -> dict:
        """把一条刺激注入大脑并处理"""
        r = self.brain.think(text, task_type="general")
        rec = {
            "cycle": self.cycle,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "input": text,
            "convergence": r.get("convergence"),
            "output": r.get("output"),
            "dont_know": r.get("should_say_dont_know"),
            "activation": r.get("activation"),
        }
        _emit(rec)
        return rec

    def maybe_sleep(self) -> bool:
        """低能耗时段 + 积累够了 → 睡眠巩固"""
        if not self.energy.should_do_heavy():
            return False
        if self.brain.buffer.stats()["size"] < 2:
            return False
        r = self.brain.sleep()
        self.slept += 1
        _log(f"💤 睡眠巩固 #{self.slept}: {json.dumps(r.get('sleep', {}), ensure_ascii=False)[:120]}")
        return True

    def tick(self, paced: bool = False):
        """一个思考周期"""
        self.cycle += 1
        # 1. 外部输入优先
        ext = self.stim.poll_external()
        if ext:
            self.external_count += 1
            rec = self.inject(ext, "external")
            _log(f"📥 外部输入 → {ext[:30]}")
            _log(f"   ↳ 收敛={rec['convergence']} 输出={'有' if rec['output'] else '无(诚实)'}")
            return
        # 2. 自发刺激（思考流不等输入）
        spon = self.stim.spontaneous()
        if spon:
            self.spontaneous_count += 1
            self.inject(spon, "spontaneous")
            if paced:
                _log(f"💭 自发思考: {spon[:30]}")
        # 3. 定期睡眠
        if self.cycle % 8 == 0:
            self.maybe_sleep()
        # 4. 定期报告
        if self.cycle % 10 == 0:
            st = self.brain.state()
            _log(f"📊 周期#{self.cycle} | 活跃={st['active']} 经历={st['self_model']['经历数']} "
                 f"缓冲={st['experience_buffer']['size']} 层级={list(st['tiers'].values())}")

    def run(self, steps: int = 0, paced: bool = False, interval: float = 1.0):
        _log(f"🧠 仿生大脑 Harness 启动 | 模式={'paced' if paced else 'daemon'}"
             f"{' | steps=' + str(steps) if steps else ''}")
        _log(f"   神经元={self.brain.state()['neurons']} ｜ inbox={INBOX.name} outbox={OUTBOX.name}")
        n = 0
        while RUNNING:
            self.tick(paced)
            n += 1
            if steps and n >= steps:
                break
            if paced:
                time.sleep(interval)
            else:
                time.sleep(0.05)     # daemon 模式：极短间隔，持续跑
        self.shutdown()

    def shutdown(self):
        _log("🛑 Harness 停止，保存大脑状态...")
        try:
            p = self.brain.save()
            _log(f"   已保存: {p}")
        except Exception as e:
            _log(f"   ⚠ 保存失败: {e}")
        st = self.brain.state()
        _log(f"✅ 本次运行：{self.cycle} 周期 | 外部输入 {self.external_count} | "
             f"自发思考 {self.spontaneous_count} | 睡眠 {self.slept} | "
             f"经历 {st['self_model']['经历数']}")


# ------------------------------------------------------------------
def _sigint(sig, frame):
    global RUNNING
    RUNNING = False


def main():
    ap = argparse.ArgumentParser(description="仿生大脑 Harness")
    ap.add_argument("--daemon", action="store_true", help="持续后台思考")
    ap.add_argument("--paced", type=float, metavar="SEC", help="每 SEC 秒一个周期")
    ap.add_argument("--once", action="store_true", help="跑固定轮数后退出")
    ap.add_argument("--steps", type=int, default=30, help="--once 时的轮数")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    h = BrainHarness()
    if args.paced:
        h.run(paced=True, interval=args.paced)
    elif args.once:
        h.run(steps=args.steps, paced=False)
    else:
        h.run(paced=False)     # daemon 默认


if __name__ == "__main__":
    main()
