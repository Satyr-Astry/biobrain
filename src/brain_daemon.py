"""
仿生大脑 · 完整思考流 (brain_daemon.py)
==========================================
阶段 C：把 B（指挥语言区）接进思考流，让它真正"活着"。

行为：
  1. 持续思考（永不停止，NORM-1）
  2. 从 brain_inbox.txt 拾取外部输入 → 大脑思考 → 指挥 LLM 回答 → 写入 outbox
  3. 无输入时自发活动（回放记忆 / 提问 / 联想）
  4. 空闲时自动睡眠（巩固 + 修剪）
  5. 所有经验持久化（重启动脑子还能记住）

用法：
    python brain_daemon.py                 # 持续运行（Ctrl+C 停止）
    python brain_daemon.py --once --steps 30
    python brain_daemon.py --paced 1       # 每秒一轮（可观察）
    python brain_daemon.py --teach "问题" "答案"   # 预先教一条
"""
from __future__ import annotations
import json
import os
import time
import signal
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

# 路径
HERE = Path(__file__).parent
INBOX = HERE / "brain_inbox.txt"
OUTBOX = HERE / "brain_outbox.jsonl"
STATE = HERE / "brain_state.pkl"
LOG = HERE / "brain_daemon.log"

RUNNING = True


def _log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _handle_stop(sig, frame):
    global RUNNING
    RUNNING = False
    _log("收到停止信号，正在保存…")


def load_brain(Conductor, n_neurons: int, use_llm: bool):
    """载入大脑（有存档就恢复，没有就新建）"""
    if STATE.exists():
        try:
            with open(STATE, "rb") as f:
                data = pickle.load(f)
            c = Conductor(n_neurons=data.get("n_neurons", n_neurons), use_llm=use_llm)
            # 恢复可塑/主体/记忆
            c.brain.base = data["base"]
            c.brain.plasticity = data["plasticity"]
            c.brain.importance = data["importance"]
            c.memory = data["memory"]
            c.tick_total = data.get("tick_total", 0)
            _log(f"恢复大脑存档：{len(c.memory)} 条记忆，{c.tick_total} ticks")
            return c
        except Exception as e:
            _log(f"存档载入失败({e})，新建大脑")
    c = Conductor(n_neurons=n_neurons, use_llm=use_llm)
    _log(f"新建大脑：{n_neurons} 神经元")
    return c


def save_brain(c, n_neurons: int):
    try:
        with open(STATE, "wb") as f:
            pickle.dump({
                "n_neurons": n_neurons,
                "base": c.brain.base,
                "plasticity": c.brain.plasticity,
                "importance": c.brain.importance,
                "memory": c.memory,
                "tick_total": c.tick_total,
            }, f)
        return True
    except Exception as e:
        _log(f"存档失败: {e}")
        return False


def read_inbox() -> Optional[str]:
    """读外部输入（读完即清空，避免重复处理）"""
    if not INBOX.exists():
        return None
    try:
        txt = INBOX.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        INBOX.write_text("", encoding="utf-8")
        return txt
    except Exception:
        return None


def write_outbox(rec: dict):
    try:
        with open(OUTBOX, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def spontaneous(c, k: int = 1):
    """自发活动：回放 / 联想 / 提问（无外部输入时的大脑内部活动）"""
    acts = []
    if c.memory and k % 3 == 0:
        # 回放一条旧记忆
        import random
        m = random.choice(c.memory)
        acts.append(("回放", f"回忆「{m['text']}」"))
    if k % 3 == 1:
        acts.append(("联想", "把最近的记忆互相联系起来"))
    if k % 3 == 2:
        acts.append(("提问", "有没有什么我还没搞懂的？"))
    for kind, desc in acts:
        c.perceive(desc)
        c.think(steps=5)
    return acts


def main():
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 · 完整思考流")
    ap.add_argument("--once", action="store_true", help="跑一段就退出")
    ap.add_argument("--steps", type=int, default=50, help="--once 模式跑多少轮")
    ap.add_argument("--paced", type=float, help="每轮间隔秒数")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--neurons", type=int, default=262144)   # ★v5：默认 26 万神经元
    ap.add_argument("--teach", nargs=2, metavar=("Q", "A"), help="预教一条经验")
    ap.add_argument("--sleep-every", type=int, default=200, help="每多少轮睡一次")
    args = ap.parse_args()

    from conductor import Conductor

    signal.signal(signal.SIGINT, _handle_stop)
    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

    _log("=" * 56)
    _log("  仿生大脑 · 完整思考流启动")
    _log("=" * 56)
    c = load_brain(Conductor, args.neurons, not args.no_llm)
    _log(f"语言区: {'✓ ' + c.llm.model if (c.llm and c.llm.available) else '✗ 不可用'}")

    if args.teach:
        ok = c.teach(args.teach[0], args.teach[1], "oracle")
        _log(f"预教: {'成功' if ok else '失败'} 「{args.teach[0]}」")

    rounds = 0
    stats = {"input": 0, "spoken": 0, "spontaneous": 0, "sleep": 0}

    while RUNNING:
        rounds += 1
        did = False

        # 1) 外部输入
        txt = read_inbox()
        if txt:
            _log(f"📥 输入: {txt[:60]}")
            r = c.respond(txt, steps=12)
            stats["input"] += 1
            if r["output"]:
                stats["spoken"] += 1
                _log(f"   🗣️  大脑说: {r['output'][:100]}")
            else:
                _log(f"   🤐 大脑保持沉默（自信={r['confidence']}）")
            write_outbox({
                "t": datetime.now().isoformat(), "type": "respond",
                "input": txt, **{k: r[k] for k in
                                 ("convergence", "confidence", "should_speak", "output")},
                "recalled": [x["text"] for x in r["recalled"]],
            })
            did = True

        # 2) 自发活动
        if not did:
            acts = spontaneous(c, rounds)
            stats["spontaneous"] += 1
            if acts:
                _log(f"💭 自发: {'; '.join(a[1] for a in acts)}")
            else:
                c.think(steps=3)

        # 3) 睡眠
        if rounds % args.sleep_every == 0:
            s = c.sleep()
            stats["sleep"] += 1
            _log(f"😴 睡眠: 合并{s['merged']}个神经元 | 记忆{s['memory']}条")

        # 4) 存状态（每 20 轮）
        if rounds % 20 == 0:
            save_brain(c, args.neurons)

        if args.paced:
            time.sleep(args.paced)
        if args.once and rounds >= args.steps:
            break

    # 收尾
    save_brain(c, args.neurons)
    _log("=" * 56)
    _log(f"结束 | {rounds} 轮 | 输入{stats['input']} 说{stats['spoken']} "
         f"自发{stats['spontaneous']} 睡眠{stats['sleep']}")
    _log(f"大脑状态: {json.dumps(c.stats(), ensure_ascii=False)}")
    _log("状态已保存，下次启动自动恢复")


if __name__ == "__main__":
    main()
