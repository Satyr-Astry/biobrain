"""
仿生大脑 · 批量 LLM 学习 (batch_learn.py)
============================================
用 LLM 批量灌输知识（比网络学习快得多、准得多）。

用法：
    python batch_learn.py --topics topics.txt     # 从文件读主题
    python batch_learn.py --preset                # 用内置主题
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

PROGRESS = HERE / "batch_learn_progress.json"

DEFAULT_TOPICS = [
    "Transformer 注意力机制的数学原理",
    "稀疏注意力机制",
    "KV 缓存与推理加速",
    "混合专家模型 MoE",
    "扩散模型的数学基础",
    "对比学习与 InfoNCE 损失",
    "向量检索与近似最近邻",
    "模型量化原理",
    "知识蒸馏方法",
    "图神经网络的消息传递机制",
    "强化学习中的 PPO 算法",
    "位置编码的设计演进",
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="批量 LLM 学习")
    ap.add_argument("--topics", type=str, help="主题文件（每行一个）")
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--rounds", type=int, default=2, help="每主题学几个方面")
    ap.add_argument("--neurons", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="最多学几个主题")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset and PROGRESS.exists():
        PROGRESS.unlink()

    topics = DEFAULT_TOPICS if args.preset else []
    if args.topics:
        topics = [l.strip() for l in Path(args.topics).read_text(
            encoding="utf-8").splitlines() if l.strip()]
    if not topics:
        ap.print_help()
        return

    # 进度
    done = []
    if PROGRESS.exists():
        try:
            done = json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", [])
        except Exception:
            pass
    pending = [t for t in topics if t not in done]
    if args.limit:
        pending = pending[:args.limit]

    print("=" * 72)
    print("  批量 LLM 学习")
    print("=" * 72)
    print(f"  主题总数: {len(topics)}")
    print(f"  已完成:   {len(done)}")
    print(f"  本次学:   {len(pending)}")
    print()

    if not pending:
        print("  全部已学完！")
        return

    # 载入
    from conductor import Conductor
    from llm_distiller import LLMDistiller

    print("载入大脑…")
    c = Conductor(n_neurons=args.neurons,
                  n_tracts=max(256, args.neurons // 16), use_llm=True)
    # 恢复记忆
    mem_file = HERE / "agent_memory.json"
    if mem_file.exists():
        try:
            for m in json.loads(mem_file.read_text(encoding="utf-8")):
                c.teach(m["text"], m["answer"], m.get("source", "oracle"))
            print(f"  恢复记忆 {len(c.memory)} 条")
        except Exception:
            pass

    d = LLMDistiller(c.llm, c, verbose=True)
    t_all = time.time()
    total_stored = 0

    for i, topic in enumerate(pending, 1):
        print()
        print("─" * 72)
        print(f"[{i}/{len(pending)}] {topic}")
        print("─" * 72)
        t0 = time.time()
        try:
            r = d.distill(topic, rounds=args.rounds, check=True)
            n = r["stats"]["stored"]
            total_stored += n
            dt = time.time() - t0
            print(f"  用时 {dt:.0f}s | 入库 {n} 条")
            if n > 0:
                done.append(topic)
        except Exception as e:
            print(f"  ✗ 异常: {type(e).__name__}: {e}")

        # 存档
        try:
            PROGRESS.write_text(json.dumps({"done": done,
                                            "ts": datetime.now().isoformat()},
                                           ensure_ascii=False, indent=2),
                                encoding="utf-8")
            mem = getattr(c, "memory", [])
            mem_file.write_text(json.dumps(
                [{"text": m["text"], "answer": m["answer"], "source": m["source"]}
                 for m in mem], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 定期睡眠
        if i % 3 == 0:
            try:
                s = c.sleep()
                print(f"  😴 睡眠: {s}")
            except Exception:
                pass

    print()
    print("=" * 72)
    print("  批量学习完成")
    print("=" * 72)
    print(f"  总用时:   {(time.time()-t_all)/60:.1f} 分钟")
    print(f"  学会主题: {len(done)}/{len(topics)}")
    print(f"  本次入库: {total_stored} 条")
    print(f"  大脑记忆: {len(getattr(c, 'memory', []))} 条")

    # 最终睡眠
    try:
        print(f"  😴 最终巩固: {c.sleep()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
