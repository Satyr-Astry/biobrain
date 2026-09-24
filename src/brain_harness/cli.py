"""
HarnessPlatform · 统一 CLI
============================
    python -m platform.cli list                       # 列出所有插件
    python -m platform.cli run  --config run.yaml     # 按配置跑
    python -m platform.cli run  --steps 30            # 跑 30 tick
    python -m platform.cli demo                       # 零配置演示
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import sys

from .platform import HarnessPlatform, load_config
from .registry import list_all, load_builtins

_RUNNING = True


def _sig(sig, frame):
    global _RUNNING
    _RUNNING = False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="brain_harness", description="BioBrain Harness Platform")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="列出所有可用插件")

    p_run = sub.add_parser("run", help="按配置运行")
    p_run.add_argument("--config", "-c", default="", help="YAML 配置文件")
    p_run.add_argument("--steps", "-n", type=int, default=0, help="跑多少 tick（0=持续）")
    p_run.add_argument("--paced", type=float, default=0.0, help="每 tick 间隔秒")
    p_run.add_argument("--state", default="", help="大脑状态文件路径")

    p_demo = sub.add_parser("demo", help="零配置演示（30 tick）")
    p_demo.add_argument("--steps", type=int, default=30)

    args = ap.parse_args(argv)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if args.cmd == "list":
        load_builtins()
        info = list_all()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "demo":
        cfg = {
            "pipeline": {
                "sources": [{"type": "static",
                             "items": ["猫是哺乳动物", "狗是哺乳动物", "量子纠缠是物理现象"]}],
                "processors": [{"type": "think"}],
                "policies": [{"type": "metrics", "every": 1}],
                "sinks": [{"type": "log", "echo": True}],
            },
            "recorder": {"path": "run/demo_recorder.jsonl"},
        }
        plat = HarnessPlatform(cfg)
        plat.build()
        print("=== 平台插件 ===")
        print(json.dumps(plat.pipeline.describe(), ensure_ascii=False, indent=2))
        stats = plat.run(steps=args.steps)
        print("\n=== 运行统计 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "run":
        cfg = load_config(args.config) if args.config else {}
        if args.state:
            cfg.setdefault("brain", {})["state_path"] = args.state
        plat = HarnessPlatform(cfg)
        plat.build()
        print(f"平台插件: {json.dumps(plat.pipeline.describe(), ensure_ascii=False)}")
        stats = plat.run(steps=args.steps, paced=args.paced)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
