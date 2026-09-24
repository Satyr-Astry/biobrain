"""
实验分析：从 recorder.jsonl 读回，回答 Q1/Q2/Q3
=================================================
用法：cd code && python analyze_experiment.py
"""
from __future__ import annotations
import json
import os
import statistics as st
import statistics
import sys

REC = "run/exp_platform/recorder.jsonl"


def load(path=REC):
    evs = []
    if not os.path.isfile(path):
        print(f"✗ 找不到 {path}")
        return evs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    evs.append(json.loads(line))
                except Exception:
                    pass
    return evs


def main():
    evs = load()
    if not evs:
        return 1
    ticks = [e for e in evs if e.get("ev") == "tick"]
    results = [e for e in evs if e.get("ev") == "result"]
    policies = [e for e in evs if e.get("ev") == "policy"]
    snaps = [e for e in evs if e.get("ev") == "snapshot"]
    errs = [e for e in evs if e.get("ev") == "error"]

    print("=" * 72)
    print("Q1 · 平台稳定性")
    print("=" * 72)
    print(f"  tick 事件   : {len(ticks)}")
    print(f"  result 事件 : {len(results)}")
    print(f"  policy 事件 : {len(policies)}")
    print(f"  snapshot    : {len(snaps)}")
    print(f"  错误        : {len(errs)}")
    if errs:
        for e in errs[:5]:
            print(f"    ⚠ [{e.get('where')}] {e.get('err')}")

    if ticks:
        ms = [t.get("ms", 0) for t in ticks]
        # 分四段看漂移
        n = len(ms) // 4 or 1
        seg = [statistics.mean(ms[i * n:(i + 1) * n]) for i in range(4)]
        print(f"\n  每 tick 耗时: 均值 {st.mean(ms):.2f}ms / 中位 {st.median(ms):.2f}ms / 最大 {max(ms):.2f}ms")
        print(f"  四段均值漂移: {[round(x, 2) for x in seg]}")
        drift = (seg[-1] - seg[0]) / (seg[0] or 1) * 100
        print(f"  漂移率: {drift:+.1f}%  {'✓ 稳定' if abs(drift) < 50 else '⚠ 漂移明显'}")

    print()
    print("=" * 72)
    print("Q2 · 历史区分度 dmin 演化（ALG-18/19 相关）")
    print("=" * 72)
    probes = [p for p in policies if p.get("name") == "workmem_probe"]
    if probes:
        print(f"  {'tick':>6} {'dmin':>10} {'dmean':>10}  判定")
        for p in probes:
            s = p.get("summary", {})
            dm = s.get("dmin")
            dme = s.get("dmean")
            if dm is None:
                continue
            mark = "✅ >0.01" if dm > 0.01 else "⚠ <0.01"
            print(f"  {p.get('tick'):>6} {dm:>10.6f} {dme:>10.6f}  {mark}")
        dms = [p["summary"]["dmin"] for p in probes
               if p.get("summary", {}).get("dmin") is not None]
        if dms:
            print(f"\n  dmin: 首次={dms[0]:.6f} 末次={dms[-1]:.6f} "
                  f"min={min(dms):.6f} max={max(dms):.6f}")
            print(f"  演化趋势: {'↑ 上升' if dms[-1] > dms[0] else '↓ 下降'}"
                  f"（{dms[-1] - dms[0]:+.6f}）")
    else:
        print("  (本次运行未触发 workmem_probe —— 检查 every 与总 tick 数)")

    print()
    print("=" * 72)
    print("Q3 · 睡眠巩固是否改变网络")
    print("=" * 72)
    sleeps = [p for p in policies if p.get("name") == "sleep"]
    ran = [p for p in sleeps if not p.get("summary", {}).get("skipped")]
    skip = [p for p in sleeps if p.get("summary", {}).get("skipped")]
    print(f"  实际睡眠: {len(ran)} 次 | 跳过: {len(skip)} 次")
    for p in ran[:5]:
        s = p.get("summary", {})
        sl = s.get("sleep", {})
        print(f"    tick {p.get('tick'):>4} → {str(sl)[:100]}")
    if skip:
        print(f"  跳过原因样本: {skip[0].get('summary', {}).get('reason')}")

    if len(snaps) >= 2:
        print(f"\n  快照对比（首 vs 末）:")
        a, b = snaps[0].get("state", {}), snaps[-1].get("state", {})
        for k in ("neurons", "tracts", "active"):
            print(f"    {k:10s}: {a.get(k)} → {b.get(k)}")

    print()
    print("=" * 72)
    print("观测汇总")
    print("=" * 72)
    convs = [r.get("convergence") for r in results if r.get("convergence") is not None]
    if convs:
        print(f"  收敛度: n={len(convs)} 均值={st.mean(convs):.4f} "
              f"min={min(convs):.4f} max={max(convs):.4f}")
    nulls = sum(1 for r in results if r.get("output") is None)
    print(f"  output None 数: {nulls}/{len(results)} （NORM-9 诚实输出）")
    by_src = {}
    for r in results:
        by_src[r.get("source")] = by_src.get(r.get("source"), 0) + 1
    print(f"  按来源: {by_src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
