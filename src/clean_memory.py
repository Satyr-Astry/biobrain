"""
记忆库清理 (clean_memory.py)
=============================
把低质量的网络学习残留清掉，只保留 LLM 学习的优质知识。

判断标准（按质量分）：
  · ai_llm / ai_distill  → 高分（LLM 提炼，含自检）
  · oracle / user_action → 高分（可信来源）
  · web_wiki / web_api   → 中分（需正文）
  · web_search / web_page→ 低分（摘要碎片，且多是词典释义）

用法：
    python clean_memory.py --dry-run     # 只看会删什么
    python clean_memory.py --apply       # 真的清理
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
MEM = HERE / "agent_memory.json"

# 质量分（越高越可信）
SOURCE_SCORE = {
    "oracle": 1.0, "user_action": 1.0, "outcome": 1.0,
    "ai_llm": 0.9, "ai_distill": 0.85,
    "web_wiki": 0.6, "web_api": 0.6,
    "web_search": 0.3, "web_page": 0.3,
}

# 明显垃圾的模式（词典释义、无关内容）
JUNK_PATTERNS = [
    r"拼音为|注音符号|词性为|部首|笔画",          # 词典释义
    r"^搜索:.*\|.*百度百科",                      # 百科词条页标题
    r"文章浏览阅读\d+",                            # CSDN 噪声头
    r"^\d{4}年\d{1,2}月\d{1,2}日 ·",               # 日期开头的摘要
]


def score(m: dict) -> float:
    s = SOURCE_SCORE.get(m.get("source", ""), 0.4)
    ans = m.get("answer", "")
    # 长度加成（越长越可能是真知识）
    if len(ans) >= 300:
        s += 0.15
    elif len(ans) < 100:
        s -= 0.2
    # 垃圾模式扣分
    for pat in JUNK_PATTERNS:
        if re.search(pat, ans[:200]):
            s -= 0.5
            break
    return max(0.0, min(1.0, s))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="记忆库清理")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not MEM.exists():
        print(f"记忆库不存在: {MEM}")
        return

    data = json.loads(MEM.read_text(encoding="utf-8"))
    print("=" * 72)
    print("  记忆库清理")
    print("=" * 72)
    print(f"  当前: {len(data)} 条")
    print(f"  阈值: 质量分 < {args.threshold} 的删除")
    print()

    keep, drop = [], []
    for m in data:
        s = score(m)
        m["_score"] = round(s, 2)
        (keep if s >= args.threshold else drop).append(m)

    print(f"  保留: {len(keep)} 条 | 删除: {len(drop)} 条")
    print()

    if drop:
        print("  将删除的低质量条目：")
        for m in drop[:12]:
            print(f"    [{m['_score']}] [{m['source']}] {m['answer'][:60]}")
        if len(drop) > 12:
            print(f"    ... 还有 {len(drop)-12} 条")
    print()

    if keep:
        print("  保留的高质量条目（前8条）：")
        for m in sorted(keep, key=lambda x: -x["_score"])[:8]:
            print(f"    [{m['_score']}] [{m['source']}] {len(m['answer'])}字 "
                  f"{m['text'][:40]}")

    if args.apply:
        # 去掉临时字段
        out = [{k: v for k, v in m.items() if k != "_score"} for m in keep]
        MEM.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print()
        print(f"  ✓ 已清理，剩余 {len(out)} 条")
    else:
        print()
        print("  （dry-run 模式，加 --apply 才真的写入）")


if __name__ == "__main__":
    main()
