"""
仿生 AI · M10-M12 P2 创新机制
================================
按《创造性改进提案》实现：
  M10 神经发生 (Neurogenesis)  —— 动态扩容 + 隔离观察期
  M11 集体潜意识层 (Collective) —— 跨实例共享 + 防同源污染
  M12 自我模型 (Self-Model)    —— 知道自己是谁 + 防幻觉

依赖: numpy
运行: python p2_mechanisms.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import json, hashlib, time

from bio_brain import (NervousSystem, Neuron, Tract, Group, Library,
                       norm, top_k_mean, clamp, ACTIVE_EPS,
                       D_DENDRITE, D_SOMA, D_AXON, NORM_CLIP)


# ============================================================
# M10: 神经发生（改进 8）—— 动态扩容 + 隔离观察
# ============================================================
@dataclass
class NewbornNeuron:
    """新生神经元：先隔离观察，验证有效才并入"""
    nid: int
    born_at: float
    matched_patterns: int = 0        # 成功匹配的模式数
    failed_patterns: int = 0         # 失败数
    isolated: bool = True            # ★隔离中：不参与主回路

    def success_rate(self) -> float:
        tot = self.matched_patterns + self.failed_patterns
        return self.matched_patterns / tot if tot else 0.0


class Neurogenesis:
    """触发条件：长期无法拟合 + 现有束都不匹配 → 生成新神经元/新束"""
    def __init__(self, ns: NervousSystem):
        self.ns = ns
        self.newborns: List[NewbornNeuron] = []
        self.promoted: List[int] = []
        self.rejected: List[int] = []
        self.failure_streak = 0
        self.TRIGGER_STREAK = 3          # 连续N次拟合失败才触发
        self.MATCH_THRESHOLD = 0.25      # 束匹配度门槛
        self.OBSERVATION_MIN = 3         # 观察期最少样本数
        self.PROMOTE_RATE = 0.6          # 成功率达标才并入
        self.REJECT_RATE = 0.3           # 低于此成功率则淘汰

    def try_fit(self, signal: np.ndarray) -> bool:
        """尝试用现有组织拟合信号；失败则累积失败计数"""
        best = 0.0
        for t in self.ns.tracts.values():
            if len(t.key) != len(signal):
                continue
            sim = abs(float(np.dot(t.key, signal))) / (norm(t.key) * norm(signal) + 1e-8)
            best = max(best, sim)
        if best >= self.MATCH_THRESHOLD:
            self.failure_streak = 0
            return True
        self.failure_streak += 1
        return False

    def should_grow(self) -> bool:
        """★触发条件：长期无法拟合"""
        return self.failure_streak >= self.TRIGGER_STREAK

    def grow_neuron(self, role: str = "inter") -> int:
        """生成新神经元（隔离态）"""
        nid = self.ns.add_neuron(role)
        self.newborns.append(NewbornNeuron(nid=nid, born_at=time.time()))
        return nid

    def grow_tract(self, member_ids: List[int]) -> int:
        """分裂/新建束来容纳新结构"""
        return self.ns.add_tract(member_ids)

    def observe(self, nid: int, success: bool):
        """观察期记录（新神经元的表现）"""
        for nb in self.newborns:
            if nb.nid == nid:
                if success:
                    nb.matched_patterns += 1
                else:
                    nb.failed_patterns += 1
                return

    def evaluate_newborns(self) -> dict:
        """★评估：够好的并入主回路，太差的淘汰"""
        promoted, rejected = [], []
        for nb in self.newborns[:]:
            total = nb.matched_patterns + nb.failed_patterns
            if total < self.OBSERVATION_MIN:
                continue                       # 观察期未满
            rate = nb.success_rate()
            if rate >= self.PROMOTE_RATE:
                nb.isolated = False            # 并入主回路
                promoted.append(nb.nid)
                self.promoted.append(nb.nid)
                self.newborns.remove(nb)
            elif rate < self.REJECT_RATE:
                # 淘汰：移除神经元
                for tid in list(self.ns.neurons[nb.nid].tract_ids):
                    if nb.nid in self.ns.tracts[tid].members:
                        self.ns.tracts[tid].members.remove(nb.nid)
                del self.ns.neurons[nb.nid]
                rejected.append(nb.nid)
                self.rejected.append(nb.nid)
                self.newborns.remove(nb)
        return {"promoted": promoted, "rejected": rejected,
                "pending": len(self.newborns)}

    def active_neurons(self) -> Set[int]:
        """参与主回路的神经元（排除隔离中的）"""
        isolated = {nb.nid for nb in self.newborns if nb.isolated}
        return set(self.ns.neurons.keys()) - isolated


# ============================================================
# M11: 集体潜意识层（改进 9）—— 共享 + 防同源污染
# ============================================================
@dataclass
class CollectiveEntry:
    """集体层条目：只收"经外部验证"的结论"""
    key: str                         # 内容指纹
    payload: dict                    # 匿名化的结构（不含个体标识）
    verified_by: str                 # 验证来源（oracle/user_action/outcome）
    contributed_at: float
    contributors: int = 1            # 多少个独立实例贡献过


class CollectiveLayer:
    """只读汇总层：个体 → 匿名化 → 验证 → 汇总 → 新个体继承"""
    VERIFIED_SOURCES = {"oracle", "user_action", "outcome"}   # ★自评禁止

    def __init__(self, storage_path: Optional[str] = None):
        self.entries: Dict[str, CollectiveEntry] = {}
        self.storage_path = storage_path
        self.stats = {"submitted": 0, "rejected_unverified": 0,
                      "rejected_duplicate_provenance": 0, "inherited": 0}

    def _fingerprint(self, payload: dict) -> str:
        """内容指纹（用于去重 + 防同源）"""
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def submit(self, payload: dict, source: str, instance_id: str) -> Optional[str]:
        """★个体提交：必须经外部验证（防同源污染）"""
        self.stats["submitted"] += 1
        if source not in self.VERIFIED_SOURCES:
            self.stats["rejected_unverified"] += 1
            return None                     # 自评/未验证 → 拒收
        key = self._fingerprint(payload)
        if key in self.entries:
            e = self.entries[key]
            e.contributors += 1             # 多实例独立验证 → 增强
        else:
            self.entries[key] = CollectiveEntry(
                key=key, payload=payload, verified_by=source,
                contributed_at=time.time(), contributors=1)
        return key

    def inherit(self, min_contributors: int = 1) -> List[dict]:
        """★新个体继承：只拿够可信的（默认至少1个独立验证）"""
        out = [e.payload for e in self.entries.values()
               if e.contributors >= min_contributors]
        self.stats["inherited"] += len(out)
        return out

    def save(self):
        if not self.storage_path:
            return
        data = {k: {"payload": e.payload, "verified_by": e.verified_by,
                    "contributors": e.contributors}
                for k, e in self.entries.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def load(self):
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.entries[k] = CollectiveEntry(
                    key=k, payload=v["payload"], verified_by=v["verified_by"],
                    contributed_at=time.time(), contributors=v["contributors"])
        except FileNotFoundError:
            pass


# ============================================================
# M12: 自我模型（改进 12）—— 知道自己是谁
# ============================================================
@dataclass
class SelfModel:
    """自我模型：能力认知 + 目标 + 学习史"""
    skills: Dict[str, float] = field(default_factory=dict)      # 能力→熟练度[0,1]
    goals: List[str] = field(default_factory=list)
    learned_recently: List[str] = field(default_factory=list)
    total_experiences: int = 0
    total_successes: int = 0

    # ---- 能力认知 ----
    def record_outcome(self, task_type: str, success: bool):
        self.total_experiences += 1
        if success:
            self.total_successes += 1
        cur = self.skills.get(task_type, 0.5)         # 先验 0.5
        # 指数平滑：新结果按 20% 权重更新
        self.skills[task_type] = clamp(cur * 0.8 + (1.0 if success else 0.0) * 0.2)

    def competence(self, task_type: str) -> float:
        """★能力估计（用于校准自信）"""
        return self.skills.get(task_type, 0.3)        # 未知任务给低分

    # ---- 自信校准（防幻觉核心）----
    def calibrated_confidence(self, task_type: str, raw_confidence: float) -> float:
        """★用能力认知校准原始自信 —— 不懂就别装懂"""
        comp = self.competence(task_type)
        # 若原始自信远超自身能力，打压它
        if raw_confidence > comp + 0.2:
            return comp * 0.7          # 打回能力水平
        return min(raw_confidence, comp + 0.2)

    def should_say_dont_know(self, task_type: str, raw_confidence: float,
                             threshold: float = 0.35) -> bool:
        """★防幻觉：能力不足且自信不高 → 说"我不知道" """
        return self.calibrated_confidence(task_type, raw_confidence) < threshold

    # ---- 目标连续性 ----
    def set_goal(self, goal: str):
        if goal not in self.goals:
            self.goals.append(goal)

    def complete_goal(self, goal: str):
        if goal in self.goals:
            self.goals.remove(goal)
            self.learned_recently.append(f"达成: {goal}")

    # ---- 学习史 ----
    def note_learning(self, item: str):
        self.learned_recently.append(item)
        if len(self.learned_recently) > 20:
            self.learned_recently.pop(0)

    def who_am_i(self) -> dict:
        """★自我描述"""
        acc = (self.total_successes / self.total_experiences
               if self.total_experiences else 0.0)
        return {
            "擅长": [k for k, v in sorted(self.skills.items(),
                                        key=lambda x: -x[1])[:3] if v > 0.6],
            "不擅长": [k for k, v in sorted(self.skills.items(),
                                          key=lambda x: x[1])[:3] if v < 0.4],
            "当前目标": list(self.goals),
            "整体成功率": round(acc, 3),
            "经历数": self.total_experiences,
        }


# ============================================================
# 演示
# ============================================================
def demo():
    print("=" * 66)
    print("  仿生 AI · M10-M12 P2 创新机制验证")
    print("=" * 66)

    # ---- M10 神经发生 ----
    print("\n[M10] 神经发生（动态扩容 + 隔离观察）...")
    ns = NervousSystem(seed=42)
    n0 = [ns.add_neuron() for _ in range(4)]
    ns.add_tract(n0)
    ng = Neurogenesis(ns)

    # 喂一个"现有组织无法拟合"的信号
    novel = np.random.default_rng(1).normal(0, 1, D_SOMA)
    for i in range(3):
        ok = ng.try_fit(novel)
        print(f"  拟合尝试{i+1}: {'成功' if ok else '失败'} (failure_streak={ng.failure_streak})")
    print(f"  是否触发神经发生: {ng.should_grow()}  ★连续3次失败应触发")

    # 生成新神经元并观察
    nb = ng.grow_neuron()
    t_new = ng.grow_tract(n0[:2] + [nb])
    print(f"  新生神经元 {nb} 已创建（隔离中）")
    print(f"  隔离验证: {nb} 在活跃集? {nb in ng.active_neurons()}  ★应False")

    # 观察期
    for s in [True, True, True, False]:
        ng.observe(nb, s)
    res = ng.evaluate_newborns()
    print(f"  观察后评估: {res}  ★成功率0.75≥0.6应并入")

    # 测试淘汰
    nb2 = ng.grow_neuron()
    ng.grow_tract(n0[:2] + [nb2])
    for s in [False, False, False, False]:
        ng.observe(nb2, s)
    res2 = ng.evaluate_newborns()
    print(f"  劣质新生评估: {res2}  ★成功率0应淘汰")
    print(f"  淘汰后神经元数: {len(ns.neurons)}")

    # ---- M11 集体层 ----
    print("\n[M11] 集体潜意识层（共享 + 防同源污染）...")
    cl = CollectiveLayer()
    # 提交未验证(自评) → 应拒收
    r1 = cl.submit({"pattern": "A"}, source="self_eval", instance_id="inst1")
    print(f"  提交自评: {r1}  ★应None(拒收)")
    # 提交验证过的 → 应接受
    r2 = cl.submit({"pattern": "A"}, source="oracle", instance_id="inst1")
    print(f"  提交oracle: {r2}  ★应接受")
    # 第二个实例独立验证同一结论 → contributors+1
    cl.submit({"pattern": "A"}, source="oracle", instance_id="inst2")
    cl.submit({"pattern": "B"}, source="user_action", instance_id="inst2")
    print(f"  统计: {cl.stats}")
    print(f"  新个体继承: {len(cl.inherit())} 条")
    print(f"  严格继承(≥2实例): {len(cl.inherit(min_contributors=2))} 条  ★应1条")

    # ---- M12 自我模型 ----
    print("\n[M12] 自我模型（能力认知 + 防幻觉）...")
    sm = SelfModel()
    # 模拟经历
    for _ in range(8):
        sm.record_outcome("写代码", success=True)
    for _ in range(7):
        sm.record_outcome("学外语", success=False)
    sm.set_goal("完成仿生AI v7")
    sm.note_learning("学会了神经发生")

    print(f"  写代码能力: {sm.competence('写代码'):.2f}")
    print(f"  学外语能力: {sm.competence('学外语'):.2f}")
    print(f"  未知任务能力: {sm.competence('量子物理'):.2f}")

    # 防幻觉测试：不懂却装懂
    raw = 0.95
    cal = sm.calibrated_confidence("学外语", raw)
    print(f"  学外语原始自信{raw} → 校准后{cal:.2f}  ★应被打压")
    print(f"  该说'我不知道'? {sm.should_say_dont_know('学外语', raw)}  ★应True")
    cal2 = sm.calibrated_confidence("写代码", 0.85)
    print(f"  写代码自信0.85 → 校准后{cal2:.2f}  ★应保持")
    print(f"  who_am_i(): {sm.who_am_i()}")

    print("\n✅ M10-M12 全部跑通")


if __name__ == "__main__":
    demo()
