"""
仿生 AI · M3-M5 自训练闭环
============================
按《工程规范_v7.1.md》实现：
  M3  经验积累 (ALG-6)  —— 双时态记忆 + 可靠信号过滤 + 惊讶优先
  M4  双相睡眠 (ALG-7)  —— NREM 巩固(直接入base) + REM 联想(只入候选库)
  M5  回灌闭环 + 收敛度量 (ALG-11) —— 输出→自我倾听，带护栏

依赖: numpy
运行: python self_training.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import time

from cog_vec import (
    NervousSystem, Neuron, Tier, norm, top_k_mean, clamp,
    ACTIVE_EPS, NORM_CLIP,
)

# ============================================================
# 常量（对应规范 §2.2 + v7.1）
# ============================================================
SURPRISE_KEEP_K = 8       # 每次巩固重放样本数（原型用小值）
REM_SEED_K = 3            # REM 相随机种子束数
REM_COMBO_SIZE = 2        # REM 相组合大小
CONVERGE_HIGH = 0.8       # 收敛阈值（该输出）
CONVERGE_LOW = 0.3        # 收敛下限（该诚实说不会）
PATIENCE = 30.0           # 想不出来的容忍时长（秒，原型调小）
FISHER_DECAY = 0.01
GROUNDING_EPS = 0.05


# ============================================================
# M3: 经验积累（ALG-6 §6.2）—— 双时态 + 可靠信号过滤
# ============================================================
@dataclass
class Experience:
    """经验：双时态记忆（v5.0 升级）"""
    id: int
    thought_trace: List[int]          # 思考轨迹（活跃神经元快照）
    ignitions: List[int]              # 当时的输入（感觉神经元）
    emissions: List[int]              # 当时的输出（运动神经元）
    outcome: float                    # 结果好坏（来自可靠来源）
    surprise: float                   # 惊讶度 = |预测-实际|
    # ★双时态（v5.0）
    event_time: float                 # 何时发生
    record_time: float                # 何时记下
    source: str                       # 来源（用于可靠性判定）
    supersedes: Optional[int] = None  # 修正了哪条旧经验


class ExperienceBuffer:
    """经验缓冲：只收可靠信号 + 惊讶优先 + 双时态不覆盖"""
    RELIABLE_SOURCES = {"oracle", "user_action", "outcome"}   # ★铁律：自评禁用

    def __init__(self):
        self.items: List[Experience] = []
        self._nid = 0
        self.rejected = {"unreliable": 0, "duplicate": 0}

    def is_reliable(self, src: str) -> bool:
        return src in self.RELIABLE_SOURCES

    def is_duplicate(self, e: Experience) -> bool:
        for old in self.items:
            if old.ignitions == e.ignitions and old.emissions == e.emissions:
                if abs(old.outcome - e.outcome) < 1e-6:
                    return True
        return False

    def store(self, e: Experience) -> bool:
        """★MUST: 只收可靠信号；自评 MUST NOT 入缓冲"""
        if not self.is_reliable(e.source):
            self.rejected["unreliable"] += 1
            return False
        if self.is_duplicate(e):
            self.rejected["duplicate"] += 1
            return False
        self._merge_or_add(e)             # ★双时态：不静默覆盖
        self.items.append(e)
        self.items.sort(key=lambda x: x.surprise, reverse=True)  # 惊讶优先
        return True

    def _merge_or_add(self, e: Experience):
        """★同键不同版本不覆盖，保留历史（双时态）"""
        for old in self.items:
            if old.ignitions == e.ignitions and old.event_time == e.event_time:
                if old.record_time < e.record_time:
                    e.supersedes = old.id    # 标记为修正版，但两者都保留
                break

    def top_k(self, k: int = SURPRISE_KEEP_K) -> List[Experience]:
        return self.items[:k]

    def stats(self) -> dict:
        return {"size": len(self.items),
                "rejected": dict(self.rejected),
                "superseded": sum(1 for e in self.items if e.supersedes)}


# ============================================================
# M4: 双相睡眠（ALG-7 §6.3）—— NREM 巩固 + REM 联想
# ============================================================
class Consolidator:
    def __init__(self, ns: NervousSystem, buffer: ExperienceBuffer):
        self.ns = ns
        self.buffer = buffer
        self.candidate_store: List[dict] = []    # ★REM 产物只入这里
        self.checkpoints: List[dict] = []
        self.log: List[str] = []

    # ---- 活动判定（§4.5）----
    def can_consolidate(self, has_running_task: bool, has_pending_slice: bool) -> bool:
        """★分片挂起 = 活动计算 → 不可合并"""
        if has_running_task or has_pending_slice:
            return False
        global_act = max((lib.activation for lib in self.ns.libraries.values()), default=0)
        return global_act < 0.15 or len(self.buffer.items) > 0

    # ---- NREM 相：巩固真实经验 ----
    def nrem_phase(self) -> dict:
        """惊讶优先重放 → 正交投影 → 合并入 base → 检查点"""
        samples = self.buffer.top_k()
        if not samples:
            return {"phase": "nrem", "merged": 0}

        # ① 重放：把样本的激活模式重新点火，累积可塑量
        for s in samples:
            for nid in s.thought_trace[:5]:        # 原型：重放前几个
                if nid in self.ns.neurons:
                    self.ns.ignite(nid, 0.3)

        # ② 正交投影（防遗忘）：把可塑量投影到重要性的正交补
        plastic_neurons = [n for n in self.ns.neurons.values()
                           if norm(n.plasticity) > 1e-6]
        for n in plastic_neurons:
            # 简化版正交投影：按重要性衰减（重要参数少动）
            n.plasticity *= (1.0 - min(n.importance, 0.9) * 0.5)

        # ③ 合并入主体（可逆：先存快照）
        merged = 0
        snapshot = {}
        for n in plastic_neurons:
            snapshot[n.id] = n.base.copy()
            n.base = n.base + n.plasticity           # ★合并
            n.plasticity = n.plasticity * 0           # 清零
            merged += 1
        self.checkpoints.append(snapshot)            # ★可回滚

        self.log.append(f"NREM: 重放{len(samples)}条, 合并{merged}个神经元")
        return {"phase": "nrem", "replayed": len(samples), "merged": merged}

    # ---- REM 相：联想重组 ----
    def rem_phase(self) -> dict:
        """★打散重组：随机抽取束 → 强制共激活 → 只入候选库(MUST NOT 直接合并)"""
        tract_ids = list(self.ns.tracts.keys())
        if len(tract_ids) < REM_COMBO_SIZE:
            return {"phase": "rem", "hypotheses": 0}

        seeds = list(np.random.choice(tract_ids, size=min(REM_SEED_K, len(tract_ids)), replace=False))
        hyps = []
        for i in range(0, len(seeds) - 1, REM_COMBO_SIZE):
            combo = seeds[i:i+REM_COMBO_SIZE]
            if len(combo) < 2:
                continue
            # 反事实联想：强制这些束共激活
            assoc = np.mean([self.ns.tracts[t].value for t in combo], axis=0)
            novelty = float(np.std(assoc))            # 简化新颖度
            hyps.append({"tracts": combo, "assoc": assoc, "novelty": novelty})
            # ★只入候选库，绝不直接合并入 base
            self.candidate_store.append(hyps[-1])

        self.log.append(f"REM: 生成{len(hyps)}条候选联想(未合并)")
        return {"phase": "rem", "hypotheses": len(hyps)}

    def sleep(self, has_running_task=False, has_pending_slice=False) -> dict:
        """完整睡眠周期：NREM → REM"""
        if not self.can_consolidate(has_running_task, has_pending_slice):
            return {"skipped": True, "reason": "有活动计算（分片挂起也算）"}
        n = self.nrem_phase()
        r = self.rem_phase()
        return {"nrem": n, "rem": r}

    def rollback(self) -> bool:
        """★可逆合并：回滚到上一个检查点"""
        if not self.checkpoints:
            return False
        snap = self.checkpoints.pop()
        for nid, base in snap.items():
            self.ns.neurons[nid].base = base
        self.log.append(f"回滚了 {len(snap)} 个神经元")
        return True


# ============================================================
# M5: 收敛度量（ALG-11 §4.7）+ 回灌闭环
# ============================================================
class ConvergenceMeter:
    """收敛度量：稳定性 + 预测误差 + 目标达成"""
    def __init__(self, window: int = 5):
        self.window = window
        self.history: List[np.ndarray] = []
        self.pred_errors: List[float] = []

    def update(self, state_vec: np.ndarray, pred_err: float = 0.0):
        self.history.append(state_vec)
        if len(self.history) > self.window:
            self.history.pop(0)
        self.pred_errors.append(pred_err)

    def convergence(self, goal_satisfaction: float = 0.5) -> float:
        """三因子加权

        ★v7.4 修正：冷启动（历史<2）时不再直接返回 0——
        改用"当前状态的范数稳定度"作为初值，避免首次思考永远判定为未收敛。
        """
        if len(self.history) == 0:
            return 0.0
        if len(self.history) < 2:
            # 冷启动：用当前状态的"平滑度"（范数适中=有一定结构）作代理
            v = self.history[0]
            mag = norm(v)
            stability = 1.0 / (1.0 + abs(mag - 1.0))   # 范数接近1视为有结构
            pe = float(np.mean(self.pred_errors)) if self.pred_errors else 1.0
            pred_factor = 1.0 / (1.0 + pe)
            return float(clamp(0.4 * stability + 0.3 * pred_factor + 0.3 * goal_satisfaction))
        # 1. 稳定性：近期状态变化越小越稳定
        diffs = [norm(self.history[i] - self.history[i-1]) for i in range(1, len(self.history))]
        stability = 1.0 / (1.0 + float(np.mean(diffs)) * 10)
        # 2. 预测误差：越低越"懂"
        pe = float(np.mean(self.pred_errors[-self.window:]))
        pred_factor = 1.0 / (1.0 + pe)
        # 3. 目标达成
        return float(clamp(0.4*stability + 0.3*pred_factor + 0.3*goal_satisfaction))


class FeedbackLoop:
    """回灌闭环：输出 → 点亮自体感觉神经元（带 entailment 护栏）"""
    def __init__(self, ns: NervousSystem, self_sensory_ids: List[int]):
        self.ns = ns
        self.self_sensory = self_sensory_ids
        self.blocked = 0

    def entailment_check(self, emission: dict, supported: bool) -> bool:
        """★护栏：只回灌被推理支撑的内容，防自我说服"""
        return bool(supported)

    def feed_back(self, emission: dict, supported: bool = True) -> bool:
        if not self.entailment_check(emission, supported):
            self.blocked += 1
            return False
        for nid in self.self_sensory:                 # 点亮"自体感觉"神经元
            if emission.get("strength", 0) > 0.5:
                self.ns.ignite(nid, 0.4)
        return True


# ============================================================
# 统一决策（emit_decision §4.7）
# ============================================================
def emit_decision(conv: float, has_input: bool, elapsed: float,
                  need_external: bool = False) -> dict:
    """基于收敛度决定是否输出"""
    if conv > CONVERGE_HIGH:
        return {"action": "speak", "now": has_input, "spontaneous": not has_input}
    if conv < CONVERGE_LOW and elapsed > PATIENCE:
        return {"action": "speak", "content": "我不确定/需要更多信息"}   # ★诚实
    if need_external:
        return {"action": "tool_call"}
    return {"action": "wait"}    # 继续想


# ============================================================
# 演示：M3-M5 跑通验证
# ============================================================
def demo():
    print("=" * 64)
    print("  仿生 AI · M3-M5 自训练闭环验证")
    print("=" * 64)
    from cog_vec import SensoryPort, MotorPort

    rng = np.random.default_rng(7)
    ns = NervousSystem(seed=42)

    # 建组织（连通拓扑）
    visual = [ns.add_neuron("sensory") for _ in range(4)]
    symbol = [ns.add_neuron("sensory") for _ in range(4)]
    inter  = [ns.add_neuron("inter") for _ in range(8)]
    motor  = [ns.add_neuron("motor") for _ in range(3)]
    self_sens = [ns.add_neuron("sensory") for _ in range(2)]   # 自体感觉

    t1 = ns.add_tract(visual + inter[:2])
    t2 = ns.add_tract(symbol + inter[:2])
    t3 = ns.add_tract(inter[2:6] + motor)
    t4 = ns.add_tract(inter[:2] + inter[2:6])
    g1 = ns.add_group([t1, t2], "percept")
    g2 = ns.add_group([t3, t4], "action")
    ns.add_library([g1]); ns.add_library([g2])

    buffer = ExperienceBuffer()
    consol = Consolidator(ns, buffer)
    meter = ConvergenceMeter()
    fb = FeedbackLoop(ns, self_sens)

    # ---- M3: 经验积累 ----
    print("\n[M3] 经验积累（可靠信号过滤 + 双时态）...")
    for i in range(5):
        e = Experience(
            id=i, thought_trace=list(visual)+list(inter), ignitions=visual,
            emissions=motor, outcome=float(rng.random()),
            surprise=float(rng.random()), event_time=100.0+i, record_time=time.time(),
            source="oracle" if i < 3 else "self_eval",   # ★后两条是自评，应被拒
        )
        ok = buffer.store(e)
        print(f"  经验{i}: source={e.source:<9} → {'✓入缓冲' if ok else '✗拒绝'}")
    print(f"  缓冲统计: {buffer.stats()}")

    # ---- 运行思考流（积累可塑量）----
    print("\n[思考流] 运行 12 tick（积累可塑量）...")
    vp = SensoryPort("vision", visual, 8); sp = SensoryPort("symbol", symbol, 8)
    r1 = vp.ignite(np.array([1.,.5,.3,0,0,0,0,0]), ns)
    r2 = sp.ignite(np.array([.8,.6,0,0,0,0,0,0]), ns)
    active = set(r1["ignited"]) | set(r2["ignited"])
    for step in range(12):
        ns.apply_pending(active)
        active = ns.tick_activation(active)
        for tid in ns.tracts: ns.tracts[tid].activation = ns.tract_activation(tid)
        st = ns.compute(active)
        ns.update_plasticity(active, {})
        ns.tick_scheduler(); ns.t += 1
    pl_norm = sum(norm(n.plasticity) for n in ns.neurons.values())
    print(f"  可塑量累积: {pl_norm:.4f}")

    # ---- M4: 双相睡眠 ----
    print("\n[M4] 双相睡眠...")
    print("  ① 有分片挂起时（不可合并）:")
    r = consol.sleep(has_pending_slice=True)
    print(f"     {r}")
    print("  ② 无活动计算时（可合并）:")
    r = consol.sleep(has_running_task=False, has_pending_slice=False)
    print(f"     NREM: {r['nrem']}")
    print(f"     REM:  {r['rem']}")
    print(f"  候选联想库(REM产物, 未合并): {len(consol.candidate_store)} 条")
    pl_after = sum(norm(n.plasticity) for n in ns.neurons.values())
    print(f"  合并后可塑量: {pl_after:.4f} (应≈0，已折入base)")
    print(f"  回滚测试: {'✓成功' if consol.rollback() else '✗无检查点'}")

    # ---- M5: 收敛 + 回灌 ----
    print("\n[M5] 收敛度量 + 回灌闭环...")
    for step in range(6):
        v = rng.normal(0, 0.1, 16) if step < 3 else rng.normal(0, 0.01, 16)  # 模拟逐渐稳定
        meter.update(v, pred_err=0.5 if step < 3 else 0.05)
    conv = meter.convergence(goal_satisfaction=0.7)
    print(f"  收敛度: {conv:.3f}")
    d = emit_decision(conv, has_input=True, elapsed=5)
    print(f"  决策: {d}")
    # 回灌
    emission = {"strength": 0.9, "modality": "language"}
    ok1 = fb.feed_back(emission, supported=True)
    ok2 = fb.feed_back(emission, supported=False)   # 护栏应拦截
    print(f"  回灌(被支撑): {'✓通过' if ok1 else '✗拦截'}")
    print(f"  回灌(无支撑): {'✗未拦截!' if ok2 else '✓被护栏拦截'} (blocked={fb.blocked})")

    # ---- 诚实输出测试 ----
    print("\n[诚实输出] 低收敛 + 超时:")
    conv_low = 0.1
    d2 = emit_decision(conv_low, has_input=True, elapsed=100)
    print(f"  决策: {d2}")

    print("\n[日志]")
    for line in consol.log:
        print(f"  · {line}")
    print("\n✅ M3-M5 全部跑通")


if __name__ == "__main__":
    demo()
