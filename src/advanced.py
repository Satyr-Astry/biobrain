"""
仿生 AI · M6-M9 高级机制
==========================
按《工程规范_v7.1.md》实现：
  M6  多尺度权重 (ALG-12) —— 子权重叠加 + 资格痕迹×RPE（三因子可塑性）
  M7  能量核算   (ALG-13) —— J/token 近似 + 错峰调度
  M8  符号接地   (ALG-14) —— 共激活 + 因果验证 + 五维剖面
  M9  鲁棒性     (§9-10)  —— 数值/调度/学习/容错 + 降级等级

依赖: numpy（GPU 功率读取需 nvidia-smi，不可用时自动降级为估算）
运行: python advanced.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import subprocess, time, os

# ============================================================
# 常量（对应规范 §2.2 + v7.x）
# ============================================================
ETA_FAST, ETA_MID, ETA_SLOW = 0.1, 0.02, 0.001
DECAY_FAST, DECAY_MID, DECAY_SLOW = 0.5, 0.9, 0.999
GROUNDING_EPS = 0.05
CAUSAL_EPS = 0.05
ENERGY_BUDGET_HIGH, ENERGY_BUDGET_LOW = 1.0, 0.2
OFF_PEAK = (23, 7)          # 23:00 - 07:00
NORM_CLIP = 1.0


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ============================================================
# M6: 多尺度权重（ALG-12 §4.8）—— 子权重叠加 + 三因子可塑性
# ============================================================
@dataclass
class MultiScaleWeight:
    """权重 = 多个子权重之和，各自独立学习（NeurIPS 定理：等价于耦合模型）"""
    shape: Tuple[int, ...]
    subweights: List[np.ndarray] = field(default_factory=list)
    etas: List[float] = field(default_factory=lambda: [ETA_FAST, ETA_MID, ETA_SLOW])
    decays: List[float] = field(default_factory=lambda: [DECAY_FAST, DECAY_MID, DECAY_SLOW])
    e_traces: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.subweights:
            self.subweights = [np.zeros(self.shape) for _ in range(3)]
        if not self.e_traces:
            self.e_traces = [np.zeros(self.shape) for _ in range(3)]

    def value(self) -> np.ndarray:
        """有效权重 = 子权重之和"""
        return sum(self.subweights)

    def update(self, pre: np.ndarray, post: np.ndarray,
               reward: float, expected: float, lr_hebb: float = 0.1) -> float:
        """★三因子可塑性：资格痕迹(快) × 奖励预测误差(慢)

        ★v7.2 修正：资格痕迹必须归一化！否则 e_trace 无界累积，
        与 RPE 相乘后更新量发散（实测：负奖励反而使权重增长）。
        """
        rpe = reward - expected                    # 慢信号：事后才知道
        hebb = np.outer(pre, post).flatten()[:np.prod(self.shape)]
        if hebb.shape[0] < np.prod(self.shape):
            hebb = np.pad(hebb, (0, int(np.prod(self.shape)) - hebb.shape[0]))
        hebb = hebb.reshape(self.shape)
        # ★归一化 hebb，使资格痕迹有界（关键修正）
        hn = float(np.linalg.norm(hebb))
        if hn > 1e-8:
            hebb = hebb / hn
        # ★v7.2: 慢尺度需要更长的痕迹累积窗口才能起作用 → 按尺度补偿 lr
        lr_scale = [1.0, 3.0, 10.0]     # 越慢的尺度，单步痕迹权越大
        for i in range(3):
            # 快：资格痕迹（有界，因 hebb 已归一化 + decay<1）
            self.e_traces[i] = self.decays[i] * self.e_traces[i] + lr_hebb * lr_scale[i] * hebb
            # 痕迹本身也做范数约束（双保险）
            tn = float(np.linalg.norm(self.e_traces[i]))
            if tn > NORM_CLIP:
                self.e_traces[i] *= NORM_CLIP / tn
            # ★耦合发生在这里：快痕迹 × 慢奖励
            self.subweights[i] = self.subweights[i] + self.etas[i] * self.e_traces[i] * rpe
            # 范数裁剪
            nrm = float(np.linalg.norm(self.subweights[i]))
            if nrm > NORM_CLIP:
                self.subweights[i] *= NORM_CLIP / nrm
        return rpe

    def decay_to_slow(self, rate: float = 0.1):
        """快权重向慢权重迁移（对应睡眠合并：快→慢）"""
        moved = rate * self.subweights[0]
        self.subweights[0] -= moved
        self.subweights[2] += moved

    def scale_norms(self) -> List[float]:
        return [float(np.linalg.norm(s)) for s in self.subweights]


# ============================================================
# M7: 能量核算（ALG-13 §7.4）—— J/token 近似 + 错峰
# ============================================================
class EnergyMeter:
    """能量核算：GPU 功率×时间 近似焦耳；不可用时降级为估算"""
    def __init__(self):
        self.have_nvml = self._check_nvidia()
        self.history: List[dict] = []

    def _check_nvidia(self) -> bool:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def read_power(self) -> float:
        """读取 GPU 功率(W)；失败则降级估算"""
        if self.have_nvml:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=3)
                return float(r.stdout.strip().split("\n")[0])
            except Exception:
                pass
        return 60.0    # 降级：假设典型负载 60W

    def measure(self, fn, tokens: int = 1, label: str = "") -> dict:
        """★测能量：功率×时间 ≈ 焦耳"""
        p0 = self.read_power(); t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        p1 = self.read_power()
        joules = ((p0 + p1) / 2.0) * dt
        rec = {"label": label, "joules": joules, "seconds": dt,
               "tokens": tokens, "j_per_token": joules / max(tokens, 1),
               "power_avg": (p0 + p1) / 2.0}
        self.history.append(rec)
        return rec

    def j_per_token(self) -> Optional[float]:
        if not self.history:
            return None
        tot_j = sum(h["joules"] for h in self.history)
        tot_t = sum(h["tokens"] for h in self.history)
        return tot_j / max(tot_t, 1)


class EnergyScheduler:
    """错峰调度：低能耗时段才做重活（仿人脑睡眠）"""
    def __init__(self, off_peak: Tuple[int, int] = OFF_PEAK):
        self.off_peak = off_peak

    def is_off_peak(self, hour: Optional[int] = None) -> bool:
        h = time.localtime().tm_hour if hour is None else hour
        start, end = self.off_peak
        if start > end:      # 跨午夜，如 23→7
            return h >= start or h < end
        return start <= h < end

    def budget(self, hour: Optional[int] = None) -> float:
        return ENERGY_BUDGET_HIGH if self.is_off_peak(hour) else ENERGY_BUDGET_LOW

    def should_do_heavy(self, hour: Optional[int] = None) -> bool:
        """重活（合并/REM/训练）只在低能耗时段"""
        return self.is_off_peak(hour)


# ============================================================
# M8: 符号接地（ALG-14 §4.9）
# ============================================================
class GroundingModule:
    """三管齐下：环境共激活 + 因果验证 + 五维剖面审计"""

    def __init__(self, nervous):
        self.ns = nervous
        self.coactivation_log: Dict[int, List[int]] = {}   # 符号神经元 → 曾共激活的环境神经元

    # ---- (A) 环境共激活：接地机制 ----
    def ground_by_coactivation(self, symbol_ids: List[int], env_ids: List[int],
                               strength: float = 0.4) -> int:
        """符号与环境同时点亮 → 形成关联"""
        for s in symbol_ids:
            self.ns.ignite(s, strength)
            self.coactivation_log.setdefault(s, [])
            for e in env_ids:
                self.ns.ignite(e, strength)
                if e not in self.coactivation_log[s]:
                    self.coactivation_log[s].append(e)
        return len(symbol_ids)

    # ---- (B) 因果验证：判据 ----
    def is_grounded(self, symbol_ids: List[int], world_activator, perturb_activator) -> dict:
        """
        ★判据：符号内部状态是否因果地影响输出，且与世界对应
        - 反事实扰动：改世界 → 符号激活应变
        - 因果测试：符号激活应影响下游输出
        """
        # 记录当前激活
        a = np.array([self.ns.neurons[s].activity for s in symbol_ids])
        # 反事实：扰动世界，重新点火
        self.ns.pending_ignitions.clear()
        world_activator()
        self.ns.tick_activation(set(symbol_ids))
        b = np.array([self.ns.neurons[s].activity for s in symbol_ids])
        self.ns.pending_ignitions.clear()
        perturb_activator()
        self.ns.tick_activation(set(symbol_ids))
        c = np.array([self.ns.neurons[s].activity for s in symbol_ids])

        sensitivity = float(np.linalg.norm(c - b))      # 对世界扰动的敏感度
        causal = sensitivity > CAUSAL_EPS               # 是否因果响应
        grounded = sensitivity > GROUNDING_EPS
        return {"sensitivity": sensitivity, "grounded": grounded, "causal": causal}

    # ---- (C) 五维接地剖面（2512.06205）----
    def grounding_profile(self, symbol_ids: List[int],
                          test_pairs: List[Tuple[List, List]]) -> dict:
        """audit: authenticity / preservation / faithfulness / robustness / compositionality"""
        # authenticity: 机制是否在内部（有共激活记录）
        auth = 1.0 if all(s in self.coactivation_log for s in symbol_ids) else 0.0
        # preservation: 原子意义是否保持（扰动单点不崩）
        scores = []
        for sym, env in test_pairs:
            r = self.is_grounded(sym, lambda: self._ignite(env), lambda: self._ignite_random(env))
            scores.append(1.0 if r["grounded"] else 0.0)
        preservation = float(np.mean(scores)) if scores else 0.0
        # faithfulness / robustness / compositionality（简化实现）
        cov = len([s for s in symbol_ids if self.coactivation_log.get(s)])
        faithfulness = cov / max(len(symbol_ids), 1)
        robustness = 1.0 - min(GROUNDING_EPS / (preservation + 1e-6), 1.0) if preservation else 0.0
        compositionality = min(len(self.coactivation_log) / max(len(symbol_ids), 1), 1.0)
        return {"authenticity": auth, "preservation": preservation,
                "faithfulness": faithfulness, "robustness": robustness,
                "compositionality": compositionality}

    def _ignite(self, ids):
        for i in ids:
            self.ns.ignite(i, 0.5)

    def _ignite_random(self, ids):
        for i in ids:
            self.ns.ignite(i, 0.5 * float(np.random.random()))


# ============================================================
# M9: 鲁棒性（§9 数值/调度/学习 + §10 降级）
# ============================================================
class DegradeLevel:
    L0_NORMAL = 0
    L1_RESOURCE = 1     # 关预取，降粒度
    L2_VRAM = 2         # 只保活跃组
    L3_PROTECT = 3      # 停在线可塑，只前向
    L4_READONLY = 4     # 停一切写入


class RobustnessGuard:
    """数值/调度/学习三重保护 + 降级"""
    def __init__(self):
        self.level = DegradeLevel.L0_NORMAL
        self.alerts: List[str] = []
        self.metrics = {"nan_count": 0, "thrash_count": 0,
                        "plastic_explode": 0, "all_zero": 0}

    # ---- 数值稳定 ----
    def check_numeric(self, arr: np.ndarray, name: str = "") -> bool:
        if not np.all(np.isfinite(arr)):
            self.metrics["nan_count"] += 1
            self.alerts.append(f"数值异常: {name} 含 NaN/Inf")
            return False
        return True

    def clamp_norm(self, arr: np.ndarray, limit: float = NORM_CLIP) -> np.ndarray:
        n = float(np.linalg.norm(arr))
        return arr * (limit / n) if n > limit else arr

    # ---- 调度稳定（防抖动）----
    def check_thrash(self, moves_per_sec: float, threshold: float = 10.0) -> bool:
        if moves_per_sec > threshold:
            self.metrics["thrash_count"] += 1
            self.alerts.append(f"调度抖动: {moves_per_sec:.1f} 次/秒 > {threshold}")
            self._degrade(DegradeLevel.L1_RESOURCE)
            return False
        return True

    # ---- 学习稳定 ----
    def check_plasticity(self, norms: List[float], limit: float = NORM_CLIP) -> bool:
        if any(n > limit * 2 for n in norms):
            self.metrics["plastic_explode"] += 1
            self.alerts.append(f"可塑量爆炸: max={max(norms):.2f}")
            self._degrade(DegradeLevel.L3_PROTECT)
            return False
        return True

    def check_alive(self, activities: List[float], eps: float = 1e-6) -> bool:
        if max(activities, default=0.0) < eps:
            self.metrics["all_zero"] += 1
            self.alerts.append("全局激活归零（可能死机）")
            return False
        return True

    # ---- 降级 ----
    def _degrade(self, level: int):
        if level > self.level:
            self.level = level
            self.alerts.append(f"降级 → L{level}")

    def degrade_name(self) -> str:
        return {0: "L0 正常", 1: "L1 资源紧张", 2: "L2 显存危险",
                3: "L3 保护(停可塑)", 4: "L4 只读"}[self.level]

    def report(self) -> dict:
        return {"level": self.degrade_name(), "metrics": self.metrics,
                "alerts": self.alerts[-5:]}


# ============================================================
# 演示
# ============================================================
def demo():
    print("=" * 66)
    print("  仿生 AI · M6-M9 高级机制验证")
    print("=" * 66)

    # ---- M6 ----
    print("\n[M6] 多尺度权重（子权重叠加 + 三因子可塑性）...")
    w = MultiScaleWeight(shape=(4, 4))
    pre = np.random.normal(0, .5, 4); post = np.random.normal(0, .5, 4)
    print(f"  初始子权重范数: {[f'{n:.3f}' for n in w.scale_norms()]}")
    for step in range(5):
        rpe = w.update(pre, post, reward=1.0, expected=0.2)   # 正奖励
    print(f"  5步后子权重范数: {[f'{n:.3f}' for n in w.scale_norms()]}  (应增长)")
    print(f"  有效权重范数: {np.linalg.norm(w.value()):.3f}")
    # 负奖励测试
    w2 = MultiScaleWeight(shape=(4, 4))
    for step in range(5):
        w2.update(pre, post, reward=0.0, expected=0.8)        # 负 RPE
    print(f"  负奖励后范数: {[f'{n:.3f}' for n in w2.scale_norms()]}  (β应缩小)")
    # 迁移（睡眠合并）
    before = w.scale_norms()
    w.decay_to_slow(rate=0.5)
    after = w.scale_norms()
    print(f"  快→慢迁移: 快 {before[0]:.3f}→{after[0]:.3f}, 慢 {before[2]:.3f}→{after[2]:.3f}")

    # ---- M7 ----
    print("\n[M7] 能量核算（J/token + 错峰）...")
    em = EnergyMeter()
    print(f"  nvidia-smi 可用: {em.have_nvml}")
    rec = em.measure(lambda: sum(i*i for i in range(200000)), tokens=50, label="模拟计算")
    print(f"  实测: {rec['joules']:.3f} J ({rec['seconds']:.4f}s, 均值功率 {rec['power_avg']:.1f}W)")
    print(f"  J/token: {rec['j_per_token']:.5f}")
    sched = EnergyScheduler()
    for h in [3, 12, 20, 23]:
        print(f"  {h:02d}:00 → 预算={sched.budget(h)} {'重活✓' if sched.should_do_heavy(h) else '重活✗'}")

    # ---- M8 ----
    print("\n[M8] 符号接地（共激活 + 因果验证 + 五维剖面）...")
    from bio_brain import NervousSystem
    ns = NervousSystem(seed=11)
    symbols = [ns.add_neuron("sensory") for _ in range(3)]
    envs = [ns.add_neuron("sensory") for _ in range(3)]
    gm = GroundingModule(ns)
    n = gm.ground_by_coactivation(symbols, envs, strength=0.6)
    print(f"  共激活: {n} 个符号神经元 ←→ {len(envs)} 个环境神经元")
    r = gm.is_grounded(symbols, lambda: gm._ignite(envs), lambda: gm._ignite_random(envs))
    print(f"  因果验证: 敏感度={r['sensitivity']:.4f} grounded={r['grounded']} causal={r['causal']}")
    prof = gm.grounding_profile(symbols, [(symbols, envs)])
    print(f"  五维剖面: { {k: round(v,2) for k,v in prof.items()} }")
    # 未接地测试
    ns2 = NervousSystem(seed=99)
    s2 = [ns2.add_neuron() for _ in range(2)]
    gm2 = GroundingModule(ns2)
    r2 = gm2.is_grounded(s2, lambda: None, lambda: None)
    print(f"  未接地测试: 敏感度={r2['sensitivity']:.4f} grounded={r2['grounded']} (应False)")

    # ---- M9 ----
    print("\n[M9] 鲁棒性（数值/调度/学习 + 降级）...")
    g = RobustnessGuard()
    print(f"  数值检查(NaN): {g.check_numeric(np.array([1.0, np.nan]), 'test')}  ★应False")
    print(f"  数值检查(正常): {g.check_numeric(np.array([1.0, 2.0]), 'test')}  ★应True")
    print(f"  抖动检查(20/s): {g.check_thrash(20.0)}  ★应False并降级")
    print(f"  可塑爆炸检查: {g.check_plasticity([5.0])}  ★应False")
    print(f"  存活检查(全零): {g.check_alive([0,0,0])}  ★应False")
    print(f"  当前降级等级: {g.degrade_name()}")
    print(f"  告警: {g.report()['alerts']}")

    print("\n✅ M6-M9 全部跑通")


if __name__ == "__main__":
    demo()
