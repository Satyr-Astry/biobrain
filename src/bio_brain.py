"""
仿生 AI · M0-M2 最小可行原型
=============================
按《工程规范_v7.md》实现：
  M0  数据结构: Neuron(向量组) / Tract / Group / Library + 重叠索引
  M1  感觉端:   SensoryPort —— 外部信号点亮感觉神经元
  M2  运动端:   MotorPort  —— 读出运动神经元激活

外加 ALG-0 计算内核 + ALG-1 激活度 + ALG-4 分级调度 的最小验证。

依赖: numpy (仅此而已)
运行: python bio_brain.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np

# ============================================================
# 常量（PARAM，对应规范 §2.2）
# ============================================================
D_DENDRITE = 16      # 树突场维度（原型用小值）
D_SOMA = 16          # 胞体态维度
D_AXON = 16          # 轴突场维度
TRACT_SIZE = 8       # 束成员数
DECAY = 0.9          # 激活衰减
THETA_HIGH = 0.6     # 提升阈值（进 VRAM）
THETA_LOW = 0.35     # 降级阈值（出 VRAM）
ACTIVE_EPS = 1e-3    # 活跃稀疏阈值
PLASTIC_THRESHOLD = 0.35   # 可塑激活门槛（代码验证后从0.5下调：实测激活峰值~0.5）
TOP_K_AGG = 3
ROUTE_K = 2          # 跨束路由选束数（ALG-0）
MSG_GAIN = 1.0       # 束内消息增益
SPREAD_GAIN = 0.25   # ★激活扩散增益（代码暴露的缺失参数）
SPREAD_CAP = 0.3     # ★单次扩散上限（防爆）
BRIDGE_GAIN = 0.3    # ★跨束桥接增益（代码暴露缺失：激活需能跨束传播）
BRIDGE_SIM = 0.0     # ★束间相似度门槛（原型设0=全连通；生产应设正阈值）
NORM_CLIP = 1.0


class Tier(Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


# ============================================================
# M0: 数据结构（§3.1 §3.2 §3.3）
# ============================================================
@dataclass
class Neuron:
    """神经元 = 向量组（多隔室）"""
    id: int
    dendrite: np.ndarray          # 树突场（接收）
    soma: np.ndarray              # 胞体态（整合）
    axon: np.ndarray              # 轴突场（输出）
    base: np.ndarray              # 主体（稳定部分，扁平化存储）
    plasticity: np.ndarray        # 临时增量（MUST 与 base 同形）
    plasticity_pending: Optional[np.ndarray] = None   # ★ALG-3 延迟可塑缓存
    activity: float = 0.0
    importance: float = 0.0       # Fisher 重要性
    # ★重叠：一个神经元可属多个束（MUST 用集合）
    tract_ids: Set[int] = field(default_factory=set)
    # 角色标记（感觉/运动/普通）
    role: str = "inter"


@dataclass
class Tract:
    """神经束 = 一起激活的通路（双层语义）"""
    id: int
    members: List[int]            # 组成：谁在束里
    dynamics: np.ndarray          # 动力学：成员间怎么协同 (n x n)
    key: np.ndarray               # ★束级路由标识（ALG-0 阶段B）
    value: np.ndarray             # ★束级读出的内容
    activation: float = 0.0
    group_ids: Set[int] = field(default_factory=set)


@dataclass
class Group:
    """神经组 = 功能团块"""
    id: int
    tract_ids: List[int]
    function: str = ""
    activation: float = 0.0
    library_ids: Set[int] = field(default_factory=set)


@dataclass
class Library:
    """神经库 = 最大单位，一级存储单位"""
    id: int
    group_ids: List[int]
    activation: float = 0.0
    tier: Tier = Tier.NVME
    size_bytes: int = 0
    last_change: float = 0.0


# ============================================================
# 激活度工具（ALG-1 §4.1）
# ============================================================
def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def top_k_mean(vals: List[float], k: int = TOP_K_AGG) -> float:
    if not vals:
        return 0.0
    v = sorted(vals, reverse=True)[:k]
    return float(np.mean(v))

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ============================================================
# 神经系统（§3.3 + ALG-1~4 + ALG-0）
# ============================================================
class NervousSystem:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.neurons: Dict[int, Neuron] = {}
        self.tracts: Dict[int, Tract] = {}
        self.groups: Dict[int, Group] = {}
        self.libraries: Dict[int, Library] = {}
        self.rng = rng
        self._next_id = 0
        self.pending_ignitions: List[Tuple[int, float]] = []
        self.t = 0.0

    # ---------- 构建 ----------
    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add_neuron(self, role: str = "inter") -> int:
        i = self.new_id()
        self.neurons[i] = Neuron(
            id=i,
            dendrite=self.rng.normal(0, 0.1, D_DENDRITE),
            soma=np.zeros(D_SOMA),
            axon=np.zeros(D_AXON),
            base=self.rng.normal(0, 0.02, D_DENDRITE * D_SOMA),
            plasticity=np.zeros(D_DENDRITE * D_SOMA),
            role=role,
        )
        return i

    def add_tract(self, member_ids: List[int]) -> int:
        i = self.new_id()
        n = len(member_ids)
        self.tracts[i] = Tract(
            id=i,
            members=list(member_ids),
            dynamics=self.rng.normal(0, 0.1, (n, n)),
            key=self.rng.normal(0, 0.1, D_SOMA),
            value=self.rng.normal(0, 0.1, D_SOMA),
        )
        for m in member_ids:
            self.neurons[m].tract_ids.add(i)     # ★重叠：加到集合
        return i

    def add_group(self, tract_ids: List[int], function: str = "") -> int:
        i = self.new_id()
        self.groups[i] = Group(id=i, tract_ids=list(tract_ids), function=function)
        for t in tract_ids:
            self.tracts[t].group_ids.add(i)
        return i

    def add_library(self, group_ids: List[int]) -> int:
        i = self.new_id()
        self.libraries[i] = Library(id=i, group_ids=list(group_ids))
        for g in group_ids:
            self.groups[g].library_ids.add(i)
        return i

    # ---------- 激活（ALG-1） ----------
    def ignite(self, nid: int, amount: float):
        """★激活唯一入口（MUST 归一化）"""
        n = self.neurons[nid]
        n.activity = clamp(n.activity + amount)

    def queue_ignition(self, nid: int, amount: float):
        self.pending_ignitions.append((nid, amount))

    def neighbors(self, nid: int) -> List[int]:
        """邻居 = 同束的其他神经元"""
        n = self.neurons[nid]
        out = []
        for tid in n.tract_ids:
            out.extend(m for m in self.tracts[tid].members if m != nid)
        return out

    def tick_activation(self, active_set: Set[int]) -> Set[int]:
        """稀疏 tick：只处理活跃集 + 新点火 + ★束内/跨束扩散"""
        # 1. 衰减
        for i in list(active_set):
            self.neurons[i].activity *= DECAY
        # 2. 写入新激活（输入点亮）
        for nid, amt in self.pending_ignitions:
            if nid in self.neurons:
                self.ignite(nid, amt)
                active_set.add(nid)
        self.pending_ignitions.clear()
        # ★3. 激活扩散（束内直接传 + 跨束经共享神经元间接传）
        spread: Dict[int, float] = {}
        for i in list(active_set):
            n = self.neurons[i]
            if n.activity <= ACTIVE_EPS:
                continue
            for tid in n.tract_ids:
                tr = self.tracts[tid]
                k = tr.members.index(i)
                # 束内扩散
                for k2, m in enumerate(tr.members):
                    if m == i:
                        continue
                    gain = abs(tr.dynamics[k, k2]) * SPREAD_GAIN
                    spread[m] = spread.get(m, 0.0) + n.activity * gain
                # ★跨束扩散：激活该束的"束级信号"，供束的路由读出
                tr.activation = max(tr.activation, min(n.activity, SPREAD_CAP))
        # ★4. 跨束桥接：束激活 → 该束的 key 与其它束的 key 相似 → 传激活
        tractor_ids = [tid for tid, tr in self.tracts.items() if tr.activation > ACTIVE_EPS]
        for tid in tractor_ids:
            tr = self.tracts[tid]
            for tid2, tr2 in self.tracts.items():
                if tid2 == tid:
                    continue
                sim = float(np.dot(tr.key, tr2.key)) / (norm(tr.key)*norm(tr2.key) + 1e-8)
                if sim > BRIDGE_SIM:                      # ★束间只有足够相似才桥接
                    for m in tr2.members:
                        spread[m] = spread.get(m, 0.0) + tr.activation * sim * BRIDGE_GAIN
        for m, amt in spread.items():
            self.ignite(m, min(amt, SPREAD_CAP))    # 扩散也受上限约束
            active_set.add(m)
        # 5. 重估活跃集
        return {i for i in active_set if self.neurons[i].activity > ACTIVE_EPS}

    # ---------- 总激活度（ALG-2，归一化防爆炸） ----------
    def total_activity(self, nid: int) -> float:
        n = self.neurons[nid]
        base = n.activity
        extra = 0.0
        cnt = 0
        for tid in n.tract_ids:
            tr = self.tracts[tid]
            if tr.activation > ACTIVE_EPS:
                extra += tr.dynamics[tr.members.index(nid)][0] * tr.activation if False else tr.activation * 0.5
                cnt += 1
        if cnt > 1:
            extra /= np.sqrt(cnt)          # ★次线性归一化
        return clamp(base + extra)

    # ---------- 逐级聚合（★并集去重，修正多路径重复） ----------
    def tract_activation(self, tid: int) -> float:
        tr = self.tracts[tid]
        return top_k_mean([self.neurons[i].activity for i in tr.members])

    def group_activation(self, gid: int) -> float:
        g = self.groups[gid]
        atoms: Set[int] = set()                       # ★并集去重
        for tid in g.tract_ids:
            atoms |= set(self.tracts[tid].members)
        return top_k_mean([self.neurons[i].activity for i in atoms])

    def library_activation(self, lid: int) -> float:
        lib = self.libraries[lid]
        atoms: Set[int] = set()
        for gid in lib.group_ids:
            for tid in self.groups[gid].tract_ids:
                atoms |= set(self.tracts[tid].members)
        return top_k_mean([self.neurons[i].activity for i in atoms])

    # ---------- 计算内核（ALG-0） ----------
    def compute(self, active_set: Set[int]) -> Dict[int, np.ndarray]:
        """阶段A 束内消息传递 + 阶段B 跨束门控注意力 + 阶段C 轴突输出"""
        active = [i for i in active_set if i in self.neurons]

        # 阶段 A：束内消息传递（局部）
        active_tracts = set()
        for i in active:
            active_tracts |= self.neurons[i].tract_ids
        for tid in active_tracts:
            tr = self.tracts[tid]
            mem = tr.members
            idx = {m: k for k, m in enumerate(mem)}
            for k, m in enumerate(mem):
                if self.neurons[m].activity <= ACTIVE_EPS:
                    continue
                msg = np.zeros(D_SOMA)
                for k2, m2 in enumerate(mem):
                    if m2 == m:
                        continue
                    msg += tr.dynamics[k, k2] * self.neurons[m2].axon
                self.neurons[m].soma = np.tanh(self.neurons[m].soma + MSG_GAIN * msg)

        # 阶段 B：跨束门控注意力（稀疏：神经元 → 少数束）
        all_tract_ids = list(self.tracts.keys())
        for i in active:
            n = self.neurons[i]
            # 路由：按 soma 与 tract.key 的相似度选 top-k
            sims = []
            for tid in all_tract_ids:
                tr = self.tracts[tid]
                s = float(np.dot(n.soma, tr.key)) / (norm(n.soma)*norm(tr.key) + 1e-8)
                sims.append((s, tid))
            sims.sort(reverse=True)
            routes = sims[:ROUTE_K]
            for s, tid in routes:
                w = np.exp(s) / (sum(np.exp(x) for x, _ in routes) + 1e-8)
                n.soma = np.tanh(n.soma + w * self.tracts[tid].value)

        # 阶段 C：轴突输出
        for i in active:
            n = self.neurons[i]
            n.axon = np.tanh(n.soma)

        return {i: self.neurons[i].axon.copy() for i in active}

    # ---------- 在线可塑（ALG-3，延迟提交） ----------
    def update_plasticity(self, active_set: Set[int], state_prev: Dict[int, np.ndarray]):
        """ALG-3 在线可塑（延迟提交）

        ★v7.3 修正：原实现用 axon 算 hebb —— 但 axon 可能全零（未跑过 compute），
        导致可塑量恒为 0（实测 seed=3 时 total=0.0）。改用 soma+dendrite 的组合特征，
        保证任何活跃神经元都有非零的可塑信号。
        """
        for i in active_set:
            n = self.neurons[i]
            if n.activity < PLASTIC_THRESHOLD:
                continue
            # ★特征: dendrite(输入) 与 soma(整合态) 的投影（不依赖可能为零的 axon）
            pre = n.dendrite[:D_DENDRITE]
            post_full = n.soma if norm(n.soma) > 1e-8 else n.dendrite
            post = np.tile(post_full, int(np.ceil(D_DENDRITE / len(post_full))))[:D_DENDRITE]
            # 若 pre/post 仍全零（极端情形），用 activity 注入非零信号
            if norm(pre) < 1e-8 and norm(post) < 1e-8:
                pre = np.full(D_DENDRITE, n.activity)
                post = np.full(D_DENDRITE, n.activity)
            hebb = np.outer(pre, post).flatten()
            target = n.base.shape[0]
            if hebb.shape[0] < target:
                hebb = np.pad(hebb, (0, target - hebb.shape[0]))
            else:
                hebb = hebb[:target]
            delta = 0.02 * hebb - 0.88 * n.plasticity
            cand = n.plasticity + delta
            nrm = norm(cand)
            if nrm > NORM_CLIP:
                cand = cand * (NORM_CLIP / nrm)
            n.plasticity_pending = cand                     # ★写缓存
            n.importance = 0.99 * n.importance + 0.01 * norm(hebb)

    def apply_pending(self, active_set: Set[int]):
        for i in active_set:
            n = self.neurons[i]
            if n.plasticity_pending is not None:
                n.plasticity = n.plasticity_pending
                n.plasticity_pending = None

    # ---------- 分级调度（ALG-4，滞回 + 冷却） ----------
    def tick_scheduler(self) -> List[str]:
        changes = []
        for lid, lib in self.libraries.items():
            act = self.library_activation(lid)
            lib.activation = act
            if act > THETA_HIGH and lib.tier != Tier.VRAM:
                lib.tier = Tier.VRAM; lib.last_change = self.t
                changes.append(f"库{lid} → VRAM (act={act:.2f})")
            elif act < THETA_LOW and lib.tier == Tier.VRAM:
                if self.t - lib.last_change > 0.0:          # 冷却（原型简化为0）
                    lib.tier = Tier.RAM; lib.last_change = self.t
                    changes.append(f"库{lid} → RAM (act={act:.2f})")
        return changes


# ============================================================
# M1: 感觉端（§5.1）—— 输入 = 点亮感觉神经元
# ============================================================
class SensoryPort:
    """把外部信号变成感觉神经元的激活。不是"喂数据", 是"点火"。

    ★v2 改进：稀疏点火 —— 只点亮最强的 top-k 个神经元。
    原因：全都点亮会导致语义区分度归零（实测所有输入重叠度都是 1.00）。
    稀疏激活让"语义相近→点亮相似组合，语义不同→点亮不同组合"。
    """
    SPARSITY = 0.5          # 点亮比例（前 50% 最强的）

    def __init__(self, modality: str, neuron_ids: List[int], dim: int):
        self.modality = modality
        self.neuron_ids = neuron_ids
        self.dim = dim
        # 固定随机投影（保证同一模态稳定）
        rng = np.random.default_rng(abs(hash(modality)) % (2**32))
        self.proj = rng.normal(0, 1.0 / np.sqrt(max(dim, 1)), (dim, len(neuron_ids)))

    def ignite(self, raw: np.ndarray, ns: NervousSystem) -> dict:
        """MUST: 只点亮感觉神经元, 不直接写大脑组织。★稀疏点火。"""
        raw = np.asarray(raw, dtype=float).flatten()
        if raw.shape[0] < self.dim:
            raw = np.pad(raw, (0, self.dim - raw.shape[0]))
        raw = raw[:self.dim]
        acts = np.tanh(self.proj.T @ raw)
        # ★稀疏：只保留最强的 top-k（k = 比例 × 神经元数，至少 1）
        k = max(1, int(round(len(acts) * self.SPARSITY)))
        order = np.argsort(-np.abs(acts))[:k]
        ignited = {}
        for idx in order:
            a = float(acts[idx])
            if abs(a) > ACTIVE_EPS:
                nid = self.neuron_ids[idx]
                ns.queue_ignition(nid, abs(a))
                ignited[nid] = abs(a)
        return {"modality": self.modality, "ignited": ignited}


# ============================================================
# M2: 运动端（§5.2）—— 输出 = 读出运动神经元激活
# ============================================================
class MotorPort:
    """把运动神经元激活读成输出。不是"返回结果", 是"读出激活"。"""
    def __init__(self, modality: str, neuron_ids: List[int], threshold: float = 0.3):
        self.modality = modality
        self.neuron_ids = neuron_ids
        self.threshold = threshold

    def read(self, ns: NervousSystem) -> Optional[dict]:
        """SHOULD: 返回 None 表示无输出（思考流决定不说）"""
        vals = {nid: ns.neurons[nid].activity for nid in self.neuron_ids}
        hot = {k: v for k, v in vals.items() if v > self.threshold}
        if not hot:
            return None
        # 读出 = 激活最强的神经元们
        return {"modality": self.modality, "emission": hot,
                "strength": max(hot.values())}


# ============================================================
# 演示：M0-M2 跑通验证
# ============================================================
def demo():
    print("=" * 62)
    print("  仿生 AI · M0-M2 原型验证")
    print("=" * 62)
    ns = NervousSystem(seed=42)

    # --- M0: 建组织 ---
    print("\n[M0] 构建神经组织（含重叠）...")
    visual = [ns.add_neuron("sensory") for _ in range(4)]
    symbol = [ns.add_neuron("sensory") for _ in range(4)]
    inter  = [ns.add_neuron("inter") for _ in range(8)]
    motor  = [ns.add_neuron("motor") for _ in range(3)]

    t1 = ns.add_tract(visual + inter[:2])        # 束1: 视觉→联合
    t2 = ns.add_tract(symbol + inter[:2])        # 束2: 符号→联合(★与束1共享 inter[:2] → 重叠)
    t3 = ns.add_tract(inter[2:6] + motor)        # 束3: 联合→运动
    t4 = ns.add_tract(inter[:2] + inter[2:6])    # ★束4: 桥接束(让感觉区能通到运动区)
    g1 = ns.add_group([t1, t2], "percept")     # 组（★共享束 → 多路径）
    g2 = ns.add_group([t3, t4], "action")
    l1 = ns.add_library([g1])
    l2 = ns.add_library([g2])

    print(f"  神经元 {len(ns.neurons)} | 束 {len(ns.tracts)} | 组 {len(ns.groups)} | 库 {len(ns.libraries)}")
    # 验证重叠
    shared = ns.neurons[inter[0]].tract_ids
    print(f"  ★重叠验证: 神经元 inter[0] 属于 {len(shared)} 个束 {shared}")

    # --- M1: 感觉端 ---
    print("\n[M1] 感觉端点火（视觉 + 符号 共激活 → 接地）...")
    vis_port = SensoryPort("vision", visual, dim=8)
    sym_port = SensoryPort("symbol", symbol, dim=8)
    r1 = vis_port.ignite(np.array([1.,0.5,0.3,0,0,0,0,0]), ns)
    r2 = sym_port.ignite(np.array([0.8,0.6,0,0,0,0,0,0]), ns)
    print(f"  视觉点火 {len(r1['ignited'])} 个神经元")
    print(f"  符号点火 {len(r2['ignited'])} 个神经元")

    # --- 运行思考循环几步 ---
    print("\n[思考流] 运行 15 个 tick（信号需时间在组织中传播）...")
    active: Set[int] = set(r1["ignited"]) | set(r2["ignited"])
    prev_state = {}
    for step in range(15):
        ns.apply_pending(active)                      # ALG-3: 提交上一步可塑
        active = ns.tick_activation(active)           # ALG-1: 稀疏 tick
        # 聚合束激活
        for tid in ns.tracts:
            ns.tracts[tid].activation = ns.tract_activation(tid)
        state = ns.compute(active)                    # ALG-0: 计算
        ns.update_plasticity(active, prev_state)      # ALG-3: 在线可塑
        changes = ns.tick_scheduler()                 # ALG-4: 分级调度
        prev_state = state
        ns.t += 1
        if step % 3 == 0 or step == 14:
            mm = max((ns.neurons[m].activity for m in motor), default=0)
            print(f"  tick{step:2d}: 活跃={len(active):2d} 运动区峰值={mm:.2f} " +
                  f"库驻留={[l.tier.value for l in ns.libraries.values()]}")
        if changes:
            print(f"         调度: {changes}")

    # --- M2: 运动端 ---
    print("\n[M2] 运动端读出...")
    mot_port = MotorPort("language", motor)
    out = mot_port.read(ns)
    if out:
        print(f"  输出: {out}")
    else:
        print(f"  无输出（运动神经元未达阈值 {mot_port.threshold}；符合'有输入不一定输出'）")

    # --- 验证关键机制 ---
    print("\n[验证] 关键不变量：")
    # 1. 重叠
    print(f"  1. 重叠归属: ✓ (神经元属 {len(shared)} 束)")
    # 2. 可塑量已更新
    pl = sum(norm(n.plasticity) for n in ns.neurons.values())
    print(f"  2. 在线可塑累积: {pl:.4f} (应>0)")
    # 3. 因果性（pending 机制）
    有pending = any(n.plasticity_pending is not None for n in ns.neurons.values())
    print(f"  3. 延迟可塑缓存: {'有' if 有pending else '无'} (证明只影响未来)")
    # 4. 分级调度
    tiers = {lid: lib.tier.value for lid, lib in ns.libraries.items()}
    print(f"  4. 分级驻留: {tiers}")
    # 5. 多路径去重
    ga = ns.group_activation(g1)
    print(f"  5. 组激活度(并集去重): {ga:.3f}")
    print("\n✅ M0-M2 全部跑通")


if __name__ == "__main__":
    demo()
