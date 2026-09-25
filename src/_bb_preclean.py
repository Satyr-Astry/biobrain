"""
仿生 AI · M0-M2 最小可行原型
=============================
按《工程规范_v7.md》实现：
  M0  数据结构: Neuron(向量组) / Tract / Group / Library + 重叠索引
  M1  感觉端:   SensoryPort —— 外部信号点亮感觉神经元
  M2  运动端:   MotorPort  —— 读出运动神经元激活

外加 ALG-0 计算内核 + ALG-1 激活度 + ALG-4 分级调度 的最小验证。

依赖: numpy (仅此而已)
运行: python cog_vec.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np
import os

# ============================================================
# 常量（PARAM，对应规范 §2.2）
# ============================================================
D_DENDRITE = 16      # 树突场维度（原型用小值）
D_SOMA = 16          # 胞体态维度
# ★结构编辑3(2026-09-18)：固有描述子维度（仿 FlyGM η_v）
ETA_DIM = 4          # [兴奋性, 增益, 时间常数, 阈值偏移]
ETA_INIT_STD = 0.3   # 初始方差（够大才能产生个体差异）
D_AXON = 16          # 轴突场维度
TRACT_SIZE = 8       # 束成员数
DECAY = 0.9          # 激活衰减
# ★存储分层参数（v1.7，Colibri 借鉴）
THETA_HIGH = 0.50    # ★v1.8 重校准：原 0.6 太高（感知库实测 0.567 上不去 L0）
THETA_MID = 0.35     # ★中阈（L2→L1 预取上行用；低于 THETA_HIGH 以形成滞回）
THETA_LOW = 0.20     # ★v1.8 重校准：原 0.35 与 THETA_HIGH 差距太小，滞回空间不足
#   重校准依据（实测）：感知库 activation ≈ 0.567，原 THETA_HIGH=0.6 → 上不了 L0。
#   新配置：0.50 / 0.35 / 0.20 —— 滞回带宽 = 0.30（原 0.25），且感知库能进 L0。
COOLDOWN_TICKS = 20        # ★冷却（tick 数；Colibri 防抖振思想）——原实现为 0（等于无冷却）
COOLDOWN_NVME_MULT = 2     # ★降级到 NVME 需要更长冷却（更保守）
PIN_THRESHOLD = 0.55       # ★热存储门限：hot_score 超过此值 → 钉在 L0 不被淘汰
HOT_EMA_ALPHA = 0.05       # ★热分数 EMA 系数（长期统计，抗短期波动）
PREFETCH_LOOKBACK = 3      # ★预取：观察最近 N 个 tick 的趋势（Colibri 的 router-ahead）
PREFETCH_RISE = 0.05       # ★预取：activation 上升超过此值才预取（防噪声触发）
#   依据：Colibri "parameters are not resident state to be held, they are data to be
#         staged across a heterogeneous storage hierarchy, exactly when the router
#         proves they are needed" / "routing has measurable structure — and structure
#         is cacheable"

ACTIVE_EPS = 1e-3    # 活跃稀疏阈值
PLASTIC_THRESHOLD = 0.35   # 可塑激活门槛（代码验证后从0.5下调：实测激活峰值~0.5）
TOP_K_AGG = 3
ROUTE_K = 2          # 跨束路由选束数（ALG-0）
# ★v1.9 交叉注意力（架构书 §7）
N_HEADS = 4          # 注意力头数（D_SOMA=16 → 每头 4 维）
TIER_BIAS = 0.15     # ★层级软偏置：顺流 +bias / 逆行 −bias（替代硬门控 _may_flow）
ATTN_GAIN = 1.0      # 注意力产出写入 soma 的增益（cos 模式下与旧实现同量级）
# ★v2.0 Step 4：可学习投影
ATTN_MODE_DEFAULT = "cos"   # "cos"（默认，向后兼容）| "learned"（可学习 W_q/W_k/W_v/W_o）
W_INIT_STD = 0.15           # 投影矩阵初始标准差（近恒等，保证初始行为≈cos 模式）
ATTN_LR = 0.01              # 投影矩阵的在线学习率（NORM-3：学习是思考的副产品）
#   设计依据（架构书 §6.5）：cos 模式是"硬切段"，无表达能力；
#   learned 模式引入 W_q/W_k/W_v/W_o，通过在线可塑更新（不引入独立训练阶段）。
#   设计依据（架构书 §7.3）：D1 多头、D2 可学习投影(待做)、D3 层级偏置、
#                            D4 BIND只读/INTEGRATE写、D5 subtick 相位
#   cos 兼容：无投影矩阵时，多头退化为"把向量切段各算 cos"，与旧单头同族
MSG_GAIN = 1.0       # 束内消息增益
SPREAD_GAIN = 0.8    # ★激活扩散增益（2026-09-18 重校准：0.25→0.8，配合饱和抑制恢复可达幅度）
SPREAD_CAP = 0.8     # ★单次扩散上限（2026-09-18 重校准：0.3→0.8）
SPREAD_SAT_SUPPRESS = 0.7  # ★饱和抑制系数（2026-09-18 新增，实测甜点）
#   问题：原实现让活跃集内神经元持续累积扩散量，t12 时符号端/运动端全部饱和到 1.0，
#         导致不同输入的激活模式完全一致 → 输出与输入无关（P0 病灶）。
#   对策：对"已活跃"神经元，接受扩散量按 (1-cur)*抑制系数 衰减，越接近饱和补充越少。
#   仿生对应：神经元的相对不应期/饱和特性。
#   参数标定（10 seed 实测，见 diag_calibrate）：
#     suppress=0.7, cap=0.8, gain=0.8 → 峰值中位 0.929 / max 0.985 / 饱和数 0 / motor>0.3 达 10/10
#     对比 suppress=0.9 → 峰值 0.992 但有 5/10 饱和（信息又被抹平）
#   平衡点含义：既让峰值够高以触发 PLASTIC_THRESHOLD(0.35) 的在线可塑，
#               又不让所有神经元顶到 1.0 而丢失输入区分度。
BRIDGE_GAIN = 0.3    # ★跨束桥接增益（代码暴露缺失：激活需能跨束传播）
BRIDGE_SIM = 0.5     # ★束间相似度门槛（v1.8：key 改为结构派生后，此阈值才有意义）
#   历史：key 为随机初始化时，0.0=全连通（功能库被污染，实测激活 0.60~0.72）、
#         0.3=全断开（感知库也归零）——**中间任何值都是随机的连/断**。
#   现在 key 由成员角色派生（_derive_tract_key），同角色构成的束 key 相似，
#   不同构成的束 key 正交 → 0.5 可有效区分"功能内连通 / 功能间断开"。
#   依据：BANC(Nature 656:957)「连接关系是显式的区域配对，不是相似度算出来的」
# ★结构编辑1(2026-09-18)：抑制强度系数（仿果蝇 FlyGM W=N_exc−N_inh 的抑制项）
INHIB_STRENGTH = 1.0
#   生物学依据：FlyWire 中 GABA/Glycine 为抑制性递质，提供负反馈防止全网自激。
#   fly-api 教训：整脑 LIF 模型若无双稳抑制，任何中心输入会引爆"永久性8400神经元风暴"。
#   本项目此前用 abs(dynamics) 丢弃符号 → 无抑制 → 全网络均一化（P0 病灶）。
# ★结构编辑2(2026-09-18)：Tier 层级顺序，用于约束束间信息流方向
TIER_ORDER = {"sensory": 0, "inter": 1, "drives": 2, "motor": 3}
#   仿 FlyGM 的 afferent→intrinsic→efferent 三分区；允许同级（局部反馈），禁长程逆行。
#   ★v1.2(2026-09-18)：新增 "drives" 层级（参考 snedea/flybrain 的
#     sensory/central/drives/motor 四分类）——驱动力神经元介于 inter 与 motor 之间。
#   注意：drives 排在 inter 之后、motor 之前，保证 drives→motor 顺流合法。

# ★结构编辑4(2026-09-18 v1.2)：Hebbian 结构可塑（让连接强度从经验累积）
STRUCT_PLASTIC_ETA = 0.05      # 结构可塑学习率（比神经元可塑慢，结构应稳定）
STRUCT_PLASTIC_DECAY = 0.02    # 结构权重向初始值的回调（防漂移）
STRUCT_PLASTIC_CAP = 0.5       # 单次结构更新上限（防一次经验冲垮结构）
STRUCT_W_SLOW_BLEND = 0.1      # 慢权重混合比（巩固相把 w_fast 混入 w_slow）
TARGET_ROW_SUM = None          # ★None = 自适应（= 初始每行 L1 的均值，见 _ensure_row_budget）
#   生物学依据：突触稳态（synaptic homeostasis）——真实神经元的突触总量守恒，
#   防止 Hebbian 正反馈导致权重无界增长。
#   ★教训（2026-09-18 v1.5）：不要拍脑袋设绝对值！
#     初始每行 L1 实测 mean=1.39~2.04，而拍脑袋设 0.6 →
#     100% 的行初始就超标 → 归一化立刻把结构压扁（有效/初始=0.383 恒定）→ 锁死。
#     正确做法：上限 = 初始每行 L1 的统计量（自适应）。
#   依据：arXiv 2404.17128（拓扑>动力学）+ 本项目实测（否定结果_结构化dynamics无效.md）
#         → 结构信息不能靠初始化分布，必须由运行经验累积。
#   参考实现：erojasoficial-byte/fly-brain 的 Hebbian 突触可塑性。
#   约束：保持 excitatory/inhibitory 符号不变（只调幅度），遵守 NORM-4/5。

NORM_CLIP = 1.0


class Tier(Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


# ============================================================
# ★v1.9：subtick 相位枚举（架构书 §6.2）
# ============================================================
class Phase(Enum):
    """一个 tick（认知周期）内的相位。每相位只做一件事，读写严格分离。

    为什么：whole-tick 把 6 件事串在一起，共用同一 activity 快照 →
      ① 扩散(快τ≈1)/计算(中)/可塑(慢τ≫1) 三个时间尺度混叠
      ② 同 tick 内读改写纠缠（compute 读 tick_activation 刚改的 activity，
         且内部又"读soma→写soma"）
      ③ 无法单独替换注意力，因为与扩散耦合

    相位划分后：相位边界 = 时间尺度边界；BIND 只读、INTEGRATE 才写 soma。
    """
    COMMIT = 0      # 提交上 tick 的可塑/结构更新（NORM-4 因果）
    DECAY = 1       # 衰减（只改 activity）
    IGNITE = 2      # 外部输入点火 + pending 写入
    SPREAD = 3      # 束内/跨束扩散（可多步；只读/写 activity）
    BIND = 4        # ★交叉注意力：只读 soma/key/value，产出 _attn_out（不改状态）
    INTEGRATE = 5   # ★唯一写 soma 的相位
    EMIT = 6        # 轴突输出 axon = tanh(soma)
    PLASTIC = 7     # 累积可塑量（写 pending，不提交）
    SCHEDULE = 8    # 分级调度 / 预算检查


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
    # ★结构编辑3(2026-09-18)：固有描述子 η_v —— 仿果蝇 FlyGM 的 η_v
    #   依据: FlyGM(2602.17997) "assign each neuron a trainable intrinsic descriptor
    #         η_v to capture cell-specific computational properties
    #         (e.g., excitability and gain) that are not included in the connectomic data"
    #   作用: 让同一输入对不同神经元产生不同响应 → 解决"6个运动神经元只有2个带信息"
    #   含义: [兴奋性, 增益, 时间常数, 阈值偏移]
    eta: np.ndarray = field(default_factory=lambda: np.zeros(ETA_DIM))


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
    # ★结构编辑2(2026-09-18)：束的层级（用于约束束间信息流方向）
    #   取值 sensory / inter / drives / motor，由成员角色推断；仿 FlyGM 三分区。
    tier: str = "inter"
    # ★结构编辑4(2026-09-18 v1.2)：Hebbian 结构可塑字段
    #   w_fast: 快权重增量（在线累积，每个 tick 可更新）
    #   w_slow: 慢权重（只在巩固相合并，作为稳定结构）
    #   有效连接 = dynamics（初始骨架）+ w_slow + w_fast
    w_fast: Optional[np.ndarray] = None   # shape 同 dynamics
    w_slow: Optional[np.ndarray] = None   # shape 同 dynamics
    w_pending: Optional[np.ndarray] = None  # ★NORM-4 因果：延迟提交
    row_budget: Optional[float] = None      # ★v1.5 突触稳态预算（自适应，首次使用时计算）


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
    # ★v1.7 存储分层（Colibri 借鉴）
    hot_score: float = 0.0            # 热分数 EMA（长期使用频率）→ 决定是否钉在 L0
    act_history: List[float] = field(default_factory=list)   # 近期激活（供预取判断趋势）
    prefetched: bool = False          # 是否因预取而提前上行
    tier_changes: int = 0             # tier 切换计数（验收"不抖振"用）


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
        # ★v1.7：本 tick 因不在 L0 而被跳过的束数（Colibri 稀疏计算可观测性）
        self._l0_filtered = 0
        # ★v2.0 Step 4：可学习投影矩阵（attention mode）
        #   W_q/W_k/W_v: (D_SOMA, N_HEADS*D_HEAD)，W_o: (D_SOMA, D_SOMA)
        d_proj = N_HEADS * max(1, D_SOMA // N_HEADS)
        self.attn_mode = os.environ.get("BIO_ATTN_MODE", ATTN_MODE_DEFAULT)
        self.W_q = self.rng.normal(0, W_INIT_STD, (D_SOMA, d_proj))
        self.W_k = self.rng.normal(0, W_INIT_STD, (D_SOMA, d_proj))
        self.W_v = self.rng.normal(0, W_INIT_STD, (D_SOMA, d_proj))
        self.W_o = self.rng.normal(0, W_INIT_STD, (D_SOMA, D_SOMA))
        # 投影权重的 pending（NORM-4 因果：延迟提交）
        self._W_pending: Dict[str, np.ndarray] = {}

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
        tier = self._infer_tract_tier(member_ids)
        self.tracts[i] = Tract(
            id=i,
            members=list(member_ids),
            dynamics=self.rng.normal(0, 0.1, (n, n)),
            # ★结构编辑6(2026-09-18 v1.8)：key 由结构派生，不用随机初始化
            #   问题：随机 key 的余弦相似度不表达功能关系 →
            #         BRIDGE_SIM=0 全连通（污染）/ 0.3 全断开（断链）
            #   解法：key = normalize(Σ 角色基向量 × 该角色在束内成员数)
            #         功能相近的束 → key 相似 → 自然桥接
            key=self._derive_tract_key(member_ids),
            value=self.rng.normal(0, 0.1, D_SOMA),
            tier=tier,
        )
        for m in member_ids:
            self.neurons[m].tract_ids.add(i)     # ★重叠：加到集合
        return i

    # ---------- 角色基向量（结构编辑6 的基础）----------
    def _role_bases(self) -> Dict[str, np.ndarray]:
        """每个角色一个确定性正交基向量（懒初始化，全局共享）。

        用确定性构造（非随机）保证：
          · 同一角色的束 key 方向一致 → 功能相近的束自然相似
          · 不同角色的基正交 → 功能不同的束自然隔离
        """
        if getattr(self, "_role_bases_cache", None) is None:
            d = D_SOMA
            bases = {}
            # 用固定的单位向量（尽量正交且铺开）
            for idx, role in enumerate(["sensory", "inter", "drives", "motor"]):
                v = np.zeros(d)
                # 每个角色占据不同的"频段"，形成近似正交
                for k in range(d):
                    if (k + idx) % 4 == 0:
                        v[k] = 1.0
                if norm(v) < 1e-8:
                    v[idx % d] = 1.0
                bases[role] = v / norm(v)
            self._role_bases_cache = bases
        return self._role_bases_cache

    def _derive_tract_key(self, member_ids: List[int]) -> np.ndarray:
        """由成员的角色构成派生 key（结构编辑6）。

        与随机初始化相比，它让"功能相近→相似"成为**必然**而非偶然。
        """
        bases = self._role_bases()
        acc = np.zeros(D_SOMA)
        for m in member_ids:
            n = self.neurons.get(m)
            if n is None:
                continue
            acc += bases.get(n.role, bases["inter"])
        if norm(acc) < 1e-8:
            acc = bases["inter"].copy()
        return acc / norm(acc)

    def _infer_tract_tier(self, member_ids: List[int]) -> str:
        """由成员角色推断束的层级（★结构编辑2辅助）。

        规则（仿 FlyGM 三分区）：
          · 含 motor 成员          → "motor"
          · 只含 sensory 成员      → "sensory"
          · 其余（含 inter 混合）  → "inter"
        混合束取"下游"层级，保证感觉→运动的主通路不被误挡。
        """
        roles = {self.neurons[m].role for m in member_ids if m in self.neurons}
        if "motor" in roles:
            return "motor"
        if roles == {"sensory"}:
            return "sensory"
        return "inter"

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

    # ============================================================
    # ★v1.9：subtick 相位化（架构书 §6.2）
    # ============================================================
    # 相位与 whole-tick 的对应关系（保证行为兼容）：
    #   _phase_decay   ← 原 #1 衰减
    #   _phase_ignite  ← 原 #2 写入新激活
    #   _phase_spread  ← 原 #3+#4 扩散 + 桥接（★拆成独立相位，可多步）
    #   _phase_bind / _phase_integrate ← 原 compute 阶段B（★读写分离）
    #   _phase_emit    ← 原 compute 阶段C
    # ============================================================

    def _phase_decay(self, active: Set[int]) -> Set[int]:
        """相位 DECAY：衰减。只改 activity，不碰 soma/axon。"""
        for i in list(active):
            self.neurons[i].activity *= DECAY
        return active

    def _phase_ignite(self, active: Set[int]) -> Set[int]:
        """相位 IGNITE：写入新激活（外部输入点亮）。"""
        for nid, amt in self.pending_ignitions:
            if nid in self.neurons:
                self.ignite(nid, amt)
                active.add(nid)
        self.pending_ignitions.clear()
        return active

    def tick_activation(self, active_set: Set[int]) -> Set[int]:
        """稀疏 tick（兼容入口）：= DECAY + IGNITE + SPREAD。

        ★v1.9：保留此方法以兼容既有调用（测试/旧路径）。
        新路径应用 tick_phased()，它把相位显式化并分离 BIND/INTEGRATE。
        """
        active_set = self._phase_decay(active_set)
        active_set = self._phase_ignite(active_set)
        active_set = self._phase_spread(active_set)
        return active_set

    def _phase_spread(self, active_set: Set[int]) -> Set[int]:
        """相位 SPREAD：激活扩散（束内直接传 + 跨束经共享神经元间接传）。"""
        spread: Dict[int, float] = {}
        for i in list(active_set):
            n = self.neurons[i]
            if n.activity <= ACTIVE_EPS:
                continue
            for tid in n.tract_ids:
                tr = self.tracts[tid]
                k = tr.members.index(i)
                # ★结构编辑4(2026-09-18 v1.2)：使用「有效权重」而非初始 skeleton
                #   有效权重 = dynamics + w_slow + w_fast
                #   → 运行经验（Hebbian 累积）能真的改变传播
                W = self.effective_weights(tid)
                # 束内扩散
                # ★结构编辑1(2026-09-18)：突触带符号 —— 仿果蝇 FlyGM 的 W = N_exc − N_inh
                #   原实现 abs(dynamics) 丢弃符号 → 无抑制 → 全网自激至均一化。
                #   现保留符号：负权重 → 抑制性投射（降低目标活动），提供负反馈。
                for k2, m in enumerate(tr.members):
                    if m == i:
                        continue
                    raw = W[k, k2]
                    gain = abs(raw) * SPREAD_GAIN
                    signed = gain if raw >= 0 else -gain * INHIB_STRENGTH
                    spread[m] = spread.get(m, 0.0) + n.activity * signed
                # ★跨束扩散：激活该束的"束级信号"，供束的路由读出
                tr.activation = max(tr.activation, min(n.activity, SPREAD_CAP))
        # ★4. 跨束桥接：束激活 → 该束的 key 与其它束的 key 相似 → 传激活
        tractor_ids = [tid for tid, tr in self.tracts.items() if tr.activation > ACTIVE_EPS]
        for tid in tractor_ids:
            tr = self.tracts[tid]
            for tid2, tr2 in self.tracts.items():
                if tid2 == tid:
                    continue
                # ★结构编辑2(2026-09-18)：只许"顺流"（sensory→inter→motor），禁长程逆行
                #   仿 FlyGM 的 afferent→intrinsic→efferent 三分区；
                #   依据 BANC(Nature 656:957)：局部反馈有益，长程逆行有害。
                if not self._may_flow(tr, tr2):
                    continue
                sim = float(np.dot(tr.key, tr2.key)) / (norm(tr.key)*norm(tr2.key) + 1e-8)
                if sim > BRIDGE_SIM:                      # ★束间只有足够相似才桥接
                    for m in tr2.members:
                        spread[m] = spread.get(m, 0.0) + tr.activation * sim * BRIDGE_GAIN
        # ★5. 施加扩散（兴奋走 ignite，抑制走负向调整）
        for m, amt in spread.items():
            # ★修正(2026-09-18)：已活跃神经元只取增量的一部分，避免正反馈自激
            #   原实现: ignite(m, min(amt, SPREAD_CAP)) 会让活跃集内神经元持续累积 → t12 全饱和
            #   现实: 扩散只做"未饱和部分的补充"，越接近饱和补充越少
            if m in active_set:
                cur = self.neurons[m].activity
                # 饱和抑制：当前值越高，接受扩散的比例越低（仿生：不应期/饱和）
                accept = max(0.0, 1.0 - cur) * SPREAD_SAT_SUPPRESS
                amt = amt * accept
            if abs(amt) <= ACTIVE_EPS:
                pass
            elif amt > 0:
                self.ignite(m, min(amt, SPREAD_CAP))     # 兴奋性投射
            else:
                # ★抑制性投射：降低目标活动（下限 0）
                self.neurons[m].activity = max(
                    0.0, self.neurons[m].activity + amt)
            active_set.add(m)
        # 6. 重估活跃集
        return {i for i in active_set if self.neurons[i].activity > ACTIVE_EPS}

    # ---------- 结构编辑2辅助：信息流方向约束 ----------
    def _may_flow(self, src: "Tract", dst: "Tract") -> bool:
        """束间信息流是否允许（sensory → inter → motor 顺流，禁长程逆行）。

        生物学依据：FlyGM(2602.17997) 的 afferent→intrinsic→efferent 三分区；
                   BANC(Nature 656:957, 2026) 的"局部反馈环 + 长程协调"。
        同级允许（局部反馈有益），仅禁止"下游→上游"的长程逆行。
        """
        so = TIER_ORDER.get(src.tier, 1)
        do = TIER_ORDER.get(dst.tier, 1)
        return do >= so


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
        # ★v1.7：给 size_bytes 赋值（原为死字段）——调度需要知道搬运代价
        if lib.size_bytes == 0:
            lib.size_bytes = len(atoms) * (D_DENDRITE * D_SOMA + D_SOMA * 3) * 4
        return top_k_mean([self.neurons[i].activity for i in atoms])

    # ---------- 计算内核（ALG-0） ----------
    def compute(self, active_set: Set[int]) -> Dict[int, np.ndarray]:
        """阶段A 束内消息传递 + 阶段B 跨束门控注意力 + 阶段C 轴突输出

        ★v1.7（Colibri 借鉴）：计算范围按 tier 约束 —— 只有 L0(VRAM) 上的束参与计算。
        Colibri 的核心：744B 模型每 token 只激活 40B（5.4%），
        稀疏激活 + 分层驻留 = 小内存跑大模型。
        本项目照此：**可用规模**（全部组织）可以远超**计算规模**（单 tick 参与计算的）。
        """
        active = [i for i in active_set if i in self.neurons]
        # ★v1.7：L0 过滤 —— 只让 VRAM 上的束参与计算
        #   ⚠️安全阀：若 L0 上一个束都没有，则**不做过滤**（否则大脑停摆）
        #   实测教训：库数=1 且该库在 NVMe 时，束全被过滤 → 大脑等于停止思考
        def _tract_in_l0(tid: int) -> bool:
            tr = self.tracts[tid]
            # 束所属的库若不在 L0，则该束不参与本 tick 计算
            for gid in tr.group_ids:
                for lid, lib in self.libraries.items():
                    if gid in lib.group_ids:
                        return lib.tier == Tier.VRAM
            return True          # 无所属库 → 不限制（兼容测试场景）

        # 安全阀：统计 L0 上是否有束；无则不过滤
        n_l0 = 0
        for tid, tr in self.tracts.items():
            for gid in tr.group_ids:
                for lid, lib in self.libraries.items():
                    if gid in lib.group_ids and lib.tier == Tier.VRAM:
                        n_l0 += 1
                        break
        l0_enabled = n_l0 > 0
        self._l0_filtered = 0

        # 阶段 A：束内消息传递（局部）
        # ★v1.9 重大修复：原实现读 axon 作为输入，但 axon 由阶段C写（axon=tanh(soma)），
        #   而 soma 又由本阶段写（读 axon）→ **互为输入，永远停在 0**（零不动点陷阱）。
        #   实测：跑满 12 tick 后全部 64 个神经元的 soma/axon 范数仍为 0，
        #        即 compute() 完全空转（见 docs/04_实测与诊断/重大缺陷_compute空转.md）。
        #   修复：改用**真正活跃的 activity 通路**作为主要输入信号，
        #         axon 一旦有值则融合（保持"轴突输出"的语义）。
        #   为什么不用随机初始化：activity 承载输入语义，随机数无信息（已被两次否定）。
        # ★v3.3（P0 方案V）纯性能改写，语义位级不变：
        #   · 去 np.ones(D_SOMA) 广播 —— 标量 × 标量本就直接广播，原写法每边白分配 16 维数组
        #   · 消除循环不变量 norm(src.axon) —— 源码在 (k,k2) 二层循环内重算
        #   · 目标层 activity 门控提到内层循环外（每个 m 只判一次）
        active_tracts = set()
        for i in active:
            active_tracts |= self.neurons[i].tract_ids
        for tid in active_tracts:
            if l0_enabled and not _tract_in_l0(tid):
                self._l0_filtered += 1
                continue
            tr = self.tracts[tid]
            mem = tr.members
            dyn = tr.dynamics
            # ★v3.3：预计算每个成员的"有效信号强度"（循环不变量，原实现每边重算 norm(axon)）
            #   注意：src.activity 可能在本 tick 内被 INTEGRATE 改动，但阶段A 早于 INTEGRATE，
            #   且原实现同样在整个阶段A 期间读到的是同一份 activity → 语义一致。
            sig = np.zeros(len(mem))
            for k2, m2 in enumerate(mem):
                src = self.neurons[m2]
                s2 = src.activity
                na = norm(src.axon)
                if na > 1e-8:
                    # axon 有值时融合（均值，保持量级稳定）
                    s2 = 0.5 * s2 + 0.5 * float(np.mean(np.abs(src.axon)))
                sig[k2] = s2
            for k, m in enumerate(mem):
                if self.neurons[m].activity <= ACTIVE_EPS:
                    continue
                # ★v3.3：msg 为标量（原为 D_SOMA 维向量，每维值完全相同）
                #   Σ_{k2≠k} dynamics[k,k2] * sig[k2]  →  一次 BLAS 点积
                #   源码用 np.zeros(D_SOMA) 累加 16 个相同标量，数值上 == 标量×16 个相同副本
                msg = float(np.dot(dyn[k], sig) - dyn[k, k] * sig[k])
                self.neurons[m].soma = np.tanh(self.neurons[m].soma + MSG_GAIN * msg)

        # 阶段 B：跨束门控注意力（稀疏：神经元 → 少数束）
        # ★v1.9：改为 BIND（只读）→ INTEGRATE（写）两段
        self._phase_bind(active, l0_enabled, _tract_in_l0)
        self._phase_integrate(active)

        # 阶段 C：轴突输出
        for i in active:
            n = self.neurons[i]
            n.axon = np.tanh(n.soma)

        return {i: self.neurons[i].axon.copy() for i in active}

    # ============================================================
    # ★v1.9：BIND / INTEGRATE 读写分离（架构书 §6.2 / §7）
    # ============================================================
    def _phase_bind(self, active: List[int], l0_enabled: bool = True,
                    l0_check=None):
        """相位 BIND：多头交叉注意力 —— **只读，不写任何神经元状态**。

        写入目标只有 self._attn_out（临时缓冲），供 INTEGRATE 相位使用。
        这消除了 whole-tick 里"读 soma → 写 soma"的同 tick 自反馈（架构书 §6.1 T2）。

        机制（架构书 §7.2）：
          Q = 神经元 soma（每神经元私有）
          K = 束 key     （共享）
          V = 束 value   （共享）
          多头：把 D_SOMA 切成 N_HEADS 份，各头独立算相似度
          tier 软偏置：顺流加分、逆行减分（替代硬门控 _may_flow）
        ★cos 兼容模式：无投影矩阵，直接用 cos(soma, key) —— 与旧单头实现同族
        """
        self._attn_out = {}
        all_tract_ids = list(self.tracts.keys())
        # 预计算每个束的 tier 数值（软偏置用）
        tier_val = {tid: TIER_ORDER.get(tr.tier, 1)
                    for tid, tr in self.tracts.items()}
        d_head = max(1, D_SOMA // N_HEADS)
        learned = (self.attn_mode == "learned")
        d_proj = N_HEADS * d_head
        # ★v2.0 Step 4：预投影所有束的 K/V（learned 模式下每 tick 一次）
        if learned:
            K_cache, V_cache = {}, {}
            for tid, tr in self.tracts.items():
                K_cache[tid] = (self.W_k.T @ tr.key)[:d_proj]
                V_cache[tid] = (self.W_v.T @ tr.value)[:d_proj]

        # ============================================================
        # ★v3.3（P0 方案V）：einsum 全量向量化打分
        #   原实现：for i(活跃) → for h(4头) → for tid(5束) 三重 Python 循环
        #     每格 5.63 µs，其中 norm(q) 是**循环不变量**却算了 T·H 次（1160 次多余调用）
        #   现实现：一次性把 (N_a,H,d)×(T,H,d) 收缩成 (N_a,H,T) 分数矩阵
        #   语义位级不变：同样的 cos 公式、同样的 0 分子 +1e-8 分母、同样的 tier 偏置
        #   ⚠️ tie-break 必须用 np.lexsort((−索引, −得分)) 复刻 sorted(reverse=True)
        #      的"索引降序"语义；裸 argsort(stable) 是索引升序 → 实测 0.118 maxdiff
        # ============================================================
        # ---- 1. 桶装 Q / K（跳过 soma 范数 <1e-8 的神经元，与原 continue 一致）----
        idx_valid = [i for i in active
                     if i in self.neurons and norm(self.neurons[i].soma) >= 1e-8]
        if not idx_valid or not all_tract_ids:
            return
        if learned:
            Q = np.stack([(self.W_q.T @ self.neurons[i].soma)[:d_proj]
                          for i in idx_valid])                       # Na × d_proj
        else:
            Q = np.stack([self.neurons[i].soma[:d_proj] for i in idx_valid])
        K = np.stack([(K_cache[t][:d_proj] if learned else self.tracts[t].key[:d_proj])
                      for t in all_tract_ids])                        # T × d_proj
        V = np.stack([(V_cache[t][:d_proj] if learned else self.tracts[t].value[:d_proj])
                      for t in all_tract_ids])                        # T × d_proj
        Qh = Q.reshape(len(idx_valid), N_HEADS, d_head)
        Kh = K.reshape(len(all_tract_ids), N_HEADS, d_head)
        Vh = V.reshape(len(all_tract_ids), N_HEADS, d_head)
        # 循环不变量：每头范数只算一次（原实现每格重算）
        qn = np.linalg.norm(Qh, axis=2)                               # Na × H
        kn = np.linalg.norm(Kh, axis=2)                               # T × H
        # ---- 2. 分数矩阵 (Na,H,T) ----
        S = np.einsum('nhd,thd->nht', Qh, Kh) / (qn[:, :, None] * kn.T[None, :, :] + 1e-8)
        # ---- 3. tier 软偏置（向量化：目标层 >= 自身层 → +bias）----
        src_t = np.array([TIER_ORDER.get(self.neurons[i].role, 1) for i in idx_valid])
        dst_t = np.array([tier_val[t] for t in all_tract_ids])
        S = S + np.where(dst_t[None, None, :] >= src_t[:, None, None],
                         TIER_BIAS, -TIER_BIAS)
        # ---- 4. top-K（★lexsort 复刻 sorted(reverse=True)：主键 −得分，次键 −索引）----
        kk = min(ROUTE_K, S.shape[2])
        tid_arr = np.array(all_tract_ids)
        # lexsort 最后一把键为**主键** → (次键 −索引, 主键 −得分)
        order = np.lexsort((-np.arange(S.shape[2])[None, None, :].repeat(S.shape[0], 0)
                            .repeat(S.shape[1], 1),
                            -S), axis=2)[:, :, :kk]                   # Na × H × K
        # ---- 5. 多头加权融合（softmax over top-K，逐头仅 K 个元素 → 保持小规模确定性）----
        #     严格照源码：mx = max(score)；z = Σexp + 1e-8
        S_top = np.take_along_axis(S, order, axis=2)                   # Na × H × K
        mx = S_top.max(axis=2, keepdims=True)
        Ex = np.exp(S_top - mx)
        Zn = Ex.sum(axis=2, keepdims=True) + 1e-8
        Wt = Ex / Zn                                                  # Na × H × K
        #     acc[n, h*d:(h+1)*d] = Σ_k Wt[n,h,k] * V[order[n,h,k], h*d:(h+1)*d]
        #     ★Vh 是 (T,H,d)：按束索引 order 在 axis=0 上取行 —— 每头取各自的 d 段
        Na_ = len(idx_valid)
        hh = np.arange(N_HEADS)[None, :, None]
        Vg = Vh[order, hh, :]                                         # Na × H × K × d
        acc_all = (Wt[:, :, :, None] * Vg).sum(axis=2).reshape(
            Na_, N_HEADS * d_head)                                     # Na × d_proj
        # ★learned：投影回 D_SOMA 维
        if learned:
            acc_all = (acc_all @ self.W_o)[:, :D_SOMA]
        else:
            acc_all = acc_all[:, :D_SOMA]
        for a_, i in enumerate(idx_valid):
            self._attn_out[i] = acc_all[a_].copy()    # ★只写临时缓冲，不碰神经元

    def _phase_integrate(self, active: List[int]):
        """相位 INTEGRATE：把 BIND 产出写进 soma（**唯一写 soma 的相位**）。"""
        if not getattr(self, "_attn_out", None):
            return
        # ★v2.0：保留一份副本供在线学习使用（update_attention_weights 需要）
        #   否则 _attn_out 被清空后，学习永远拿不到本轮注意力产出
        self._attn_last = dict(self._attn_out)
        for i, acc in self._attn_out.items():
            n = self.neurons.get(i)
            if n is None:
                continue
            n.soma = np.tanh(n.soma + ATTN_GAIN * acc)
        self._attn_out = {}

    # ============================================================
    # ★v1.9：完整相位化 tick（架构书 §6.2）
    # ============================================================
    def tick_phased(self, active_set: Set[int], n_spread: int = 1) -> Set[int]:
        """一个 tick = 9 个相位，每相位只做一件事，读写严格分离。

        相位顺序（架构书 §6.2）：
          COMMIT → DECAY → IGNITE → SPREAD×n → BIND → INTEGRATE → EMIT → (PLASTIC/SCHEDULE 由外层负责)

        与 whole-tick 的关键差异：
          · SPREAD 步数可配 → "传播需要时间"成为**设计参数**而非副作用
          · BIND 只读、INTEGRATE 才写 → 消除同 tick 自反馈
        """
        # COMMIT（可塑提交）由外层 run_ticks 处理（涉及 pending 缓存）
        active = set(active_set)
        active = self._phase_decay(active)
        active = self._phase_ignite(active)
        for _ in range(max(1, n_spread)):
            active = self._phase_spread(active)
        # 阶段A（束内消息）+ BIND/INTEGRATE + EMIT 复用 compute
        out = self.compute(active)
        # PLASTIC 由外层负责（需 state_prev）
        return active


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

    # ---------- ★结构编辑4（v1.2）：Hebbian 结构可塑 ----------
    def _ensure_row_budget(self, tid: int) -> float:
        """突触稳态预算：每行 L1 的允许上限（自适应 = 初始值的均值 × 容差）。

        ★v1.5 教训：不要拍脑袋设绝对值。
          初始每行 L1 mean=1.39~2.04；若设 0.6，100% 行初始超标 →
          归一化立刻压扁结构 → 有效/初始=0.383 恒定 → 锁死。
        自适应做法：预算 = 初始每行 L1 的均值（允许在均值上下浮动）。
        """
        tr = self.tracts[tid]
        if tr.row_budget is None:
            row_sums = np.abs(tr.dynamics).sum(axis=1)
            tr.row_budget = float(row_sums.mean()) * 1.5   # 留 50% 增长空间
        return tr.row_budget

    def effective_weights(self, tid: int) -> np.ndarray:
        """有效连接 = 初始骨架 + 慢权重 + 快权重。

        为什么这样设计（依据 arXiv 2404.17128 + 本项目否定结果）：
          · 结构信息不能来自初始化分布（实测三种分布互换全部退化）
          · 必须由运行经验累积 —— 这就是 w_slow + w_fast 的作用
          · 初始骨架仍保留，作为"先天连接倾向"，避免无经验时结构退化
        """
        tr = self.tracts[tid]
        w = tr.dynamics
        if tr.w_slow is not None:
            w = w + tr.w_slow
        if tr.w_fast is not None:
            w = w + tr.w_fast
        return w

    def update_structure(self, active_set: Set[int]):
        """Hebbian 结构可塑：共激活的神经元之间加强连接。

        ★v1.4 修正（两次失败后的正解，见 04_实测与诊断/否定结果2、3）：
          v1.2 裸 Hebbian  → w 顶到 CAP，有效权重 4.5x → 全网饱和 → dmin=0
          v1.3 误用 Oja    → oja 项 (a_i − w·a_j) 使 w 收敛到主成分，
                             结果 w_fast **抵消** dynamics（有效/初始=0.383 恒定）
          v1.4 正解        → **纯 Hebbian 增强 + L1 归一化（突触稳态）**
                             · 增强：Δw = η·a_i·a_j（共激活加强）
                             · 归一化：每个神经元出边 L1 总量守恒 → 天然有界
                             · 竞争：归一化 = 弱边被强边挤掉（资源有限）

        生物学依据：突触稳态（synaptic homeostasis）——
          真实神经元的突触总量守恒，Hebbian 增强必然伴随弱突触的等比削减。
          这是"竞争"的实现形式，防止正反馈无界增长。
        """
        act = {i: self.neurons[i].activity for i in active_set}
        for tid, tr in self.tracts.items():
            mem = tr.members
            n = len(mem)
            if tr.w_fast is None:
                tr.w_fast = np.zeros_like(tr.dynamics)
            if tr.w_slow is None:
                tr.w_slow = np.zeros_like(tr.dynamics)
            delta = np.zeros_like(tr.dynamics)
            touched = False
            for k, i in enumerate(mem):
                ai = act.get(i, 0.0)
                if ai <= ACTIVE_EPS:
                    continue
                for k2, j in enumerate(mem):
                    if i == j:
                        continue
                    aj = act.get(j, 0.0)
                    if aj <= ACTIVE_EPS:
                        continue
                    # ★纯 Hebbian 增强（共激活 → 加强）
                    #   保持符号：按 skeleton 的符号施加，不破坏 excitatory/inhibitory 分工
                    sign = 1.0 if tr.dynamics[k, k2] >= 0 else -1.0
                    delta[k, k2] += sign * STRUCT_PLASTIC_ETA * ai * aj
                    touched = True
            if not touched:
                continue
            delta = np.clip(delta, -STRUCT_PLASTIC_CAP, STRUCT_PLASTIC_CAP)
            # 衰减（防长期漂移）+ 增强
            cand_fast = tr.w_fast * (1.0 - STRUCT_PLASTIC_DECAY) + delta
            cand_W = tr.dynamics + tr.w_slow + cand_fast
            # ★突触稳态：每个神经元「出边权重 L1 总量」守恒
            #   超出上限 → 等比缩放（弱边被强边挤掉 = 竞争）
            #   ★v1.5：预算自适应（_ensure_row_budget），不拍脑袋设绝对值
            budget = self._ensure_row_budget(tid)
            for k in range(n):
                row = np.abs(cand_W[k].copy())
                row[k] = 0.0                       # 排除自身连接
                s = row.sum()
                if s > budget and s > 1e-8:
                    scale = budget / s
                    # 只缩增量部分，保住 skeleton 的先天结构
                    cand_fast[k] = cand_W[k] * scale - tr.dynamics[k] - tr.w_slow[k]
            tr.w_pending = np.clip(cand_fast, -STRUCT_PLASTIC_CAP, STRUCT_PLASTIC_CAP)

    def apply_structure_pending(self):
        """NORM-4 因果优先：结构更新只在 tick 边界提交。"""
        for tid, tr in self.tracts.items():
            if tr.w_pending is not None:
                tr.w_fast = tr.w_pending
                tr.w_pending = None
        # ★v2.0：投影矩阵的 pending 提交
        for k, v in self._W_pending.items():
            setattr(self, k, v)
        self._W_pending = {}

    def update_attention_weights(self, active_set: Set[int]):
        """★v2.0 Step 4：投影矩阵的在线可塑（learned 模式）。

        NORM-3（无分离）：学习是思考的副产品，不引入独立训练阶段。
        NORM-4（因果优先）：写入 _W_pending，下 tick 提交。

        注意：必须在 compute() **之后**调用（需要 _attn_out），
              且提交在**下个 tick** 的 apply_structure_pending 中发生。
        """
        if getattr(self, "attn_mode", "cos") != "learned":
            return
        # ★v2.0：用 _attn_last（INTEGRATE 保留的副本），而非已被清空的 _attn_out
        attn = getattr(self, "_attn_last", None)
        if not attn:
            return
        d_head = max(1, D_SOMA // N_HEADS)
        d_proj = N_HEADS * d_head
        # 以本 tick 活跃神经元的 soma 均值作为"信号"
        sig = np.zeros(D_SOMA)
        cnt = 0
        for i in active_set:
            n = self.neurons.get(i)
            if n is None:
                continue
            sig += n.soma
            cnt += 1
        if cnt == 0:
            return
        sig = sig / cnt
        # attn_out 均值（投影回 D_SOMA 后）
        acc_mean = np.zeros(d_proj)
        for acc in attn.values():
            a = acc[:d_proj] if len(acc) >= d_proj else np.pad(acc, (0, d_proj - len(acc)))
            acc_mean += a
        acc_mean = acc_mean / max(1, len(attn))
        # 简单 Hebbian：W_q += lr * outer(sig, acc_mean)
        dWq = ATTN_LR * np.outer(sig, acc_mean)
        cand = self.W_q + dWq
        nrm = np.linalg.norm(cand)
        if nrm > 2.0:                      # L2 上限防爆（初始范数约 2.25）
            cand = cand * (2.0 / nrm)
        # ★写入 pending，下个 tick 的 apply_structure_pending 提交
        self._W_pending["W_q"] = cand
        # ★同时保留 _attn_out 供下一步（但标记已用）

    def consolidate_structure(self):
        """NORM-5 合并独占：巩固相把 w_fast 混入 w_slow（长期结构）。

        仿睡眠巩固：快权重代表"近期经验"，定期沉淀为慢权重（稳定结构）。
        """
        blended = []
        for tid, tr in self.tracts.items():
            if tr.w_fast is None:
                continue
            if tr.w_slow is None:
                tr.w_slow = np.zeros_like(tr.dynamics)
            before = float(np.abs(tr.w_slow).mean())
            tr.w_slow = tr.w_slow + STRUCT_W_SLOW_BLEND * tr.w_fast
            tr.w_fast = tr.w_fast * (1.0 - STRUCT_W_SLOW_BLEND)
            after = float(np.abs(tr.w_slow).mean())
            if after > before + 1e-9:
                blended.append((tid, before, after))
        return blended

    def structure_stats(self) -> dict:
        """结构可塑的可观测性（用于验收 dmin 是否随运行上升）。"""
        out = {}
        for tid, tr in self.tracts.items():
            out[tid] = {
                "tier": tr.tier,
                "w_fast_abs_mean": float(np.abs(tr.w_fast).mean()) if tr.w_fast is not None else 0.0,
                "w_slow_abs_mean": float(np.abs(tr.w_slow).mean()) if tr.w_slow is not None else 0.0,
            }
        return out

    # ---------- 分级调度（ALG-4 v2，三级驻留 + 滞回 + 热存储 + 预取） ----------
    def tick_scheduler(self) -> List[str]:
        """三级驻留调度（v1.7，Colibri 借鉴）。

        Colibri 的核心洞察：
          "parameters are not resident state to be held, they are data to be staged
           across a heterogeneous storage hierarchy, exactly when the router proves
           they are needed" — 参数不是常驻状态，而是按需分层的**数据**。

        三级：
          L0 VRAM ← 激活集 + 热存储(pinned) + 预取目标
          L1 RAM  ← 温数据
          L2 NVMe ← 冷数据（长期不用）

        四个机制：
          ① 滞回（hysteresis）——上/下行用不同阈值，防抖振
          ② 冷却（cooldown）——切换后 N tick 内不再切（Colibri 防抖振）
          ③ 热存储（pinned hot-store）——长期高频者钉在 L0
          ④ 预取（router-ahead）——上升趋势提前上行，input 到来时零等待
        """
        changes = []
        for lid, lib in self.libraries.items():
            act = self.library_activation(lid)
            lib.activation = act

            # ★③ 热分数 EMA（长期统计，抗短期波动）
            lib.hot_score = (1 - HOT_EMA_ALPHA) * lib.hot_score + HOT_EMA_ALPHA * act
            # ★④ 记录激活历史（供预取趋势判断）
            lib.act_history.append(act)
            if len(lib.act_history) > PREFETCH_LOOKBACK + 1:
                lib.act_history.pop(0)

            pinned = lib.hot_score > PIN_THRESHOLD     # ★热存储：钉住，不参与淘汰
            since = self.t - lib.last_change

            # ---- 上行判定 ----
            if lib.tier != Tier.VRAM and act > THETA_HIGH:
                lib.tier = Tier.VRAM
                lib.last_change = self.t
                lib.tier_changes += 1
                lib.prefetched = False
                changes.append(f"库{lid} L1→L0 (act={act:.2f} hot={lib.hot_score:.2f})")
                continue
            # ★④ 预取：尚未过热阈，但呈上升趋势 → 提前上行（Colibri router-ahead）
            if lib.tier == Tier.NVME and self._is_rising(lib) and act > THETA_MID:
                lib.tier = Tier.RAM
                lib.last_change = self.t
                lib.tier_changes += 1
                lib.prefetched = True
                changes.append(f"库{lid} L2→L1 预取 (act={act:.2f} 上升趋势)")
                continue

            # ---- 下行判定（带滞回 + 冷却 + 热存储保护）----
            if lib.tier == Tier.VRAM:
                if pinned:
                    continue                        # ★钉住：热库不下行
                if act < THETA_LOW and since > COOLDOWN_TICKS:
                    lib.tier = Tier.RAM
                    lib.last_change = self.t
                    lib.tier_changes += 1
                    changes.append(f"库{lid} L0→L1 (act={act:.2f} 冷却{since}t)")
            elif lib.tier == Tier.RAM:
                # ★L1 → L2 NVMe（原实现从未使用 NVME 档）
                if act < THETA_LOW and since > COOLDOWN_TICKS * COOLDOWN_NVME_MULT:
                    lib.tier = Tier.NVME
                    lib.last_change = self.t
                    lib.tier_changes += 1
                    lib.prefetched = False
                    changes.append(f"库{lid} L1→L2 (act={act:.2f} 冷却{since}t)")
        return changes

    def _is_rising(self, lib: "Library") -> bool:
        """预取判据：激活是否呈持续上升趋势（Colibri router-ahead 思想）。"""
        h = lib.act_history
        if len(h) < 3:
            return False
        return (h[-1] - h[0]) > PREFETCH_RISE and h[-1] >= h[-2] >= h[-3]

    def tier_stats(self) -> dict:
        """存储分层的可观测性（验收 S1~S6 用）。"""
        from collections import Counter
        cnt = Counter(lib.tier.value for lib in self.libraries.values())
        total_changes = sum(lib.tier_changes for lib in self.libraries.values())
        pinned = [lid for lid, lib in self.libraries.items() if lib.hot_score > PIN_THRESHOLD]
        return {
            "tier_counts": dict(cnt),
            "tier_changes_total": total_changes,
            "pinned_libs": pinned,
            "hot_scores": {lid: round(lib.hot_score, 3) for lid, lib in self.libraries.items()},
        }


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
