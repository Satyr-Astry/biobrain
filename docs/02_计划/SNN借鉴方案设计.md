# SNN 借鉴方案设计（v1.0）

> 落笔：2026-09-19
> 依据：`code/` 实测 + SNN 领域实证文献（附 arXiv 编号可复查）
> 状态：**设计文档**。按 NORM-8，本文件先于代码；批准后写入 `总架构书.md`，再动代码。

---

## 0. 动机：两个实测确认的结构性缺陷

### 0.1 缺陷 A：稀疏度严重不足（实测）

| 阈值 | 本项目活跃率（实测） | 生物/SNN 参考 | 差距 |
|---|---|---|---|
| 0.01 | **93.75%** | 1-5% | **19-94×** |
| 0.10 | **90.62%** | 1-5% | **18-91×** |
| 0.20 | **89.06%** | 1-5% | **18-89×** |
| 0.50 | **73.44%** | 1-5% | **15-73×** |

**实测命令**：
```python
b = CogVec(seed=1); b.think("猫是哺乳动物")
acts = np.array([n.activity for n in b.ns.neurons.values()])
np.mean(acts > 0.1)   # → 0.9062
```

**结论**：架构建了 `ACTIVE_EPS` 稀疏机制，但**它没有在工作** —— 即使阈值提到 0.5，仍有 73% 活跃。

### 0.2 缺陷 B：无目标驱动学习（实测）

```
grep -n "backward|gradient|loss|optimizer|Adam|SGD|def train|def fit" bio_brain.py
→ 只命中 2 处注释（bio_brain.py:77 / :938），都在说"不引入独立训练阶段"
```

**项目现状**：
| 三因子 | 项目状态 |
|---|---|
| ① 前/后活动 | ✅ 有（`activity`） |
| ② Hebbian 相关性 | ✅ 有（`update_structure`） |
| ③ **目标/误差信号** | ❌ **完全没有** ← **这就是"无训练机制"的根源** |

**结论**：项目只有**相关性学习（Hebbian）**，**没有误差信号** → 网络**不知道自己对不对**。

---

## 1. 本项目与 SNN 的关系（定位澄清）

### 1.1 本项目不是 SNN（实测证据）

| SNN 判定标准 | 本项目实测 | 符合？ |
|---|---|---|
| **离散脉冲（0/1）** | `activity` 64 个神经元 **61 个不同值**（0.0~0.9374） | ❌ **连续值** |
| **膜电位积分**（LIF 方程） | 无 `τ dV/dt = -V + I`；`soma = tanh(soma + ATTN_GAIN*acc)` 是**注意力加权和** | ❌ |
| **阈值触发 + reset** | 无发放阈值、无 reset 机制 | ❌ |
| **时序编码**（脉冲精确时刻） | 有 tick/subtick 时序，但**不是脉冲时刻编码** | ❌ 部分 |

### 1.2 正确定位

> **本项目 = 生物启发的连续值向量神经元网络**
> （biologically-inspired continuous-valued vector-neuron network）
> ≈ SNN 的**连续松弛版本** + 注意力机制

**可借鉴 vs 不可借鉴**：

| SNN 特性 | 可借鉴性 | 理由 |
|---|---|---|
| **稀疏编码机制**（k-WTA/侧抑制） | ✅ **高度可借鉴** | 与连续值**不冲突**，是动力学层 |
| **三因子学习规则** | ✅ **高度可借鉴** | 不需要脉冲，只需"调制因子" |
| **dual-timescale trace** | ✅ **可借鉴** | 项目已有 `w_fast`/`w_slow` 基础 |
| **Homeostatic 阈值** | ✅ **可借鉴** | 纯动力学 |
| 脉冲二值化 | ❌ **不建议** | 会**丧失可微性**，且与"无分离学习"冲突 |
| 代理梯度训练 | ❌ **不建议** | 违反 NORM-3 |
| 神经形态硬件映射 | ❌ **不适用** | 项目目标不是低功耗硬件 |

**关键理由（实证）**：SNN 有硬伤 ——
> "Spike-based activation of SNNs is **not differentiable**, thus gradient descent-based BP is not available."
> —— Wikipedia `Spiking_neural_network`（引 Maass 1997）

---

## 2. 借鉴方案（4 项，按优先级与风险排序）

### 🥇 S1 — k-WTA + 侧抑制 + Homeostatic 阈值【治缺陷 A】

#### 2.1.1 实证依据

| 文献 | 关键结论 |
|---|---|
| `arXiv:2205.10338` | "**a network of 150 spiking neurons can efficiently represent objects with as little as 40 spikes**"（≈27% 活跃）；用 **spike-latency coding + WTA inhibition (WTA-I)** |
| `PMC13406919` | 皮层 **E:I ≈ 4:1**；**"unsupervised competitive learning frameworks based on cooperative action of STDP and lateral inhibition have been shown to promote emergence of neuronal selectivity"**；稀疏编码是皮层表示自然刺激的重要策略 |
| `PubMed 31905147` | **"sparse coding allows only a few neurons to respond from input images"**；**"After sending out a spike, the threshold value of postsynaptic neuron increases due to homeostasis"** |
| `arXiv:2501.17266` | **"hard Winner-Takes-All (WTA) competition, Gaussian lateral inhibition mechanisms and Bienenstock–Cooper–Munro (BCM) learning rule in a single model"**；"neural competition... prevents redundant feature learning" |

#### 2.1.2 机制设计

```
当前（无效）：
    activity 全体衰减（DECAY=0.9）→ 无竞争，90% 保持活跃

改进（k-WTA + 侧抑制 + 稳态）：
    相位 INHIB（新增，紧跟 DECAY 之后）：
        ① 排序：按 activity 降序
        ② 取前 K 名胜出（K = ceil(N_active × SPARSE_TARGET)，SPARSE_TARGET≈0.05）
        ③ 败者被侧抑制：activity *= INHIB_FACTOR（如 0.1）
        ④ 稳态调节（防复燃）：
           胜者：threshold += HOMEOSTASIS_UP      （发放后阈值升高）
           全体：threshold -= HOMEOSTASIS_DECAY   （静默时缓慢回落）
        ⑤ 返回 winners 作为新的 active_set
```

#### 2.1.3 伪代码（可直接落地）

```python
# bio_brain.py 新增常量
SPARSE_TARGET = 0.05        # 目标活跃率（生物 1-5%，取上界稳健）
INHIB_FACTOR = 0.10         # 侧抑制强度（败者活动乘子）
HOMEOSTASIS_UP = 0.02       # 胜者发放后阈值增量
HOMEOSTASIS_DECAY = 0.005   # 静默阈值回落
THETA_MIN, THETA_MAX = 0.05, 0.80   # 阈值上下界

class Phase(Enum):
    ...
    DECAY  = 1
    INHIB  = 11       # ★新增：k-WTA + 侧抑制（只改 activity/threshold，不碰 soma/axon）
    IGNITE = 2
    ...

def _phase_inhibit(self, active: Set[int]) -> Set[int]:
    """相位 INHIB：k-WTA 竞争 + 侧抑制 + 稳态阈值。

    ★只改 activity 与 threshold，不碰 soma/axon（遵守读写分离铁律）。
    ★不引入权重改动 → 属拓扑/动力学层（项目已证"有效"的方向）。
    """
    if not active:
        return active
    acts = sorted(active, key=lambda i: self.neurons[i].activity, reverse=True)
    K = max(1, int(round(len(acts) * SPARSE_TARGET)))
    # 但不低于下限（防 64 神经元时只剩 3 个）
    K = max(K, SPARSE_MIN)

    winners = acts[:K]
    losers = acts[K:]

    # ① 侧抑制：败者被压回
    for i in losers:
        self.neurons[i].activity *= INHIB_FACTOR

    # ② 稳态：胜者阈值升高（防"永远赢"）
    for i in winners:
        n = self.neurons[i]
        n.threshold = min(THETA_MAX, n.threshold + HOMEOSTASIS_UP)

    # ③ 稳态：全体缓慢回落
    for i in active:
        n = self.neurons[i]
        n.threshold = max(THETA_MIN, n.threshold - HOMEOSTASIS_DECAY)

    return set(winners)
```

**插入位置**：`tick_phased` 里 `DECAY` 之后、`IGNITE` 之前。

#### 2.1.4 预期效果与验收

| 指标 | 现状 | 目标 | 怎么测 |
|---|---|---|---|
| **活跃率** | 90.6% | **5-10%** | `np.mean(acts > 0.1)` |
| **计算量** | 全量 | **降 10-18×** | 计时对比（叠加在 v3.3 的 7.33× 之上） |
| **dmin（区分度）** | 0.0088 | **不退化** | `diag_discrim.py`（**注意：稀疏化会改变 dmin，是行为变更非 bug，必须重标定**）|
| 回归测试 | 37 passed | **37 passed** | `PYTHONHASHSEED=0 python -m pytest test_bio_brain.py -q` |

#### 2.1.5 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **k 太小导致信息坍缩** | 64 神经元 × 5% = 3.2 个 → 可能只有 3 个神经元表达一切 | `SPARSE_MIN` 下限（建议 ≥8）；**A/B 测试 k=8/12/20** |
| **需要重标定** | 稀疏化后 `PLASTIC_THRESHOLD`/`SPREAD_*`/`THETA_*` 全部失准 | **必须重标定**，不可用旧参数 |
| **与 v3.3 优化冲突** | 上一轮的 208/208 路由一致性是在**稠密**条件下验证的 | 稀疏化后**必须重跑** `_p0_accept.py` |
| **历史教训** | 本项目已证"权重层无效"，但 **k-WTA 是动力学层**（改 activity/threshold，不改权重） | 严格守住"不碰 `dynamics`/`w_fast`" |

---

### 🥈 S2 — 三因子学习规则（加 neuromodulator）【治缺陷 B】

#### 2.2.1 实证依据

| 文献 | 关键结论 |
|---|---|
| `arXiv:2605.00402` | **"biologically motivated learning framework that combines: (i) population-based winner-take-all (WTA) teaching signals at the output layer, (ii) fixed random broadcast alignment feedback pathways, and (iii) low-dimensional modulatory neuron populations that gate synaptic updates through three-factor learning rules with eligibility traces"** —— **无需反向传播或代理梯度** |
| `arXiv:2509.14447` | **"local three-factor learning rules with dual-timescale eligibility traces... combines error-modulated Hebbian updates, fast/slow trace consolidation, and adaptive learning rate control, requiring only O(1) memory versus O(T) for BPTT"** |
| `arXiv:2601.08526` | SADP：把类标签编成输出脉冲模式，**用 Cohens κ 衡量隐层神经元与正确类输出的一致性** → "**without any gradient computation or external reward signal**"；比 STDP **提升 23.66 个百分点**，**训练快 1.47×** |
| `arXiv:2603.00710` | 三因子 STDP 方程 = 标准理论动机；**"dopamine-like reward can serve as a third factor linking delayed outcomes to local eligibility traces"** |
| R-STDP 综述 | **"STDP 窗口由乙酰胆碱/多巴胺在不同相位门控"**（exploration/exploitation 分离） |

#### 2.2.2 机制设计（核心公式）

```
Δw = η · e(t) · M(t)
     ↑     ↑      ↑
     学习率  资格迹  调制因子（第三因子）

其中：
  e(t) = 资格迹（eligibility trace）—— Hebbian 相关性的时间累积
  M(t) = 调制因子 —— 目标/误差/奖励信号
```

**项目现状对应**：
| 因子 | 项目现状 | 缺什么 |
|---|---|---|
| η 学习率 | ✅ `Neuron.eta`（已加未接线） | 接线 |
| e(t) 资格迹 | ✅ `plasticity` + `plasticity_pending`（ALG-3 延迟可塑） | 复用 |
| **M(t) 调制** | ❌ **完全没有** | **新增** |

#### 2.2.3 伪代码（**关键：不违反 NORM-3 和 NORM-6**）

```python
# 新增：调制因子（第三因子）
class NervousSystem:
    def __init__(self, ...):
        self.neuromod = 0.0          # 当前调制强度
        self._neuromod_source = None # 来源标记（审计用）

    def set_neuromodulator(self, signal: float, source: str) -> bool:
        """设置调制因子（第三因子）。

        ★NORM-6 铁律：只有外部可验证信号才能置非零，自评一律拒绝。
        ★NORM-3 铁律：这不是"训练阶段"，是思考流程里自然产生的调制信号。
        """
        if source not in RELIABLE_SOURCES:
            # 自评禁用（项目已有此铁律，见 ExperienceBuffer.RELIABLE_SOURCES）
            return False
        self.neuromod = float(np.clip(signal, -1.0, 1.0))
        self._neuromod_source = source
        return True

# update_structure 里使用（改动极小）
def update_structure(self, active_set):
    ...现有 Hebbian 逻辑...
    # ★第三因子：调制 Hebbian 更新的强度与方向
    if abs(self.neuromod) > 1e-6:
        pending *= (1.0 + self.neuromod)    # 正向调制 → 加强；负向 → 削弱
    ...提交...
```

**为什么这不违反 NORM-3**：
> NORM-3 说"学习是思考的副产品，**不引入独立训练阶段**"。
> 三因子规则**不需要独立阶段** —— `neuromod` 由思考流程里的**外部反馈**自然设置
> （如 oracle 判定、用户行为、任务结果），是**思考的副产品**，**符合 NORM-3 精神**。

#### 2.2.4 关键前提：**需要外部可验证信号源**

| 信号源 | 项目现状 | 可用于 S2？ |
|---|---|---|
| `oracle` | 无（`_produce_output()` 返回 None） | ❌ 暂不可用 |
| `user_action` | HTTP API 有 `/teach` 端点 | ⚠️ 理论可用 |
| `outcome` | `self_model.record_outcome()` 存在 | ✅ **可用** |
| 自评 | `ExperienceBuffer.RELIABLE_SOURCES` 已禁用 | ❌ 铁律禁止 |

**⚠️ 诚实结论**：**S2 需要有"能判对错"的外部源**，
而项目**当前没有生成后端**（`has_generation_backend()` 返回 False），
所以 **S2 的完整落地依赖"接生成后端"或"定义可验收任务"**（见 `实际任务支持方案_v1.md`）。

**可先做的部分**：把 `set_neuromodulator()` 接口建好，用 `outcome` 作为临时信号源做验证。

---

### 🥉 S3 — Dual-timescale eligibility trace（衔接 Zenke 规则）

#### 2.3.1 实证依据

| 文献 | 关键结论 |
|---|---|
| `arXiv:2509.14447` | **"dual-timescale eligibility traces"** + **"fast/slow trace consolidation"** |
| `Nogarx/Spark` 的 `zenke_rule.py`（**源码已核**） | `pre_tau=20` / `post_tau=20` / **`post_slow_tau=100`** / **`target_tau=1200000`**；`c=-400`（异突触项，负值→竞争）；`p=20`（双势阱）；注释：**"the target moves on the time scale of consolidation rather than of activity"** |

#### 2.3.2 与项目现有字段的对应

| 时间尺度 | Zenke 参数 | 项目字段 | 状态 |
|---|---|---|---|
| 快（活动） | `post_tau=20` | `plasticity`（ALG-3） | ✅ 已用 |
| **慢（短期历史）** | `post_slow_tau=100` | **`w_slow`** | ⚠️ **字段已存在但未用** |
| **最慢（巩固）** | `target_tau=1200000` | 无 | ❌ **需新增 `w_target`** |

**这正是项目缺的"慢速时间尺度"**，也正好实现 **NORM-3 的"睡眠时目标移动、活动时迹移动"**。

#### 2.3.3 落地

```python
# Tract 新增字段
w_target: Optional[np.ndarray] = None   # 权重目标（最慢时间尺度）

# COMMIT 相位里（每 tick）
tau_ratio = 1.0 / TARGET_TAU_STEPS      # 如 1/10000
tr.w_target += tau_ratio * (tr.w_slow - tr.w_target)   # 目标缓慢追迹
```

**Trigger 睡眠鞏固的判据**（替代拍脑袋的"累积 N 条"）：
```python
drift = float(np.abs(tr.w_fast - tr.w_target).mean())
if drift > SLEEP_TRIGGER_DRIFT:   # 漂移够大 → 需要巩固
    consolidator.sleep()
```

---

### 4️⃣ S4 — 分层调制的探索/利用切换（R-STDP 的 ACh/DA 门控）

**实证**：R-STDP 综述 —— **"Sequential neuromodulation uses STDP windows gated by acetylcholine and dopamine in separate phases for exploration and exploitation"**

**项目可用**：`neuromod` 的**符号**决定探索/利用：

| neuromod | 模式 | 对更新的影响 |
|---|---|---|
| > 0 | **利用**（exploit） | 加强当前 Hebbian 方向 |
| < 0 | **探索**（explore） | 反转/削弱当前方向 |
| ≈ 0 | 中性 | 纯 Hebbian |

**风险**：低（只是调制符号）。

---

## 3. 实施顺序（严格按风险排序）

```
Step 0（NORM-8）：把本方案要点写入 总架构书.md
        · 新增 §2.3「稀疏度实测」与 §2.4「学习机制实测」
        · ALG 总表新增 ALG-20（k-WTA 侧抑制）、ALG-21（三因子调制）
        · 缺陷表新增 P-ARCH-18（稀疏度不足）、P-ARCH-19（无目标驱动学习）

Step 1  S1：k-WTA + 侧抑制 + Homeostatic 阈值【最高优先】
        · 改：bio_brain.py 新增 Phase.INHIB + _phase_inhibit + 5 个常量
        · 先备份 code/_backup_before_inhib/
        · 验收：活跃率 90%→5-10%，37 passed，dmin 重标定后再评估
        · ⚠️ 必须先做 A/B（k=8/12/20）找最优 k

Step 2  S3：dual-timescale trace【与 S1 独立，可并行】
        · 改：Tract 加 w_target 字段 + COMMIT 相位加目标追迹
        · 验收：w_target 随 tick 漂移；sleep 触发判据改为 drift

Step 3  S2：三因子调制【需外部信号源】
        · 前置：先有可验收任务或接生成后端
        · 改：set_neuromodulator() + update_structure 里的调制
        · 验收：neuromod != 0 时 Hebbian 更新确实被调制

Step 4  S4：探索/利用切换【S2 之后】
```

**每步独立可回滚**（项目铁律）。

---

## 4. 【不要做什么】

| 禁止项 | 理由 |
|---|---|
| ❌ **做脉冲二值化（activity → 0/1）** | SNN 的硬伤是**不可微**（Wikipedia/Maass 1997）；项目有"无分离学习"铁律，二值化会摧毁在线可塑 |
| ❌ **引入代理梯度/反向传播** | 违反 NORM-3「不引入独立训练阶段」 |
| ❌ **改 `dynamics` / `w_fast` / `w_slow` 的数值** | 本项目七项实证明：**权重层改动全部无效**（-39%~-92%） |
| ❌ **照搬 SNN 的稀疏度到 1-5% 不验证** | 64 神经元 × 1% = 0.64 个 → **必须设 `SPARSE_MIN` 下限**；且要 A/B 找最优 |
| ❌ **稀疏化后用旧参数** | 上一轮子代理明确警告："稀疏化会改变 dmin（是行为变更非 bug），**必须重标定参数**" |
| ❌ **用自评信号当 neuromod** | NORM-6 铁律（`ExperienceBuffer.RELIABLE_SOURCES` 已禁用自评） |
| ❌ **同时改多个相位** | 一次只改一个，否则无法归因（历史四轮互相污染就是这么来的） |

---

## 5. 可验证的验收指标

| ID | 指标 | 现状实测 | 目标 | 测法 |
|---|---|---|---|---|
| **V1** | 活跃率（阈值 0.1） | **90.62%** | **5-10%** | `np.mean(acts > 0.1)` |
| **V2** | 活跃率（阈值 0.5） | **73.44%** | **< 15%** | 同上 |
| **V3** | 计算量 | 基线 | **降 ≥5×**（叠在 v3.3 之上） | `_p0_accept.py` 计时 |
| **V4** | 回归测试 | 37 passed | **37 passed** | `PYTHONHASHSEED=0 python -m pytest test_bio_brain.py -q` |
| **V5** | 路由一致性 | 208/208 | **208/208**（稀疏化后重跑） | `_p0_accept.py cos` |
| **V6** | dmin（区分度） | 0.0088 | **不退化**（需重标定后评估） | `diag_discrim.py` |
| **V7** | `soma`/`axon` 非零 | 52/52 | **不退化** | guard.py A1 |
| **V8** | **neuromod 生效** | N/A | `neuromod != 0` 时更新确实被调制 | 单元测试 |
| **V9** | **w_target 漂移** | N/A | 随 tick 单调趋近 `w_slow` | 打印时序 |
| **V10** | **睡眠自动触发** | 从不触发 | `drift > 阈值` 时自动 `sleep()` | 日志 |

### 反判据（防"用错的数字骗过验收"）

| 反判据 | 说明 |
|---|---|
| **活跃率低 ≠ 稀疏化成功** | 可能是"全都死了"（activity 全 0）。**必须同时检查 dmin 不退化** |
| **快 ≠ 对了** | v3.3 的教训：**必须位级/数值一致性断言**，不能只看加速比 |
| **k-WTA 后 dmin 升高 ≠ 变好** | 样本少时 dmin 是**极值统计量**（噪声比 **0.77**）→ 必须用 **dmean** + 多 seed 聚合 |
| **neuromod 非零 ≠ 学到东西** | 必须跑**对照实验**（`neuromod=0` vs `≠0` 的差分） |

---

## 6. 数据来源（可复查）

| 内容 | 来源 |
|---|---|
| SNN 三大定义、不可微 | Wikipedia `Spiking_neural_network`；Maass 1997, Neural Networks 10(9):1659-1671 |
| "SNN 是第三代神经网络" | `arXiv:2204.07050` "Recent Advances and New Frontiers in SNNs" |
| 150 神经元用 40 脉冲 | `arXiv:2205.10338` |
| E:I = 4:1、侧抑制促选择性 | `PMC13406919` |
| WTA + 发放后阈值升高（稳态） | `PubMed 31905147` |
| Hard-WTA + Gaussian 侧抑制 + BCM | `arXiv:2501.17266` |
| 三因子 + WTA 教学 + 广播对齐（无 BP） | `arXiv:2605.00402` |
| dual-timescale eligibility trace | `arXiv:2509.14447` |
| SADP（Cohen's κ 一致性，快 1.47×） | `arXiv:2601.08526` |
| R-STDP 的 ACh/DA 门控 | EmergentMind "Reward-Modulated STDP" |
| Zenke 三迹规则参数 | `github.com/Nogarx/Spark` → `spark/nn/components/plasticity/zenke_rule.py`（源码已核，91 行） |

---

## 7. 一句话总结

> **项目不是 SNN，但有 SNN 能解的两个病**：
> ① 稀疏度差 18-94 倍 → 借鉴 **k-WTA + 侧抑制 + 稳态阈值**（S1，最高优先）
> ② 无目标驱动学习 → 借鉴 **三因子规则**（S2，需外部信号源，不违反 NORM-3）
> **明确不做**：脉冲二值化、反向传播、任何权重层改动。
> **全部改动落在动力学/拓扑层** —— 这是本项目七项实证证明**唯一有效**的方向。
