# SNN（脉冲神经网络）专项评估：现状 · 优劣 · 与 CogVec 的关系

> **落笔**：2026-09-19
> **定位**：SNN 是独立于 ANN/CogVec 的**第三极**（"第三代神经网络"）
> **铁律**：所有结论**均有 arXiv/URL 出处**（web_search 实证）
> **★最震撼的发现**：**QKFormer 已在 ImageNet 上超过 DeiT-B/Swin-T**（同参数量），能耗仅 1/2

---

# 第一部分 · SNN 是什么（三句话）

```
① 神经元发【离散脉冲】（0/1），不是连续值
② 有时序：膜电位累积 → 超阈值 → 发放 → 重置
③ 稀疏 + 事件驱动：只有发放时才计算（accumulate 而非 multiply）
```

**官方定位**（arXiv:2409.02111 综述）：
> *"SNNs... regarded as the **third generation** of neural networks"*

---

# 第二部分 · ★★ 硬数据：SNN 已到什么水平（这是最重要的部分）

## 2.1 ★★★ ImageNet 分类：**SNN 已经超过同等 ANN**

### QKFormer（arXiv:2403.16552）—— **里程碑**

| 模型 | 类型 | 参数 | 时间步 | 精度 | 能耗 |
|---|---|---|---|---|---|
| **QKFormer** | **SNN** | **64.96M** | **4** | **85.65%** | **113.64 mJ** |
| Swin Transformer | ANN | 88M | 1 | 84.5% | 216.20 mJ |
| DeiT-B | ANN | 86M | 1 | 83.1% | 254.84 mJ |
| ViT | ANN | 85.59M | 1 | 77.9% | 254.84 mJ |
| Spikformer（前作） | SNN | 66.34M | 4 | 74.81% | — |

**★原文的关键论断**：
> *"Under the same experiment conditions **without pre-training or extra training data**, our QKFormer has **surpassed the most well-known Transformer-based ANNs** in performance while maintaining **high energy efficiency**."*

**★这是第一次"直接训练的 SNN 超过 85%"**：
> *"the **first time** that a directly training SNN has achieved an accuracy of over **85%** on ImageNet-1K"*

## 2.2 其他 SOTA（都在快速进步）

| 模型 | 参数 | 时间步 | ImageNet Top-1 | 出处 |
|---|---|---|---|---|
| **SAFformer** | 26.58M | 4 | **80.44%** | arXiv:2605.08270 |
| Max-Former | 28.65M | — | 79.86% | 同上 |
| Spikformer V2 | 172M | **1** | **81.10%** | arXiv:2401.02020 |
| Spikformer V2-8-512 | — | 4 | 80.38% | 同上 |

**★SAFformer 的性价比（最亮眼）**：
```
SAFformer:  26.58M 参数 → 80.44%，能耗 5.88 mJ
对比 S-Transformer v2: 31.30M → 77.20%（参数少 15.1%，精度高 2.24%）
★ 能耗 5.88 mJ（DeiT-S 是 21.20 mJ，DeiT-B 是 80.50 mJ）
```

## 2.3 ★★ 语言方向（最接近生产）

### WD-Spikingformer（arXiv:2604.11321）

| 模型 | 参数 | 准确率 | 能耗 |
|---|---|---|---|
| **WD-Spikingformer** | **1.0B** | **43.6%** | **574.7 mJ** |
| Qwen-1.5B | 1.5B | 44.3% | **3398.3 mJ** |

**★结论**：
```
准确率： 43.6% vs 44.3%   ← 差 0.7%（几乎持平）
能耗：   574.7 vs 3398.3  ← ★只有 17%
★ 关键：用 WTA 替代 softmax → Top-K 机制能效高 8×
```

### NeuTransformer（arXiv:2510.00133）—— **GPT-2 转换**

> *"between **64.71% and 85.28% reduction** in estimated energy consumption when implementing the self-attention"*

**★但诚实承认天花板**：
> *"for **model sizes greater than 300M parameters**, the performance of the converted SNN **degrades beyond an acceptable threshold**"*

---

# 第三部分 · ★ SNN 的五大优势（带实证）

## 优势 1：**能耗**（核心卖点，但有条件）

| 项 | 数据 | 出处 |
|---|---|---|
| 神经形态硬件上 | **38-57×** 能效 | arXiv:2604.15769 |
| 实测（CIFAR-10） | 最高 **3×** 能效（vs 匹配 ANN） | MDPI 2673-4117/6/11/304 |
| 硬件实测（Xpikeformer） | **13×** 能耗降低（同吞吐） | arXiv:2408.08794 |
| 人脑对比 | **20 瓦** vs 数据中心 100+ MW | NIST/IBM |

**★但注意**：
> *"SNNs derive their efficiency primarily from **event-driven sparsity**: computation is **triggered only upon spike events** rather than being executed at every timestep"*（Springer s11063-025-11832-z）

## 优势 2：**时序处理**（天然优势）

> *"SNN neurons possess **intrinsic temporal dynamics** that binary ANNs inherently lack. Through **membrane integration, leakage, thresholding, and reset**, SNN neurons maintain internal states that evolve over time, enabling **nonlinear temporal filtering** that cannot be replicated by static binary activation functions."*（Springer）

**★关键**：**即使没有显式循环连接**，神经元级动力学也有时序能力

## 优势 3：**事件驱动稀疏**（异步）

> *"computation is triggered **only upon spike events**"* → 异步，不用每个时间步算全部
> **★能耗优势来自"异步时间稀疏"，不是来自二值化本身**

## 优势 4：**神经形态硬件生态在成熟**

| 硬件 | 规模 | 状态 |
|---|---|---|
| **Intel Loihi 2** | **1M 神经元 / 120M 突触** | ✅ **已发布**，功耗 ~1W |
| **Hala Point**（Intel） | **1.15B 神经元**（1152 个 Loihi 2） | ✅ **世界最大神经形态平台** |
| IBM TrueNorth / NorthPole | — | ✅ |
| SpiNNaker | — | ✅ |

**★2026 年已"主流化"**（Wedbush 报告）：
> *"The **mainstreaming of neuromorphic computing in 2026** marks the end of the 'silicon status quo'"*
> 提到 **Intel Loihi 3** 和 **IBM 的下一代**

## 优势 5：**生物合理性**（可用于神经科学验证）

> 因为 SNN 模拟真实神经元动力学，**可用于脑科学研究**（真实大脑数据的对照）

---

# 第四部分 · ★ SNN 的五大劣势（带实证）

## 劣势 1：**脉冲不可导 → 训练困难**（最根本）

> *"spikes are **non-differentiable**, complicating gradient-based optimization"*（MDPI 教程）

**四种绕法及代价**（MDPI 2673-4117）：
| 方法 | 代价 |
|---|---|
| **代理梯度（SG）** | 近似，随深度/时长失效 |
| **精确伴随** | 计算昂贵 |
| **局部可塑（STDP 等）** | 性能有限 |
| **ANN→SNN 转换** | 需要长时步，失去时序优势 |

**★代理梯度的内在限制**（arXiv:2602.01978）：
> *"In theory, SGs also **intrinsically limit the degree of sparseness** in a network, as **excessive sparsity amplifies gradient vanishing**, collapsing training at a critical transition point"*

**★★ 这条极重要**：
> **SNN 的稀疏性和可训练性直接冲突** ——
> **越稀疏 → 梯度越容易消失 → 训练崩溃**

## 劣势 2：**BPTT 内存爆炸**（O(LT) vs ANN 的 O(L)）

**ICLR 2026 论文**（*Towards Lossless Memory-efficient Training of SNN*）：
> *"BPTT requires storing all intermediate results for **L layers expanded over T time steps**, leading to a memory complexity of **O(LT)**, whereas similar ANN structures only require **O(L)**"*

**★实测细节**：
> *"For an SNN with **T=4**, intermediate features (input spikes per layer + internal neuron states) account for **over 96% of peak memory**"*

**★解法与代价**（该论文）：
```
梯度检查点 + 无损脉冲压缩 + 多级调整
→ 峰值内存降到 0.12×-0.47×，但速度慢 <20%
```

**SLTT（ICCV 2023）**：
> *"memory cost and training time are reduced by more than **70% and 50%**"*
> 但**忽略时序反向传播的"不重要路径"** → 有损

## 劣势 3：**规模天花板**（★最致命）

| 路线 | 天花板 | 出处 |
|---|---|---|
| **ANN→SNN 转换** | **>300M 参数性能超阈值** | arXiv:2510.00133 |
| **直接训练 CNN/Transformer SNN** | 目前 **~100M 级**（QKFormer 65M） | arXiv:2403.16552 |
| **SNN 语言模型** | **1.0B**（WD-Spikingformer，43.6%） | arXiv:2604.11321 |
| 对比：LLM | **1T+** | — |

**arXiv:2409.02111 综述的诚实评价**：
> *"Compared to state-of-the-art ANNs that have **billions of parameters**, deep SNNs are still **limited in the number of parameters**... typically employ **millions**"*

## 劣势 4：**注意力能力退化**（★很关键）

> *"Although spiking self-attention has shown success in constructing Spiking Transformers, **it has reduced capabilities compared to vanilla self-attention due to the removal of non-linearity**"*（arXiv:2409.02111）

**★具体困难**（Edge 综述 arXiv:Middleton2026b）：
> *"spiking Transformers face limitations including the **difficulty of accurately approximating softmax operations with discrete spikes**, **high memory demands from temporal unrolling**, and **performance gaps on large-scale complex tasks**"*

**★还有一句很犀利的判断**：
> *"The current trends surrounding spiking Transformers is, **at least in part, misplaced** when the deployment context is the constrained edge. **Transformers were conceived as scaling architectures**, and their memory and compute footprint, even in spiking form, **sits awkwardly with strict size, weight, and power budgets**. **Convolutional and recurrent SNN architectures, though less fashionable, may ultimately prove better matched** to the environments this field is most motivated to serve."*
>
> **⇒ 权威意见：SNN 该做"卷积/循环"而非"Transformer"**（因为 Transformer 是 scaling 架构）

## 劣势 5：**生态不成熟**

> *"software and hardware ecosystems for SNNs are still **maturing**, and standardized, energy-aware benchmarks remain **limited**"*（MDPI 教程）
>
> *"event-based vision lacks comparable standardisation. This absence **hinders objective comparison** of different approaches"*（Edge 综述）

---

# 第五部分 · ★★ 最关键的实证：稀疏上限（arXiv:2607.26648）

**这篇论文用"只换隐藏单元"的方法，给出了 SNN 的适用边界**

| 任务 | 可稀疏到 | 判定 |
|---|---|---|
| **低负载前馈感知** | **5%** | ✅ **SNN 赢在这里** |
| **循环语言模型** | ❌ **不能低于 ~50%** | *"the recurrent state must stay active to carry information"* |
| **脉冲 Transformer** | 2% | ⚠️ 但要**存全量 KV**（换来"内存墙"） |

**★原文的两句关键结论**：
> *"the ceiling is a property of **recurrent compression**, not sequence modeling"*
>
> *"on neuromorphic hardware, **recurrence and attention pay on different axes, neither escapes**"*

**★信息论界**：
```
ρ ≥ H_b^{-1}(log₂ M / H)

放电下限：
  · 随【记忆负载】↑
  · 随【状态宽度】↓
  · 随【任务难度】↑（★反驳"只看内存"的简单解读）
```

**★最终结论（原文）**：
> *"isolating **event-driven perception as where neuromorphic hardware wins**"*
> **⇒ SNN 的战场是事件驱动感知，不是循环语言推理**

---

# 第六部分 · ★ 三方对比（完整版）

| 维度 | **CogVec**（连续值向量神经元） | **SNN**（脉冲） | **Transformer** |
|---|---|---|---|
| **基本单元** | 连续值向量 | **0/1 脉冲** | 实数激活 |
| **时序** | 12 tick 单轮 | ✅ **膜电位动力学** | ⚠️ 位置编码 |
| **稀疏性** | ❌ **90%+ 活跃** | ✅ **1-5%（感知）/ 50%（循环）** | ❌ 稠密 |
| **能耗** | ❌ 未测（64 神经元） | ✅ **38-57×（神经形态）** / 3×（实测） | ❌ 基准 |
| **规模** | ❌ 64 神经元 | ⚠️ **65M-1B**（SOTA） | ✅ **1T+** |
| **ImageNet** | ❌ 未测 | ✅ **85.65%**（QKFormer） | ✅ 84.5-91.1% |
| **训练** | ❌ **纯 Hebbian**（无误差信号） | ⚠️ 代理梯度（近似） | ✅ 反向传播 |
| **长程依赖** | ❌ 12 tick | ⚠️ 中等 | ✅ 天然 |
| **硬件** | ❌ CPU/GPU | ⚠️ **神经形态已可用**（Loihi 2/Hala Point） | ✅ GPU 原生 |
| **生态** | ❌ 无 | ⚠️ 成熟中 | ✅ 完整 |
| **生物合理性** | ⚠️ 部分（无脉冲） | ✅ **高** | ❌ 低 |
| **持续学习** | ✅ NORM-3 | ⚠️ 可做（但训练难） | ❌ 固定权重 |

## ★ 三方"各自的最佳战场"

| 战场 | 赢家 | 证据 |
|---|---|---|
| **事件驱动感知**（边缘/低功耗） | ✅ **SNN** | arXiv:2607.26648：可稀疏到 5% |
| **ImageNet 分类（等参数）** | ✅ **SNN 已追平/超越** | QKFormer 85.65% > DeiT-B 83.1% |
| **大规模语言** | ✅ **Transformer** | 1T+ vs 1B |
| **长程依赖** | ✅ **Transformer** | SNN "not transformers (yet)" |
| **神经科学验证** | ✅ **SNN** | 生物合理性 |
| **持续学习** | ⚠️ **都不成熟** | CogVec 有意愿但无机制 |

---

# 第七部分 · ★★ 对 CogVec 的启示（最重要的部分）

## 7.1 残酷的真相：**CogVec 现在是"三方中最弱的"**

| 维度 | CogVec | 差距 |
|---|---|---|
| 稀疏性 | 90%+ 活跃 | **比 SNN 差 20-90×** |
| 训练 | 纯 Hebbian | **SNN 至少有代理梯度** |
| 规模 | 64 神经元 | **比 SNN 小 6 个数量级** |
| 硬件 | CPU | **SNN 有 Loihi 2 / Hala Point** |
| 结果 | ❌ **无任何基准分数** | **SNN 有 ImageNet 85.65%** |

**⇒ CogVec 目前**：
- **没脉冲**（不是 SNN）
- **没梯度**（不是 ANN）
- **没规模**（不是大模型）
- **没基准**（无法证明任何优势）

## 7.2 ★ 但 SNN 的困境**正好说明了 CogVec 的可能机会**

**SNN 的两个核心矛盾**（都是 SNN 自己解决不了的）：

### 矛盾 1：**稀疏 vs 可训练**
> *"excessive sparsity amplifies gradient vanishing, **collapsing training**"*（arXiv:2602.01978）

**★SNN 用梯度 → 必须牺牲稀疏**

### 矛盾 2：**BPTT 内存墙**
> *"memory complexity of **O(LT)**, whereas ANN only requires **O(L)**"*（ICLR 2026）

**★SNN 要时序 → 必须存全图 → 内存爆炸**

**⇒ 这两条正好是 CogVec "无梯度 + 局部可塑" 路线的**理论空间**：
```
SNN 的困境：要梯度就得稠密，要时序就得存全图
CogVec 的可能：不用梯度（三因子局部规则）→ 不受"稀疏-可训练"矛盾约束
                 因果优先（NORM-4）→ 不需存全图
```

**★但前提**：**CogVec 必须真的实现"局部规则"**（现在的纯 Hebbian **不算** —— 无误差信号 = 无学习目标）

## 7.3 ★★★ 权威建议：**别做 SNN-Transformer**

**Edge 综述（arXiv:Middleton2026b）的判断**：
> *"**Transformers were conceived as scaling architectures**, and their memory and compute footprint, **even in spiking form, sits awkwardly with strict size, weight, and power budgets**. **Convolutional and recurrent SNN architectures, though less fashionable, may ultimately prove better matched** to the environments this field is most motivated to serve."*

**⇒ 对 CogVec**：
> **不要追"大而全"**（做不了，SNN 自己都做不了）
> **CogVec 的优势方向是"小、稀疏、事件驱动、持续学习"**

## 7.4 ★ 具体可借鉴（按优先级）

| 优先级 | 借鉴 | 依据 | 对 CogVec |
|---|---|---|---|
| **P0** | **膜电位动力学**（LIF） | SNN 时序能力的来源 | CogVec 现在**无膜电位**（只有 activity） |
| **P0** | **事件驱动计算**（只有活跃突触算） | Springer：能耗优势来自此 | CogVec 90% 活跃 → 无事件驱动 |
| **P1** | **WTA 替代 softmax** | WD-Spikingformer：能效 8× | CogVec 用注意力（softmax） |
| **P1** | **双时间尺度**（快激活 + 慢结构） | RSGN：15× 参数少 | CogVec 有结构可塑，但未验证 |
| **P2** | **ANN→SNN 转换思路** | 免从头训练 | 可用于"给 CogVec 灌能力" |
| **P3** | **梯度检查点**（治 O(LT) 内存） | ICLR 2026 | 如果 CogVec 将来要 BPTT |

## 7.5 ★★★ 最该做的一件事：**跟 SNN 做同一个基准**

**SNN 有明确的**：
- **基准**：ImageNet / CIFAR-10 / CIFAR10-DVS / DVS128 Gesture
- **指标**：Top-1 精度 / 参数 / 时间步 / 能耗（mJ）
- **对照**：同参数量的 ANN

**⇒ CogVec 应该**：
```
在同一任务（如 CIFAR-10，或 DVS128 Gesture）上：
  ① 报告 Top-1 精度
  ② 报告参数量
  ③ 报告活跃率（对应 SNN 的放电率）
  ④ 估算能耗
  ⑤ 与 QKFormer/SAFformer 对照

★ 因为 SNN 社区的基准是【公认的】，不像 CogVec 自造的 dmin/dmean
```

---

# 第八部分 · 诚实标注（NORM-9）

| 项 | 状态 |
|---|---|
| 所有引用（arXiv/DOI/URL） | ✅ **已核实** |
| QKFormer 85.65% > DeiT-B 83.1% | ✅ **原文明确"under the same experiment conditions"** |
| SNN 能耗 38-57× | ⚠️ **仅限神经形态硬件** |
| "SNN 已超过 Transformer" | ⚠️ **仅在 ImageNet 分类（等参数）**，不是全面超越 |
| 稀疏上限公式 | ⚠️ **一篇论文（arXiv:2607.26648）**，需独立复现 |
| Loihi 2 / Hala Point 状态 | ✅ **已发布**（Intel 官方 + Open Neuromorphic） |
| **CogVec 的任何基准** | ❌ **完全没有**（这是最大空缺） |

---

# 附 · 一句话总结

> **SNN 现状**：**比大多数人以为的强得多**
> - **ImageNet 85.65%**（QKFormer，**超过 DeiT-B/Swin-T**）
> - **能耗 38-57×**（神经形态硬件）/ **3×**（实测）/ **17%**（WD-Spikingformer vs Qwen-1.5B）
> - **硬件已可用**（Loihi 2：1M 神经元 / Hala Point：1.15B 神经元）
>
> **SNN 的五大劣势**：
> 1. **脉冲不可导**（代理梯度近似，且**稀疏-可训练矛盾**）
> 2. **BPTT 内存 O(LT)**（比 ANN 的 O(L) 差一个 T 倍）
> 3. **规模天花板**（转换法 300M / 直接训练 ~100M）
> 4. **注意力能力退化**（离散脉冲难近似 softmax）
> 5. **生态不成熟**
>
> **★稀疏上限（最关键的边界）**：
> - 事件驱动感知 → **可稀疏到 5%** ✅ **SNN 的战场**
> - 循环语言推理 → **不能低于 50%** ❌
>
> **★对 CogVec 的判断**：
> - **现状是三方中最弱的**（无脉冲/无梯度/无规模/无基准）
> - **但 SNN 的两个困境（稀疏-可训练矛盾、BPTT 内存墙）正是 CogVec "无梯度+因果优先"路线的理论空间**
> - **前提**：必须真的实现"局部学习规则"（纯 Hebbian 不算）
> - **最该做**：**跟 SNN 用同一个基准**（而非自造 dmin/dmean）
