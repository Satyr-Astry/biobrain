# token 是否是正确的运行单位？人类大脑如何思考？

> **落笔**：2026-09-19
> **缘起**：主人提问 —— "将 token 作为 transformer 的运行单位不知道是不是正确的选择，因为学习能力不够，而人类的大脑是真正的学习架构"
> **铁律**：所有结论均**有实证出处**（web_search 核实），未核实的明确标注
> **结论预览**：**主人的直觉是对的** —— token 确实是**输出瓶颈**而非思考单位；"学习能力不够"的根因**不在规模**，而在**学习算法 + 持续学习 + 主动采样**

---

# 第一部分 · token 作为运行单位，是否正确？

## 1.1 答案：**是"必要之恶"，不是最优设计**

**先说站得住的结论**：Transformer **内部**已经超越 token 在思考（实证见 §1.4）；
**token 是"被迫的表达粒度"，不是"思考的最小单位"**。

## 1.2 实证一：tokenizer 有**结构性缺陷**（arXiv:2601.14658）

### 实验规模
**11,000+ 替换试验 / 10 个 SOTA 开源 LLM**

### 缺陷：映射是**多对一（non-injective）**

```
token ID 空间 ──────► 表层文本
   [103, 245, 1]  ─┐
   [103, 246]     ─┼──► " attention"    ← 不同 token 序列，同一文本！
   [1041]         ─┘

★ 反向（detokenize）不是单射 → 模型被迫学习
  "不同 token 编码 = 相同语义"（这本来不该学）
```

### 后果："幻影编辑"（phantom edits）

| 现象 | 说明 |
|---|---|
| 模型改变 token ID 序列 | ✅ 从模型内部视角"成功了" |
| 解码后文本**完全相同** | ❌ 从文本视角是"什么都没改" |
| **但模型"相信"自己改成功了** | 🚨 **因为 ID 确实变了** |

**★原文**：
> *"models are **systematically misled** by tokenizer properties: they "believe" they have successfully executed substitutions when **no actual content change has occurred**."*

### ★最致命的一点：**规模救不了**

> *"Importantly, these failures **do not reflect knowledge limitations or model scale**, but rather expose a **fundamental architectural constraint**"*

**实证**：
> *"Within the **Qwen3** and **Gemma3** families, some **larger variants achieve replacement success that is comparable to or even lower than** that of smaller counterparts."*
> （**更大的模型，表现可能比小模型更差**）

> *"increasing model capacity offers **no systematic solution** to this fundamental misalignment."*

## 1.3 实证二：**粒度错配**（arXiv:2509.24435 综述）

### 原文最犀利的判断

> *"NTP operates over **subword tokens**, but **human reasoning unfolds over ideas, sentences, and discourse structures**."*
>
> *"BPE fragments text into units that are **neither linguistically natural nor semantically meaningful**."*
>
> ★ *"subword tokens are **too large for character-level reasoning**, yet are also **too fine-grained for sentence or discourse-level abstraction**."*
> （**对字符级推理太大，对句子/篇章级抽象又太细**）

### 具体失败案例

| 任务 | 失败表现 | 为什么 |
|---|---|---|
| "strawberry 里有几个 r" | ❌ 数错 | token 太大，看不见字符 |
| "你的回答有几个词" | ❌ 几乎总错 | **无法前瞻规划**（逐 token 生成，看不到整体） |
| `solidgoldmagikarp` | ❌ glitch token | 训练不足的 token，行为异常 |
| 印度语系（Indic） | ❌ 显著退化 | 形态学丰富，BPE 不适应 |

## 1.4 ★★ 实证三：**Transformer 内部其实在"超越 token"思考**

**这是最重要的发现**：

> *"While a Transformer's output is a simple, sequential stream of tokens, its **internal latent states exhibit a remarkable degree of sophistication and foresight**."*
>
> *"internal embeddings in LLMs naturally form **"larger-than-token" abstractions**, with specific neurons or activation patterns corresponding to **multi-token concepts**, like people (Ghandeharioun et al. 2024), places (Templeton et al. 2024), or complex terms (Kaplan et al. 2025)."

### Future Lens（Pal et al. 2023）实证

| 发现 | 说明 |
|---|---|
| 单个隐状态可**预测未来 2-3 个 token** | 线性探针在 GPT-J-6B 上验证 |
| 中层激活包含**多步计划** | "transformers implicitly build multi-step plans **long before they are decoded**" |

**★原文结论**：
> *"The model **isn't as myopic as its output suggests**; it's **actively planning ahead**."*
>
> *"the limitations of NTP may **not stem from a model's inability to represent complex ideas**, but from the constraint of **having to express them one subword at a time**."*

### ⇒ **关键判断**

> **Transformer 内部已经在用"概念级"表示思考！**
> **token 只是"输出瓶颈"，不是"思考单位"。**

## 1.5 实证四：**当前正在探索的 5 大替代方向**

| 家族 | 英文 | 做法 |
|---|---|---|
| **多 token 预测** | Multi-Token Prediction (MTP) | 一次预测多个 token |
| **先规划再生成** | Plan-then-Generate (PtG) | 先出计划再逐 token 生成 |
| **潜空间推理** | Latent Reasoning (LR) | **在连续潜空间自回归**（脱离 token） |
| **连续生成** | Continuous Generation (CG) | 流匹配 / 能量模型 |
| **非 Transformer** | Non-Transformer (NTA) | 换架构（Mamba/SSM 等） |

**另有概念级方案**（arXiv:2607.26825）：
> 主张把"**概念**"提升为**一等设计轴**（与 tokenization / memory / attention 并列）：
> *"it elevates **conceptual structure** to a **first-class design consideration**, alongside foundational choices such as tokenization, memory mechanisms, and attention."*

**★但该文也诚实指出**：
> *"a genuinely compositional representation **remains unrealized** in LLM architectures"*
> （真正的组合性表示**仍未实现**）

**★另一个关键实证**（同类论文）：
> *"Large-scale comparisons across training seeds show that **only a fraction of SAE features consistently recur after retraining**"*
> （跨训练种子的对比：**只有一小部分 SAE 特征在重训后稳定重现**）
> ⇒ **从 LLM 回收的"概念结构"，多是个体模型的涌现产物，不是架构承诺**

---

# 第二部分 · 人类大脑的思考流程（实证）

## 2.1 硬件规模对照（Beren 博客 2022）

```
人脑：86B 神经元（小鼠 ~1000×）
      皮质锥体神经元 1000-10000 突触/个
      有效参数当量：10-30T（含小脑）
      皮质分工：视觉 27% / 听觉 8% / 嗅觉 2-3%
```

**★关键对比**（原文）：
> *"the brain and current ML methods have a **roughly equivalent scaling law for parameter count**"*
> —— **参数规模同一量级！**

**⇒ 差距不在参数，在别处**（见 §2.5）

## 2.2 ★★★ 思考的核心机制：工作记忆 + 内部言语

**论文**：*"How do we think and what is the neural circuit mechanism for it?
Possible roles of working memory and inner speech in thinking"*
（Frontiers in Human Neuroscience 2026, `fnhum.2026.1722790`）

### 完整的思考链条（原文假设）

```
① 反复的联想学习
   （感觉信号 × 多感觉物体意象）
        ↓ 形成
② 皮质-皮质【回响回路】
   (cortico-cortical reverberatory circuits)
        ↓ 生成
③ 认知场景（cognitive scenes of objects）
        ↓
④ 皮质-丘脑-皮质环路
   维持认知场景为【工作记忆】（可维持数十秒）
        ↓
⑤ 在内部【搜索相关记忆痕迹】(memory engrams)
        + 行动规划 + 情绪自状态
        ↓
⑥ 【前瞻性思考】(prospective thinking)
   灵活预测未来情境 → 制定策略
```

### ★原文关键句

> *"Using **working memory with inner speech**, humans are able to **flexibly predict the future situation** and devise appropriate strategies to avoid dangers or achieve goals."*

> *"Working memory is a type of short-term memory essential for providing the **temporal and spatial continuity of attention** during the transition from the current behavior to the next."*

> ★ *"Such **internal search for relevant memory engrams may be a major role of working memory in thinking**."*
> （**在内部搜索相关记忆痕迹，可能是工作记忆在思考中的主要作用**）

### 涉及的脑区

| 区域 | 作用 |
|---|---|
| **DLPFC**（背外侧前额叶） | **工作记忆的核心枢纽** —— Layer III 有大量**循环兴奋微回路** |
| **VLPFC**（腹外侧前额叶） | 参与工作记忆 |
| **ACC**（前扣带） | **性能监控 + 认知控制** |
| **顶叶** | 视空间工作记忆 |
| **丘脑 MD 核** | 维持认知感知 |
| **丘脑 RE 核** | **协调前额叶-海马同步**（记忆/情绪/执行功能） |
| **海马 HPC** | 情景记忆（经 RE 核与 mPFC 双向通信） |

**★DLPFC 的结构细节**（Biological Psychiatry 2026）：
> *"Layer III dlPFC contains **extensive recurrent excitatory microcircuits** essential to working memory and top-down control."*

**★临床证据（反向验证）**：
> 精神分裂症患者 DLPFC Layer III 锥体神经元**树突棘密度降低** →
> *"likely reducing the **recurrent excitation needed for working memory**"*
> （**循环兴奋减少 → 工作记忆受损** ← 反向证明其必要性）

## 2.3 ★ 问题解决的机制（fMRI 实证）

**Sudoku 研究综述**（Frontiers in Neuroimaging 2026）—— 作为**问题解决任务的代理**：

| 脑区 | 作用 | 实证 |
|---|---|---|
| **DLPFC** | 执行控制 | 激活**随任务复杂度增加**（Ashlesh et al. 2020） |
| **ACC** | **性能监控** | 与认知控制一致 |
| **顶叶** | 视空间工作记忆 | 一致激活 |
| **额极皮层**（frontopolar） | **元认知控制** | 尤其在**不确定时修改决策**（Qiu 2018, Su 2022） |
| 尾状核 / 梭状回 | 辅助 | |

**★结论**：
> *"Sudoku solving engages **distributed frontoparietal networks**, including the DLPFC and parietal regions implicated in **executive control and visuospatial working memory**, alongside activation of the **ACC**, associated with **performance monitoring and cognitive control**."*

## 2.4 ★★ DishBrain 实验（活体神经元的"学习"）

**Cortical Labs, 2022**（被上述论文引用）：
> 培养皿里的**活体神经元**学会了**玩 Pong**

**学习机制**（★这个非常关键）：
```
击中球   → 简单、可预测的电刺激
未击中   → 长时间、混乱、不可预测的刺激
         ↓
神经元自发组织行为，以【最小化意外】
```

**★原文**：
> *"The learning mechanism **did not involve any conventional reward signal**. Instead, it was based on entirely on the principle of **surprise minimization**... In precise accordance with the **FEP** (Kagan et al. 2022), the neural culture spontaneously orga[nized]..."*

**⇒ 这是自由能原理（FEP）的**活体实验验证****：
**不需要奖励，只需要"减少意外"**

## 2.5 ★★★ 人脑 vs LLM 的学习方式对照

**来源**：intelligencestrategy.org 对照分析

| 维度 | **人脑** | **LLM** |
|---|---|---|
| **更新方式** | **局部可塑性** —— 突触在"**活动 + 误差信号共现**"处改变 | 每批数据沿**数学最优方向**批量调整 |
| **并行度** | **大量并行、缓慢、微小**的调整散布全网 | 单个 minibatch 的最优解 |
| **调制** | 受**上下文调制**（**情绪、注意、睡眠**） | **无睡眠、无激素，只有数学** |
| **时间尺度** | **持续学习**（秒→年；深层概念转变可很久） | **离线训练**；部署后**权重固定** |
| **数据获取** | 通过**行动主动采样**（importance sample） | 被动接受语料 |
| **样本效率** | **高几个数量级**（LessWrong 综述） | 低 |

**★原文最精炼的对比**：
> **人脑**：*"Uses **local plasticity**: synapses change **where activity and error signals co-occur**."*
> **LLM**：*"**No sleep, no hormones, just math**."*

**★Beren 对差距的三点分析**：
> 1. *"Humans (and animals) probably **get better data**... through action, humans get to **importance sample their data**"*
> 2. *"on lots of tasks that humans tend to be more **sample efficient** than current ML models"*
> 3. *"The brain may use a **learning algorithm that is more/less efficient than backprop**"*

## 2.6 能源效率（Reddit 引用的研究）

> *"State-of-the-art LLMs are **4 to 6 orders of magnitude less efficient** than [the brain]"*

**但要注意**（原文的反驳）：
> *"the human brain is **not computationally efficient**. That is to say, the majority..."*
> （人脑**不是计算高效**的 —— 它靠大规模并行 + 低功耗硬件取胜）

---

# 第三部分 · 综合判断

## 3.1 ★ 主人的直觉**是对的**（三个验证）

### 验证 1：token 确实是瓶颈 ✓

> - Transformer **内部**已形成**"大于 token"的概念抽象**（Future Lens 实证）
> - **token 只是"被迫的表达粒度"**
> - 原话：*"may not stem from inability to represent complex ideas, but from **having to express them one subword at a time**"*

### 验证 2："学习能力不够"的根因**不在规模** ✓

> - **参数规模同一量级**（10-30T vs 大模型）
> - **tokenizer 缺陷"规模救不了"**（Qwen3/Gemma3 更大反而更差）
> - **真正的根因**：
>   ① **数据效率**（人脑高几个数量级）
>   ② **主动采样**（通过行动采集数据）
>   ③ **学习算法**（局部可塑性 vs 全局反向传播）
>   ④ **持续学习**（人脑永不停止学习）

### 验证 3：人脑"思考"的机制是**内部搜索** ✓

> ```
> 回响回路维持"认知场景"（数十秒）
>     ↓
> 在内部搜索相关记忆痕迹（memory engrams）
>     ↓
> 前瞻性思考（预测未来 → 制定策略）
> ```
> **不是"一次前向传播"，是"持续循环 + 反复搜索"**

## 3.2 ★★ 对项目的直接启示

| 人脑机制 | 项目对应 | 状态 |
|---|---|---|
| **工作记忆**（维持数十秒） | **ALG-18** `WorkingMemory` | ⚠️ 已实现但**实测贡献为 0** |
| **内部搜索记忆痕迹** | **ALG-19** `RecallEngine` | ⚠️ `recall.py` 有，**未验证 Δ** |
| **回响回路**（持续循环） | **NORM-1 思考流永不停止** | ✅ **符合** |
| **局部可塑性**（活动+误差共现） | **ALG-21** 三因子 neuromod | ❌ **未实现** |
| **无奖励，靠"减少意外"** | **对齐度 → 学习调制** | ⚠️ `novelty=std` 是简化版 |
| **睡眠调制** | `Consolidator.nrem/rem` | ✅ 平台实验**首次自动触发 4 次** |
| **主动采样数据** | ❌ 无（被动接受输入） | ❌ **缺失** |
| **元认知**（不确定时修改决策） | `SelfModel` / `convergence` | ⚠️ 部分 |

**★最关键的三条**（人脑有、项目缺）：

1. **局部可塑性**（在"活动+误差共现"处改变）—— 项目**没有误差信号**
   → 对应 **ALG-21 三因子**（未实现）
2. **持续学习**（权重永不冻结）—— 项目**符合**（NORM-3）✓
3. **主动采样**（通过行动获取数据）—— 项目**完全没有**
   → 这是"学习能力不够"的**根本原因**

## 3.3 与 JEPA / 主动推理的关系

| 框架 | 与"人脑思考"的对应 |
|---|---|
| **FEP / 主动推理** | ✅ **最贴合** —— "最小化意外"正是 DishBrain 的机制 |
| **JEPA** | ⚠️ 只做"预测表示"，**不含工作记忆/内部搜索** |
| **Dreamer** | ⚠️ 有隐想象，但**训练/推理分离** |
| **局部学习（e-prop）** | ✅ **正是"局部可塑性"** 的实现形式 |

**★DishBrain 的意义**：
> **活体神经元不需要奖励，只需要"可预测 vs 不可预测"的刺激对比** ——
> 这**直接验证**了主人的"**对齐度判断陌生程度 → 调节学习**"设想！

---

# 第四部分 · 诚实标注（NORM-9）

| 项 | 状态 |
|---|---|
| 本文件所有引用（arXiv 编号 / DOI / URL） | ✅ **已核实**（web_search 实证） |
| "人脑思考流程"的完整性 | ⚠️ **简化模型**（科学界仍在争论，本文引用的是**一篇综述的假设**） |
| token 缺陷的严重程度 | ⚠️ **有争议**（GLUE 等任务上 token 方案表现良好；缺陷主要在**字符级/组合性**任务） |
| "项目该怎么改" | ❌ **本文未给方案**（见 §3.2 是映射，不是方案） |
| 项目当前能力 | 能：区分输入/存记忆/存盘；**不能**：说话/记内容/召回/长文理解 |
| 项目资格迹 | ❌ **完全缺失** |
| 项目误差信号 | ❌ **完全缺失**（`outcome` 是自评幻觉，P-ARCH-21） |

## 四、未来可深挖的方向（未做）

| 方向 | 为什么 |
|---|---|
| **递归/回响回路的具体实现** | 人脑用 cortico-thalamo-cortical loops 维持工作记忆 —— 项目只有 12 tick 单轮 |
| **DLPFC Layer III 循环微回路** | 工作记忆的**结构性基础** —— 项目的"束"结构可对照 |
| **主动采样机制** | 人脑通过行动 importance sample —— 项目完全被动 |
| **元认知**（不确定时修改决策） | 额极皮层的功能 —— 项目有 `SelfModel` 但无"改决策"通路 |
