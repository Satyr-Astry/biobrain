# BioBrain：一种持续思考、在思考中学习的仿生认知架构
# BioBrain: A Bionic Cognitive Architecture That Thinks Continuously and Learns While Thinking

> 中文 / English bilingual paper · 仿生AI项目设计 · 2026-09

---

## 摘要

当代大语言模型（LLM）遵循"先训练后使用"范式：训练结束后权重冻结，推理过程不产生任何学习。这与生物大脑"持续思考、激活即学习、睡眠巩固"的运行方式存在根本差异。本文提出 **BioBrain**——一种仿生认知架构，以张量化神经组织为核心，实现永不停止的"思考流"（thinking stream），使学习成为思考的副产品而非独立阶段。架构包含三层：**感觉端**（文本→稀疏神经激活）、**神经认知层**（16384 个张量化神经元组织成 1024 个重叠神经束，执行消息传递、侧向抑制、赢者通吃与在线可塑性）、**运动端**（由认知层指挥的 LLM 语言区，权重保持冻结，仅负责将大脑意图转化为流利语言）。记忆以 BM25+语义混合检索支持经验召回，置信度门控防止幻觉，双相睡眠（NREM 巩固 + REM 联想）在低活动期合并可塑量。系统对外暴露 OpenAI 兼容 API，可被任意 LLM 客户端或编排框架（如 Hermes Agent）直接调用。实验表明，系统在 27 条蒸馏记忆上达到 0.67–0.95 的置信度区分、约 3.8 秒的端到端响应，并正确拒答未知领域问题。本文还梳理了工程实现中暴露的 13 项规范缺陷（C1–C13）及其修复，展示了"规范—实现—回归测试"闭环的开发方法论。

**关键词**：仿生认知架构；持续学习；神经组织；双相睡眠；符号接地；LLM 编排

## Abstract

Contemporary large language models follow a "train-then-use" paradigm: weights are frozen after training, and inference produces no learning. This fundamentally differs from biological brains, which think continuously, learn whenever activated, and consolidate during sleep. We propose **BioBrain**, a bionic cognitive architecture built on a tensorized neural tissue that runs a never-ending "thinking stream," making learning a byproduct of thinking rather than a separate phase. The architecture has three layers: a **sensory end** (text → sparse neural activation), a **neural cognitive layer** (16,384 tensorized neurons organized into 1,024 overlapping tracts with message passing, lateral inhibition, winner-take-all dynamics, and online plasticity), and a **motor end** (an LLM language area commanded by the cognitive layer; its weights stay frozen and it only translates brain intent into fluent language). Memory uses BM25+semantic hybrid retrieval for experience recall; a confidence gate prevents hallucination; biphasic sleep (NREM consolidation + REM association) merges plasticity during low-activity phases. The system exposes an OpenAI-compatible API and can be invoked by any LLM client or orchestration framework (e.g., Hermes Agent). Experiments show confidence discrimination of 0.67–0.95 across 27 distilled memories, end-to-end latency of ≈3.8 s, and correct refusal on out-of-domain questions. We also document 13 specification defects (C1–C13) exposed by implementation and their fixes, demonstrating a closed-loop "specification–implementation–regression-test" development methodology.

**Keywords**: bionic cognitive architecture; continual learning; neural tissue; biphasic sleep; symbol grounding; LLM orchestration

---

## 1. 引言 / Introduction

### 1.1 动机 / Motivation

主流 LLM 的三个结构性限制与生物大脑形成鲜明对比：

1. **权重冻结**：LLM 训练结束后权重不再变化，新知识只能通过上下文窗口或微调注入，推理时不学习。而大脑的突触在每一次激活中都会发生可塑性变化。
2. **无自发活动**：LLM 是"被调用才运行"的被动系统；大脑即使在无外部输入时也持续进行默认模式活动（default-mode activity）——回放、联想、预演。
3. **训练与推理分离**：LLM 的训练是独立、昂贵、离线的阶段；大脑的学习与使用是同一过程的两个侧面。

Three structural limitations of mainstream LLMs contrast sharply with biological brains: (1) **frozen weights** — no learning happens at inference time; (2) **no spontaneous activity** — LLMs are passive systems that run only when invoked, while brains exhibit persistent default-mode activity; (3) **separated training and inference** — training is a distinct, costly, offline phase, whereas in brains, learning and use are two sides of one process.

### 1.2 贡献 / Contributions

1. 提出**思考流（thinking stream）**设计——唯一且永不停止的系统状态，计算相/巩固相/休眠相在其上切换；
2. 提出**神经组织四级层级**（神经库→组→束→神经元），其中神经元为向量组（树突场/胞体态/轴突场），并以矩阵运算实现张量化；
3. 提出**置信度门控**的防幻觉机制：大脑"知道自己不知道"，不熟则拒答；
4. 提出**双相睡眠**的在线巩固机制与**多尺度权重**（快/中/慢子权重叠加）解决时间尺度耦合；
5. 提出**LLM 作为语言区**的混合架构：认知决策在神经组织中完成，LLM 权重冻结只做语言生成；
6. 完整实现为 OpenAI 兼容服务，演示了与 Hermes Agent 等编排框架的接入。

---

## 2. 架构总览 / Architecture Overview

```
                 ┌─────────────────────────────────┐
                 │  思考流（永不停止，唯一状态）      │
 感觉神经组 ◀─────│  计算相 ⇄ 巩固相 ⇄ 休眠相        │──────▶ 运动神经组
 (文本/嵌入)      │  · 计算内核 ALG-0                │        (LLM 语言区)
                 │  · 在线可塑 ALG-3（激活即学）      │
                 │  · 收敛度量 ALG-11（何时输出）     │
                 └────────────┬────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │  神经组织库（分级加载）        │
               │   库→组→束→神经元(向量组)     │
               │   VRAM / RAM / NVMe          │
               └──────────────┬──────────────┘
                              │
                 ┌────────────┴────────────┐
                 │  双相睡眠合并 ALG-7        │
                 │  NREM 巩固 + REM 联想      │
                 │  + 多尺度权重 ALG-12       │
                 │  + 符号接地 ALG-14         │
                 └──────────────────────────┘
```

一句话总纲 / One-line summary：
> **矩阵=神经元，输入=耳朵，输出=嘴巴；思考不停，激活即学，睡眠巩固。**
> *Matrices are neurons, input is the ear, output is the mouth; thinking never stops, activation is learning, sleep consolidates.*

### 2.1 统一术语 / Unified Terminology

| 术语 Term | 定义 Definition |
|---|---|
| 思考态/思考流 Thinking stream | 唯一且持续的状态；主循环永不停止 |
| 神经元 Neuron | 一个向量组（树突场/胞体态/轴突场），最小组织单位 |
| 神经束 Tract | 一起激活的通路；双层语义：`members`（组成）+ `dynamics`（协同） |
| 神经组 Group / 神经库 Library | 功能团块 / 最大组织单位，一级存储单位 |
| 激活度 Activity | 组织"有多活跃"，[0,1] |
| 多尺度权重 Multi-scale weights | `w = w_fast + w_mid + w_slow`，各尺度独立学习 |
| 可塑量 Plasticity | 在线累积的临时增量（与 base 同形） |
| 主体 Base | 稳定权重（慢权重），睡眠期被合并更新 |
| 双相睡眠 Biphasic sleep | NREM 相（巩固事实）+ REM 相（联想重组） |
| 收敛度量 Convergence | 判断"想清楚了没"（稳定性+预测误差+目标达成） |
| 接地 Grounding | 符号与环境共激活、因果对应的程度 |
| 能量 Energy | J/token 或 GPU 功率×时间；用于错峰调度 |

---

## 3. 核心机制 / Core Mechanisms

### 3.1 计算内核（ALG-0）/ Computation Kernel

束内消息传递 + 束级门控注意力。每个 tick 中，束内神经元沿 `members` 拓扑交换消息（稀疏矩阵乘实现），束之间通过 `dynamics` 矩阵协同。跨束桥接仅由最热的前 256 个源束发起，连接 top-32 最相似束（BRIDGE_TOPQ），避免 O(T²) 的全连接灾难。

Intra-tract message passing with tract-level gated attention. Neurons inside a tract exchange messages along the membership topology (implemented as sparse matrix multiplication); tracts coordinate through their dynamics matrices. Cross-tract bridging is initiated only by the hottest 256 source tracts and connects to the top-32 most similar tracts, avoiding the O(T²) full-connectivity catastrophe.

### 3.2 激活动力学（ALG-1/2）/ Activation Dynamics

- **激活度计算**：多隔室能量聚合 + 稀疏 tick（只更新活跃集，与全模型规模解耦）。
- **扩散**：激活沿束传播（束内传播），且跨束桥接——二者缺失曾被实测暴露为"神经元全熄火"缺陷（C1/C2）。
- **侧向抑制**：仿 GABA 抑制性神经元，每束抑制最强 3 个（INHIB_K=3），赢者通吃压制塔尖以下 30%（WTA_RATIO=0.30），硬上限约束活跃神经元占比 ≤8%。
- **重叠叠加**：一个神经元可属多束；贡献按 `1/√(归属数)` 次线性归一化，防止激活爆炸。

Lateral inhibition mimics GABAergic interneurons; winner-take-all suppresses the bottom 30% below the peak; a hard cap constrains active-neuron ratio to ≤8%. Overlapping membership contributes with sublinear `1/√k` normalization to prevent activation explosion.

### 3.3 在线可塑（ALG-3）/ Online Plasticity

延迟可塑（pending 缓存）：用 t−1 的突触后激活更新，t+1 生效，保证严格因果（NORM-4）。三因子可塑性：资格痕迹 × RPE（奖励预测误差），正 RPE 强化、负 RPE 弱化（已代码验证方向正确，C9）。资格痕迹 MUST 归一化，防止无界累积导致更新量发散（C6）。

Delayed plasticity with a pending buffer: updates use postsynaptic activation from t−1 and take effect at t+1, enforcing strict causality. Three-factor plasticity (eligibility trace × reward prediction error) strengthens on positive RPE and weakens on negative RPE (verified in code, C9). Eligibility traces MUST be normalized to prevent unbounded accumulation and divergent updates (C6).

### 3.4 思考流主循环（ALG-5）/ Thinking-Stream Main Loop

唯一状态、永不退出（NORM-1）。"永不停止"指循环不停，而非前向计算不停：通过**相（phase）切换**在计算相/巩固相/休眠相之间轮转，合并只在无活动计算时进行（NORM-5）。

A single state that never exits (NORM-1). "Never stops" refers to the loop, not the forward pass: phase switching rotates among compute/consolidate/dormant phases, and consolidation only occurs when no active computation is running (NORM-5).

### 3.5 置信度门控（防幻觉核心）/ Confidence Gate

大脑对输入计算置信度 = 最相似经验的语义相似度（cosine over embeddings）。闸门规则：置信度 >0.6 才允许 LLM 回答；0.4–0.65 要求 LLM 声明不确定；<0.4 明确禁止编造。这实现了"我知不知道自己知道"的元认知。

The brain computes confidence as the semantic similarity to the most similar stored experience. The gate: only confidence >0.6 lets the LLM answer; 0.4–0.65 forces a cautious disclaimer; <0.4 explicitly forbids fabrication. This realizes "knowing what you know" metacognition.

### 3.6 双相睡眠（ALG-7）/ Biphasic Sleep

NREM 相：把可塑量合并进主体 base（事实巩固）。REM 相：跨束联想重组（对抗路径依赖）。合并独占（NORM-5），只在无活动计算时进行。

NREM phase merges plasticity into the base weights (fact consolidation). REM phase performs cross-tract associative reorganization (countering path dependence). Merging is exclusive (NORM-5), occurring only when no active computation runs.

### 3.7 多尺度权重（ALG-12）/ Multi-Scale Weights

权重 = 快/中/慢子权重之和（NeurIPS 定理：耦合时间尺度模型 ≡ 独立子权重之和）。按尺度补偿学习率 `lr_scale=[1,3,10]`，解决慢尺度几乎不学习的实测问题（C7）。

Weights are the sum of fast/mid/slow sub-weights (a coupled-timescale model is equivalent to a sum of independent sub-weights). Scale-compensated learning rates `lr_scale=[1,3,10]` fix the measured problem that slow scales barely learn (C7).

### 3.8 符号接地（ALG-14）/ Symbol Grounding

三管齐下：环境 token 共激活 + 因果验证 + 五维接地剖面审计。只让外部可验证信号进入主体（NORM-6），自评禁用——防止"自己说自己对"的自噬。

Three-pronged grounding: co-activation with environment tokens, causal verification, and a five-dimensional grounding-profile audit. Only externally verifiable signals may enter the base (NORM-6); self-evaluation is forbidden, preventing self-cannibalization.

---

## 4. 实现 / Implementation

### 4.1 张量化 / Tensorization

原始实现是逐神经元 Python 循环（64 神经元即慢）。张量化后所有状态变为 numpy 矩阵：胞体/轴突/树突为 n×d 矩阵，束内消息传递为稀疏矩阵乘，跨束路由为矩阵 top-k，激活聚合向量化。性能目标：10,000 神经元 @ 单 tick <50ms。

### 4.2 模块清单 / Module Inventory

| 模块 Module | 职责 Responsibility |
|---|---|
| `tensor_brain.py` | 张量化神经组织核心（tick/可塑/调度/分级存储） |
| `conductor.py` | 认知层指挥：记忆召回 + 激活语义反解 + 置信度调制 |
| `cortex.py` | LLM 语言区接口（Ollama 后端，R1 思维链清理） |
| `encoder.py` | 语义编码器（文本 ↔ 嵌入） |
| `bm25.py` | BM25+语义混合检索器 |
| `api_server.py` | OpenAI 兼容服务（chat/completions、models、embeddings） |
| `server.py` | 总装 + CLI + 持久化 |
| `p2_mechanisms.py` | 神经发生/集体潜意识/自我模型（M10–M12） |
| `self_training.py` | 三因子可塑性的自训练验证 |
| `test_bio_brain.py` | 回归测试套件（32 项） |

### 4.3 OpenAI 兼容服务 / OpenAI-Compatible Service

纯标准库（http.server）实现，零新依赖：`POST /v1/chat/completions`（含流式）、`GET /v1/models`、`POST /v1/embeddings`、`GET /health`、`GET /brain/state`。任何 OpenAI 客户端可直连 `http://127.0.0.1:8701/v1`，模型名 `bio-brain`。已演示接入 Hermes Agent（`providers.bio-brain` 配置，主模型保持 deepseek 不动）。

Pure standard library (http.server), zero new dependencies. Any OpenAI client can connect directly; integration with Hermes Agent has been demonstrated via a `providers.bio-brain` configuration entry while the primary model stays untouched.

---

## 5. 实验 / Experiments

### 5.1 环境 / Setup

- 硬件：AMD Ryzen 7 5700X3D / 16GB RAM / RTX 4070 Ti 12GB / Windows
- 神经组织：16,384 神经元 / 1,024 神经束（当前为原型规模；设计上限见规划文档）
- 语言区：deepseek-r1-abliterated 14B（Ollama，Q4 量化）
- 记忆：27 条 LLM 蒸馏知识（稀疏注意力/MoE/扩散模型/KV缓存等主题）

### 5.2 结果 / Results

| 查询 Query | 置信度 Confidence | 耗时 Latency | 结果 Outcome |
|---|---|---|---|
| KV缓存是什么 | 0.772 | 3.8s | ✅ 正确回答（含哈希分桶细节） |
| MoE 混合专家模型 | 0.947 | 3.8s | ✅ 正确回答（gating/softmax） |
| 扩散模型与热传导方程 | 0.670 | 3.7s | ✅ 正确回答（扩散方程） |
| 未知领域问题 | <0.4 | — | ✅ 诚实拒答（不编造） |

观察：置信度与知识覆盖度正相关；拒答机制在记忆缺失时可靠触发；端到端延迟约 3.8s（含 14B 本地生成）。

### 5.3 工程方法论 / Engineering Methodology

"规范—实现—回归测试"闭环：32 项回归测试覆盖 13 项已暴露缺陷（C1–C13）与 10 项机制不变量；反向验证（注入 bug）使 4–5 项测试失败，证明测试有效性。统计断言（多种子 ≥80% 成功率）消除随机种子导致的 flaky 测试。

---

## 6. 局限与未来工作 / Limitations & Future Work

1. **规模**：当前 16K 神经元为原型规模；规范设计目标为亿级神经元（分级存储：VRAM/RAM/NVMe 三层，激活度驱动加载）。
2. **语言区依赖**：当前由 14B 量化模型担任语言区，输出质量受其上限约束；架构上语言区可替换为任意更强模型。
3. **记忆规模**：27 条经验远未覆盖开放域；蒸馏管线（ai_distiller/llm_distiller）可持续扩充。
4. **自发活动（静默态/默认模式）**：无任务时自发回放/联想/预演/整合的调度策略处于设计阶段。
5. **评估**：缺乏系统性的多任务基准；未来工作包括推理/输出解耦异步、两模块通信协议、回放作业调度。

---

## 7. 结论 / Conclusion

BioBrain 演示了一条与"先训练后使用"范式不同的路径：**学习是思考的副产品**。张量化神经组织以矩阵运算承载持续思考流，置信度门控提供防幻觉元认知，双相睡眠实现在线巩固，LLM 语言区保留通用语言能力。系统以 OpenAI 兼容 API 融入现有 LLM 生态，可被编排框架直接调用。项目同时沉淀了一套"规范—实现—回归测试"的闭环工程方法论（v1.0→v7.4 八次大版本迭代、13 项缺陷回填），为后续亿级神经元扩展提供工程基础。

BioBrain demonstrates a path distinct from the train-then-use paradigm: **learning as a byproduct of thinking**. Tensorized neural tissue sustains a continuous thinking stream through matrix operations; the confidence gate provides anti-hallucination metacognition; biphasic sleep realizes online consolidation; the LLM language area preserves general language ability. The system integrates into the existing LLM ecosystem through an OpenAI-compatible API and can be invoked by orchestration frameworks directly. The project also establishes a closed-loop "specification–implementation–regression-test" methodology (eight major version iterations from v1.0 to v7.4, thirteen defect back-fills), laying an engineering foundation for future billion-neuron scaling.

---

*仿生AI项目设计 · BioBrain Project · 2026-09 · Author: Satyr_Astry*
