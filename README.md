# 🧠 BioBrain — 仿生认知架构 / Bionic Cognitive Architecture

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**A bionic cognitive architecture that thinks continuously and learns while thinking — instead of "train-then-use".**

| | |
|---|---|
| **Paradigm** | Learning is a byproduct of thinking (NORM-3) |
| **Core** | Tensorized neural tissue: 16,384 neurons / 1,024 overlapping tracts (matrix ops) |
| **Memory** | BM25 + semantic hybrid retrieval with confidence gating |
| **Language** | LLM language area (frozen weights, commanded by the cognitive layer) |
| **API** | OpenAI-compatible (`POST /v1/chat/completions`, works with any client / Hermes Agent) |
| **Docs** | [Paper (EN/中文)](paper/biobrain_paper.md) · [Engineering spec v7.4 (中文)](docs/工程规范_v7.3.md) |

### Quick Start

```bash
# 1. Install Ollama + pull a local LLM (the "language area")
ollama pull deepseek-r1-abliterated:14b

# 2. Install deps
pip install numpy scikit-learn

# 3. Start the brain as an OpenAI-compatible service
cd src
python api_server.py --port 8701 --preload

# 4. Talk to it (any OpenAI client)
curl -X POST http://127.0.0.1:8701/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bio-brain","messages":[{"role":"user","content":"What is KV caching?"}]}'
```

### Architecture

```
             ┌─────────────────────────────────┐
             │   Thinking stream (never stops)  │
 sensory ◀───│   compute ⇄ consolidate ⇄ sleep │───▶ motor
 (text)      │  · kernel ALG-0                  │    (LLM area)
             │  · online plasticity ALG-3       │
             │  · convergence ALG-11            │
             └────────────┬────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │  Neural library (tiered)     │
           │  library→group→tract→neuron  │
           │  VRAM / RAM / NVMe           │
           └──────────────┬──────────────┘
                          │
             ┌────────────┴────────────┐
             │  Biphasic sleep ALG-7     │
             │  NREM consolidate + REM   │
             │  + multi-scale ALG-12     │
             │  + grounding ALG-14       │
             └──────────────────────────┘
```

### Key Mechanisms

| ID | Mechanism | What it does |
|---|---|---|
| ALG-0 | Computation kernel | Intra-tract message passing + tract-level gated attention |
| ALG-3 | Online plasticity | Delayed plasticity (causal), eligibility × RPE |
| ALG-7 | Biphasic sleep | NREM consolidation + REM associative reorganization |
| ALG-10 | Activation budget | Scenario-dependent budget (task: tighten, association: relax) |
| ALG-11 | Convergence | "Have I thought enough?" — decides when to output |
| ALG-12 | Multi-scale weights | w = w_fast + w_mid + w_slow, independent learning per scale |
| ALG-14 | Symbol grounding | Co-activation + causal verification + 5-D audit |

**Anti-hallucination**: the confidence gate makes the brain *know what it doesn't know* — confidence < 0.6 suppresses the LLM and forces an honest "I don't know".

### Integrate with Hermes Agent

```yaml
# config.yaml — add as a provider (do NOT replace the main model)
providers:
  bio-brain:
    name: bio-brain
    base_url: http://127.0.0.1:8701/v1
    model: bio-brain
    api_key: brain-local
```

### Repository Layout

```
repo/
├── src/                 # core code (tensor_brain, conductor, cortex, api_server, ...)
├── docs/                # design documents (Chinese, v1 → v7.4 evolution)
├── paper/               # bilingual paper
├── scripts/             # launcher
└── .gitignore
```

### Roadmap

- [ ] Billion-neuron scaling (tiered VRAM/RAM/NVMe, activity-driven loading)
- [ ] Spontaneous default-mode activity (replay / association / rehearsal)
- [ ] Decoupled inference/output with async two-module protocol
- [ ] Larger distilled memory corpus

---

<a name="中文"></a>
## 中文

**一个持续思考、在思考中学习的仿生认知架构——而不是"先训练后使用"。**

| | |
|---|---|
| **范式** | 学习是思考的副产品（NORM-3 无分离） |
| **核心** | 张量化神经组织：16384 神经元 / 1024 重叠神经束（矩阵运算） |
| **记忆** | BM25+语义混合检索 + 置信度门控 |
| **语言** | LLM 语言区（权重冻结，受认知层指挥） |
| **接口** | OpenAI 兼容（任意客户端 / Hermes Agent 可直接调用） |
| **文档** | [论文（中英）](paper/biobrain_paper.md) · [工程规范 v7.4](docs/工程规范_v7.3.md) |

### 快速开始

```bash
# 1. 安装 Ollama 并拉取本地模型（"语言区"）
ollama pull deepseek-r1-abliterated:14b

# 2. 安装依赖
pip install numpy scikit-learn

# 3. 以 OpenAI 兼容服务启动大脑
cd src
python api_server.py --port 8701 --preload

# 4. 对话（任意 OpenAI 客户端）
curl -X POST http://127.0.0.1:8701/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bio-brain","messages":[{"role":"user","content":"KV缓存是什么"}]}'
```

### 核心机制

| ID | 机制 | 作用 |
|---|---|---|
| ALG-0 | 计算内核 | 束内消息传递 + 束级门控注意力 |
| ALG-3 | 在线可塑 | 延迟可塑（严格因果）+ 资格痕迹×RPE |
| ALG-7 | 双相睡眠 | NREM 巩固事实 + REM 联想重组 |
| ALG-10 | 激活预算 | 分场景预算（任务收紧/联想放开） |
| ALG-11 | 收敛度量 | "想清楚了没"——决定何时输出 |
| ALG-12 | 多尺度权重 | 快/中/慢子权重独立学习 |
| ALG-14 | 符号接地 | 共激活 + 因果验证 + 五维审计 |

**防幻觉**：置信度门控让大脑"知道自己不知道"——置信度 <0.6 压制 LLM，诚实说"不确定"。

### 接入 Hermes Agent

```yaml
# config.yaml —— 作为 provider 添加（不替换主模型）
providers:
  bio-brain:
    name: bio-brain
    base_url: http://127.0.0.1:8701/v1
    model: bio-brain
    api_key: brain-local
```

### 演进历史

```
v1.0 静默态/双轨/状态机     → v2.0 神经元四级组织+思考流
→ v3.0 感觉/运动神经元+自训练 → v4.0 可实现规范（ALG-1~9）
→ v5.0 双相睡眠+激活预算   → v6.0 计算内核+收敛度量
→ v7.0 多尺度权重+能量+接地 → v7.4 服务层落地（当前）
```

### 铁律（NORM）

1. 无待机：只有思考态，主循环永不退出
2. 无接口：输入输出是感觉端/运动端，非外挂 API
3. 无分离：训练是思考的副产品
4. 因果优先：在线更新只影响未来
5. 合并独占：只在无活动计算时合并
6. 可靠信号：只让外部可验证信号进主体，自评禁用
7. 资源约束：不当 Hermes 主会话；12GB 不并排两 base

---

*仿生AI项目设计 · 2026-09 · 作者：Satyr_Astry*
