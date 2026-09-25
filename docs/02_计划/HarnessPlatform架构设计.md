# CogVec Harness Platform · 架构设计（v1.0）

> 落笔：2026-09-19
> 状态：**设计稿**（NORM-8：先改架构书，确认后才写代码）
> 目标：把 13 个散落的入口收敛成**一个可扩展的平台**

---

## 1. 为什么要平台（现状实测）

### 1.1 现状：13 个独立入口，各干各的

| 文件 | 行数 | 干什么 | 问题 |
|---|---|---|---|
| `harness.py` | 242 | 思考流驱动（inbox→think→outbox） | **硬编码**：刺激源/睡眠策略/日志全写死 |
| `brain_daemon.py` | 224 | 把 Conductor 接进思考流 | 与 harness **重复** |
| `brain_sharded.py` | 234 | 分片版（亿级） | 独立入口 |
| `batch_learn.py` | 159 | 用 LLM 批量灌输 | 独立入口 |
| `auto_evolve.py` | 213 | 自动优化 | 独立入口 |
| `self_iterate.py` | 237 | 持续自主迭代 | 独立入口 |
| `autonomous_agent.py` | 394 | 自主智能体 | 独立入口 |
| `server.py` | 573 | HTTP 服务 + CLI | 独立入口 |
| `api_server.py` | 369 | OpenAI 兼容（已降级薄代理） | 独立入口 |
| `llm_distiller.py` | 297 | LLM 蒸馏 | 独立入口 |
| `ai_distiller.py` | 335 | AI 蒸馏 | 独立入口 |
| `clean_memory.py` | 204 | 记忆清理 | 独立入口 |
| `brain_daemon` / `cortex` / `conductor` | — | 集成路线② | 独立入口 |

### 1.2 三个具体痛点（实测）

**痛点 1：无法组合**
想"边跑边学 + 同时提供 HTTP + 定期睡眠"→ 要**同时开 3 个进程**，**各自持有独立的 CogVec 实例** → **状态分裂**（P-ARCH-1 的同源病）。

**痛点 2：加新能力要改核心代码**
`harness.py` 的 `tick()` 里**硬编码**了：
```python
ext = self.stim.poll_external()      # 输入源写死 inbox 文件
if self.cycle % 8 == 0:              # 睡眠周期写死 8
if self.cycle % 10 == 0:             # 报告周期写死 10
```
→ 想加"HTTP 输入"或"每 3 tick 睡眠" → **必须改 harness.py 本身**。

**痛点 3：无法观测/复现**
每个入口的日志格式不同、状态无统一视图、跑不出可对比的实验记录。

---

## 2. 平台设计

### 2.1 一句话

> **把 "大脑" 与 "驱动它的各种东西" 解耦** ——
> 大脑只有一个实例；**输入源、处理阶段、策略、输出端**都是**可插拔的组件**，
> 由 **Pipeline** 按声明式配置组装并驱动。

### 2.2 架构图

```
                    ┌─────────────────────────────────────┐
                    │      HarnessPlatform (单例)          │
                    │  ┌───────────────────────────────┐  │
                    │  │   BrainHandle                 │  │
                    │  │   （唯一的 CogVec 实例）     │  │
                    │  └───────────────────────────────┘  │
                    │                                     │
                    │   Pipeline（声明式组装）              │
                    │   ┌──────────┐                      │
   输入源 ──────────┼──▶│  Source  │  FileSource          │
   (可插拔)         │   │  (产生   │  HTTPSource          │
                    │   │   刺激)  │  ClockSource         │
                    │   └────┬─────┘  MemoryReplaySource  │
                    │        │                            │
                    │   ┌────▼─────┐                      │
                    │   │ Processor│  ThinkProcessor      │
                    │   │ (处理    │  ObserveProcessor    │
                    │   │  刺激)   │  TeachProcessor      │
                    │   └────┬─────┘                      │
                    │        │                            │
                    │   ┌────▼─────┐                      │
                    │   │  Policy  │  SleepPolicy         │
                    │   │ (旁路    │  EvolvePolicy        │
                    │   │  策略)   │  DistillPolicy       │
                    │   └────┬─────┘                      │
                    │        │                            │
                    │   ┌────▼─────┐                      │
   输出端 ◀─────────┼───│   Sink   │  JSONLSink           │
   (可插拔)         │   │ (消费    │  HTTPSink（对外）     │
                    │   │  结果)   │  LogSink / MemorySink│
                    │   └──────────┘                      │
                    │                                     │
                    │   Recorder（统一观测/复现）           │
                    └─────────────────────────────────────┘
```

### 2.3 四个扩展点（插件协议）

```python
class Source(Protocol):
    """刺激来源：产出待处理的刺激"""
    name: str
    def poll(self, ctx: TickContext) -> Optional[Stimulus]: ...
    def close(self) -> None: ...

class Processor(Protocol):
    """刺激处理器：把刺激变成结果"""
    name: str
    def process(self, stim: Stimulus, brain, ctx: TickContext) -> Result: ...

class Policy(Protocol):
    """旁路策略：不产出刺激，但可影响大脑（睡眠/进化/蒸馏）"""
    name: str
    def should_run(self, ctx: TickContext) -> bool: ...
    def run(self, brain, ctx: TickContext) -> dict: ...

class Sink(Protocol):
    """结果消费者：日志/HTTP/记忆/外发"""
    name: str
    def handle(self, result: Result, ctx: TickContext) -> None: ...
    def close(self) -> None: ...
```

**关键**：**四个协议都用 `Protocol`（结构化子类型）**，插件**无需继承基类**，只要方法签名匹配即可 → **零耦合扩展**。

### 2.4 声明式配置（YAML）

```yaml
# platform.yaml
brain:
  seed: 1
  state_path: "run/brain.json"
  neuron_scale: 1.0          # 可选扩规模

pipeline:
  sources:
    - type: file
      path: "run/inbox.txt"
      weight: 1.0            # 优先级
    - type: clock
      interval: 5.0          # 每 5 tick 自发一次
  processors:
    - type: think
      task_type: general
  policies:
    - type: sleep
      every: 8
      min_buffer: 2
    - type: evolve
      every: 50
  sinks:
    - type: log
    - type: jsonl
      path: "run/outbox.jsonl"
    - type: http
      port: 8642
    - type: metrics
      path: "run/metrics.json"

recorder:
  path: "run/recorder.jsonl"
  snapshot_every: 100
```

### 2.5 与现有代码的关系（**不破坏**）

| 现有 | 平台化后 |
|---|---|
| `harness.py` | **保留为兼容 CLI**，内部改为调 Platform（或标记 deprecated） |
| `brain_daemon.py` 等 | **不改**（P1：先并存，验证后再收敛） |
| `server.py` / `api_server.py` | **不改**，平台可通过 `HTTPSink` 复用 |
| `auto_evolve` / `self_iterate` | **改为 Policy 插件**（不删，包一层） |

**原则**：**新增，不替换**（避免又一次"改一处坏一处"）。

---

## 3. 实施计划（分阶段）

### 阶段 1：平台内核（本次）
```
platform/
  __init__.py
  context.py      # TickContext / Stimulus / Result
  protocols.py    # 4 个 Protocol
  brain_handle.py # 唯一 CogVec 实例 + 线程安全
  pipeline.py     # 组装 + 驱动
  recorder.py     # 统一观测
  registry.py     # 插件注册表（type 字符串 → 类）
  sources/        # 内置 Source 实现
  processors/     # 内置 Processor 实现
  policies/       # 内置 Policy 实现
  sinks/          # 内置 Sink 实现
  cli.py          # 统一入口
```
**验收**：能用 YAML 配置跑起来，产出与 `harness.py --once 30` 等价的结果。

### 阶段 2：迁移现有入口（后续）
- `harness.py` → 薄封装
- `auto_evolve` / `self_iterate` → Policy 插件
- `batch_learn` / `llm_distiller` → Processor 插件

### 阶段 3：可观测性（后续）
- Web 面板（复用 `server.py` 的 HTTP）
- 指标导出（Prometheus 格式）

---

## 4. 架构约束（遵守项目铁律）

| 铁律 | 如何遵守 |
|---|---|
| **NORM-1 无待机** | 平台主循环永不退出；`--once` 只用于测试 |
| **NORM-2 无接口** | HTTP 是 Sink（输出端），不作为大脑的"外挂 API" |
| **NORM-3 无分离** | 所有学习都是 tick 的副产品，平台只驱动不训练 |
| **NORM-4 因果优先** | 平台的写入在 tick 边界提交（复用现有 pending 机制） |
| **NORM-7 资源约束** | 平台**不引入新的大模型**；单实例 |
| **NORM-8 架构先行** | 本文档先行；实施后更新 `总架构书.md` |
| **NORM-9 诚实输出** | 无生成后端时 Sink 收到 `output=None`，如实记录 |
| **NORM-10 基线就近** | 平台验收用"改造前"的实测基线对比 |

---

## 5. 验收标准（可量化）

| 项 | 目标 |
|---|---|
| **可扩展性** | 新增一个 Source **只需写一个类 + 注册**，**不改平台核心** |
| **组合能力** | 同一进程内 **HTTP + 学习 + 睡眠** 并行，**只 1 个 CogVec 实例** |
| **等价性** | 用平台跑 30 步，与 `harness.py --once 30` 的**行为等价**（同 seed 同输入→同收敛序列） |
| **回归** | `pytest test_bio_brain.py` **仍 37 passed** |
| **可观测** | 统一 `recorder.jsonl`，含每 tick 的 source/processor/policy 来源标记 |
| **不改坏** | 现有 13 个入口**全部仍可独立运行** |

---

## 6. 风险

| 风险 | 缓解 |
|---|---|
| 平台变成"又一个入口" | 明确目标：**是收敛器，不是第 14 个** |
| 抽象过度 | 阶段 1 只做**最小可用**（4 协议 + 3 内置实现） |
| 性能开销 | tick 级插件的开销**实测 < 5%**（否则回退） |
| 线程安全 | CogVec 非线程安全 → `BrainHandle` 用**单锁**串行化 |
