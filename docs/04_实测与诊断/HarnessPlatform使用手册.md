# HarnessPlatform 使用手册

> 代码位置：`code/brain_harness/`（包名**不是** `platform` —— 那会遮蔽 Python 标准库）
> 设计文档：`docs/02_计划/HarnessPlatform架构设计.md`

---

## 1. 快速开始

```bash
cd code

# 看有哪些插件
python -m brain_harness.cli list

# 零配置演示（8 tick）
python -m brain_harness.cli demo --steps 8

# 按 YAML 配置跑 30 tick
python -m brain_harness.cli run -c brain_harness/config.example.yaml -n 30

# 持续跑（Ctrl-C 停）
python -m brain_harness.cli run -c brain_harness/config.example.yaml
```

---

## 2. 内置插件

| 类别 | 名字 | 作用 | 主要参数 |
|---|---|---|---|
| **Source** | `file` | 读文件一行（消费掉） | `path` |
| | `clock` | 每 N tick 自发一次 | `every` |
| | `static` | 按列表依次产出 | `items`, `repeat` |
| | `http` | 收 POST 文本入队 | `port`, `path` |
| **Processor** | `think` | `brain.think()` | `task_type` |
| | `observe` | 多模态观察 | — |
| | `teach` | 知识注入 | `source_name` |
| **Policy** | `sleep` | 定期睡眠巩固 | `every`, `min_buffer` |
| | `metrics` | 采集指标 | `every` |
| | `evolve` | 结构演化 | `every` |
| | `workmem_probe` | 历史区分度探针 | `every` |
| **Sink** | `log` | 控制台/日志 | `echo`, `path` |
| | `jsonl` | 结构化结果 | `path` |
| | `http` | 只读 HTTP 端点 | `port` |
| | `memory` | 记忆用 JSONL | `path` |

**HTTPSink 端点**：`GET /health` `/latest` `/results` `/state`

---

## 3. 扩展：写自己的插件（**不改平台核心**）

```python
from brain_harness import source, processor, policy, sink, Stimulus, Result

@source("my_sensor")                     # 1. 注册
class MySensor:                          # 2. 写类（无需继承）
    name = "my_sensor"
    def __init__(self, device="/dev/ttyS0", **kw):   # 3. 配置即 kwargs
        self.device = device
    def poll(self, ctx):
        data = read_from(self.device)     # 你的逻辑
        if data is None:
            return None
        return Stimulus(text=data, source=self.name)
    def close(self): pass
```

然后在 YAML 里用：

```yaml
pipeline:
  sources:
    - type: my_sensor
      device: "/dev/ttyS0"
```

**关键**：
- `type` 字符串 = 装饰器参数
- 其余字段**原样作为构造参数**（`cls(**spec)`）
- **无需改 `registry.py` / `pipeline.py` / `platform.py`**

---

## 4. 与现有入口的关系

**平台不替换任何现有入口**（全部仍可独立运行）：

| 现有 | 平台里的对应 |
|---|---|
| `harness.py --daemon` | `cli run -c platform/config.example.yaml` |
| `brain_daemon.py` | 待迁移（阶段 2） |
| `auto_evolve.py` | `Policy: evolve` |
| `batch_learn` / `llm_distiller` | 待变成 Processor（阶段 2） |

---

## 5. 验收结果（实测）

| 项 | 目标 | 结果 |
|---|---|---|
| **可扩展性** | 新插件不改核心 | ✅ 实测（`countdown`/`upper`/`counter` 全部生效） |
| **单实例** | HTTP+学习+睡眠同进程仅 1 个 CogVec | ✅ `/state` 返回 `neurons: 64`，handle 唯一 |
| **HTTP 并发** | 端点可用 | ✅ 4/4（`/health` `/latest` `/results` `/state`） |
| **回归** | pytest 仍 37 passed | 见 `验收报告` |
| **开销** | <1.05× | 见 `验收报告` |

---

## 6. 已知限制

| 限制 | 说明 |
|---|---|
| **单大锁** | `BrainHandle` 用一把 `RLock` —— tick 频率高时会串行化。当前 tick 17ms，可接受 |
| **无热重载** | 改插件要重启进程 |
| **`evolve` Policy** | 依赖 `auto_evolve.py` 暴露 `evolve_once`/`run_once`，若没有则跳过 |
| **YAML 依赖** | 有 pyyaml 用真解析；没有则用内置极简解析（只支持平台配置形状） |

---

## 7. Web UI 观测面板（v1.1 新增）

### 启动

```bash
cd code
python -m brain_harness.cli run -c brain_harness/config.ui.yaml --paced 0.5
```

然后浏览器打开 **http://127.0.0.1:8765**

### 界面内容

| 区域 | 显示 |
|---|---|
| **顶栏** | 连接状态灯（绿=在线）· `neurons / tracts / active` · 刷新次数 |
| **左列** | 大脑状态 KPI（版本/种子/神经元/束/群组/库/活跃/ticks/体验缓冲/经历数/成功率）· **输入注入框** |
| **中列** | **收敛度趋势图**（Canvas 自绘，最近 60 条）· **思考流表格**（tick / 自发·外部标签 / 输入→输出 / 收敛度） |
| **右列** | 流水线统计（ticks/刺激/结果/策略执行/错误 + 按来源/按策略）· 指标历史 |

### HTTP 端点

| 端点 | 说明 |
|---|---|
| `GET /` | 单页 UI |
| `GET /api/snapshot` | **一次拿全**（state + results + stats + metrics）—— UI 主轮询 |
| `GET /api/results` | 最近 40 条结果 |
| `GET /api/state` | 大脑状态 |
| `GET /api/metrics` | 指标历史（`metrics` policy 的 history） |
| `GET /api/stats` | pipeline 统计 |
| `POST /api/inject` | **注入一条输入**（JSON: `{"text": "..."}`） |
| `GET /health` | 健康检查 |

### 配置

```yaml
pipeline:
  sinks:
    - type: webui
      port: 8765
      keep: 500        # 内存中保留的结果条数
```

### ⚠️ 踩坑记录

| 坑 | 说明 |
|---|---|
| **curl 传中文会乱码** | 终端编码破坏 UTF-8（显示 `Сڲƽ̨`）。**用 Python `urllib` + `ensure_ascii=False` 才准** |
| **注入后要马上查** | `/api/results` 只留 40 条，`--paced 0.5` 时 20 秒就滑出窗口 |
| **`webui` 注册依赖 `load_builtins`** | 新增 Sink 模块后必须在 `registry.load_builtins()` 里 import，否则 `list` 看不到 |

### 验证过的行为

```
GET  /health        → {"ok": true, "sink": "webui", "port": 8765}
GET  /api/snapshot  → neurons 64 / tracts 5 / active 60 / metrics 4条
POST /api/inject    → {"ok": true, "injected": "小悠在测试平台UI"}
                      下一 tick: tick115 [inject] 小悠在测试平台UI  conv=0.496
```
