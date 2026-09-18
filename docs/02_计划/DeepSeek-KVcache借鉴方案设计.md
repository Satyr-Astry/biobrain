# DeepSeek KV Cache 借鉴方案设计（v1.0）

> 落笔：2026-09-19
> 依据论文：**DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression**（arXiv:**2609.19969**，2026-09-17，DeepSeek-AI）
> 辅助：**MISA: Mixture of Indexer Sparse Attention**（arXiv:2605.07363，2026-05-08）
> 状态：**设计文档**。按 NORM-8 先行；批准后写入 `总架构书.md`，再动代码。

---

## 0. 论文事实（实证摘录，可复查）

### 0.1 核心成就（原文数据）

| 指标 | 数值 |
|---|---|
| **单 token 全局 KV 占用** | **890 bytes** |
| vs DeepSeek-V4-Flash | **1/4** |
| vs DeepSeek-V1 | **1/437** |
| **持久化 KV**（SSD/主机内存） | **1/8** |
| Decode FLOPs | **随上下文长度近乎恒定** |

> 原文："These designs reduce its global KV cache footprint (always in HBM) to **890 bytes per token**,
> roughly **1/4** of the corresponding footprint of DeepSeek-V4-Flash."

### 0.2 机制一：CSA2（Compressed Sparse Attention 2）—— 跨层复用

**三种静态分配模式**：

| 模式 | 复用 | 计算 | 成本 |
|---|---|---|---|
| **Full** | 不复用 | 算 main KV + indexer K + indexer Q + Top-K 索引（**全路径**） | 最贵 |
| **Reindex** | **复用上游 main KV + indexer K** | **只用自己 indexer Q 重新打分** → 新 Top-K | 中 |
| **Reuse** | **复用上游 main KV + Top-K 索引** | **直接做稀疏注意力**，不算 indexer | 最便宜 |

> 原文："**Full Mode** generates global KV and performs indexing.
> **Reindex Mode** reuses the global KV from a preceding layer, and uses its own indexer Q
> to rescore the shared indexer K and select fresh Top-K indices.
> **Reuse Mode** reuses both global KV and the Top-K indices in a preceding layer,
> and directly performs sparse attention."
>
> "Sharing global KV and indexer K **reduces duplicated cache storage**."

### 0.3 机制二：Hierarchical Sparse Indexer（分层稀疏索引器）

> 原文："For each query, the **first layer assigned to Full Mode constructs a candidate pool**
> that later re-indexing layers use as their **search domain**."
>
> "This first Full Mode layer scores **all causally visible main KV positions** and produces
> the Top-K indices for its own attention. It also performs **blockwise candidate selection**:
> each block is assigned the **maximum index score** among its positions, and the blocks with
> the highest scores are selected."
>
> "For a fixed candidate-pool size, the number of positions scored per query by each subsequent
> indexer is **bounded independently of context length**."

**流程**：
```
第一层（Full）：扫描全范围 → 选 Top-512 + 按块建候选池
     ↓
后续层（Reindex）：只在候选池内打分 → 成本与上下文长度解耦
```

### 0.4 机制三：FP4 + SWA Bounded Replay

- **FP4 main KV cache**（训练时就用，性能损失"marginal"）
- **SWA Bounded Replay**：只重放最近 `n_win` token（而非 `L × n_win`）
  → 持久化 KV 降到 **1/8**

### 0.5 MISA 的补充洞察

> 原文："the indexer uses **many query heads** (for example, **64 on DeepSeek-V3.2**) that
> share the same selected token set; **this multi-head design is precisely what makes the
> indexer the dominant cost on long contexts**."

**MISA 方案**：把 indexer heads 当 MoE 专家池 → 轻量 router 只选少数活跃 head
→ **8 个活跃 head 追平 dense DSA，kernel 加速 3.82×**

---

## 1. 项目现状对照（实测）

| 项 | 项目实测 | DeepSeek 对照 |
|---|---|---|
| **注意力打分** | `BIND` 相位 O(N_active·H·T) = **1160 次打分**，5.63 µs/打分，占 compute **26.6%** | CSA2 用三模式把大部分层降为 Reuse |
| **头数** | `N_HEADS=4` | 64 heads（V3.2）→ MISA 收益大 |
| **束间冗余** | **T65↔T66 key 余弦 = 1.00**（两束在算同一件事） | 跨层 KV 共享消除重复存储 |
| **稀疏化尝试（已失败）** | 限制自身束 → **只有 5/208 = 2.4% 路由一致**；根因：真值 top-K 仅 **31.2%** 落在自身束内 | Hierarchical Indexer：**粗筛候选池 + 池内精排** |
| **存盘** | `neurons.npz` 123 KB（float32） | FP4 → 体积 1/4 |

---

## 2. 借鉴方案（4 项，按优先级）

### 🥇 D1 — CSA2 式三模式（给 Tract 加 `attn_mode`）

#### 2.1.1 机制设计

```python
# bio_brain.py — Tract 新增字段
@dataclass
class Tract:
    ...
    attn_mode: str = "full"      # "full" | "reindex" | "reuse"

# 静态分配规则（按 tier 层级，仿 CSA2 的静态模式）
#   sensory（第 0 tier）        → full    （必须产生索引）
#   inter 第一束                → full    （产生候选池）
#   inter 其余 / drives         → reindex （复用 KV，重排索引）
#   motor                       → reuse   （直接白嫖上游索引）
```

#### 2.1.2 伪代码

```python
def _phase_bind_csa2(self, active: Set[int]):
    """BIND 相位：按 CSA2 三模式分层复用。"""
    # ① 按 tier 排序束
    tracts_by_tier = sorted(self.tracts.items(),
                            key=lambda kv: TIER_ORDER.get(kv[1].tier, 1))

    for tid, tr in tracts_by_tier:
        mode = tr.attn_mode
        if mode == "full":
            # 全路径：算 Q/K/V + Top-K
            q, k, v = self._compute_qkv(tr, active)
            scores = self._score_all(q, k)          # 贵
            topk = self._topk(scores, ROUTE_K)
            self._cache[(tid, "qkv")] = (q, k, v)   # 供下游复用
            self._cache[(tid, "topk")] = topk
            self._attn_out.update(self._apply_attn(tr, topk, v))
        elif mode == "reindex":
            # 复用上游 K/V，只用自己 Q 重排
            up_q, up_k, up_v = self._cache.get(("upstream", "qkv"), (None,)*3)
            if up_k is None:
                # 回退到 full（上游还没算）
                v = self._full_path(tr, active)
            else:
                q = self._compute_q(tr, active)
                scores = self._score(q, up_k)        # 中
                topk = self._topk(scores, ROUTE_K)
                self._cache[(tid, "topk")] = topk
                self._attn_out.update(self._apply_attn(tr, topk, up_v))
        else:  # reuse
            # 直接白嫖最近一次 topk
            topk = self._cache.get(("upstream", "topk"))
            if topk is None:
                v = self._full_path(tr, active)
            else:
                up_v = self._cache[("upstream", "qkv")][2]
                self._attn_out.update(self._apply_attn(tr, topk, up_v))
```

#### 2.1.3 预期收益与验收

| 指标 | 现状 | 目标 | 测法 |
|---|---|---|---|
| **BIND 打分次数** | 1160 | **降 50-75%**（仅 1-2 个束 full） | 计数器 |
| **compute 耗时** | 基线 | **降 15-20%**（BIND 占 26.6%） | `_p0_accept.py` |
| **路由一致性** | 208/208 | ⚠️ **会变**（这是行为变更） | 需**重新建立基线** |
| 回归测试 | 37 passed | 37 passed | pytest |

#### 2.1.4 风险

| 风险 | 缓解 |
|---|---|
| **改变路由语义**（不再是"每束独立"） | 提供 `BIO_CSA2=0` 回退开关；先 A/B 对比 |
| 上游无缓存时回退逻辑复杂 | 显式 fallback 到 full path（已在伪代码中） |
| 与 `TIER_BIAS=+0.15` 冲突 | `TIER_BIAS` 可降为 0（CSA2 取代手工偏置） |

---

### 🥈 D2 — 两阶段路由（Hierarchical Sparse Indexer）★ 治"2.4% 陷阱"

#### 2.2.1 这是**之前失败的正解**

| 项目失败做法 | DeepSeek 正解 |
|---|---|
| **硬性限制在自身束内** → 只有 5/208 = 2.4% 一致 | **全范围粗筛候选池 → 池内精排** |
| 丢失跨束目标（真值 top-K 仅 31.2% 在自身束内） | **保留跨束可能**（粗筛是全范围） |
| 一次性决定 | **两阶段**：粗筛 + 精排 |
| 成本随规模增长 | **候选池固定 → 成本与规模解耦** |

#### 2.2.2 伪代码

```python
def _phase_bind_hierarchical(self, active: Set[int]):
    """两阶段路由：粗筛候选池 + 池内精排（仿 DeepSeek Hierarchical Sparse Indexer）。"""
    # ===== 阶段 1：粗筛（低频，每 CANDIDATE_REFRESH_TICKS tick 做一次）=====
    if self.tick_count % CANDIDATE_REFRESH_TICKS == 0 or self._candidate_pool is None:
        # 全范围打分（贵，但低频）
        full_scores = self._score_all_tracts(active)
        # 按块选：每块取最大分（仿原文 "each block is assigned the maximum index score"）
        block_scores = self._blockwise_max(full_scores, block_size=CAND_BLOCK)
        # 取前 K 块 → 展开成候选池
        top_blocks = np.argsort(-block_scores)[:CAND_BLOCKS]
        self._candidate_pool = self._expand_blocks(top_blocks)

    # ===== 阶段 2：精排（每 tick，只在候选池内）=====
    cand = self._candidate_pool
    scores = self._score_subset(cand, active)        # 便宜，池大小固定
    topk = self._topk(scores, ROUTE_K)
    return topk
```

#### 2.2.3 关键参数（仿原文）

| 参数 | 建议值 | 依据 |
|---|---|---|
| `CANDIDATE_BLOCKS` | 32（原文用 Top-512 位置 → 我们规模小） | 原文 "Top-512 indices" |
| `CAND_BLOCK`（块大小） | 4~8 | 原文用 blockwise candidate selection |
| `CANDIDATE_REFRESH_TICKS` | 20（与 `COOLDOWN_TICKS` 同量级） | 复用项目已有节拍 |

#### 2.2.4 预期收益与验收

| 指标 | 现状 | 目标 | 测法 |
|---|---|---|---|
| **路由一致率** | 2.4%（束内限制）| **接近 100%**（候选池来自全范围） | `_c2_final.py` 改造 |
| **精排成本** | O(N·H·T) | **O(CAND·H)**，与束数解耦 | 计数器 |
| dmin（区分度） | 0.0088 | **不退化**（需重标定） | `diag_discrim.py` |

#### 2.2.5 风险

| 风险 | 缓解 |
|---|---|
| **粗筛频率太低 → 候选池过期** | `CANDIDATE_REFRESH_TICKS` 可调；先设 20 tick |
| 池太小 → 丢目标 | A/B 测试 `CANDIDATE_BLOCKS = 16/32/64` |
| 与 D1（CSA2）叠加复杂 | **先做 D1，稳定后再加 D2** |

---

### 🥉 D3 — 跨束 KV 共享（治 T65/T66 冗余）

#### 2.3.1 实测依据

```
key 相似度实测：
  T65 ↔ T66 = 1.00   ← 完全重合！（16 sensory + 2 inter 构成相同）
  T67 ↔ T68 = 1.00
  T65 ↔ T67 = 0.119  （远低于 0.5 阈值）
```

**原文对照**："Sharing global KV and indexer K **reduces duplicated cache storage**."

#### 2.3.2 设计（**关键转变**）

> **不要把"key 余弦 = 1.00"当 bug 修，而是当"复用机会"用。**

```python
# 若两束 key 相似度 > REUSE_THRESHOLD：
#   → 判定为"同构束"，让下游束直接复用上游束的 KV
#   → 不重复计算
def _classify_tract_modes(self):
    """按 key 相似度自动分配 attn_mode（替代手工静态分配）。"""
    for tid, tr in self.tracts.items():
        sim = max(key_sim(tr, other) for other in self.tracts if other is not tr)
        if sim > REUSE_THRESHOLD:      # 如 0.95
            tr.attn_mode = "reuse"      # 有同构伙伴 → 复用
        elif tr.tier == "sensory":
            tr.attn_mode = "full"
        else:
            tr.attn_mode = "reindex"
```

#### 2.3.3 预期收益

| 项 | 现状 | 目标 |
|---|---|---|
| T65/T66 重复计算 | **100% 冗余** | 消除 |
| BIND 打分 | 1160 | **可降 40%**（2 对同构束） |

---

### 4️⃣ D4 — FP4 存盘量化（治体积，**仅存档**）

#### 2.4.1 设计原则（**关键**）

> **只对"存档"用 FP4，不对"运行时"用** ——
> 存盘时压成 FP4，加载时还原成 float32 → **不影响运行时语义**。

```python
# persistence.py
def save_state(brain, state_path):
    ...
    # 量化存盘（4-bit）
    np.savez_compressed(
        os.path.join(d, "neurons.npz"),
        **{k: _to_fp4(v) for k, v in neu.items()}    # ★量化
    )

def _to_fp4(arr):
    """float32 → 4-bit（16 个可表示值）→ 打包存储。"""
    # 用 np.float16 + 位打包，或直接存 4-bit 索引 + scale
    scale = np.abs(arr).max() / 7.0 if np.abs(arr).max() > 0 else 1.0
    q = np.clip(np.round(arr / scale), -7, 7).astype(np.int8)   # 4-bit 有符号
    return q, scale   # 存储 (量化值, 缩放因子)
```

#### 2.4.2 预期收益与风险

| 项 | 现状 | FP4 后 |
|---|---|---|
| `neurons.npz` | 123 KB | **~15 KB** |
| 存盘/加载 | 基线 | **快 4-8×** |

**⚠️ 风险（重要）**：
| 风险 | 说明 | 缓解 |
|---|---|---|
| **破坏"位级一致"铁律** | 项目刚做完 `PYTHONHASHSEED` 确定性修复，要求可复现 | **量化只在存档，不在运行时** → 内存中始终 float32 |
| **量化误差累积** | 多次存/读循环会累积误差 | 加断言：往返误差 < 阈值 |
| **C17 测试会挂** | 现有 C17 要求"逐元素一致 < 1e-5" | 需**放宽到 FP4 精度**（或改为"关闭量化时严格一致"）|

---

## 3. 实施顺序（按风险排序）

```
Step 0（NORM-8）：本方案写入 总架构书.md
        · 新增 ALG-26（CSA2 三模式）、ALG-27（两阶段路由）、ALG-28（跨束 KV 共享）
        · 缺陷表新增 P-ARCH-24（无跨层复用抽象）

Step 1  D1：CSA2 三模式【最高优先，风险最低】
        · 改：Tract 加 attn_mode 字段 + _phase_bind 分支
        · 备份 code/_backup_before_csa2/
        · 回退开关 BIO_CSA2=0
        · 验收：BIND 打分降 ≥40%；37 passed；A/B 对比路由差异

Step 2  D2：两阶段路由【治 2.4% 陷阱，需 D1 稳定后】
        · 验收：路由一致率 2.4% → >90%；dmin 不退化

Step 3  D3：跨束 KV 共享【自动模式分配】
        · 验收：T65/T66 冗余消除

Step 4  D4：FP4 存档【最后，需处理 C17 冲突】
```

---

## 4. 【不要做什么】

| 禁止项 | 理由 |
|---|---|
| ❌ **硬性按束裁剪注意力** | 已实测 **2.4% 一致率**（灾难）—— 用 D2 的"粗筛+精排"替代 |
| ❌ **运行时用 FP4** | 会破坏 `somma`/`axon` 连续向量语义 + 位级一致铁律 |
| ❌ **照搬 Top-512 的规模** | 本项目只有 5 束 / 64 神经元 → 需按 §2.2.3 重定标 |
| ❌ **同时改 D1+D2** | 一次只改一个，否则无法归因（历史四轮互相污染教训） |
| ❌ **去掉上游 fallback** | 上游无缓存时必须回退 full path，否则首 tick 无输出 |
| ❌ **期待 MISA 式收益** | MISA 针对 64 头，项目只有 4 头 → **收益有限，不优先做** |

---

## 5. 可验证的验收指标

| ID | 指标 | 现状实测 | 目标 | 测法 |
|---|---|---|---|---|
| **W1** | BIND 打分次数 | **1160** | 降 **≥40%** | 计数器 |
| **W2** | compute 耗时 | 基线 | 降 **≥15%** | `_p0_accept.py` |
| **W3** | 路由一致率（对 full baseline） | 208/208（baseline） | 记录**差异率**（行为变更） | A/B 对比脚本 |
| **W4** | 两阶段路由 vs 全范围 | — | **>90%** 一致 | 改造 `_c2_final.py` |
| **W5** | 回归测试 | 37 passed | **37 passed** | `PYTHONHASHSEED=0 python -m pytest test_bio_brain.py -q` |
| **W6** | `soma`/`axon` 非零 | 52/52 | 不退化 | guard.py A1 |
| **W7** | T65/T66 冗余 | 100% | 消除 | key 相似度 + 计算计数 |
| **W8** | 存盘体积（FP4） | 123 KB | **<30 KB** | 文件大小 |
| **W9** | 存盘往返（FP4） | ~1e-8 | **< FP4 精度上限** | `diag_persist_e2e.py`（需放宽） |

### 反判据

| 反判据 | 说明 |
|---|---|
| **打分少了 ≠ 变快了** | 必须**同时**验证 `compute` 总耗时下降（W2） |
| **路由一致率高 ≠ 对** | CSA2 本身就是**行为变更**，目标是"**降成本**"不是"保持一致"；一致率只需**记录**（W3） |
| **FP4 体积小了 ≠ 没损失** | 必须验证**往返误差**（W9），且确认运行时不参与 |
| **别用 dmin 单次判定** | `dmin` 噪声比 **0.77** → 用 `dmean` + 多 seed |

---

## 6. 数据来源（可复查）

| 内容 | 来源 |
|---|---|
| CSA2 三模式定义 | **arXiv:2609.19969** §2.3.1 "Cross-Layer KV and Index Reuse" |
| Full/Reindex/Reuse 原文 | 同上（"Full Mode generates global KV and performs indexing..."） |
| Hierarchical Sparse Indexer | 同上 §2.3.2 |
| "candidate pool" / "bounded independently of context length" | 同上 |
| 890 bytes/token、1/4、1/437 | 同上 Abstract + Figure 1(b) |
| SWA Bounded Replay | 同上 §3.2.2 |
| FP4 main KV | 同上 §2.4.4 |
| MISA（indexer 64 头是主要成本） | **arXiv:2605.07363** |
| 项目实测（1160 次打分 / 26.6% / 2.4% 一致率 / T65↔T66=1.00） | 本项目子代理实测报告 |

---

## 7. 一句话总结

> **DeepSeek 的 KV cache 思路本质是"分层复用"：**
> **`Full` 层产出 → `Reindex` 层重排 → `Reuse` 层白嫖**
>
> **而项目现在的做法是"每束都 Full"** —— 这是**架构层缺失**，不是参数问题。
>
> 三条可借鉴：
> 1. **CSA2 三模式** → 治 BIND 26.6% 成本 + T65/T66 冗余
> 2. **Hierarchical Indexer** → **正是"2.4% 陷阱"的正解**（粗筛+精排，而非硬裁剪）
> 3. **FP4** → 只用于存档，不碰运行时
>
> **明确不做**：MISA 式 indexer MoE（项目只有 4 头，收益有限）。
