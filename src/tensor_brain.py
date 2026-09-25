"""
仿生大脑 · 张量化核心 v2
==========================
把原 cog_vec.py 的「逐神经元 Python 循环」重写为「矩阵运算」，
使神经元数量可以上千/上万而仍然够快。

核心变化：
  · 神经元状态 → numpy 矩阵（一次性算一层）
  · 束内消息传递 → 稀疏矩阵乘
  · 跨束路由 → 矩阵 top-k
  · 激活度 → 向量化聚合

设计保持与规范一致：
  · 神经元 = 向量组（dendrite/soma/axon 各为矩阵的一行）
  · 束 = 成员索引 + 动力学矩阵
  · 重叠 = 一个神经元可属多束（用 CSR 风格索引）
  · 激活度分级加载（vram/ram/nvme）

性能目标：10000 神经元 @ 单 tick < 50ms（原版 64 神经元就慢）
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

# ---------------- 常量 ----------------
D_SOMA = 64          # 胞体维度（张量化的受益点：维度可以大）
D_AXON = 64
DECAY = 0.9
ACTIVE_EPS = 1e-4
NORM_CLIP = 1.0
SPREAD_GAIN = 0.25
SPREAD_CAP = 0.3
SPREAD_GAIN_V2 = 0.08      # ★v2：张量版扩散增益（比逐神经元版低，防全网点亮）
PLASTIC_MIN = 0.15         # ★v2：可塑门槛（原0.3在张量摊薄下落不实）
INHIB_GAIN = 0.35          # ★v3：侧向抑制增益（仿生：GABA 抑制性神经元）
INHIB_K = 3                # ★v3：每束内抑制最强的几个
WTA_RATIO = 0.30           # ★v3：赢者通吃——塔尖以下按比例压制
CAP_ACTIVE_RATIO = 0.08    # ★v3：硬上限——活跃神经元占比不得超过此值
BRIDGE_TOPQ = 32           # ★v4：桥接只连最相似的 32 个束（原为全 T×T 矩阵，O(T²) 灾难）
MAX_BRIDGE_SRC = 256       # ★v5：桥接源束上限——只有最热的 256 个束发起跨区通信
                           #      仿生依据：只有强激活区域才需要远程投射，弱激活是局部事件
BRIDGE_GAIN = 0.3
ROUTE_K = 8
THETA_HIGH, THETA_LOW = 0.25, 0.12   # ★v2：按实测束激活度分布校准


class Tier:
    VRAM, RAM, NVME = "vram", "ram", "nvme"


# ==================================================================
# 张量化神经系统
# ==================================================================
class TensorBrain:
    """张量化仿生大脑：所有状态都是矩阵，批量运算。"""

    def __init__(self, n_neurons: int = 2048, n_tracts: int = 128,
                 tract_size: int = 24, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n = n_neurons
        self.rng = rng

        # ---- 神经元状态（矩阵：n × d）----
        self.soma = np.zeros((self.n, D_SOMA), dtype=np.float32)
        self.axon = np.zeros((self.n, D_AXON), dtype=np.float32)
        self.dendrite = rng.normal(0, 0.1, (self.n, D_SOMA)).astype(np.float32)
        self.activity = np.zeros(self.n, dtype=np.float32)      # 激活度向量
        # 主体权重（扁平化）+ 临时可塑量
        self.base = rng.normal(0, 0.02, (self.n, 256)).astype(np.float32)
        self.plasticity = np.zeros((self.n, 256), dtype=np.float32)
        self.plasticity_pending = np.zeros_like(self.plasticity)
        self.has_pending = np.zeros(self.n, dtype=bool)
        self.importance = np.zeros(self.n, dtype=np.float32)
        # 角色标记
        self.role = np.array(["inter"] * self.n, dtype=object)

        # ---- 神经束（重叠：每个神经元可属多束，但要受控）----
        self.n_tracts = n_tracts
        self.tract_size = tract_size
        # ★v2修正：不能纯随机采样（会导致 78% 神经元同时活跃 = 激活爆炸）
        #   改为：把神经元划分到束，再让少量神经元跨束（受控重叠）
        #   每个神经元主属 1 束 + 少量随机跨束 → 总归属数 ≈ n_tracts*tract_size/n ≈ 1.5~3
        self.tract_members = np.zeros((n_tracts, tract_size), dtype=np.int32)
        perm = rng.permutation(self.n)
        for t in range(n_tracts):
            base = perm[(t * tract_size) % self.n: (t * tract_size) % self.n + tract_size]
            if base.size < tract_size:      # 环绕补足
                base = np.concatenate([base, perm[:tract_size - base.size]])
            self.tract_members[t] = base
        # 束动力学（tract_size × tract_size，共享一份以省内存）
        self.tract_dyn = rng.normal(0, 0.1, (tract_size, tract_size)).astype(np.float32)
        # 束级 key/value（用于跨束路由）
        self.tract_key = rng.normal(0, 0.1, (n_tracts, D_SOMA)).astype(np.float32)
        self.tract_val = rng.normal(0, 0.1, (n_tracts, D_SOMA)).astype(np.float32)
        self.tract_act = np.zeros(n_tracts, dtype=np.float32)

        # ---- 神经元 → 束 的反向索引（重叠）----
        # 构建 CSR 风格：neuron_tract_ptr / neuron_tract_idx
        self._build_overlap_index()

        # ---- 分级驻留 ----
        self.tier = np.array([Tier.NVME] * n_tracts, dtype=object)
        self.lib_act = np.zeros(1, dtype=np.float32)   # 简化：一个库

        self.t = 0

    # ---------------- 重叠索引 ----------------
    def _build_overlap_index(self):
        """建立 neuron → tracts 的反向索引（支持重叠）"""
        flat = self.tract_members.flatten()
        tract_ids = np.repeat(np.arange(self.n_tracts), self.tract_size)
        order = np.argsort(flat, kind="stable")
        self._n2t_neuron = flat[order]
        self._n2t_tract = tract_ids[order]
        counts = np.bincount(self._n2t_neuron, minlength=self.n)
        self._n2t_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=self._n2t_ptr[1:])

    def tracts_of(self, nid: int) -> np.ndarray:
        return self._n2t_tract[self._n2t_ptr[nid]:self._n2t_ptr[nid + 1]]

    # ---------------- 激活 ----------------
    def ignite(self, nid: int, amount: float):
        self.activity[nid] = min(1.0, float(self.activity[nid]) + amount)

    def ignite_batch(self, ids: np.ndarray, amounts: np.ndarray):
        np.add.at(self.activity, ids, amounts)
        np.clip(self.activity, 0, 1, out=self.activity)

    # ---------------- 张量化 tick ----------------
    def tick(self, dt: float = 1.0):
        """一次完整思考周期（全部矩阵运算）"""
        self.t += 1
        # 1) 衰减
        self.activity *= DECAY

        # 2) 束内扩散（向量化）
        #    mem_act: (T, S) —— 每束成员的当前激活
        mem_act = self.activity[self.tract_members]              # (T, S)
        #    束内消息强度：每个成员收到 = Σ_j |dyn[i,j]| * act[j] * gain
        #    用 (T,S) 与 (S,S) 相乘 → (T,S)
        msg_in = np.abs(self.tract_dyn).sum(axis=1, keepdims=True)      # (S,1) 每列影响力
        # ★v2修正：增益调低（原0.25会把整网点亮），并做"只有源本身活跃才扩散"的门控
        src_gate = (mem_act > ACTIVE_EPS).astype(np.float32)
        recv = (mem_act * src_gate) @ np.abs(self.tract_dyn).T * SPREAD_GAIN_V2
        np.clip(recv, 0, SPREAD_CAP, out=recv)

        # 3) 束激活度 = 成员激活的 top-k 均值
        k = max(1, self.tract_size // 3)
        srt = np.sort(mem_act, axis=1)[:, -k:]
        self.tract_act = srt.mean(axis=1).astype(np.float32)

        # 4) 把束内扩散量累加到神经元（重叠 → np.add.at 累加）
        spread = np.zeros(self.n, dtype=np.float32)
        np.add.at(spread, self.tract_members.flatten(), recv.flatten())
        np.clip(spread, 0, SPREAD_CAP, out=spread)

        # 5) 跨束桥接：活跃束 → 最相似的前 Q 个束的成员
        #    ★v4：原实现算 T×T 全相似度矩阵，26万神经元时耗时 1304ms/1308ms（99%！
        #    改用"只对活跃束求 top-Q 邻居"，复杂度从 O(T²) 降到 O(H·T + H·Q)
        bridge = np.zeros(self.n, dtype=np.float32)
        hot = np.where(self.tract_act > ACTIVE_EPS)[0]
        # ★v5：活跃束过多时，只取最热的前 MAX_BRIDGE_SRC 个（否则 argpartition 476ms 拖垮全局）
        if hot.size > MAX_BRIDGE_SRC:
            hot = hot[np.argpartition(-self.tract_act[hot], MAX_BRIDGE_SRC - 1)[:MAX_BRIDGE_SRC]]
        if hot.size and hot.size < self.n_tracts * 0.5:
            keyn = self.tract_key / (np.linalg.norm(self.tract_key, axis=1, keepdims=True) + 1e-8)
            w = self.tract_act[hot]                               # (H,)
            # 只算活跃束的 key 对全体 key 的相似度：(H, T) 而非 (T, T)
            sim_h = keyn[hot] @ keyn.T                            # (H, T)
            # 每行取 top-Q
            Q = min(BRIDGE_TOPQ, sim_h.shape[1])
            topq = np.argpartition(-sim_h, Q - 1, axis=1)[:, :Q]  # (H, Q)
            topv = np.take_along_axis(sim_h, topq, axis=1)        # (H, Q)
            # 按目标束聚合贡献（H·Q 个条目，远小于 T²）
            contrib = np.zeros(self.n_tracts, dtype=np.float32)
            flat_t = topq.flatten()
            flat_c = (topv * w[:, None]).flatten() * BRIDGE_GAIN / max(self.tract_size, 1)
            np.add.at(contrib, flat_t, flat_c)
            np.add.at(bridge, self.tract_members.flatten(),
                      np.repeat(contrib, self.tract_size))

        self.activity = np.clip(self.activity + spread + bridge, 0, 1)

        # ★v3：侧向抑制（仿生 GABA）—— 没有它，激活会指数爆炸到全网 72%
        #   每束内：最强的 k 个抑制其余；全局：只保留最强的 8%
        self._apply_inhibition()

        # 6) 计算内核：胞体整合 + 轴突输出（只算活跃神经元）
        active = np.where(self.activity > ACTIVE_EPS)[0]
        if active.size:
            a = self.activity[active][:, None]
            self.soma[active] = np.tanh(self.soma[active] * 0.5 + self.dendrite[active] * a)
            self.axon[active] = np.tanh(self.soma[active])
        return active

    # ---------------- ★v3 侧向抑制 ----------------
    def _apply_inhibition(self):
        """仿生侧向抑制：束内竞争 + 全局活跃上限。

        为什么需要：没有抑制时，1 个神经元经束内扩散能点亮 ~36 个邻居，
        3 步内 72% 的神经元都活跃 → 失去稀疏性 → 大规模下彻底失效。
        人脑用 GABA 抑制性中间神经元做同样的事。
        """
        # (a) 束内竞争：每束里最强的 INHIB_K 个，压制同束其他成员
        mem = self.activity[self.tract_members]                    # (T, S)
        if mem.shape[1] > INHIB_K:
            thr = np.partition(mem, -INHIB_K, axis=1)[:, -INHIB_K][:, None]
            supp = np.where(mem < thr, INHIB_GAIN, 1.0).astype(np.float32)
            new_mem = (mem * supp).flatten()
            flat_idx = self.tract_members.flatten()
            # ★v3.1：用 bincount 求"每神经元被压制的最大次数"不可行（要 max 不是 sum）
            #   改用：重叠束数极少(≈1.5)，直接取最后一次写入即可，
            #   概率上等价（同一神经元在不同束中被压制的程度相近）
            self.activity[flat_idx] = new_mem

        # (b) 全局活跃上限：只保留最强的 CAP_ACTIVE_RATIO
        #   ★v3.1：用 partition 直接取阈值，省掉一次 sum 扫描
        cap = max(int(self.n * CAP_ACTIVE_RATIO), 64)
        if cap < self.n:
            thr = np.partition(self.activity, -cap)[-cap]
            if thr > ACTIVE_EPS:      # 只有确实溢出才裁剪
                self.activity[self.activity < thr] = 0.0

    # ---------------- 在线可塑（向量化）----------------
    def update_plasticity(self, active: np.ndarray):
        """激活即优化：只更新活跃神经元（延迟提交）"""
        if active.size == 0:
            return
        act = self.activity[active]
        sel = active[act >= PLASTIC_MIN]
        if sel.size == 0:
            return
        a = self.activity[sel][:, None]
        # hebb ≈ 外积的扁平化：用 soma 与 dendrite 的逐元素积拼成 256 维
        feat = np.concatenate([
            self.soma[sel] * a, self.dendrite[sel] * a
        ], axis=1)[:, :256]
        if feat.shape[1] < 256:
            feat = np.pad(feat, ((0, 0), (0, 256 - feat.shape[1])))
        delta = 0.02 * feat - 0.88 * self.plasticity[sel]
        cand = self.plasticity[sel] + delta
        # 范数裁剪
        nrm = np.linalg.norm(cand, axis=1, keepdims=True)
        scale = np.minimum(1.0, NORM_CLIP / (nrm + 1e-8))
        cand = cand * scale
        # ★v2修正：在线可塑立即生效（"激活即优化"），pending 仅作快照用途
        self.plasticity[sel] = cand
        self.plasticity_pending[sel] = cand
        self.has_pending[sel] = True
        self.importance[sel] = 0.99 * self.importance[sel] + 0.01 * np.linalg.norm(delta, axis=1)

    def apply_pending(self):
        m = self.has_pending
        if m.any():
            self.plasticity[m] = self.plasticity_pending[m]
            self.has_pending[m] = False

    # ---------------- 睡眠合并 ----------------
    def consolidate(self) -> dict:
        """NREM：把可塑量折入主体（可逆快照）"""
        pl = np.linalg.norm(self.plasticity, axis=1)
        sel = np.where(pl > 1e-6)[0]
        if sel.size == 0:
            return {"merged": 0}
        snapshot = self.base[sel].copy()
        # 正交保护：重要神经元少动
        factor = (1.0 - np.minimum(self.importance[sel], 0.9) * 0.5)[:, None]
        self.base[sel] += self.plasticity[sel] * factor
        self.plasticity[sel] = 0
        self._last_snapshot = (sel, snapshot)
        return {"merged": int(sel.size)}

    def rollback(self) -> bool:
        if not hasattr(self, "_last_snapshot"):
            return False
        sel, snap = self._last_snapshot
        self.base[sel] = snap
        return True

    # ---------------- 调度 ----------------
    def schedule(self) -> np.ndarray:
        """按束激活度分级（滞回）"""
        changed = []
        for i in range(self.n_tracts):
            a = self.tract_act[i]
            cur = self.tier[i]
            if a > THETA_HIGH and cur != Tier.VRAM:
                self.tier[i] = Tier.VRAM; changed.append(i)
            elif a < THETA_LOW and cur == Tier.VRAM:
                self.tier[i] = Tier.RAM; changed.append(i)
        return np.array(changed, dtype=int)

    # ---------------- 状态 ----------------
    def stats(self) -> dict:
        tiers = {}
        for t in (Tier.VRAM, Tier.RAM, Tier.NVME):
            tiers[t] = int((self.tier == t).sum())
        return {
            "neurons": self.n, "tracts": self.n_tracts,
            "active": int((self.activity > ACTIVE_EPS).sum()),
            "ticks": self.t, "tiers": tiers,
            "plastic_norm": float(np.linalg.norm(self.plasticity)),
        }


# ==================================================================
# 自测：性能 + 行为
# ==================================================================
if __name__ == "__main__":
    import time
    print("=" * 60)
    print("  张量化大脑 · 性能与行为自测")
    print("=" * 60)

    for n in (512, 2048, 8192):
        b = TensorBrain(n_neurons=n, n_tracts=max(32, n // 16))
        b.ignite_batch(np.arange(50), np.full(50, 0.5))
        t0 = time.perf_counter()
        for _ in range(20):
            active = b.tick()
            b.update_plasticity(active)
        dt = (time.perf_counter() - t0) / 20
        print(f"  {n:>6} 神经元: {dt*1000:6.1f} ms/tick  | 活跃={active.size:>5} "
              f"| 可塑范数={np.linalg.norm(b.plasticity):.3f}")

    print("\n=== 行为验证（2048 神经元）===")
    b = TensorBrain(2048, 128)
    b.ignite_batch(np.arange(80), np.full(80, 0.6))
    for i in range(15):
        act = b.tick()
        b.update_plasticity(act)
        if i % 5 == 0:
            print(f"  tick{i:2d}: 活跃={act.size:>5} 可塑={np.linalg.norm(b.plasticity):.4f} "
                  f"层级={list(b.stats()['tiers'].values())}")
    r = b.consolidate()
    print(f"  合并: {r}  | 合并后可塑={np.linalg.norm(b.plasticity):.4f}（应≈0）")
    print(f"  回滚: {'✓' if b.rollback() else '✗'}")
    print(f"\n最终: {b.stats()}")
