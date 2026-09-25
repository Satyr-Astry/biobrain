"""真存盘（v3.1，修 R1）——完整神经状态持久化

★背景：原 save()/load() 只存元数据（version/seed/tick/自模型统计/一个 base_sum 标量），
      **完全没存神经状态与记忆库内容** → 重启即失忆（见架构书 §8.2 R1）。

设计：分层持久化（JSON + npz 分离）
  ├── meta.json      元数据（版本/seed/tick/自模型统计/束与库的字符串字段）
  ├── neurons.npz    神经状态（base/plasticity/activity/soma/axon/dendrite/eta）
  ├── tracts.npz     连接（dynamics/key/value/w_fast/w_slow）
  ├── libraries.json 分层状态（tier/size_bytes/hot_score/act_history）
  ├── memory.jsonl   记忆库内容（Experience 逐条）★之前完全丢失
  ├── collective.json 集体层
  └── attn.npz       投影矩阵 W_q/W_k/W_v/W_o

为什么 npz+json 分离而不是一个大 JSON：
  · 体积：浮点矩阵 JSON 化膨胀 3-5 倍
  · 精度：float 经 JSON 序列化会丢精度
  · 增量：记忆库用 jsonl 可追加
  · 兼容：旧 state.json 仍能读
"""
from __future__ import annotations
import json
import os
import glob
from dataclasses import asdict
from typing import Optional, Dict, List, Tuple
import numpy as np

STATE_VERSION = "3.1"
_DIR_SUFFIX = ".brainstate"


def _dir_for(state_path: str) -> str:
    """把单个 state_path 映射为状态目录。

    ★注意：Windows 上 tempfile.gettempdir() 可能返回 8.3 短路径名
    （如 C:\\Users\\ADMINI~1\\AppData\\Local\\Temp），os.makedirs 在这种
    路径下可能静默失败。所以调用方必须用 _verify_dir() 校验。
    """
    base = os.path.splitext(state_path)[0]
    return base + _DIR_SUFFIX


def _verify_dir(d: str) -> bool:
    """校验目录真的创建成功了（防止短路径名/权限导致的静默失败）。"""
    if not os.path.isdir(d):
        return False
    try:
        probe = os.path.join(d, "_w_test")
        with open(probe, "w") as f:
            f.write("1")
        os.remove(probe)
        return True
    except Exception:
        return False


# ============================================================
# 写
# ============================================================
def save_state(brain, state_path: str) -> str:
    """完整存盘。返回实际写入的目录。

    ★v3.2：解决 Windows 8.3 短路径名（如 C:\\Users\\ADMINI~1\\...）导致
    makedirs 静默失败的问题 —— 失败时自动回退到长路径名，并最终校验目录可写。
    """
    d = _dir_for(state_path)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    # ★短路径名回退：用长路径名重试
    if not _verify_dir(d):
        try:
            long_base = os.path.splitext(os.path.abspath(state_path))[0]
            if os.path.isdir(os.path.dirname(long_base)):
                d2 = long_base + _DIR_SUFFIX
                os.makedirs(d2, exist_ok=True)
                if _verify_dir(d2):
                    d = d2
        except Exception:
            pass
    # ★最终校验：目录不可写就报错（不再静默返回路径）
    if not _verify_dir(d):
        raise IOError(
            f"状态目录创建失败或不可写: {d} "
            f"(state_path={state_path!r}, 可能是 Windows 8.3 短路径名或权限问题)"
        )

    ns = brain.ns
    meta = {
        "version": STATE_VERSION,
        "seed": getattr(brain, "seed", 0),
        "tick_count": brain.tick_count,
        "ns_t": float(getattr(ns, "t", 0.0)),
        "attn_mode": getattr(ns, "attn_mode", "cos"),
        # 自模型
        "skills": getattr(brain.self_model, "skills", {}),
        "goals": getattr(brain.self_model, "goals", []),
        "experiences": getattr(brain.self_model, "total_experiences", 0),
        "successes": getattr(brain.self_model, "total_successes", 0),
        # 结构（id 列表 + 角色）
        "neuron_ids": sorted(ns.neurons.keys()),
        "neuron_roles": {str(i): ns.neurons[i].role for i in ns.neurons},
        "tract_ids": sorted(ns.tracts.keys()),
        "tract_members": {str(t): ns.tracts[t].members for t in ns.tracts},
        "tract_tier": {str(t): ns.tracts[t].tier for t in ns.tracts},
        "tract_row_budget": {str(t): ns.tracts[t].row_budget for t in ns.tracts},
        "group_ids": sorted(ns.groups.keys()),
        "group_tracts": {str(g): ns.groups[g].tract_ids for g in ns.groups},
        "group_function": {str(g): ns.groups[g].function for g in ns.groups},
        "library_ids": sorted(ns.libraries.keys()),
        "library_groups": {str(l): ns.libraries[l].group_ids for l in ns.libraries},
        "last_sim_time": float(getattr(brain, "_last_sim_time", 0.0)),
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # ---- 神经状态（npz）----
    # ★注意：activity/importance 是**标量**，必须保持标量（否则 load 后变 array 破坏类型）
    nids = meta["neuron_ids"]
    neu = {}
    for field in ["base", "plasticity", "soma", "axon", "dendrite", "eta"]:
        arrs = []
        for i in nids:
            n = ns.neurons[i]
            v = getattr(n, field, None)
            if v is None:
                v = np.zeros(1)
            arrs.append(np.asarray(v, dtype=float).ravel())
        # 统一长度（不同字段维度可能不同）
        L = max(len(a) for a in arrs) if arrs else 0
        mat = np.zeros((len(arrs), L), dtype=np.float32)
        for k, a in enumerate(arrs):
            mat[k, :len(a)] = a
        neu[field] = mat
    # ★标量字段单独存为 1 维数组（load 时取 [k] 得到标量）
    neu["activity"] = np.array(
        [float(getattr(ns.neurons[i], "activity", 0.0)) for i in nids], dtype=np.float32)
    neu["importance"] = np.array(
        [float(getattr(ns.neurons[i], "importance", 0.0)) for i in nids], dtype=np.float32)
    np.savez_compressed(os.path.join(d, "neurons.npz"), **neu)

    # ---- 连接（npz）----
    tids = meta["tract_ids"]
    con = {}
    for field in ["dynamics", "key", "value", "w_fast", "w_slow"]:
        arrs = []
        for t in tids:
            v = getattr(ns.tracts[t], field, None)
            if v is None:
                v = np.zeros(1)
            arrs.append(np.asarray(v, dtype=float).ravel())
        L = max(len(a) for a in arrs) if arrs else 0
        mat = np.zeros((len(arrs), L), dtype=np.float32)
        for k, a in enumerate(arrs):
            mat[k, :len(a)] = a
        con[field] = mat
    np.savez_compressed(os.path.join(d, "tracts.npz"), **con)

    # ---- 分层状态（json）----
    libs = {}
    for l in meta["library_ids"]:
        lib = ns.libraries[l]
        libs[str(l)] = {
            "tier": lib.tier.value,
            "size_bytes": int(lib.size_bytes),
            "hot_score": float(lib.hot_score),
            "act_history": list(lib.act_history),
            "activation": float(lib.activation),
            "tier_changes": int(getattr(lib, "tier_changes", 0)),
            "prefetched": bool(getattr(lib, "prefetched", False)),
            "last_change": float(lib.last_change),
            "group_ids": lib.group_ids,
        }
    with open(os.path.join(d, "libraries.json"), "w", encoding="utf-8") as f:
        json.dump(libs, f, ensure_ascii=False)

    # ---- 记忆库（jsonl）★之前完全丢失 ----
    with open(os.path.join(d, "memory.jsonl"), "w", encoding="utf-8") as f:
        for e in _iter_experiences(brain):
            rec = {
                "id": e.id, "thought_trace": list(e.thought_trace),
                "ignitions": list(e.ignitions), "emissions": list(e.emissions),
                "outcome": float(e.outcome), "surprise": float(e.surprise),
                "event_time": float(e.event_time), "record_time": float(e.record_time),
                "source": e.source, "supersedes": e.supersedes,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 集体层 ----
    coll = {str(k): {"payload": v.payload, "verified_by": v.verified_by,
                     "contributors": v.contributors}
            for k, v in getattr(brain.collective, "entries", {}).items()}
    with open(os.path.join(d, "collective.json"), "w", encoding="utf-8") as f:
        json.dump(coll, f, ensure_ascii=False)

    # ---- 注意力投影（npz）----
    attn = {}
    for nm in ["W_q", "W_k", "W_v", "W_o"]:
        v = getattr(ns, nm, None)
        if v is not None:
            attn[nm] = np.asarray(v, dtype=np.float32)
    if attn:
        np.savez_compressed(os.path.join(d, "attn.npz"), **attn)

    return d


def _iter_experiences(brain) -> List:
    """取出记忆库里的全部经验（兼容不同内部结构）。"""
    buf = getattr(brain, "buffer", None)
    if buf is None:
        return []
    out = []
    for attr in ["items", "entries", "_items", "buffer", "_buffer"]:
        v = getattr(buf, attr, None)
        if isinstance(v, list):
            out.extend(v)
    if not out:
        # 有些实现用 dict 存
        for attr in ["by_id", "_by_id"]:
            v = getattr(buf, attr, None)
            if isinstance(v, dict):
                out.extend(v.values())
    # 去重（按 id）
    seen, uniq = set(), []
    for e in out:
        if getattr(e, "id", None) in seen:
            continue
        seen.add(getattr(e, "id", None))
        uniq.append(e)
    return uniq


# ============================================================
# 读
# ============================================================
def load_state(brain, state_path: str) -> Tuple[bool, str]:
    """完整读盘。返回 (成功, 说明)。支持旧格式向后兼容。"""
    d = _dir_for(state_path)
    # ---- 旧格式兼容 ----
    if not os.path.isdir(d):
        if os.path.isfile(state_path):
            return _load_legacy(brain, state_path)
        return False, "no state found"

    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"meta read failed: {e}"

    ns = brain.ns
    note = []

    # ---- 神经状态 ----
    p = os.path.join(d, "neurons.npz")
    if os.path.isfile(p):
        z = np.load(p)
        nids = meta.get("neuron_ids", [])
        VEC_FIELDS = ["base", "plasticity", "soma", "axon", "dendrite", "eta"]
        for k, i in enumerate(nids):
            n = ns.neurons.get(i)
            if n is None:
                continue
            for field in VEC_FIELDS:
                if field not in z:
                    continue
                v = z[field][k]
                cur = getattr(n, field, None)
                if cur is None:
                    setattr(n, field, v.astype(float))
                else:
                    cur = np.asarray(cur, dtype=float).ravel()
                    if cur.shape[0] >= v.shape[0]:
                        cur[:v.shape[0]] = v
                        setattr(n, field, cur)
                    else:
                        setattr(n, field, v[:cur.shape[0]].astype(float))
            # ★标量字段保持标量
            if "activity" in z:
                n.activity = float(z["activity"][k])
            if "importance" in z:
                n.importance = float(z["importance"][k])
        note.append(f"neurons={len(nids)}")

    # ---- 连接 ----
    p = os.path.join(d, "tracts.npz")
    if os.path.isfile(p):
        z = np.load(p)
        tids = meta.get("tract_ids", [])
        tier_map = meta.get("tract_tier", {})
        rb_map = meta.get("tract_row_budget", {})
        for k, t in enumerate(tids):
            tr = ns.tracts.get(t)
            if tr is None:
                continue
            n_mem = len(tr.members)
            for field in ["dynamics", "key", "value"]:
                if field not in z:
                    continue
                flat = np.asarray(z[field][k], dtype=float).ravel()
                cur = np.asarray(getattr(tr, field), dtype=float)
                want_shape = cur.shape           # ★保持原 shape（dynamics 是 n×n）
                need = int(np.prod(want_shape))
                if flat.shape[0] >= need:
                    setattr(tr, field, flat[:need].reshape(want_shape))
                else:
                    padded = np.zeros(need)
                    padded[:flat.shape[0]] = flat
                    setattr(tr, field, padded.reshape(want_shape))
            # w_fast/w_slow 是 n×n 增量（可能为 None）
            for field in ["w_fast", "w_slow"]:
                if field not in z:
                    continue
                flat = np.asarray(z[field][k], dtype=float).ravel()
                if float(np.abs(flat).max()) > 0.0:
                    setattr(tr, field, flat[: n_mem * n_mem].reshape(n_mem, n_mem))
            tr.tier = tier_map.get(str(t), "inter")
            rb = rb_map.get(str(t))
            if rb is not None:
                tr.row_budget = float(rb)
        note.append(f"tracts={len(tids)}")

    # ---- 分层 ----
    p = os.path.join(d, "libraries.json")
    if os.path.isfile(p):
        from cog_vec import Tier
        with open(p, encoding="utf-8") as f:
            libs = json.load(f)
        for ls, v in libs.items():
            lib = ns.libraries.get(int(ls))
            if lib is None:
                continue
            try:
                lib.tier = Tier(v["tier"])
            except Exception:
                pass
            lib.size_bytes = int(v.get("size_bytes", 0))
            lib.hot_score = float(v.get("hot_score", 0.0))
            lib.act_history = list(v.get("act_history", []))
            lib.activation = float(v.get("activation", 0.0))
            lib.tier_changes = int(v.get("tier_changes", 0))
            lib.prefetched = bool(v.get("prefetched", False))
            lib.last_change = float(v.get("last_change", 0.0))
        note.append(f"libraries={len(libs)}")

    # ---- 记忆库 ----
    p = os.path.join(d, "memory.jsonl")
    if os.path.isfile(p):
        from self_training import Experience
        cnt = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                e = Experience(
                    id=rec["id"], thought_trace=list(rec.get("thought_trace", [])),
                    ignitions=list(rec.get("ignitions", [])),
                    emissions=list(rec.get("emissions", [])),
                    outcome=float(rec.get("outcome", 0.0)),
                    surprise=float(rec.get("surprise", 0.0)),
                    event_time=float(rec.get("event_time", 0.0)),
                    record_time=float(rec.get("record_time", 0.0)),
                    source=rec.get("source", "outcome"),
                    supersedes=rec.get("supersedes"),
                )
                _restore_experience(brain, e)
                cnt += 1
        note.append(f"memory={cnt}")

    # ---- 集体层 ----
    p = os.path.join(d, "collective.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            coll = json.load(f)
        for _, v in coll.items():
            try:
                brain.collective.submit(v["payload"], source=v["verified_by"],
                                        instance_id="restored")
            except Exception:
                pass

    # ---- 注意力投影 ----
    p = os.path.join(d, "attn.npz")
    if os.path.isfile(p):
        z = np.load(p)
        for nm in ["W_q", "W_k", "W_v", "W_o"]:
            if nm in z:
                setattr(ns, nm, z[nm].astype(float))
        note.append("attn=yes")

    # ---- 元数据 ----
    from cog_vec import Tier  # noqa
    brain.tick_count = meta.get("tick_count", brain.tick_count)
    ns.t = float(meta.get("ns_t", getattr(ns, "t", 0.0)))
    ns.attn_mode = meta.get("attn_mode", getattr(ns, "attn_mode", "cos"))
    if hasattr(brain, "_last_sim_time"):
        brain._last_sim_time = float(meta.get("last_sim_time", 0.0))
    for k in ["skills", "goals", "experiences", "successes"]:
        if k in meta:
            attr = {"experiences": "total_experiences", "successes": "total_successes"}.get(k, k)
            setattr(brain.self_model, attr, meta[k])

    return True, ", ".join(note)


def _restore_experience(brain, e):
    """把一条经验塞回记忆库（兼容不同实现）。"""
    buf = getattr(brain, "buffer", None)
    if buf is None:
        return
    for attr in ["items", "entries", "_items", "buffer", "_buffer"]:
        v = getattr(buf, attr, None)
        if isinstance(v, list):
            v.append(e)
            return
    for attr in ["by_id", "_by_id"]:
        v = getattr(buf, attr, None)
        if isinstance(v, dict):
            v[getattr(e, "id", 0)] = e
            return
    # 兜底：用 store()
    try:
        buf.store(e)
    except Exception:
        pass


def _load_legacy(brain, state_path: str) -> Tuple[bool, str]:
    """读旧格式（单个 json，只有元数据）。"""
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"legacy read failed: {e}"
    brain.self_model.skills = data.get("skills", {})
    brain.self_model.goals = data.get("goals", [])
    brain.self_model.total_experiences = data.get("experiences", 0)
    brain.self_model.total_successes = data.get("successes", 0)
    for k, v in data.get("collective", {}).items():
        try:
            brain.collective.submit(v["payload"], source=v["verified_by"],
                                    instance_id="restored")
        except Exception:
            pass
    return True, "legacy (metadata only)"
