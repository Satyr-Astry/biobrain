"""
仿生 AI · 回归测试套件
========================
把《工程规范_v7.2.md》中 9 个"代码暴露的漏洞"变成回归测试。
每个测试对应一个曾经的真实缺陷 —— 若重现则失败。

运行:  python -m pytest test_cog_vec.py -v
      （在 code/ 目录下执行）

对应关系：
  C1  束内扩散缺失             → test_c1_spread_exists
  C2  跨束传播断链             → test_c2_cross_tract_bridge
  C3  PLASTIC_THRESHOLD 过高   → test_c3_plastic_threshold_calibrated
  C4  扩散参数缺失             → test_c4_param_exists
  C5  信号传播需要时间         → test_c5_signal_needs_time
  C6  资格痕迹无界累积         → test_c6_trace_normalized
  C7  慢尺度不学习             → test_c7_slow_scale_learns
  C8  能量读取容错             → test_c8_energy_fallback
  C9  三因子方向正确           → test_c9_three_factor_direction
  + 机制不变量测试（重叠/因果/自评拦截/护栏/诚实输出）
"""
import numpy as np
import pytest
import time

import cog_vec as bb
from cog_vec import (NervousSystem, SensoryPort, MotorPort,
                       ACTIVE_EPS, NORM_CLIP)
from self_training import (Experience, ExperienceBuffer, Consolidator,
                           ConvergenceMeter, FeedbackLoop, emit_decision,
                           CONVERGE_HIGH, CONVERGE_LOW, PATIENCE)
from advanced import (MultiScaleWeight, EnergyMeter, EnergyScheduler,
                      GroundingModule, RobustnessGuard, DegradeLevel)


# ============================================================
# 公共夹具
# ============================================================
def make_brain(seed=42):
    """建一个连通的大脑（含重叠+桥接束）"""
    ns = NervousSystem(seed=seed)
    visual = [ns.add_neuron("sensory") for _ in range(4)]
    symbol = [ns.add_neuron("sensory") for _ in range(4)]
    inter  = [ns.add_neuron("inter") for _ in range(8)]
    motor  = [ns.add_neuron("motor") for _ in range(3)]
    self_s = [ns.add_neuron("sensory") for _ in range(2)]
    t1 = ns.add_tract(visual + inter[:2])
    t2 = ns.add_tract(symbol + inter[:2])       # 与 t1 共享 → 重叠
    t3 = ns.add_tract(inter[2:6] + motor)
    t4 = ns.add_tract(inter[:2] + inter[2:6])   # 桥接束
    g1 = ns.add_group([t1, t2], "percept")
    g2 = ns.add_group([t3, t4], "action")
    ns.add_library([g1]); ns.add_library([g2])
    return ns, {"visual": visual, "symbol": symbol, "inter": inter,
                "motor": motor, "self_s": self_s}


def run_ticks(ns, active, n=15):
    for _ in range(n):
        ns.apply_pending(active)
        active = ns.tick_activation(active)
        for tid in ns.tracts:
            ns.tracts[tid].activation = ns.tract_activation(tid)
        ns.compute(active)
        ns.update_plasticity(active, {})
        ns.tick_scheduler()
        ns.t += 1
    return active


# ============================================================
# C1: 束内扩散必须存在（否则神经元全熄火）
# ============================================================
def test_c1_spread_exists():
    """C1: 激活必须能沿束扩散 —— 无扩散则神经元只会衰减到 0"""
    ns, ids = make_brain()
    ns.ignite(ids["visual"][0], 0.8)          # 只点亮一个神经元
    active = {ids["visual"][0]}
    # 跑几步，看邻居是否被点亮
    for _ in range(3):
        active = ns.tick_activation(active)
    spread_to = [n for n in ids["visual"][1:] if ns.neurons[n].activity > ACTIVE_EPS]
    assert spread_to, "C1 回归：激活没有扩散到束内其他神经元（全熄火 bug 重现）"


# ============================================================
# C2: 跨束传播必须存在（否则激活传不到远处组织）
# ============================================================
def test_c2_cross_tract_bridge():
    """C2: 激活必须能跨束传播到运动区 —— 否则感觉区与运动区断链"""
    ns, ids = make_brain()
    vp = SensoryPort("vision", ids["visual"], 8)
    sp = SensoryPort("symbol", ids["symbol"], 8)
    r1 = vp.ignite(np.array([1., .5, .3, 0, 0, 0, 0, 0]), ns)
    r2 = sp.ignite(np.array([.8, .6, 0, 0, 0, 0, 0, 0]), ns)
    active = set(r1["ignited"]) | set(r2["ignited"])
    run_ticks(ns, active, n=15)
    motor_max = max(ns.neurons[m].activity for m in ids["motor"])
    assert motor_max > 0.3, f"C2 回归：激活未能跨束传到运动区 (max={motor_max:.3f})"


# ============================================================
# C3: PLASTIC_THRESHOLD 必须经实测校准（不能过高）
# ============================================================
def test_c3_plastic_threshold_calibrated():
    """C3: 可塑门槛必须低于实际可达激活峰值，否则在线学习永不触发

    ★v2 修正：改为多次种子测试，避免单一随机初始化导致 flaky。
    """
    # ★v3 修正: 用统计断言（多seed多数成功），避免单seed随机性造成flaky
    ok_peak, ok_plastic = 0, 0
    peaks = []
    for seed in range(10):
        ns, ids = make_brain(seed=seed)
        vp = SensoryPort("vision", ids["visual"], 8)
        r = vp.ignite(np.array([1., .5, .3, 0, 0, 0, 0, 0]), ns)
        active = set(r["ignited"])
        run_ticks(ns, active, n=15)
        peak = max(n.activity for n in ns.neurons.values())
        total = sum(float(np.linalg.norm(n.plasticity)) for n in ns.neurons.values())
        peaks.append(peak)
        if bb.PLASTIC_THRESHOLD < peak:
            ok_peak += 1
        if total > 0:
            ok_plastic += 1
    # 门槛必须绝大多数情况下可被超过（≥80%），否则门槛过高
    assert ok_peak >= 8, \
        f"C3 回归：仅 {ok_peak}/10 seed 的峰值超过门槛({bb.PLASTIC_THRESHOLD})，peaks={[round(x,2) for x in peaks]}"
    assert ok_plastic >= 8, f"C3 回归：仅 {ok_plastic}/10 seed 产生可塑量"


# ============================================================
# C4: 扩散相关参数必须存在
# ============================================================
def test_c4_param_exists():
    """C4: SPREAD_* / BRIDGE_* 参数必须定义"""
    for name in ["SPREAD_GAIN", "SPREAD_CAP", "BRIDGE_GAIN", "BRIDGE_SIM"]:
        assert hasattr(bb, name), f"C4 回归：缺少参数 {name}"


# ============================================================
# C5: 信号传播需要时间（仿生特性，不是缺陷）
# ============================================================
def test_c5_signal_needs_time():
    """C5: 长时间运行后信号应比短时间更强（传播需要时间）"""
    ns1, ids1 = make_brain()
    vp1 = SensoryPort("vision", ids1["visual"], 8)
    sp1 = SensoryPort("symbol", ids1["symbol"], 8)
    a1 = set(vp1.ignite(np.array([1.,.5,.3,0,0,0,0,0]), ns1)["ignited"]) | \
         set(sp1.ignite(np.array([.8,.6,0,0,0,0,0,0]), ns1)["ignited"])
    run_ticks(ns1, a1, n=5)
    short = max(ns1.neurons[m].activity for m in ids1["motor"])

    ns2, ids2 = make_brain()
    vp2 = SensoryPort("vision", ids2["visual"], 8)
    sp2 = SensoryPort("symbol", ids2["symbol"], 8)
    a2 = set(vp2.ignite(np.array([1.,.5,.3,0,0,0,0,0]), ns2)["ignited"]) | \
         set(sp2.ignite(np.array([.8,.6,0,0,0,0,0,0]), ns2)["ignited"])
    run_ticks(ns2, a2, n=15)
    long = max(ns2.neurons[m].activity for m in ids2["motor"])
    assert long > short, f"C5 回归：长跑({long:.3f}) 未强于短跑({short:.3f})"


# ============================================================
# C6: 资格痕迹必须归一化（否则更新发散）
# ============================================================
def test_c6_trace_normalized():
    """C6: 连续更新后子权重范数不得爆表（痕迹归一化生效）"""
    rng = np.random.default_rng(1)
    pre = rng.normal(0, .5, 4); post = rng.normal(0, .5, 4)
    w = MultiScaleWeight(shape=(4, 4))
    for _ in range(200):                       # 大量更新（会放大发散）
        w.update(pre, post, reward=1.0, expected=0.0)
    norms = w.scale_norms()
    assert all(n <= NORM_CLIP * 1.01 for n in norms), \
        f"C6 回归：痕迹/权重发散 (norms={[round(n,2) for n in norms]})"


# ============================================================
# C7: 慢尺度必须能学习（不能永远 0）
# ============================================================
def test_c7_slow_scale_learns():
    """C7: 慢尺度权重必须增长（否则长期知识永远学不到）"""
    rng = np.random.default_rng(2)
    pre = rng.normal(0, .5, 4); post = rng.normal(0, .5, 4)
    w = MultiScaleWeight(shape=(4, 4))
    for _ in range(20):
        w.update(pre, post, reward=1.0, expected=0.0)
    n_fast, n_mid, n_slow = w.scale_norms()
    assert n_slow > 1e-3, f"C7 回归：慢尺度几乎不学习 (slow={n_slow:.5f})"
    assert n_mid > n_slow * 0.5, "C7 回归：中/慢尺度比例失衡"


# ============================================================
# C8: 能量读取必须容错
# ============================================================
def test_c8_energy_fallback():
    """C8: nvidia-smi 不可用时必须降级为估算，不能抛异常"""
    em = EnergyMeter()
    em.have_nvml = False                         # 强制模拟无 GPU
    p = em.read_power()
    assert p > 0, "C8 回归：无 GPU 时功率读取未降级"
    rec = em.measure(lambda: sum(range(1000)), tokens=10, label="t")
    assert rec["joules"] >= 0 and "j_per_token" in rec


# ============================================================
# C9: 三因子可塑性方向正确
# ============================================================
def test_c9_three_factor_direction():
    """C9: 正 RPE 强化、负 RPE 弱化（方向相反）"""
    rng = np.random.default_rng(0)
    pre = rng.normal(0, .5, 4); post = rng.normal(0, .5, 4)
    w_pos = MultiScaleWeight(shape=(4, 4))
    w_neg = MultiScaleWeight(shape=(4, 4))
    for _ in range(5):
        w_pos.update(pre, post, reward=1.0, expected=0.2)   # RPE=+0.8
        w_neg.update(pre, post, reward=0.0, expected=0.8)   # RPE=-0.8
    s_pos = float(np.sum(w_pos.subweights[0]))
    s_neg = float(np.sum(w_neg.subweights[0]))
    assert s_pos * s_neg < 0, \
        f"C9 回归：正负 RPE 未产生相反方向 (pos={s_pos:.4f}, neg={s_neg:.4f})"


# ============================================================
# 机制不变量测试
# ============================================================
def test_overlap_membership():
    """不变量：单个神经元可属于多个束"""
    ns, ids = make_brain()
    assert len(ns.neurons[ids["inter"][0]].tract_ids) >= 2, "重叠归属丢失"


def test_latency_causality():
    """不变量：在线可塑必须经 pending 缓存（只影响未来）

    ★v2 修正：直接构造活跃神经元验证机制，不依赖随机信号强度。
    """
    ns, ids = make_brain()
    # 强制点亮一组神经元到可塑阈值以上（确定性）
    active = set(ids["inter"])
    for nid in active:
        ns.neurons[nid].activity = 0.9
    # 调用可塑更新
    ns.update_plasticity(active, {})
    # 更新后必须有 pending，且 plasticity 本体尚未改变（延迟提交）
    changed_now = [nid for nid in active if np.linalg.norm(ns.neurons[nid].plasticity) > 1e-9]
    has_pending = any(ns.neurons[nid].plasticity_pending is not None for nid in active)
    assert has_pending, "因果性破坏：更新未走 pending 缓存"
    assert not changed_now, "因果性破坏：plasticity 被立即修改（应延迟提交）"
    # 提交后才生效
    ns.apply_pending(active)
    after = [nid for nid in active if np.linalg.norm(ns.neurons[nid].plasticity) > 1e-9]
    assert after, "提交后 plasticity 未生效"


def test_unreliable_rejected():
    """不变量：自评信号不得进入经验缓冲（防自噬）"""
    buf = ExperienceBuffer()
    e_self = Experience(id=0, thought_trace=[], ignitions=[1], emissions=[2],
                        outcome=1.0, surprise=1.0, event_time=1., record_time=1.,
                        source="self_eval")
    assert buf.store(e_self) is False, "自噬防护失效：自评经验入库"
    e_oracle = Experience(id=1, thought_trace=[], ignitions=[3], emissions=[4],
                          outcome=1.0, surprise=1.0, event_time=1., record_time=1.,
                          source="oracle")
    assert buf.store(e_oracle) is True, "可靠经验被误拒"


def test_consolidation_blocked_when_sliced():
    """不变量：分片挂起时不可合并（§4.5）"""
    ns, ids = make_brain()
    buf = ExperienceBuffer()
    c = Consolidator(ns, buf)
    r = c.sleep(has_running_task=False, has_pending_slice=True)
    assert r.get("skipped") is True, "分片挂起时仍执行了合并"


def test_rem_not_merged():
    """不变量：REM 产物只入候选库，不得直接合并入 base"""
    ns, ids = make_brain()
    buf = ExperienceBuffer()
    c = Consolidator(ns, buf)
    c.rem_phase()
    assert len(c.candidate_store) > 0, "REM 未产生候选"
    # base 不应因 REM 改变（简化验证：候选存在且合并记录为空）
    assert len(c.checkpoints) == 0, "REM 直接改动了 base（违反 MUST）"


def test_feedback_guard():
    """不变量：无支撑的回灌必须被护栏拦截"""
    ns, ids = make_brain()
    fb = FeedbackLoop(ns, ids["self_s"])
    ok_bad = fb.feed_back({"strength": 0.9}, supported=False)
    assert ok_bad is False and fb.blocked >= 1, "回灌护栏失效"


def test_honest_output():
    """不变量：低收敛+超时必须诚实说"不确定" """
    d = emit_decision(conv=0.1, has_input=True, elapsed=PATIENCE + 100)
    assert d.get("content") == "我不确定/需要更多信息", "诚实输出机制失效"
    d2 = emit_decision(conv=0.95, has_input=True, elapsed=1)
    assert d2["action"] == "speak", "高收敛时应输出"


def test_robustness_degrade():
    """不变量：异常必须触发降级，且降级不可逆升级"""
    g = RobustnessGuard()
    g.check_thrash(100.0)                      # 触发 L1
    assert g.level >= DegradeLevel.L1_RESOURCE
    g.check_plasticity([99.0])                 # 触发 L3
    assert g.level >= DegradeLevel.L3_PROTECT


def test_double_platform_no_grounding():
    """不变量：未接地的符号必须被判为 False"""
    ns = NervousSystem(seed=5)
    syms = [ns.add_neuron() for _ in range(2)]
    gm = GroundingModule(ns)
    r = gm.is_grounded(syms, lambda: None, lambda: None)
    assert r["grounded"] is False, "未接地符号被误判为已接地"


def test_energy_scheduler_offpeak():
    """不变量：错峰调度跨午夜正确"""
    s = EnergyScheduler(off_peak=(23, 7))
    assert s.should_do_heavy(3) is True
    assert s.should_do_heavy(12) is False
    assert s.should_do_heavy(23) is True


# ============================================================
# M10-M12 P2 机制测试
# ============================================================
from p2_mechanisms import Neurogenesis, CollectiveLayer, SelfModel


def test_m10_neurogenesis_trigger():
    """M10: 长期无法拟合必须触发神经发生"""
    ns = NervousSystem(seed=42)
    n0 = [ns.add_neuron() for _ in range(4)]
    ns.add_tract(n0)
    ng = Neurogenesis(ns)
    novel = np.random.default_rng(1).normal(0, 1, 16)
    assert not ng.should_grow(), "初始不应触发"
    for _ in range(3):
        ng.try_fit(novel)
    assert ng.should_grow(), "M10 回归：连续失败未触发神经发生"


def test_m10_newborn_isolated():
    """M10: 新生神经元必须先隔离观察"""
    ns = NervousSystem(seed=42)
    n0 = [ns.add_neuron() for _ in range(4)]
    ns.add_tract(n0)
    ng = Neurogenesis(ns)
    nb = ng.grow_neuron()
    assert nb not in ng.active_neurons(), "M10 回归：新生神经元未隔离"


def test_m10_promote_and_reject():
    """M10: 成功率高的并入、低的淘汰"""
    ns = NervousSystem(seed=42)
    n0 = [ns.add_neuron() for _ in range(4)]
    ns.add_tract(n0)
    ng = Neurogenesis(ns)
    g = ng.grow_neuron(); ng.grow_tract(n0[:2] + [g])
    for s in [True, True, True, False]:
        ng.observe(g, s)
    b = ng.grow_neuron(); ng.grow_tract(n0[:2] + [b])
    for _ in range(4):
        ng.observe(b, False)
    res = ng.evaluate_newborns()
    assert g in res["promoted"], "M10 回归：优质新生未并入"
    assert b in res["rejected"], "M10 回归：劣质新生未淘汰"
    assert b not in ns.neurons, "M10 回归：淘汰后神经元仍存在"


def test_m11_reject_unverified():
    """M11: 未验证内容必须拒收（防同源污染）"""
    cl = CollectiveLayer()
    assert cl.submit({"p": 1}, source="self_eval", instance_id="i1") is None, \
        "M11 回归：自评内容入库"
    assert cl.submit({"p": 1}, source="oracle", instance_id="i1") is not None, \
        "M11 回归：验证内容被误拒"


def test_m11_multi_instance_trust():
    """M11: 多实例独立验证应提升可信度"""
    cl = CollectiveLayer()
    cl.submit({"p": 2}, source="oracle", instance_id="i1")
    cl.submit({"p": 2}, source="outcome", instance_id="i2")
    strict = cl.inherit(min_contributors=2)
    assert len(strict) == 1, "M11 回归：多实例验证未提升可信度"


def test_m12_dont_know():
    """M12: 能力不足时必须抑制过度自信（防幻觉）"""
    sm = SelfModel()
    for _ in range(8):
        sm.record_outcome("code", success=False)
    assert sm.should_say_dont_know("code", 0.95), \
        "M12 回归：能力不足却未抑制过度自信"
    cal = sm.calibrated_confidence("code", 0.95)
    assert cal < 0.5, f"M12 回归：过度自信未被校准 (cal={cal:.2f})"


def test_m12_competence_preserved():
    """M12: 擅长的领域自信不应被无理打压"""
    sm = SelfModel()
    for _ in range(10):
        sm.record_outcome("code", success=True)
    cal = sm.calibrated_confidence("code", 0.85)
    assert cal > 0.7, f"M12 回归：擅长的自信被误压 (cal={cal:.2f})"


def test_m12_unknown_low_competence():
    """M12: 未知任务应给低能力分"""
    sm = SelfModel()
    assert sm.competence("从未见过") < 0.4, "M12 回归：未知任务能力分过高"


# ============================================================
# 服务层测试（M-Server）
# ============================================================
from server import CogVec


def test_server_think():
    """服务层：思考应返回结构化结果"""
    b = CogVec(seed=1)
    for _ in range(5):
        b.self_model.record_outcome("chat", success=True)
    r = b.think("你好", task_type="chat")
    assert "convergence" in r and "output" in r
    assert r["convergence"] > 0, "服务层回归：收敛度恒为0（冷启动bug）"


def test_server_dont_know():
    """服务层：不擅长的领域必须诚实说不知道"""
    b = CogVec(seed=1)
    for _ in range(8):
        b.self_model.record_outcome("quantum", success=False)
    r = b.think("解释量子力学", task_type="quantum")
    assert r["should_say_dont_know"] is True, "服务层回归：未诚实说不知道"
    assert r["output"] is None, "服务层回归：不该硬编却输出了"


def test_server_sleep_and_state():
    """服务层：睡眠 + 状态查询可用"""
    b = CogVec(seed=1)
    b.think("测试")
    s = b.sleep()
    assert "sleep" in s
    st = b.state()
    assert st["neurons"] > 0 and "self_model" in st


def test_server_save_load():
    """服务层：存盘/加载往返"""
    import os, tempfile, shutil, glob
    path = os.path.join(tempfile.gettempdir(), "bio_test_state.json")
    b = CogVec(seed=1, state_path=path)
    for _ in range(5):
        b.self_model.record_outcome("coding", success=True)
    b.save()
    b2 = CogVec(seed=1, state_path=path)
    ok = b2.load()
    assert ok, "服务层回归：加载失败"
    assert b2.self_model.competence("coding") > 0.6, "服务层回归：技能未恢复"
    # ★v3.1：存盘现在是目录，清理时兼容两种形态
    _cleanup_state(path)


def _cleanup_state(path):
    """清理状态（兼容"单文件"与"目录"两种形态）。"""
    import os, shutil, glob
    if os.path.isfile(path):
        os.remove(path)
    for d in glob.glob(os.path.splitext(path)[0] + ".brainstate"):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


# ============================================================
# C17: 真存盘往返（R1 验收）—— 神经状态与记忆库必须完整恢复
# ============================================================
def test_c17_full_state_roundtrip():
    """C17: save/load 必须完整往返神经状态与记忆库（修 R1）。

    ★背景：原 save() 只存元数据（含一个 base_sum 标量），
    完全没存 base/plasticity/soma/axon/dynamics/记忆库内容
    → 重启即失忆（架构书 §8.2 R1）。
    """
    import os, tempfile, shutil, glob
    import numpy as np
    path = os.path.join(tempfile.gettempdir(), "bio_c17_state.json")
    _cleanup_state(path)

    b = CogVec(seed=1, state_path=path)
    # 跑起来，产生真实状态
    for t in ["猫", "狗是哺乳动物", "注意力机制"]:
        b.think(t)
    # 快照（save 前）
    snap = {}
    for i, n in b.ns.neurons.items():
        snap[i] = dict(base=np.array(n.base, copy=True),
                       soma=np.array(n.soma, copy=True),
                       axon=np.array(n.axon, copy=True),
                       activity=float(n.activity))
    tract_snap = {t: np.array(tr.dynamics, copy=True) for t, tr in b.ns.tracts.items()}
    mem_before = len([e for e in _all_experiences(b)])

    b.save()

    # 新大脑加载
    b2 = CogVec(seed=1, state_path=path)
    ok = b2.load()
    assert ok, "C17：加载失败"

    # S1: 神经状态逐元素一致
    for i, s in snap.items():
        n2 = b2.ns.neurons.get(i)
        assert n2 is not None, f"C17：神经元 {i} 未恢复"
        for f in ["base", "soma", "axon"]:
            d = float(np.abs(np.asarray(getattr(n2, f)) - s[f]).max())
            assert d < 1e-5, f"C17：神经元 {i}.{f} 往返误差 {d:.6f}"
        assert abs(float(n2.activity) - s["activity"]) < 1e-5, f"C17：activity 未恢复"

    # S2: 连接恢复
    for t, arr in tract_snap.items():
        tr2 = b2.ns.tracts.get(t)
        assert tr2 is not None, f"C17：束 {t} 未恢复"
        d = float(np.abs(np.asarray(tr2.dynamics) - arr).max())
        assert d < 1e-5, f"C17：束 {t}.dynamics 往返误差 {d:.6f}"

    # S3: 记忆库内容恢复
    mem_after = len([e for e in _all_experiences(b2)])
    assert mem_after >= mem_before, (
        f"C17：记忆库未恢复（save 前 {mem_before} 条，load 后 {mem_after} 条）")

    # S4: 注意力权重恢复
    assert abs(np.linalg.norm(b2.ns.W_q) - np.linalg.norm(b.ns.W_q)) < 1e-4, \
        "C17：投影矩阵 W_q 未恢复"

    # S5: 端到端 —— 加载后思考结果一致
    r1 = b.think("猫")
    r2 = b2.think("猫")
    assert abs(r1["convergence"] - r2["convergence"]) < 0.2, \
        f"C17：加载后思考结果漂移（{r1['convergence']} vs {r2['convergence']}）"

    _cleanup_state(path)


def _all_experiences(b):
    """取出记忆库里全部经验（兼容不同内部结构）。"""
    buf = getattr(b, "buffer", None)
    if buf is None:
        return []
    out = []
    for attr in ["items", "entries", "_items", "buffer", "_buffer"]:
        v = getattr(buf, attr, None)
        if isinstance(v, list):
            out.extend(v)
    for attr in ["by_id", "_by_id"]:
        v = getattr(buf, attr, None)
        if isinstance(v, dict):
            out.extend(v.values())
    return out


def test_c18_save_creates_full_artifacts():
    """C18: 存盘必须产出完整的分层文件（不是只有一个 json）。"""
    import os, tempfile
    path = os.path.join(tempfile.gettempdir(), "bio_c18_state.json")
    _cleanup_state(path)
    b = CogVec(seed=1, state_path=path)
    b.think("测试存盘")
    d = b.save()
    assert os.path.isdir(d), f"C18：存盘未产出目录（{d}）"
    for f in ["meta.json", "neurons.npz", "tracts.npz", "libraries.json",
              "memory.jsonl", "attn.npz"]:
        p = os.path.join(d, f)
        assert os.path.isfile(p), f"C18：缺少 {f}"
        assert os.path.getsize(p) > 0, f"C18：{f} 为空"
    _cleanup_state(path)



def test_server_observe():
    """服务层：多模态观察"""
    import numpy as np
    b = CogVec(seed=1)
    r = b.observe(image_vec=np.zeros(16), text="猫", source="oracle")
    assert r["ignited"] > 0, "服务层回归：观察未点火"


# ============================================================
# C14: compute 数据通路必须真的活着（soma/axon 非零）
# ============================================================
def test_c14_compute_is_alive():
    """C14: 计算内核(ALG-0)的产物 soma/axon 必须非零。

    ★背景（2026-09-18 重大缺陷）：原实现中 soma ← f(axon) 而 axon ← tanh(soma)，
    互为输入形成**零不动点陷阱** → 全部神经元的 soma/axon 恒为 0，
    compute() 完全空转，但只测 activity 的诊断完全看不出来。

    本测试专门断言「compute 的产物非零」，防止此类沉默缺陷再次出现。
    """
    import numpy as np
    b = CogVec(seed=1)
    ign = b.sensory_symbol.ignite(b._encode_text("猫"), b.ns)
    b.active |= set(ign["ignited"])
    b.run_ticks(12)
    soma_max = max(np.linalg.norm(n.soma) for n in b.ns.neurons.values())
    axon_max = max(np.linalg.norm(n.axon) for n in b.ns.neurons.values())
    assert soma_max > 0.01, (
        f"C14 回归：compute 空转 —— soma 恒为 0 (max={soma_max:.4f})。"
        f"检查阶段A 的输入信号是否形成零不动点。")
    assert axon_max > 0.01, (
        f"C14 回归：compute 空转 —— axon 恒为 0 (max={axon_max:.4f})。")


# ============================================================
# C15: subtick 相位读写分离（BIND 只读）
# ============================================================
def test_c15_bind_is_readonly():
    """C15: BIND 相位必须只读 —— 不修改任何神经元状态。

    架构书 §6.2：BIND 只产 attn_out，INTEGRATE 才写 soma。
    这是消除"同 tick 自反馈"的关键约束。
    """
    import hashlib
    import numpy as np
    b = CogVec(seed=1)
    ign = b.sensory_symbol.ignite(b._encode_text("测试"), b.ns)
    b.active |= set(ign["ignited"])
    b.run_ticks(8)

    def soma_hash(ns):
        h = hashlib.md5()
        for i in sorted(ns.neurons):
            h.update(ns.neurons[i].soma.tobytes())
        return h.hexdigest()

    active = [i for i in b.active if i in b.ns.neurons]
    before = soma_hash(b.ns)
    b.ns._phase_bind(active)
    after = soma_hash(b.ns)
    assert before == after, "C15 回归：BIND 相位篡改了 soma（应只读）"
    assert len(b.ns._attn_out) > 0, "C15 回归：BIND 未产出 attn_out"


# ============================================================
# C16: 交叉注意力多头必须非退化
# ============================================================
def test_c16_multihead_not_degenerate():
    """C16: N_HEADS 个头必须产生不同的注意力分布（否则多头无意义）。"""
    import numpy as np
    import cog_vec as bb
    b = CogVec(seed=1)
    ign = b.sensory_symbol.ignite(b._encode_text("注意力机制"), b.ns)
    b.active |= set(ign["ignited"])
    b.run_ticks(8)

    d_head = max(1, bb.D_SOMA // bb.N_HEADS)
    diff_neurons = 0
    checked = 0
    for i in list(b.active)[:8]:
        n = b.ns.neurons.get(i)
        if n is None or np.linalg.norm(n.soma) < 1e-8:
            continue
        checked += 1
        picks = []
        for h in range(bb.N_HEADS):
            lo, hi = h * d_head, (h + 1) * d_head
            q = n.soma[lo:hi]
            sc = []
            for tid, tr in b.ns.tracts.items():
                kv = tr.key[lo:hi]
                c = float(np.dot(q, kv)) / (np.linalg.norm(q)*np.linalg.norm(kv)+1e-8)
                sc.append((c, tid))
            sc.sort(reverse=True)
            picks.append(tuple(tid for _, tid in sc[:bb.ROUTE_K]))
        if len(set(picks)) > 1:
            diff_neurons += 1
    assert checked > 0, "C16 回归：无可用神经元（soma 全零？见 C14）"
    assert diff_neurons > 0, (
        f"C16 回归：{checked} 个神经元的所有注意力头选择完全相同 → 多头退化")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
