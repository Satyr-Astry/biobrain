"""
仿生 AI · 服务层（BrainServer）
================================
把仿生大脑包装成可用服务：
  · HTTP API  —— POST /think  /observe  /sleep  /state  /teach
  · CLI       —— python server.py --think "..."  / --serve / --demo
  · 持久化    —— 状态可存盘/加载（JSON）

依赖: 仅标准库 + numpy（HTTP 用 http.server，避免额外依赖）
运行:
  python server.py --serve            # 起 HTTP 服务 (默认 :8642)
  python server.py --think "你好"      # 单次思考
  python server.py --demo             # 完整演示
  python server.py --state            # 查看状态
"""
from __future__ import annotations
import json
import os
import sys
import argparse
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from dataclasses import asdict
import numpy as np

from bio_brain import (NervousSystem, SensoryPort, MotorPort, Tier,
                       ACTIVE_EPS, D_SOMA, norm)
from encoder import get_encoder
from self_training import (Experience, ExperienceBuffer, Consolidator,
                           ConvergenceMeter, FeedbackLoop, emit_decision,
                           CONVERGE_HIGH, CONVERGE_LOW, PATIENCE)
from advanced import EnergyMeter, EnergyScheduler, GroundingModule, RobustnessGuard
from p2_mechanisms import Neurogenesis, CollectiveLayer, SelfModel


# ============================================================
# 核心：仿生大脑（把各模块组装成一个整体）
# ============================================================
class BioBrain:
    """一个可直接使用的仿生大脑实例"""

    STATE_VERSION = "v7.3"

    def __init__(self, seed: int = 42, n_sensory: int = 32, n_inter: int = 24,
                 n_motor: int = 6, state_path: Optional[str] = None):
        self.seed = seed
        self.state_path = state_path or "bio_brain_state.json"
        self.ns = NervousSystem(seed=seed)
        self._build(n_sensory, n_inter, n_motor)
        # 子系统
        self.buffer = ExperienceBuffer()
        self.consolidator = Consolidator(self.ns, self.buffer)
        self.meter = ConvergenceMeter()
        self.feedback = FeedbackLoop(self.ns, self.self_sensory)
        self.energy = EnergyMeter()
        self.scheduler = EnergyScheduler()
        self.grounding = GroundingModule(self.ns)
        self.guard = RobustnessGuard()
        self.neurogenesis = Neurogenesis(self.ns)
        self.collective = CollectiveLayer()
        self.self_model = SelfModel()
        # ★语义编码器（自动选最优：真嵌入 或 n-gram 兜底）
        self.encoder = get_encoder()
        # 端口
        self.sensory_symbol = SensoryPort("symbol", self.symbol_ids, D_SOMA)
        self.sensory_vision = SensoryPort("vision", self.vision_ids, D_SOMA)
        self.motor_lang = MotorPort("language", self.motor_ids, threshold=0.45)
        self.active: set = set()
        self.tick_count = 0
        # ★v1.2：Eon 规范「计时区与 I/O 分离」——最近一次纯计算耗时
        self._last_sim_time = 0.0

    # ---------- 构建 ----------
    def _build(self, n_sensory, n_inter, n_motor):
        ns = self.ns
        half = n_sensory // 2
        self.vision_ids = [ns.add_neuron("sensory") for _ in range(half)]
        self.symbol_ids = [ns.add_neuron("sensory") for _ in range(n_sensory - half)]
        self.inter_ids = [ns.add_neuron("inter") for _ in range(n_inter)]
        self.motor_ids = [ns.add_neuron("motor") for _ in range(n_motor)]
        self.self_sensory = [ns.add_neuron("sensory") for _ in range(2)]
        # 束：感觉→联合、联合→运动、桥接
        # ★v3.2：self_sensory(63/64) 此前从未被任何束收编 → 并入感知束 T65/T66（自感觉端回到主通路）
        t1 = ns.add_tract(self.vision_ids + self.inter_ids[:2] + self.self_sensory)
        t2 = ns.add_tract(self.symbol_ids + self.inter_ids[:2] + self.self_sensory)      # 与t1重叠
        # ★结构编辑5(2026-09-18 v1.6)：运动端解耦 —— 拓扑层改动
        #   问题（实测）：6 个运动神经元全挤在一个含 20 个 inter 的束里
        #                → 信号被平均掉 → 只有 2/6 携带信息（另 3 个恒为 0）
        #   依据：BANC(Nature 656:957)「效应神经元主要受同体部位感觉神经元影响，
        #         形成局部反馈环」—— 运动神经元应按功能分组，而非共享平均化通道
        #   做法：把运动神经元拆成 2 组，各带不同的 inter 子集 → 两组独立通路
        mid = self.inter_ids[2:max(3, n_inter - 2)]
        half_m = max(1, n_motor // 2)
        motors_a = self.motor_ids[:half_m]
        motors_b = self.motor_ids[half_m:]
        mid_a = mid[:len(mid)//2] if len(mid) > 1 else mid
        mid_b = mid[len(mid)//2:] if len(mid) > 1 else mid
        # 两组运动通路（各自带不同的 inter 子集 → 不共享平均化通道）
        t3 = ns.add_tract(mid_a + motors_a)
        t4 = ns.add_tract(mid_b + motors_b)
        # 桥接束：让两组之间保持协调（BANC：长程连接用于协调局部环）
        # ★结构编辑6(2026-09-18 v3.2 修复 P0 断路)：补齐 4 个孤立神经元
        #   问题（实测）：inter_ids 尾部 2 个（55/56）掉进三个切片
        #        [:2] / [2:max(3,n-2)] / mid[:2] 的缝隙 → tract_ids=[]，入度=出度=0
        #        另 self_sensory（63/64）在 84~100 行的 5 个 add_tract 中一次未被引用
        #        ⇒ 4 个神经元从未被任何束收编，感觉端可达率只有 62/64
        #   依据：BANC(Nature 656:957) 局部环需有长程协调输入；本项目 §6.1「拓扑层有效」
        #   做法：只把尾部 inter 并入桥接束 + 把自感觉端并入 T65/T66
        #        —— 不做全连接（避免重蹈 P-ARCH-12 污染）
        t_bridge = ns.add_tract(self.inter_ids[:2] + mid[:2] + self.inter_ids[-2:]) if len(mid) >= 2 else None

        g1 = ns.add_group([t1, t2], "percept")
        g2 = ns.add_group([t3, t4], "action")
        ns.add_library([g1]); ns.add_library([g2])

    # ---------- 输入编码（文本 → 激活模式）----------
    def _encode_text(self, text: str) -> np.ndarray:
        """★真语义编码（可插拔：semantic 优先，自动降级 n-gram）

        语义相近的文本 → 激活模式相近，这是"理解"而非"字面匹配"的基础。
        """
        return self.encoder.encode(text)

    # ---------- 输出强度估计（★2026-09-18 新增，修 P0）----------
    def has_generation_backend(self) -> bool:
        """是否已接上"生成后端"（嘴）。

        当前实现尚未接生成模型 → 返回 False，
        此时 think() 会诚实地回答"不知道"，而不是谎报 confidence=1.0。

        接上后端后（如 llama.cpp server / 本地 Qwen），把这里改成实际探测。
        """
        gen_url = os.environ.get("BIO_GEN_BACKEND_URL", "").strip()
        if not gen_url:
            return False
        try:
            import urllib.request
            with urllib.request.urlopen(gen_url.rstrip("/") + "/health", timeout=1.0):
                return True
        except Exception:
            return False

    def _emission_strength(self, emission) -> float:
        """从"运动端激活 vs 符号端激活"导出输出强度 ∈ [0,1]。

        为什么不能直接用 emission['strength']：
          运动端激活会随扩散累积，容易饱和到 1.0，导致任何输入都给满分置信度。

        改为相对量：
          raw = motor_max / (motor_max + symbol_mean + eps)
        这样同样是 0.12 的运动端激活，在符号端整体活跃时得低分、安静时得高分，
        且天然落在 (0,1)，不会恒为 1。
        """
        if not emission:
            return 0.0
        mot_vals = list(emission["emission"].values())
        mot_max = max(mot_vals) if mot_vals else 0.0
        sym_vals = [self.ns.neurons[i].activity for i in self.symbol_ids]
        sym_mean = float(np.mean(sym_vals)) if sym_vals else 0.0
        raw = mot_max / (mot_max + sym_mean + 1e-8)
        return float(max(0.0, min(1.0, raw)))

    # ---------- 主循环：跑 N 个 tick ----------
    def run_ticks(self, n: int = 15) -> dict:
        """★v1.2：接入结构可塑 + 计时区与 I/O 分离（Eon 工程规范）。

        计时区：只统计计算时间（apply_pending / tick_activation / compute / ...）
        I/O 区：日志/状态写盘等，不计入 sim_time
        """
        # ★Eon 规范：计时区与 I/O 分离
        t0 = time.perf_counter()
        for _ in range(n):
            self.ns.apply_pending(self.active)
            # ★结构编辑4：提交上 tick 的结构更新（NORM-4 因果优先）
            self.ns.apply_structure_pending()
            self.active = self.ns.tick_activation(self.active)
            for tid in self.ns.tracts:
                self.ns.tracts[tid].activation = self.ns.tract_activation(tid)
            state = self.ns.compute(self.active)
            self.ns.update_plasticity(self.active, {})
            # ★结构可塑（v1.2~v1.5）默认关闭 —— 四次迭代实测无效/退化
            #   见 docs/04_实测与诊断/结构可塑四次迭代教训.md
            #   结论：有效改动在「拓扑层」（谁连谁/方向/符号），无效改动在「权重层」
            #   保留能力但不默认启用；如需实验可设 BIO_STRUCT_PLASTIC=1
            if os.environ.get("BIO_STRUCT_PLASTIC", "0") == "1":
                self.ns.update_structure(self.active)
            self.ns.tick_scheduler()
            self.ns.t += 1
            self.tick_count += 1
        sim_time = time.perf_counter() - t0          # 计时区结束

        # ---- 以下为 I/O 与守护区，不计入 sim_time（Eon 规范）----
        self._last_sim_time = sim_time
        acts = [n_.activity for n_ in self.ns.neurons.values()]
        self.guard.check_alive(acts)
        return {"ticks": n, "active": len(self.active), "t": self.tick_count,
                "sim_time": round(sim_time, 4)}

    # ---------- 对外 API ----------
    def think(self, text: str, task_type: str = "general") -> dict:
        """输入文本 → 思考 → 输出（大脑自主决定是否输出）"""
        # 1. 编码 + 点火（感觉端）
        vec = self._encode_text(text)
        ign = self.sensory_symbol.ignite(vec, self.ns)
        self.active |= set(ign["ignited"])
        # 2. 跑思考流
        self.run_ticks(12)
        # 3. 聚合状态，更新收敛度量
        state_vec = np.array([self.ns.neurons[i].activity
                              for i in sorted(self.active)]) if self.active else np.zeros(4)
        if len(state_vec) < 4:
            state_vec = np.pad(state_vec, (0, 4 - len(state_vec)))
        # ★修正: 先 update 再算 convergence（原顺序导致永远为0）
        self.meter.update(state_vec[:16] if len(state_vec) >= 16
                          else np.pad(state_vec, (0, 16 - len(state_vec)))[:16],
                          pred_err=max(0.0, 1.0 - float(np.mean([self.ns.neurons[i].activity for i in self.active]))) if self.active else 1.0)
        conv = self.meter.convergence(goal_satisfaction=0.6)
        # 4. 大脑决定是否输出（运动端读出）
        emission = self.motor_lang.read(self.ns)
        decision = emit_decision(conv, has_input=True, elapsed=1.0)
        # 5. 自我模型校准（防幻觉）
        #   ★修正(2026-09-18)：raw_conf 不能直接用 motor strength（恒为饱和值1.0）。
        #   改为：以"运动端激活相对符号端激活的归一化强度"作为输出强度估计，
        #         这样不同输入会得到不同置信度，且不会无脑满分。
        raw_conf = self._emission_strength(emission)
        cal_conf = self.self_model.calibrated_confidence(task_type, raw_conf)
        dont_know = self.self_model.should_say_dont_know(task_type, raw_conf)
        # ★修正(2026-09-18)：无生成后端时绝不宣称高置信（ALG-17 能力认知 / NORM-6 可靠信号）
        if not self.has_generation_backend():
            cal_conf = 0.0
            dont_know = True
            emission = None
            decision["emit"] = False
            decision["reason"] = "no_generation_backend"
        # 6. 回灌（带护栏）
        fed = False
        if emission:
            fed = self.feedback.feed_back(emission, supported=(not dont_know))
        # 7. 记录经验（来源=outcome，可靠）
        self.buffer.store(Experience(
            id=self.tick_count, thought_trace=list(self.active)[:8],
            ignitions=list(ign["ignited"]), emissions=list(emission["emission"]) if emission else [],
            outcome=raw_conf, surprise=abs(1.0 - conv),
            event_time=time.time(), record_time=time.time(), source="outcome"))
        self.self_model.record_outcome(task_type, success=(raw_conf > 0.45))
        return {
            "input": text,
            "activation": len(self.active),
            "convergence": round(conv, 3),
            "emission": bool(emission),
            "confidence_raw": round(raw_conf, 3),
            "confidence_calibrated": round(cal_conf, 3),
            "should_say_dont_know": dont_know,
            # ★修正(2026-09-18)：原实现是把输入 text[:20] 回显 + 前缀，等于假生成。
            #   现在：有生成后端 → 走后端；无后端 → 诚实返回 None 并说明原因。
            "output": self._produce_output(text, emission, dont_know),
            "feedback": fed,
            "tick": self.tick_count,
            "generation_backend": self.has_generation_backend(),
            "decision": decision,
        }

    def _produce_output(self, text: str, emission, dont_know: bool):
        """产出对外文字。

        ★铁律（NORM-2 无接口 / ALG-17 防幻觉）：
          · 有生成后端 → 由后端生成（生成结果仍需 entailment 校验）
          · 无生成后端 → 返回 None。不得拿输入回显冒充输出。
        """
        if dont_know or not emission:
            return None
        if not self.has_generation_backend():
            # 没有嘴 → 不出声。这是诚实，不是失败。
            return None
        # ---- 以下是接上后端后的路径（现在不会走到）----
        gen_url = os.environ.get("BIO_GEN_BACKEND_URL", "").rstrip("/")
        # 生成输入 = 激活模式摘要 + 记忆召回（不是原始问题，避免退化成普通 RAG）
        act_digest = self._activation_digest()
        prompt = self._build_generation_prompt(text, act_digest)
        try:
            import json as _json
            import urllib.request
            req = urllib.request.Request(
                gen_url + "/v1/chat/completions",
                data=_json.dumps({"messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": 256}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = _json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None if not str(e) else None  # 生成失败 → 宁可不说

    def _activation_digest(self, top_k: int = 8) -> str:
        """把当前激活模式压成可读摘要（供生成后端使用）。"""
        items = sorted(((i, self.ns.neurons[i].activity) for i in self.active),
                       key=lambda kv: -kv[1])[:top_k]
        return ", ".join(f"N{i}:{a:.3f}" for i, a in items)

    def _build_generation_prompt(self, text: str, digest: str) -> str:
        return (f"【大脑激活模式】{digest}\n"
                f"【待回应】{text}\n"
                f"请基于激活模式所指向的内容作答；若激活模式为空或无关，直接回答'我不知道'。")

    def observe(self, image_vec: Optional[np.ndarray] = None,
                text: Optional[str] = None, source: str = "outcome") -> dict:
        """多模态观察（共激活以接地）"""
        ignited = []
        if image_vec is not None:
            r = self.sensory_vision.ignite(np.asarray(image_vec).flatten(), self.ns)
            ignited += list(r["ignited"])
        if text:
            r = self.sensory_symbol.ignite(self._encode_text(text), self.ns)
            ignited += list(r["ignited"])
        self.active |= set(ignited)
        # 共激活 → 接地
        if image_vec is not None and text:
            g = self.grounding.ground_by_coactivation(
                self.symbol_ids, self.vision_ids, strength=0.5)
        self.run_ticks(8)
        grounded = bool(image_vec is not None and text is not None)
        return {"ignited": len(ignited), "active": len(self.active),
                "grounded": grounded,
                "coactivation_links": len(self.grounding.coactivation_log)}

    def sleep(self, has_pending_task: bool = False) -> dict:
        """睡眠巩固（双相：NREM + REM）"""
        if not self.scheduler.should_do_heavy():
            heavy = "非低能耗时段，只做轻量巩固"
        else:
            heavy = "低能耗时段，可做重活"
        r = self.consolidator.sleep(has_running_task=False,
                                    has_pending_slice=has_pending_task)
        return {"sleep": r, "energy_note": heavy,
                "candidates": len(self.consolidator.candidate_store)}

    def teach(self, payload: dict, source: str = "oracle") -> dict:
        """教它一条知识（进集体层）"""
        key = self.collective.submit(payload, source=source, instance_id=f"inst{self.seed}")
        return {"accepted": key is not None, "key": key}

    def state(self) -> dict:
        """完整状态"""
        tiers = {}
        for lid, lib in self.ns.libraries.items():
            tiers[f"lib{lid}"] = lib.tier.value
        return {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "neurons": len(self.ns.neurons),
            "tracts": len(self.ns.tracts),
            "groups": len(self.ns.groups),
            "libraries": len(self.ns.libraries),
            "active": len(self.active),
            "ticks": self.tick_count,
            "tiers": tiers,
            "experience_buffer": self.buffer.stats(),
            "self_model": self.self_model.who_am_i(),
            "energy_j_per_token": self.energy.j_per_token(),
            "robustness": self.guard.report()["level"],
            "newborns": len(self.neurogenesis.newborns),
            "collective_entries": len(self.collective.entries),
            "candidates": len(self.consolidator.candidate_store),
        }

    # ---------- 持久化 ----------
    def save(self):
        """存盘（★v3.1 真存盘：完整神经状态 + 记忆库内容）。

        修复 R1：原实现只存元数据（version/seed/tick/自模型统计/一个 base_sum 标量），
                 完全没存神经状态与记忆库 → 重启即失忆。见架构书 §8.2/§8.5。
        """
        try:
            from persistence import save_state
            d = save_state(self, self.state_path)
            return d
        except Exception as e:
            # 兜底：退回旧格式（至少保住元数据）
            data = {
                "version": self.STATE_VERSION, "seed": self.seed,
                "tick_count": self.tick_count,
                "skills": self.self_model.skills, "goals": self.self_model.goals,
                "experiences": self.self_model.total_experiences,
                "successes": self.self_model.total_successes,
                "collective": {k: {"payload": e.payload, "verified_by": e.verified_by,
                                   "contributors": e.contributors}
                               for k, e in self.collective.entries.items()},
                "base_sum": float(sum(norm(n.base) for n in self.ns.neurons.values())),
                "save_error": str(e),
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            return self.state_path

    def load(self) -> bool:
        """读盘（★v3.1：完整恢复，含向后兼容旧格式）。"""
        try:
            from persistence import load_state
            ok, note = load_state(self, self.state_path)
            return bool(ok)
        except Exception:
            # 兜底：尝试旧格式
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.self_model.skills = data.get("skills", {})
                self.self_model.goals = data.get("goals", [])
                self.self_model.total_experiences = data.get("experiences", 0)
                self.self_model.total_successes = data.get("successes", 0)
                for k, v in data.get("collective", {}).items():
                    self.collective.submit(v["payload"], source=v["verified_by"],
                                           instance_id="restored")
                return True
            except Exception:
                return False


# ============================================================
# HTTP API
# ============================================================
class BrainHTTPHandler(BaseHTTPRequestHandler):
    brain: BioBrain = None      # 由 serve() 注入

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            self._send(200, self.brain.state())
        elif self.path == "/health":
            self._send(200, {"ok": True, "version": BioBrain.STATE_VERSION})
        else:
            self._send(404, {"error": "not found", "try": ["/state", "/health"]})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw or "{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})

        if self.path == "/think":
            r = self.brain.think(data.get("text", ""), data.get("task_type", "general"))
            self._send(200, r)
        elif self.path == "/observe":
            img = np.array(data["image"]) if data.get("image") else None
            r = self.brain.observe(image_vec=img, text=data.get("text"),
                                   source=data.get("source", "outcome"))
            self._send(200, r)
        elif self.path == "/sleep":
            self._send(200, self.brain.sleep(data.get("has_pending_task", False)))
        elif self.path == "/teach":
            self._send(200, self.brain.teach(data.get("payload", {}),
                                             data.get("source", "oracle")))
        elif self.path == "/save":
            self._send(200, {"saved": self.brain.save()})
        else:
            self._send(404, {"error": "not found",
                             "try": ["/think", "/observe", "/sleep", "/teach", "/state"]})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[brain] {self.address_string()} {fmt % args}\n")


def serve(port: int = 8642, state_path: Optional[str] = None):
    brain = BioBrain(state_path=state_path)
    if brain.load():
        print(f"已加载历史状态: {state_path}")
    BrainHTTPHandler.brain = brain
    srv = HTTPServer(("127.0.0.1", port), BrainHTTPHandler)
    print(f"仿生大脑服务已启动: http://127.0.0.1:{port}")
    print("接口: GET /state /health | POST /think /observe /sleep /teach /save")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务，保存状态...")
        brain.save()
        srv.server_close()


# ============================================================
# CLI
# ============================================================
def cli():
    ap = argparse.ArgumentParser(description="仿生 AI 大脑服务")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--port", type=int, default=8642, help="服务端口")
    ap.add_argument("--think", type=str, help="单次思考")
    ap.add_argument("--task", type=str, default="general", help="任务类型")
    ap.add_argument("--state", action="store_true", help="查看状态")
    ap.add_argument("--sleep", action="store_true", help="执行睡眠巩固")
    ap.add_argument("--demo", action="store_true", help="完整演示")
    ap.add_argument("--state-file", type=str, default="bio_brain_state.json")
    args = ap.parse_args()

    if args.serve:
        return serve(args.port, args.state_file)

    brain = BioBrain(state_path=args.state_file)
    brain.load()

    if args.think:
        r = brain.think(args.think, args.task)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.state:
        print(json.dumps(brain.state(), ensure_ascii=False, indent=2))
    elif args.sleep:
        print(json.dumps(brain.sleep(), ensure_ascii=False, indent=2))
    else:
        demo(brain)


def demo(brain: Optional[BioBrain] = None):
    print("=" * 66)
    print("  仿生 AI · 服务层演示（BioBrain）")
    print("=" * 66)
    b = brain or BioBrain()
    print("\n[初始状态]")
    print(json.dumps(b.state(), ensure_ascii=False, indent=1)[:600])

    print("\n[1] 思考：它擅长的领域")
    for _ in range(6):
        b.self_model.record_outcome("chat", success=True)
    r = b.think("你好，请介绍一下自己", task_type="chat")
    print(json.dumps(r, ensure_ascii=False, indent=1))

    print("\n[2] 思考：它不擅长的领域（测试诚实输出）")
    for _ in range(6):
        b.self_model.record_outcome("quantum", success=False)
    r2 = b.think("请解释量子纠缠的数学基础", task_type="quantum")
    print(json.dumps(r2, ensure_ascii=False, indent=1))

    print("\n[3] 多模态观察（共激活→接地）")
    r3 = b.observe(image_vec=np.random.default_rng(0).normal(0, 1, D_SOMA),
                   text="猫", source="oracle")
    print(json.dumps(r3, ensure_ascii=False, indent=1))

    print("\n[4] 睡眠巩固（双相）")
    r4 = b.sleep()
    print(json.dumps(r4, ensure_ascii=False, indent=1))

    print("\n[5] 存盘")
    print("已保存:", b.save())

    print("\n[最终状态]")
    print(json.dumps(b.state(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cli()
