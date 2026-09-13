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
        t1 = ns.add_tract(self.vision_ids + self.inter_ids[:2])
        t2 = ns.add_tract(self.symbol_ids + self.inter_ids[:2])      # 与t1重叠
        mid = self.inter_ids[2:max(3, n_inter - 2)]
        t3 = ns.add_tract(mid + self.motor_ids)
        t4 = ns.add_tract(self.inter_ids[:2] + mid)                  # 桥接束
        g1 = ns.add_group([t1, t2], "percept")
        g2 = ns.add_group([t3, t4], "action")
        ns.add_library([g1]); ns.add_library([g2])

    # ---------- 输入编码（文本 → 激活模式）----------
    def _encode_text(self, text: str) -> np.ndarray:
        """★真语义编码（可插拔：semantic 优先，自动降级 n-gram）

        语义相近的文本 → 激活模式相近，这是"理解"而非"字面匹配"的基础。
        """
        return self.encoder.encode(text)

    # ---------- 主循环：跑 N 个 tick ----------
    def run_ticks(self, n: int = 15) -> dict:
        for _ in range(n):
            self.ns.apply_pending(self.active)
            self.active = self.ns.tick_activation(self.active)
            for tid in self.ns.tracts:
                self.ns.tracts[tid].activation = self.ns.tract_activation(tid)
            state = self.ns.compute(self.active)
            self.ns.update_plasticity(self.active, {})
            self.ns.tick_scheduler()
            self.ns.t += 1
            self.tick_count += 1
            # 数值守护
            acts = [n_.activity for n_ in self.ns.neurons.values()]
            self.guard.check_alive(acts)
        return {"ticks": n, "active": len(self.active), "t": self.tick_count}

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
        raw_conf = emission["strength"] if emission else 0.0
        cal_conf = self.self_model.calibrated_confidence(task_type, raw_conf)
        dont_know = self.self_model.should_say_dont_know(task_type, raw_conf)
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
            "output": (None if not emission or dont_know else
                       f"[激活{emission['strength']:.2f}] {text[:20]}..."),
            "feedback": fed,
            "tick": self.tick_count,
        }

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
        """存盘（权重 + 自模型 + 集体层）"""
        data = {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "tick_count": self.tick_count,
            "skills": self.self_model.skills,
            "goals": self.self_model.goals,
            "experiences": self.self_model.total_experiences,
            "successes": self.self_model.total_successes,
            "collective": {k: {"payload": e.payload, "verified_by": e.verified_by,
                               "contributors": e.contributors}
                           for k, e in self.collective.entries.items()},
            "base_sum": float(sum(norm(n.base) for n in self.ns.neurons.values())),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return self.state_path

    def load(self) -> bool:
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
        except FileNotFoundError:
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
