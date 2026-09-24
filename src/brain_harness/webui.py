"""
HarnessPlatform · Web UI 服务（后端）
========================================
在 HTTPSink 基础上扩展，提供 UI 需要的全部端点：
  GET /                → 单页 UI（内嵌 HTML，零外部依赖）
  GET /api/snapshot    → 一次拿全（state + 最近结果 + 统计）★UI 主轮询
  GET /api/results     → 最近结果
  GET /api/state       → 大脑状态
  GET /api/metrics     → 指标历史（metrics policy 的 history）
  GET /api/stats       → pipeline 统计
  POST /api/inject     → 注入一条输入（外部刺激）

设计约束（NORM-7）：只用标准库，不引 Flask/FastAPI。
"""
from __future__ import annotations
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional

from .context import TickContext, Result
from .registry import sink


@ sink("webui")
class WebUISink:
    """带 Web UI 的 Sink（同时承担"输出端 + 观测面板"）"""

    def __init__(self, port: int = 8765, keep: int = 500, **kw):
        self.name = f"webui:{port}"
        self.port = port
        self.keep = keep
        self._results: List[Dict[str, Any]] = []
        self._state_fn: Optional[Callable] = None
        self._stats_fn: Optional[Callable] = None
        self._metrics_fn: Optional[Callable] = None
        self._inject_fn: Optional[Callable] = None
        self._started_at = None
        self._server = None
        self._lock = threading.Lock()
        self._start()

    # ---------- 绑定回调（平台组装时注入） ----------
    def bind_state(self, fn: Callable) -> None:
        self._state_fn = fn

    def bind_extra(self, stats_fn=None, metrics_fn=None, inject_fn=None) -> None:
        self._stats_fn = stats_fn
        self._metrics_fn = metrics_fn
        self._inject_fn = inject_fn

    # ---------- Sink 协议 ----------
    def handle(self, r: Result, ctx: TickContext) -> None:
        with self._lock:
            self._results.append(r.to_dict())
            if len(self._results) > self.keep:
                del self._results[:len(self._results) - self.keep]

    def close(self) -> None:
        try:
            if self._server:
                self._server.shutdown()
        except Exception:
            pass

    # ---------- HTTP ----------
    def _start(self):
        holder = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _json(self, obj, code=200):
                body = json.dumps(obj, ensure_ascii=False, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _html(self, text):
                body = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                p = self.path.split("?")[0]
                try:
                    if p == "/":
                        self._html(PAGE)
                    elif p == "/api/snapshot":
                        with holder._lock:
                            results = list(holder._results[-60:])
                        self._json({
                            "state": _call(holder._state_fn),
                            "results": results,
                            "stats": _call(holder._stats_fn),
                            "metrics": _call(holder._metrics_fn),
                            "n_results": len(holder._results),
                        })
                    elif p == "/api/results":
                        with holder._lock:
                            self._json({"items": list(holder._results[-40:])})
                    elif p == "/api/state":
                        self._json(_call(holder._state_fn) or {})
                    elif p == "/api/metrics":
                        self._json({"items": _call(holder._metrics_fn) or []})
                    elif p == "/api/stats":
                        self._json(_call(holder._stats_fn) or {})
                    elif p == "/health":
                        self._json({"ok": True, "sink": "webui",
                                    "port": holder.port})
                    else:
                        self.send_error(404)
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

            def do_POST(self):
                p = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n).decode("utf-8", "ignore") if n else ""
                try:
                    if p == "/api/inject":
                        data = json.loads(raw) if raw.strip() else {}
                        text = str(data.get("text", "")).strip()
                        if not text:
                            self._json({"ok": False, "err": "text 为空"}, 400)
                            return
                        if holder._inject_fn is None:
                            self._json({"ok": False, "err": "无可注入通道"}, 503)
                            return
                        holder._inject_fn(text)
                        self._json({"ok": True, "injected": text})
                    else:
                        self.send_error(404)
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

            def log_message(self, *a):
                pass

        self._server = HTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()


def _call(fn):
    if fn is None:
        return None
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BioBrain Harness Platform</title>
<style>
 :root{--bg:#0b0f14;--card:#131a23;--bd:#243040;--fg:#e6edf5;--mut:#8b9bb0;
       --acc:#4ec9b0;--warn:#e0af68;--err:#f7768e;--ok:#9ece6a}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{display:flex;align-items:center;gap:14px;padding:10px 16px;
        background:var(--card);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:9}
 h1{font-size:15px;margin:0;font-weight:600}
 .dot{width:9px;height:9px;border-radius:50%;background:var(--err);display:inline-block}
 .dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
 .mut{color:var(--mut)}
 .wrap{display:grid;grid-template-columns:260px 1fr 300px;gap:12px;padding:12px;align-items:start}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px}
 .card h2{font-size:12px;margin:0 0 9px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;font-weight:600}
 .kpi{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed #1e2836}
 .kpi:last-child{border:0}
 .kpi b{color:var(--acc);font-variant-numeric:tabular-nums}
 .rows{max-height:62vh;overflow:auto}
 .row{display:grid;grid-template-columns:52px 62px 1fr 66px;gap:8px;padding:5px 6px;
      border-bottom:1px solid #1a232f;font-size:12px;align-items:center}
 .row:hover{background:#182231}
 .row .tk{color:var(--mut);text-align:right;font-variant-numeric:tabular-nums}
 .tag{font-size:10px;padding:1px 5px;border-radius:3px;border:1px solid var(--bd);color:var(--mut);text-align:center}
 .tag.spon{color:var(--warn);border-color:#4a3a1a}
 .tag.ext{color:var(--acc);border-color:#1d4a42}
 .conv{text-align:right;color:var(--ok);font-variant-numeric:tabular-nums}
 .none{color:var(--mut);font-style:italic;font-size:11px}
 input,button{font:inherit;background:#0e141b;color:var(--fg);border:1px solid var(--bd);
              border-radius:6px;padding:7px 9px}
 button{background:#1b2733;cursor:pointer}
 button:hover{border-color:var(--acc);color:var(--acc)}
 .flex{display:flex;gap:8px}
 .flex input{flex:1}
 .bar{height:7px;background:#1b2733;border-radius:4px;overflow:hidden;margin-top:3px}
 .bar i{display:block;height:100%;background:linear-gradient(90deg,#4ec9b0,#e0af68)}
 table{width:100%;border-collapse:collapse;font-size:11.5px}
 td{padding:2.5px 0;border-bottom:1px dashed #1e2836}
 td:last-child{text-align:right;color:var(--acc);font-variant-numeric:tabular-nums}
 #spark{width:100%;height:52px;display:block}
 .err{color:var(--err)}
 details summary{cursor:pointer;color:var(--mut);font-size:11.5px;margin-top:6px}
 pre{white-space:pre-wrap;font-size:11px;color:var(--mut);margin:6px 0 0}
</style></head><body>
<header>
  <span class="dot" id="dot"></span>
  <h1>BioBrain · Harness Platform</h1>
  <span class="mut" id="hdr">连接中…</span>
  <span style="flex:1"></span>
  <span class="mut" id="poll">—</span>
</header>

<div class="wrap">
  <!-- 左列 -->
  <div>
    <div class="card">
      <h2>大脑状态</h2>
      <div id="state"></div>
    </div>
    <div class="card" style="margin-top:12px">
      <h2>注入输入</h2>
      <div class="flex">
        <input id="txt" placeholder="输入一段文字…" autocomplete="off">
        <button onclick="inject()">发送</button>
      </div>
      <div class="mut" style="margin-top:6px;font-size:11px" id="injMsg"></div>
    </div>
  </div>

  <!-- 中列 -->
  <div>
    <div class="card">
      <h2>收敛度趋势（最近 60 条）</h2>
      <canvas id="spark"></canvas>
    </div>
    <div class="card" style="margin-top:12px">
      <h2>思考流 <span class="mut" id="nres"></span></h2>
      <div class="rows" id="rows"><div class="mut">等待数据…</div></div>
    </div>
  </div>

  <!-- 右列 -->
  <div>
    <div class="card">
      <h2>流水线</h2>
      <div id="stats"></div>
    </div>
    <div class="card" style="margin-top:12px">
      <h2>指标历史</h2>
      <div id="metrics" class="rows" style="max-height:36vh"></div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let okCount = 0, lastTick = 0;

function fmt(v){ return v===null||v===undefined ? '—' : v; }

function draw(st){
  const a=$('#state');
  if(!st){ a.innerHTML='<span class="mut">无状态</span>'; return; }
  const sm = st.self_model || {};
  const eb = st.experience_buffer || {};
  const rows = [
    ['版本', st.version], ['种子', st.seed],
    ['神经元', st.neurons], ['束', st.tracts],
    ['群组', st.groups], ['库', st.libraries],
    ['活跃', st.active], ['ticks', st.ticks],
    ['体验缓冲', eb.size], ['经历数', sm['经历数']],
    ['成功率', sm['整体成功率']],
  ];
  a.innerHTML = rows.map(([k,v])=>
    `<div class="kpi"><span class="mut">${k}</span><b>${fmt(v)}</b></div>`).join('');
}

function drawStats(s){
  const el=$('#stats');
  if(!s){ el.innerHTML='<span class="mut">无</span>'; return; }
  const bs = s.by_source||{}, bp = s.by_policy||{};
  let h = `<table>
    <tr><td>ticks</td><td>${fmt(s.ticks)}</td></tr>
    <tr><td>刺激</td><td>${fmt(s.stimuli)}</td></tr>
    <tr><td>结果</td><td>${fmt(s.results)}</td></tr>
    <tr><td>策略执行</td><td>${fmt(s.policies_run)}</td></tr>
    <tr><td>错误</td><td class="${s.errors?'err':''}">${fmt(s.errors)}</td></tr></table>`;
  if(Object.keys(bs).length){
    h += `<div class="mut" style="margin-top:8px;font-size:11px">按来源</div><table>`;
    for(const k in bs) h += `<tr><td>${k}</td><td>${bs[k]}</td></tr>`;
    h += `</table>`;
  }
  if(Object.keys(bp).length){
    h += `<div class="mut" style="margin-top:8px;font-size:11px">按策略</div><table>`;
    for(const k in bp) h += `<tr><td>${k}</td><td>${bp[k]}</td></tr>`;
    h += `</table>`;
  }
  el.innerHTML = h;
}

function drawRows(items){
  const el=$('#rows');
  if(!items || !items.length){ el.innerHTML='<div class="mut">暂无结果（无生成后端时 output 恒为 null，符合 NORM-9）</div>'; return; }
  el.innerHTML = items.slice().reverse().map(r=>{
    const spon = r.extra && r.extra.spontaneous;
    const cls = spon ? 'spon' : 'ext';
    const lab = spon ? '自发' : '外部';
    const out = (r.output===null||r.output===undefined)
      ? '<span class="none">null (诚实)</span>' : String(r.output);
    const cv = (r.convergence===null||r.convergence===undefined)?'—':Number(r.convergence).toFixed(3);
    return `<div class="row">
      <span class="tk">${r.tick}</span>
      <span class="tag ${cls}">${lab}</span>
      <span title="${escapeHtml(r.input)}">${escapeHtml(short(r.input,42))}
        <div class="mut" style="font-size:10.5px">→ ${out}</div></span>
      <span class="conv">${cv}</span></div>`;
  }).join('');
}

function drawSpark(items){
  const c = $('#spark'); if(!c) return;
  const dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = 52;
  c.width = w*dpr; c.height = h*dpr;
  const g = c.getContext('2d'); g.scale(dpr,dpr);
  g.clearRect(0,0,w,h);
  const vals = items.map(r=>r.convergence).filter(v=>typeof v==='number');
  if(vals.length<2) return;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const rng = (hi-lo)||1;
  g.beginPath();
  vals.forEach((v,i)=>{
    const x = i/(vals.length-1)*(w-4)+2;
    const y = h-4-((v-lo)/rng)*(h-12);
    i?g.lineTo(x,y):g.moveTo(x,y);
  });
  g.strokeStyle='#4ec9b0'; g.lineWidth=1.6; g.stroke();
  g.lineTo(w-2,h); g.lineTo(2,h); g.closePath();
  g.fillStyle='rgba(78,201,176,.12)'; g.fill();
  g.fillStyle='#8b9bb0'; g.font='10px monospace';
  g.fillText('max '+hi.toFixed(3), 4, 11);
  g.fillText('min '+lo.toFixed(3), 4, h-3);
}

function drawMetrics(m){
  const el=$('#metrics');
  if(!m || !m.length){ el.innerHTML='<span class="mut">无</span>'; return; }
  el.innerHTML = '<table>' + m.slice(-24).reverse().map(x=>
    `<tr><td>#${x.tick}</td><td>act ${fmt(x.active)} · buf ${fmt(x.buffer)} · exp ${fmt(x.experiences)}</td></tr>`
  ).join('') + '</table>';
}

function short(s,n){ s=String(s||''); return s.length>n ? s.slice(0,n)+'…' : s; }
function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function tick(){
  try{
    const r = await fetch('/api/snapshot', {cache:'no-store'});
    const d = await r.json();
    okCount++;
    $('#dot').classList.add('on');
    const st = d.state||{};
    $('#hdr').textContent = `neurons ${st.neurons||'—'} · tracts ${st.tracts||'—'} · active ${st.active||'—'}`;
    $('#poll').textContent = '已刷新 ' + okCount + ' 次';
    draw(st); drawStats(d.stats); drawRows(d.results);
    drawSpark(d.results||[]); drawMetrics(d.metrics);
    $('#nres').textContent = d.n_results!==undefined ? `（${d.n_results} 条）` : '';
  }catch(e){
    okCount=0; $('#dot').classList.remove('on');
    $('#poll').textContent = '连接失败';
  }
}

async function inject(){
  const t = $('#txt').value.trim();
  if(!t) return;
  $('#injMsg').textContent='发送中…';
  try{
    const r = await fetch('/api/inject', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text:t})});
    const d = await r.json();
    $('#injMsg').textContent = d.ok ? ('✓ 已注入：'+d.injected) : ('✗ '+(d.err||d.error||'失败'));
    if(d.ok) $('#txt').value='';
  }catch(e){ $('#injMsg').textContent = '✗ '+e; }
}

$('#txt').addEventListener('keydown', e=>{ if(e.key==='Enter') inject(); });
setInterval(tick, 1200); tick();
</script></body></html>
"""
