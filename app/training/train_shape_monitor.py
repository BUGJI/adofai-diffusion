"""train_shape_monitor.py — 实时训练进度页（只读 train_shape.log，不干扰训练）。

启动：venv/Scripts/python.exe app/training/train_shape_monitor.py
端口：8099
浏览器打开 http://localhost:8099 即可看实时 loss 曲线 + 提特征进度。
"""
from __future__ import annotations
import os, re, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # app/
LOG = ROOT.parent / "data" / "checkpoints" / "train_shape.log"

EP_RE = re.compile(r"ep (\d+)/(\d+)\s+loss=([\d.]+)")
FEAT_RE = re.compile(r"\[feat\] (\d+)/(\d+)\s+(.*)")
DONE_RE = re.compile(r"done tiles=(\d+) \(累计样本 (\d+)\)")
SKIP_RE = re.compile(r"\[skip\]")
SAVE_RE = re.compile(r"已保存 -> (.*)")

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>ShapeModel 训练实时监控</title>
<style>
  body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:18px}
  h1{font-size:18px;margin:0 0 10px}
  .cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
  .card{background:#1a1d24;border:1px solid #2a2e38;border-radius:10px;padding:12px 16px;min-width:150px}
  .card .k{font-size:12px;color:#8b93a1;margin-bottom:4px}
  .card .v{font-size:22px;font-weight:700}
  .status{color:#ffcc66}
  .ok{color:#5ee07a}
  canvas{background:#1a1d24;border:1px solid #2a2e38;border-radius:10px;width:100%;height:300px}
  pre{background:#161922;border:1px solid #2a2e38;border-radius:8px;padding:10px;height:240px;overflow:auto;font-size:12px;line-height:1.5;color:#9fb4c9}
  .bar{height:8px;background:#2a2e38;border-radius:4px;overflow:hidden;margin-top:6px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#5ee07a);width:0}
</style></head><body>
<h1>🛠️ ShapeModel 训练实时监控</h1>
<div class="cards">
  <div class="card"><div class="k">阶段</div><div class="v status" id="phase">—</div></div>
  <div class="card"><div class="k">提取特征</div><div class="v" id="feat">0/0</div><div class="bar"><i id="featbar"></i></div></div>
  <div class="card"><div class="k">当前 Epoch</div><div class="v" id="ep">0/0</div></div>
  <div class="card"><div class="k">最新 loss</div><div class="v" id="loss">—</div></div>
  <div class="card"><div class="k">样本数</div><div class="v" id="samples">0</div></div>
  <div class="card"><div class="k">状态</div><div class="v" id="state">运行中</div></div>
</div>
<canvas id="chart"></canvas>
<h3 style="font-size:14px;margin:14px 0 6px;color:#8b93a1">实时日志（末尾）</h3>
<pre id="log"></pre>
<script>
const cv=document.getElementById('chart'),ctx=cv.getContext('2d');
let lossPts=[];
function draw(){
  const w=cv.width=cv.clientWidth*devicePixelRatio, h=cv.height=cv.clientHeight*devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  if(lossPts.length<1) return;
  const max=Math.max(...lossPts), min=Math.min(...lossPts);
  const pad=20*dpr;
  ctx.strokeStyle='#2a2e38';ctx.lineWidth=1*dpr;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke();
  ctx.strokeStyle='#5ee07a';ctx.lineWidth=2*dpr;ctx.beginPath();
  lossPts.forEach((v,i)=>{
    const x=pad+(w-2*pad)*(i/(lossPts.length-1||1));
    const y=h-pad-(h-2*pad)*((v-min)/((max-min)||1));
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  });
  ctx.stroke();
  ctx.fillStyle='#8b93a1';ctx.font=(12*dpr)+'px sans-serif';
  ctx.fillText('loss '+min.toFixed(3),pad,h-pad+16*dpr);
  ctx.fillText('loss '+max.toFixed(3),pad,pad-6*dpr);
}
const dpr=devicePixelRatio||1;
async function poll(){
  try{
    const r=await fetch('/api');const d=await r.json();
    document.getElementById('phase').textContent=d.phase;
    document.getElementById('feat').textContent=d.feat_cur+'/'+d.feat_total;
    document.getElementById('featbar').style.width=(d.feat_total?100*d.feat_cur/d.feat_total:0)+'%';
    document.getElementById('ep').textContent=d.ep_cur+'/'+d.ep_total;
    document.getElementById('loss').textContent=d.loss?d.loss.toFixed(4):'—';
    document.getElementById('samples').textContent=d.samples;
    document.getElementById('state').textContent=d.done?'已完成 ✅':'运行中';
    document.getElementById('state').className=d.done?'v ok':'v';
    document.getElementById('log').textContent=d.tail.join('\\n');
    lossPts=d.loss_hist;draw();
  }catch(e){}
}
setInterval(poll,1500);poll();
</script>
</body></html>"""


def parse():
    if not LOG.exists():
        return {"phase": "无日志", "feat_cur": 0, "feat_total": 0, "ep_cur": 0,
                "ep_total": 0, "loss": None, "samples": 0, "done": False,
                "loss_hist": [], "tail": []}
    text = LOG.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    feat_cur = feat_total = ep_cur = ep_total = samples = 0
    loss = None
    loss_hist = []
    done = False
    for ln in lines:
        m = FEAT_RE.search(ln)
        if m:
            feat_cur = int(m.group(1)); feat_total = int(m.group(2))
        m = EP_RE.search(ln)
        if m:
            ep_cur = int(m.group(1)); ep_total = int(m.group(2))
            loss = float(m.group(3)); loss_hist.append(loss)
        m = DONE_RE.search(ln)
        if m:
            samples = int(m.group(2))
        if SAVE_RE.search(ln):
            done = True
    if done:
        phase = "训练完成"
    elif ep_total:
        phase = "训练 epoch"
    elif feat_total:
        phase = "提取特征"
    else:
        phase = "启动中"
    tail = lines[-40:]
    return {"phase": phase, "feat_cur": feat_cur, "feat_total": feat_total,
            "ep_cur": ep_cur, "ep_total": ep_total, "loss": loss,
            "samples": samples, "done": done, "loss_hist": loss_hist, "tail": tail}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(parse(), ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = 8099
    print(f"[monitor] http://localhost:{port}  (log={LOG})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
