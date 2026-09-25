"""
Generator for llm_output/seq_models_cnn_rnn_wavenet_grok.html

Stdlib-only (no numpy/matplotlib). Run with:  py .kiro/build_seq_models_html.py

Draws three sequence architectures on the SAME 8-char input:
  - CNN   (causal 1D conv, kernel 2)   -> local, parallel, shallow
  - RNN   (recurrent cell, shared)     -> chain / baton, whole history, sequential
  - WaveNet (dilated causal conv tree) -> log-depth tree, whole history, parallel

Interactive: click any node -> highlights every input that influences it
(its "receptive field"), plus the path. Shows the core difference visually.
"""

import json

CHARS = list("machines")  # 8 input tokens
N = len(CHARS)

# input x positions (shared layout across panels)
X0, STEP = 70, 96
XS = [X0 + i * STEP for i in range(N)]
NW, NH = 54, 34  # node box size

nodes = {}   # id -> dict(cx, cy, label, cls, panel)
edges = []   # (src_id, dst_id)


def add(nid, cx, cy, label, cls, panel):
    nodes[nid] = dict(cx=cx, cy=cy, label=label, cls=cls, panel=panel)


def link(a, b):
    edges.append((a, b))


def mid(a, b):
    return (a + b) / 2.0


# ----------------------------------------------------------------- CNN panel
# inputs (top) -> outputs (below). Each output sees a window of 2 inputs.
CNN_IN_Y, CNN_OUT_Y = 60, 180
for i, ch in enumerate(CHARS):
    add(f"cnn_in{i}", XS[i], CNN_IN_Y, ch, "n-input", "cnn")
for t in range(1, N):  # output t sees inputs t-1, t
    ox = mid(XS[t - 1], XS[t])
    add(f"cnn_o{t}", ox, CNN_OUT_Y, f"o{t}", "n-cnn", "cnn")
    link(f"cnn_in{t-1}", f"cnn_o{t}")
    link(f"cnn_in{t}", f"cnn_o{t}")

# ----------------------------------------------------------------- RNN panel
# inputs (top) -> hidden chain (middle) -> final prediction
RNN_IN_Y, RNN_H_Y, RNN_PRED_Y = 60, 180, 300
for i, ch in enumerate(CHARS):
    add(f"rnn_in{i}", XS[i], RNN_IN_Y, ch, "n-input", "rnn")
for t in range(N):
    add(f"rnn_h{t}", XS[t], RNN_H_Y, f"h{t+1}", "n-rnn", "rnn")
    link(f"rnn_in{t}", f"rnn_h{t}")          # input feeds cell
    if t > 0:
        link(f"rnn_h{t-1}", f"rnn_h{t}")     # the baton
add("rnn_pred", XS[N - 1], RNN_PRED_Y, "next", "n-pred", "rnn")
link(f"rnn_h{N-1}", "rnn_pred")

# ------------------------------------------------------------- WaveNet panel
# inputs (bottom) -> tree fuses pairs up to a single node that sees all 8
WN_IN_Y, WN_L1_Y, WN_L2_Y, WN_L3_Y = 300, 210, 120, 40
for i, ch in enumerate(CHARS):
    add(f"wn_in{i}", XS[i], WN_IN_Y, ch, "n-input", "wn")
l1 = []
for k in range(N // 2):
    x = mid(XS[2 * k], XS[2 * k + 1])
    nid = f"wn_l1_{k}"
    add(nid, x, WN_L1_Y, "·", "n-wl1", "wn")
    link(f"wn_in{2*k}", nid)
    link(f"wn_in{2*k+1}", nid)
    l1.append((nid, x))
l2 = []
for j in range(len(l1) // 2):
    x = mid(l1[2 * j][1], l1[2 * j + 1][1])
    nid = f"wn_l2_{j}"
    add(nid, x, WN_L2_Y, "·", "n-wl2", "wn")
    link(l1[2 * j][0], nid)
    link(l1[2 * j + 1][0], nid)
    l2.append((nid, x))
x_top = mid(l2[0][1], l2[1][1])
add("wn_top", x_top, WN_L3_Y, "next", "n-wl3", "wn")
link(l2[0][0], "wn_top")
link(l2[1][0], "wn_top")

# ----------------------------------------------------------- build parents map
parents = {}
for a, b in edges:
    parents.setdefault(b, []).append(a)

meta = {nid: {"panel": d["panel"], "label": d["label"]} for nid, d in nodes.items()}
input_ids = [nid for nid in nodes if "_in" in nid]


def endpoints(src, dst):
    s, d = nodes[src], nodes[dst]
    if abs(s["cy"] - d["cy"]) < 1:  # horizontal baton
        return (s["cx"] + NW / 2, s["cy"], d["cx"] - NW / 2, d["cy"])
    if d["cy"] > s["cy"]:           # target below
        return (s["cx"], s["cy"] + NH / 2, d["cx"], d["cy"] - NH / 2)
    return (s["cx"], s["cy"] - NH / 2, d["cx"], d["cy"] + NH / 2)  # target above


def render_panel(panel, width, height):
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px">',
        '<defs><marker id="arw" markerWidth="9" markerHeight="9" refX="7" '
        'refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/>'
        '</marker></defs>',
    ]
    # edges first
    for a, b in edges:
        if nodes[a]["panel"] != panel:
            continue
        x1, y1, x2, y2 = endpoints(a, b)
        eid = f"edge__{a}__{b}"
        parts.append(
            f'<line id="{eid}" class="edge" x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" marker-end="url(#arw)"/>'
        )
    # nodes
    for nid, d in nodes.items():
        if d["panel"] != panel:
            continue
        x = d["cx"] - NW / 2
        y = d["cy"] - NH / 2
        parts.append(
            f'<g class="node" onclick="showRF(\'{nid}\')" '
            f'style="cursor:pointer">'
            f'<rect id="node__{nid}" class="{d["cls"]}" x="{x:.1f}" '
            f'y="{y:.1f}" width="{NW}" height="{NH}" rx="7"/>'
            f'<text x="{d["cx"]:.1f}" y="{d["cy"]+5:.1f}" '
            f'text-anchor="middle">{d["label"]}</text></g>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


svg_cnn = render_panel("cnn", 860, 230)
svg_rnn = render_panel("rnn", 860, 350)
svg_wn = render_panel("wn", 860, 350)

DATA = (
    "const PARENTS=" + json.dumps(parents) + ";\n"
    "const META=" + json.dumps(meta) + ";\n"
    "const INPUTS=" + json.dumps(input_ids) + ";\n"
)

JS = """
function clearHl(){
  document.querySelectorAll('.hl-node').forEach(e=>e.classList.remove('hl-node'));
  document.querySelectorAll('.hl-edge').forEach(e=>e.classList.remove('hl-edge'));
}
function showRF(id){
  clearHl();
  const seen = new Set();
  const stack = [id];
  seen.add(id);
  while(stack.length){
    const n = stack.pop();
    (PARENTS[n]||[]).forEach(p=>{
      const eln = document.getElementById('edge__'+p+'__'+n);
      if(eln) eln.classList.add('hl-edge');
      if(!seen.has(p)){ seen.add(p); stack.push(p); }
    });
  }
  let nInputs = 0;
  seen.forEach(n=>{
    const el = document.getElementById('node__'+n);
    if(el) el.classList.add('hl-node');
    if(INPUTS.includes(n)) nInputs++;
  });
  const panel = META[id].panel;
  const names = {cnn:'CNN', rnn:'RNN', wn:'WaveNet'};
  document.getElementById('banner').innerHTML =
    '<b>'+names[panel]+'</b> \\u2014 node <b>'+META[id].label+
    '</b> is influenced by <b>'+nInputs+' of 8</b> inputs.';
}
"""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CNN vs RNN vs WaveNet</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;
       margin:0 auto;padding:24px;color:#1e293b;background:#f8fafc;line-height:1.55}
  h1{font-size:24px} h2{font-size:19px;margin-top:6px}
  .panel{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
         padding:18px 20px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}
  .tag{display:inline-block;font-size:12px;font-weight:600;padding:2px 10px;
       border-radius:999px;margin-left:8px;vertical-align:middle}
  .t-local{background:#fef3c7;color:#92400e}
  .t-chain{background:#dcfce7;color:#166534}
  .t-tree{background:#f3e8ff;color:#6b21a8}
  .lead{color:#475569;font-size:14px;margin:2px 0 10px}
  text{font-size:14px;font-weight:600;fill:#0f172a;pointer-events:none}
  .edge{stroke:#cbd5e1;stroke-width:1.6}
  .n-input{fill:#dbeafe;stroke:#3b82f6;stroke-width:1.6}
  .n-cnn{fill:#fde68a;stroke:#d97706;stroke-width:1.6}
  .n-rnn{fill:#bbf7d0;stroke:#16a34a;stroke-width:1.6}
  .n-pred{fill:#fecaca;stroke:#dc2626;stroke-width:1.6}
  .n-wl1{fill:#e9d5ff;stroke:#9333ea;stroke-width:1.6}
  .n-wl2{fill:#ddd6fe;stroke:#7c3aed;stroke-width:1.6}
  .n-wl3{fill:#c4b5fd;stroke:#6d28d9;stroke-width:1.6}
  .hl-node{fill:#fb923c!important;stroke:#c2410c!important;stroke-width:2.4!important}
  .hl-edge{stroke:#f97316!important;stroke-width:3.2!important}
  #banner{position:sticky;top:0;background:#0f172a;color:#e2e8f0;padding:10px 16px;
          border-radius:10px;font-size:14px;z-index:5}
  .hint{font-size:13px;color:#64748b;font-style:italic}
  .legend{font-size:12.5px;color:#475569;margin-top:8px}
  .sw{display:inline-block;width:12px;height:12px;border-radius:3px;
      vertical-align:middle;margin:0 4px 0 10px}
  code{background:#eef2ff;padding:1px 6px;border-radius:5px;font-size:13px}
</style></head><body>

<h1>Three ways to read a sequence</h1>
<p class="lead">Same 8-character input <b>m a c h i n e s</b> in every panel. Blue = input
characters. Colored boxes = computed values (same color = <b>same shared weights</b>).
<b>Click any node</b> to light up which inputs it actually depends on.</p>
<div id="banner">Click a colored node to see its receptive field (which inputs feed it).</div>

<div class="panel">
  <h2>1. CNN <span class="tag t-local">local &middot; parallel &middot; shallow</span></h2>
  <p class="lead">One small filter (kernel 2) slides across. Each output is computed
  <b>independently</b> from a fixed 2-input window &mdash; no baton passed between them,
  so they could all be computed at once. Click an <code>o</code> box: only its 2 inputs
  light up. Reach is fixed and small.</p>
  __CNN__
  <div class="legend"><span class="sw" style="background:#fde68a"></span>
  all outputs share the <b>one</b> filter's weights.</div>
</div>

<div class="panel">
  <h2>2. RNN <span class="tag t-chain">whole history &middot; sequential &middot; deep</span></h2>
  <p class="lead">One cell, reused at every step, carries a running memory
  <code>h</code> forward &mdash; the baton (green horizontal arrows). Each
  <code>h</code> = <code>tanh(Linear(concat(prev h, this input)))</code>. Click the red
  <b>next</b> node: <b>all 8</b> inputs light up, because the memory chains back through
  every step. That long chain is also why it is a "very very deep network" and why it's
  slow (step 3 must wait for step 2).</p>
  __RNN__
  <div class="legend"><span class="sw" style="background:#bbf7d0"></span>
  every hidden box reuses the <b>same one</b> cell's weights.</div>
</div>

<div class="panel">
  <h2>3. WaveNet <span class="tag t-tree">whole history &middot; parallel &middot; log-depth</span></h2>
  <p class="lead">A tree of small conv layers fuses pairs, then pairs-of-pairs, doubling
  reach at each level. Click the top <b>next</b> node: <b>all 8</b> inputs light up like
  the RNN &mdash; but it got there in only <b>3 layers</b> (log&#8322;8), and each layer's
  boxes are independent so they run in parallel like a CNN. WaveNet is the bridge:
  RNN's long memory, CNN's parallel speed.</p>
  __WN__
  <div class="legend">
  <span class="sw" style="background:#e9d5ff"></span>layer 1
  <span class="sw" style="background:#ddd6fe"></span>layer 2
  <span class="sw" style="background:#c4b5fd"></span>layer 3
  &mdash; weights shared <i>within</i> a layer (across positions), different per layer.</div>
</div>

<div class="panel" style="background:#0f172a;color:#e2e8f0">
  <h2 style="color:#fff">The one-glance summary</h2>
  <p style="color:#cbd5e1;font-size:14px">To let position <b>t</b> depend on the very
  first character:<br>
  &bull; <b>CNN</b> (single layer): <b>can't</b> &mdash; reach is stuck at the window size.<br>
  &bull; <b>RNN</b>: yes, but the signal travels <b>8 sequential steps</b> down the chain
  (deep &rarr; gradients fade &mdash; the disease the LSTM cures).<br>
  &bull; <b>WaveNet</b>: yes, in <b>3 parallel layers</b> (log-depth tree).</p>
</div>

<script>
__DATA__
__JS__
</script>
</body></html>
"""

HTML = (HTML
        .replace("__CNN__", svg_cnn)
        .replace("__RNN__", svg_rnn)
        .replace("__WN__", svg_wn)
        .replace("__DATA__", DATA)
        .replace("__JS__", JS))

with open("llm_output/seq_models_cnn_rnn_wavenet_grok.html", "w", encoding="utf-8") as f:
    f.write(HTML)

print("wrote llm_output/seq_models_cnn_rnn_wavenet_grok.html")
print("nodes:", len(nodes), "edges:", len(edges))
