"""
Stdlib-only generator -> llm_output/bigram_forward_pipeline_grok.html
Visualizes the bigram forward pass for the dummy word "ravi":
  "ravi" -> indices [1,4,2,5] -> Embedding layer (VxV) -> logits (4xV)
         -> softmax -> probs (4xV) -> cross-entropy -> loss
All numbers computed with real math (random.gauss seed 1337), so exact.
Run with:  py .kiro/build_bigram_forward_html.py
"""
import math, random

random.seed(1337)

# ---- dummy setup -----------------------------------------------------------
vocab = ['.', 'r', 'v', 'x', 'a', 'i']   # index 0..5
V = len(vocab)
word = "ravi"
idx = [1, 4, 2, 5]        # r=1, a=4, v=2, i=5   (the dummy numbers)
targets = [4, 2, 5, 0]    # bigram next-char: r->a, a->v, v->i, i->.

# ---- embedding table (the "layer"): V x V of small random numbers ----------
E = [[round(random.gauss(0, 1), 2) for _ in range(V)] for _ in range(V)]

# ---- forward math ----------------------------------------------------------
logits = [E[i][:] for i in idx]                      # (4, V) plucked rows

def softmax_naive(row):
    exps = [math.exp(v) for v in row]
    s = sum(exps)
    return exps, s, [e / s for e in exps]

probs, exps_all, sums_all = [], [], []
for r in logits:
    e, s, p = softmax_naive(r)
    exps_all.append(e); sums_all.append(s); probs.append(p)

losses = [-math.log(probs[p][targets[p]]) for p in range(len(idx))]
loss = sum(losses) / len(losses)

def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"

# ============================================================================
# SVG helpers
# ============================================================================
def matrix_svg(x0, y0, data, cw=56, ch=30, hi_rows=None, hi_cols=None,
               row_labels=None, col_labels=None, star_cells=None,
               base="#ffffff", hi_fill="#ffe6c7", stroke="#94a3b8",
               fontsize=13, num_fmt=f2):
    hi_rows = hi_rows or []
    star_cells = star_cells or []
    s = []
    R = len(data); C = len(data[0])
    lx = x0
    ly = y0
    # column labels
    if col_labels:
        for c, lab in enumerate(col_labels):
            s.append(f'<text x="{lx + c*cw + cw/2}" y="{ly-8}" text-anchor="middle" '
                     f'font-size="12" fill="#64748b" font-weight="600">{lab}</text>')
    for r in range(R):
        if row_labels:
            s.append(f'<text x="{lx-10}" y="{ly + r*ch + ch/2 + 4}" text-anchor="end" '
                     f'font-size="12" fill="#64748b" font-weight="600">{row_labels[r]}</text>')
        for c in range(C):
            fill = hi_fill if r in hi_rows else base
            cx = lx + c*cw
            cy = ly + r*ch
            s.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            if (r, c) in star_cells:
                s.append(f'<rect x="{cx+1.5}" y="{cy+1.5}" width="{cw-3}" height="{ch-3}" '
                         f'fill="none" stroke="#dc2626" stroke-width="3"/>')
            s.append(f'<text x="{cx+cw/2}" y="{cy+ch/2+4}" text-anchor="middle" '
                     f'font-size="{fontsize}" fill="#1e293b">{num_fmt(data[r][c])}</text>')
    return "\n".join(s)

def mini_grid(x0, y0, R, C, cell=16, hi_rows=None, base="#eef2f7",
              hi_fill="#f59e0b", stroke="#cbd5e1"):
    hi_rows = hi_rows or []
    s = []
    for r in range(R):
        for c in range(C):
            fill = hi_fill if r in hi_rows else base
            s.append(f'<rect x="{x0+c*cell}" y="{y0+r*cell}" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="0.7"/>')
    return "\n".join(s)

def arrow(x1, y, x2, label="", color="#475569"):
    s = [f'<line x1="{x1}" y1="{y}" x2="{x2-9}" y2="{y}" stroke="{color}" stroke-width="2"/>',
         f'<path d="M {x2-9},{y-5} L {x2},{y} L {x2-9},{y+5} Z" fill="{color}"/>']
    if label:
        s.append(f'<text x="{(x1+x2)/2}" y="{y-10}" text-anchor="middle" '
                 f'font-size="11.5" fill="{color}" font-weight="600">{label}</text>')
    return "\n".join(s)

# ============================================================================
# 1) MASTER PIPELINE (everything at once)
# ============================================================================
def master_svg():
    W, H = 1330, 320
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="min-width:{W}px">']
    cy = 150

    # "ravi" token box
    s.append(f'<rect x="20" y="120" width="90" height="60" rx="8" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>')
    s.append(f'<text x="65" y="156" text-anchor="middle" font-size="26" font-weight="700" fill="#1e40af">ravi</text>')
    s.append(f'<text x="65" y="200" text-anchor="middle" font-size="11" fill="#64748b">the word (T=4)</text>')

    s.append(arrow(112, cy, 175, "encode"))

    # indices column
    ix = 180
    for r, v in enumerate(idx):
        yy = 95 + r*30
        s.append(f'<rect x="{ix}" y="{yy}" width="46" height="30" fill="#eff6ff" stroke="#3b82f6"/>')
        s.append(f'<text x="{ix+23}" y="{yy+20}" text-anchor="middle" font-size="14" fill="#1e40af" font-weight="600">{v}</text>')
        s.append(f'<text x="{ix-8}" y="{yy+20}" text-anchor="end" font-size="12" fill="#64748b">{word[r]}</text>')
    s.append(f'<text x="{ix+23}" y="235" text-anchor="middle" font-size="11" fill="#64748b">idx (T,)</text>')

    s.append(arrow(230, cy, 300, "look up rows"))

    # embedding layer (the LAYER in between) - big highlighted box
    ex, ey = 305, 70
    gw = V*16
    s.append(f'<rect x="{ex-14}" y="{ey-30}" width="{gw+28}" height="{V*16+70}" rx="10" '
             f'fill="#fff7ed" stroke="#f59e0b" stroke-width="2.5"/>')
    s.append(f'<text x="{ex+gw/2}" y="{ey-12}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b45309">EMBEDDING LAYER</text>')
    s.append(mini_grid(ex, ey, V, V, cell=16, hi_rows=idx))
    s.append(f'<text x="{ex+gw/2}" y="{ey+V*16+22}" text-anchor="middle" font-size="11" fill="#b45309">weights  (V x V) = ({V} x {V})</text>')
    s.append(f'<text x="{ex+gw/2}" y="{ey+V*16+38}" text-anchor="middle" font-size="10.5" fill="#92400e">orange rows = the 4 plucked</text>')

    s.append(arrow(305+gw+20, cy, 470, "pluck 4 rows"))

    # logits mini grid (4 x V)
    lx, ly = 478, 108
    s.append(mini_grid(lx, ly, 4, V, cell=16, base="#ede9fe", hi_fill="#8b5cf6", stroke="#c4b5fd"))
    s.append(f'<text x="{lx+V*16/2}" y="{ly-8}" text-anchor="middle" font-size="12" font-weight="700" fill="#6d28d9">logits</text>')
    s.append(f'<text x="{lx+V*16/2}" y="{ly+4*16+16}" text-anchor="middle" font-size="11" fill="#6d28d9">(T x V) = (4 x {V})</text>')

    s.append(arrow(lx+V*16+6, cy, 660, "softmax", color="#059669"))
    s.append(f'<text x="{(lx+V*16+6+660)/2}" y="{cy+16}" text-anchor="middle" font-size="10" fill="#059669">exp &#247; sum</text>')

    # probs mini grid (4 x V)
    px, py = 668, 108
    s.append(mini_grid(px, py, 4, V, cell=16, base="#dcfce7", hi_fill="#22c55e", stroke="#86efac"))
    s.append(f'<text x="{px+V*16/2}" y="{py-8}" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">probs</text>')
    s.append(f'<text x="{px+V*16/2}" y="{py+4*16+16}" text-anchor="middle" font-size="11" fill="#15803d">(T x V), rows sum to 1</text>')

    s.append(arrow(px+V*16+6, cy, 900, "pick target", color="#dc2626"))
    s.append(f'<text x="{(px+V*16+6+900)/2}" y="{cy+16}" text-anchor="middle" font-size="10" fill="#dc2626">-log, mean</text>')

    # targets column
    tx = 905
    for r, v in enumerate(targets):
        yy = 95 + r*30
        s.append(f'<rect x="{tx}" y="{yy}" width="46" height="30" fill="#fee2e2" stroke="#dc2626"/>')
        s.append(f'<text x="{tx+23}" y="{yy+20}" text-anchor="middle" font-size="14" fill="#991b1b" font-weight="600">{v}</text>')
    s.append(f'<text x="{tx+23}" y="235" text-anchor="middle" font-size="11" fill="#991b1b">targets (T,)</text>')
    s.append(f'<text x="{tx+23}" y="80" text-anchor="middle" font-size="11" fill="#991b1b">true next</text>')

    s.append(arrow(tx+50, cy, 1030, ""))

    # loss box
    s.append(f'<rect x="1035" y="118" width="120" height="64" rx="10" fill="#1e293b"/>')
    s.append(f'<text x="1095" y="145" text-anchor="middle" font-size="12" fill="#94a3b8">cross-entropy</text>')
    s.append(f'<text x="1095" y="170" text-anchor="middle" font-size="22" font-weight="700" fill="#fff">{f3(loss)}</text>')

    s.append('</svg>')
    return "\n".join(s)

# ============================================================================
# 2) detailed: encoding
# ============================================================================
def encode_svg():
    W, H = 620, 150
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:{W}px">']
    # vocab table
    s.append(f'<text x="10" y="20" font-size="13" font-weight="700" fill="#334155">the vocab (V={V})</text>')
    for i, ch in enumerate(vocab):
        x = 10 + i*70
        s.append(f'<rect x="{x}" y="32" width="60" height="34" fill="#f1f5f9" stroke="#94a3b8"/>')
        s.append(f'<text x="{x+30}" y="54" text-anchor="middle" font-size="15" fill="#1e293b" font-weight="600">\'{ch}\'</text>')
        s.append(f'<text x="{x+30}" y="84" text-anchor="middle" font-size="12" fill="#64748b">idx {i}</text>')
    # arrow to ravi
    s.append(f'<text x="10" y="118" font-size="13" fill="#334155">"ravi" &#8594; look up each char &#8594; '
             f'<tspan font-weight="700" fill="#1e40af">[1, 4, 2, 5]</tspan></text>')
    s.append('</svg>')
    return "\n".join(s)

# ============================================================================
# 3) detailed: embedding table + pluck -> logits
# ============================================================================
def embed_svg():
    W, H = 900, 300
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:{W}px">']
    col_labels = [f"'{c}'" for c in vocab]
    row_labels = [f"'{c}'  {i}" for i, c in enumerate(vocab)]
    s.append(f'<text x="40" y="24" font-size="13" font-weight="700" fill="#b45309">Embedding weight matrix E  (V x V = {V} x {V})</text>')
    s.append(matrix_svg(120, 50, E, cw=56, ch=30, hi_rows=idx,
                        row_labels=row_labels, col_labels=col_labels))
    s.append(f'<text x="120" y="{50+V*30+28}" font-size="12" fill="#64748b">'
             f'Each ROW = "given this char, my raw scores for the next char." '
             f'We only pluck rows {idx} (r,a,v,i).</text>')
    s.append('</svg>')
    return "\n".join(s)

def logits_svg():
    W, H = 620, 220
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:{W}px">']
    col_labels = [f"'{c}'" for c in vocab]
    row_labels = [f"{word[r]} ({idx[r]})" for r in range(4)]
    s.append(f'<text x="90" y="22" font-size="13" font-weight="700" fill="#6d28d9">logits = the 4 plucked rows  (T x V = 4 x {V})</text>')
    s.append(matrix_svg(120, 46, logits, cw=56, ch=30, hi_rows=[0,1,2,3],
                        row_labels=row_labels, col_labels=col_labels,
                        hi_fill="#ede9fe", stroke="#c4b5fd"))
    s.append('</svg>')
    return "\n".join(s)

# ============================================================================
# 4) detailed: softmax on ONE row (last one: 'i')
# ============================================================================
def softmax_svg():
    p = 3  # position of 'i'
    row = logits[p]; e = exps_all[p]; ssum = sums_all[p]; pr = probs[p]
    W, H = 620, 260
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:{W}px">']
    col_labels = [f"'{c}'" for c in vocab]
    s.append(f'<text x="120" y="20" font-size="13" font-weight="700" fill="#334155">softmax on row for \'i\' (idx 5)</text>')
    s.append(matrix_svg(120, 34, [row], cw=56, ch=28, col_labels=col_labels,
                        row_labels=["logit"], hi_fill="#ede9fe"))
    s.append(f'<text x="60" y="98" font-size="12" fill="#059669">1) exp() each &#8594;</text>')
    s.append(matrix_svg(120, 86, [e], cw=56, ch=28, row_labels=["exp"], hi_fill="#dcfce7", num_fmt=f3))
    s.append(f'<text x="120" y="150" font-size="12" fill="#059669">2) sum of exps = {f3(ssum)}</text>')
    s.append(f'<text x="60" y="186" font-size="12" fill="#15803d">3) &#247; sum &#8594;</text>')
    s.append(matrix_svg(120, 174, [pr], cw=56, ch=28, row_labels=["prob"], hi_fill="#bbf7d0", num_fmt=f3))
    s.append(f'<text x="120" y="236" font-size="12" fill="#64748b">these are probabilities: they sum to 1.00. No weights here &#8212; pure arithmetic.</text>')
    s.append('</svg>')
    return "\n".join(s)

# ============================================================================
# 5) detailed: cross-entropy over all 4 positions
# ============================================================================
def ce_svg():
    W, H = 720, 300
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:{W}px">']
    col_labels = [f"'{c}'" for c in vocab]
    row_labels = [f"{word[r]}&#8594;{vocab[targets[r]]}" for r in range(4)]
    stars = [(r, targets[r]) for r in range(4)]
    s.append(f'<text x="120" y="20" font-size="13" font-weight="700" fill="#334155">'
             f'probs (4 x {V}) &#8212; red box = probability at the TRUE next char</text>')
    s.append(matrix_svg(120, 40, probs, cw=56, ch=30, col_labels=col_labels,
                        row_labels=row_labels, star_cells=stars,
                        hi_fill="#dcfce7", num_fmt=f3))
    yb = 40 + 4*30 + 30
    for r in range(4):
        ptrue = probs[r][targets[r]]
        s.append(f'<text x="120" y="{yb + r*24}" font-size="12" fill="#334155">'
                 f'pos {r} ({word[r]}&#8594;{vocab[targets[r]]}):  -log({f3(ptrue)}) = '
                 f'<tspan font-weight="700" fill="#dc2626">{f3(losses[r])}</tspan></text>')
    s.append(f'<text x="120" y="{yb + 4*24 + 14}" font-size="13" fill="#1e293b">'
             f'loss = mean(...) = <tspan font-weight="700" fill="#dc2626" font-size="15">{f3(loss)}</tspan></text>')
    s.append('</svg>')
    return "\n".join(s)

# ============================================================================
# assemble HTML
# ============================================================================
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bigram forward pass: "ravi"</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background:#f8fafc; color:#1e293b; line-height:1.55; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 80px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color:#64748b; margin: 0 0 22px; font-size:14px; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px;
          padding:20px 22px; margin:18px 0; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  .card h2 {{ font-size:17px; margin:0 0 4px; }}
  .card .note {{ color:#475569; font-size:14px; margin:6px 0 14px; }}
  .scroll {{ overflow-x:auto; }}
  .master {{ background:linear-gradient(180deg,#ffffff,#f1f5f9);
             border:2px solid #cbd5e1; }}
  .legend {{ font-size:12.5px; color:#64748b; margin-top:8px; }}
  .legend span {{ display:inline-block; margin-right:14px; }}
  .chip {{ display:inline-block; width:12px; height:12px; border-radius:3px;
           vertical-align:-1px; margin-right:4px; border:1px solid #94a3b8;}}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:13px;}}
  .key {{ background:#fffbeb; border-left:4px solid #f59e0b; padding:10px 14px;
          border-radius:6px; font-size:14px; margin-top:14px;}}
</style></head>
<body><div class="wrap">
  <h1>Bigram forward pass, spelled out with "ravi"</h1>
  <p class="sub">Dummy vocab of {V} chars. Watch the shapes change: word &#8594; indices &#8594; through the
     <b>embedding layer</b> &#8594; logits &#8594; softmax &#8594; probs &#8594; one loss number.</p>

  <div class="card master">
    <h2>The whole pipeline at once</h2>
    <p class="note">The <b style="color:#b45309">orange box</b> is the only layer with weights.
       Everything after it (softmax, -log, mean) is pure arithmetic with nothing to learn.</p>
    <div class="scroll">{master_svg()}</div>
    <div class="legend">
      <span><i class="chip" style="background:#dbeafe"></i>token / indices</span>
      <span><i class="chip" style="background:#f59e0b"></i>embedding layer (weights)</span>
      <span><i class="chip" style="background:#8b5cf6"></i>logits</span>
      <span><i class="chip" style="background:#22c55e"></i>probs</span>
      <span><i class="chip" style="background:#fee2e2"></i>targets</span>
    </div>
  </div>

  <div class="card">
    <h2>Step 1 &mdash; encode: chars become numbers</h2>
    <p class="note">Every char is just its position in the vocab. "ravi" &#8594; <code>[1, 4, 2, 5]</code>.</p>
    <div class="scroll">{encode_svg()}</div>
  </div>

  <div class="card">
    <h2>Step 2 &mdash; the embedding layer: a lookup table of rows</h2>
    <p class="note">This is the matrix you asked about. It's <b>V &times; V</b>. Row <code>i</code> holds the raw
       next-char scores for token <code>i</code>. The indices just <b>pluck out rows</b> &mdash; no matrix multiply.</p>
    <div class="scroll">{embed_svg()}</div>
    <div class="key">Feeding <code>[1,4,2,5]</code> grabs rows 1, 4, 2, 5 (the orange ones) and stacks
       them &#8594; that stack <b>is</b> the logits.</div>
  </div>

  <div class="card">
    <h2>Step 3 &mdash; logits: the plucked rows (T &times; V)</h2>
    <p class="note">Shape <code>(4, {V})</code>: one row per input char, {V} raw scores each. Still just numbers &mdash;
       not probabilities yet.</p>
    <div class="scroll">{logits_svg()}</div>
  </div>

  <div class="card">
    <h2>Step 4 &mdash; softmax turns one row of logits into probabilities</h2>
    <p class="note">exp() each score, then divide by their sum. Shown for the last row ('i'). This lives
       <b>inside</b> cross-entropy.</p>
    <div class="scroll">{softmax_svg()}</div>
  </div>

  <div class="card">
    <h2>Step 5 &mdash; cross-entropy: grade each prediction against the true next char</h2>
    <p class="note">For each position, take the probability at the <b>true</b> index (red box),
       do <code>-log</code>, then average all 4. That average is the loss.</p>
    <div class="scroll">{ce_svg()}</div>
    <div class="key">loss = <b>{f3(loss)}</b>. Lower = the model put more probability on the right next char.
       Only the <b>orange embedding weights</b> get nudged to shrink this &mdash; softmax &amp; -log have nothing to learn.</div>
  </div>

</div></body></html>"""

out = "llm_output/bigram_forward_pipeline_grok.html"
with open(out, "w") as fh:
    fh.write(html)
print("wrote", out)
print("loss =", f3(loss))
