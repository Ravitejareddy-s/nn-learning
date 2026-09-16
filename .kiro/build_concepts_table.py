#!/usr/bin/env python3
"""Generate a color-coded concept map for Karpathy's NN Zero-to-Hero series.
Two ratings per concept: Importance (how central for an ML researcher) and
Hardness (how hard to truly grok). Pure stdlib -> inline SVG + HTML.
Dark theme + custom interactive hover tooltips.
Run with: .venv/bin/python .kiro/build_concepts_table.py
"""
import math
import html as _html
from collections import defaultdict

# (lecture_no, lecture_title, concept, importance 1-5, hardness 1-5, note)
DATA = [
    # ---- Lecture 1: micrograd ----
    (1, "micrograd (backprop)", "Derivative / what a gradient means", 5, 2, "The single scalar that says 'nudge this, loss moves that much'."),
    (1, "micrograd (backprop)", "Chain rule", 5, 2, "Multiply local slopes along the path. The engine of all of DL."),
    (1, "micrograd (backprop)", "Computation graph (Value objects)", 5, 3, "Every op remembers its parents; the graph IS the program."),
    (1, "micrograd (backprop)", "Forward pass", 4, 1, "Just evaluate the expression left to right."),
    (1, "micrograd (backprop)", "Backpropagation (backward pass)", 5, 3, "Chain rule applied automatically over the whole graph."),
    (1, "micrograd (backprop)", "Neuron / Layer / MLP from scratch", 4, 2, "A neuron is w.x+b then squash. Layers stack them."),
    (1, "micrograd (backprop)", "Loss (MSE)", 4, 1, "One number measuring how wrong you are."),
    (1, "micrograd (backprop)", "Gradient descent / param update", 5, 2, "Step downhill: p -= lr * p.grad."),
    (1, "micrograd (backprop)", "tanh / nonlinearity", 4, 3, "Without it, stacked layers collapse to one linear layer."),
    (1, "micrograd (backprop)", "Topological sort for autograd", 3, 3, "Backprop must visit nodes in the right order."),
    (1, "micrograd (backprop)", "zero_grad / the += accumulation bug", 3, 3, "Grads accumulate; forget to reset and training silently rots."),

    # ---- Lecture 2: bigram makemore ----
    (2, "makemore bigram", "Bigram counting model", 3, 1, "Just tally which char follows which."),
    (2, "makemore bigram", "Character-level modeling", 4, 1, "Predict the next character, one at a time."),
    (2, "makemore bigram", "Probability distribution / normalization", 4, 1, "Counts -> divide by row sum -> probabilities."),
    (2, "makemore bigram", "Sampling (torch.multinomial)", 4, 2, "Roll a weighted die to generate names."),
    (2, "makemore bigram", "Likelihood / log-likelihood / NLL loss", 5, 3, "Product of probs -> take log -> negate -> the loss."),
    (2, "makemore bigram", "One-hot encoding", 3, 1, "A row of zeros with a single 1 to index a char."),
    (2, "makemore bigram", "Bigram as a neural net (1 linear layer)", 4, 2, "Counting and a 1-layer net learn the same table."),
    (2, "makemore bigram", "Softmax", 5, 2, "logits -> exp -> normalize -> a probability distribution."),
    (2, "makemore bigram", "torch.Tensor basics + broadcasting", 5, 3, "The rules for how shapes stretch to line up."),
    (2, "makemore bigram", "Counting == gradient-based equivalence", 3, 3, "Two very different methods land on the same answer."),
    (2, "makemore bigram", "Regularization / smoothing", 4, 2, "Nudge weights toward uniform; avoid zero-prob explosions."),

    # ---- Lecture 3: MLP ----
    (3, "makemore MLP", "Embeddings / lookup table", 5, 3, "Each char -> a learned vector, not a one-hot."),
    (3, "makemore MLP", "Context / block size", 4, 1, "How many previous chars you condition on."),
    (3, "makemore MLP", "MLP language model (Bengio 2003)", 4, 2, "Concat embeddings -> hidden tanh -> logits."),
    (3, "makemore MLP", "Train / dev / test split", 5, 1, "Separate data to tune vs to honestly judge."),
    (3, "makemore MLP", "Over- / under-fitting", 5, 2, "Memorizing vs failing to learn."),
    (3, "makemore MLP", "Hyperparameters", 4, 2, "Knobs you set, not the net learns."),
    (3, "makemore MLP", "Learning-rate finding (lr sweep)", 4, 2, "Sweep lr, plot loss, pick the sweet spot."),
    (3, "makemore MLP", "Minibatches", 4, 2, "Noisy-but-cheap gradient from a subset."),
    (3, "makemore MLP", "Learning-rate decay", 3, 1, "Big steps early, tiny steps to settle."),
    (3, "makemore MLP", "F.cross_entropy numerical stability", 4, 3, "Why the fused op beats manual softmax+log."),

    # ---- Lecture 4: Activations, Gradients, BatchNorm ----
    (4, "Activations & BatchNorm", "Weight init / scaling (Kaiming)", 5, 4, "Wrong scale at init and the net is dead before step 1."),
    (4, "Activations & BatchNorm", "Saturated tanh / dead neurons", 4, 3, "Flat regions -> zero grad -> neuron never learns."),
    (4, "Activations & BatchNorm", "Activation statistics (forward)", 4, 4, "Reading histograms to see if signal survives."),
    (4, "Activations & BatchNorm", "Vanishing / exploding gradients", 5, 4, "Grads shrink or blow up as they flow back."),
    (4, "Activations & BatchNorm", "Batch Normalization (forward + why)", 5, 4, "Normalize activations per batch to keep them healthy."),
    (4, "Activations & BatchNorm", "Running mean/std at inference", 3, 3, "Train-time batch stats vs a stored average at test."),
    (4, "Activations & BatchNorm", "update:data ratio diagnostic", 3, 4, "Is each step nudging weights ~0.1%? A health meter."),
    (4, "Activations & BatchNorm", "Softening final logits at init", 3, 3, "Start unconfident so early loss isn't a hockey stick."),

    # ---- Lecture 5: Backprop Ninja ----
    (5, "Backprop Ninja", "Manual backprop through cross-entropy", 4, 4, "Derive dlogits by hand; softmax-minus-onehot magic."),
    (5, "Backprop Ninja", "Manual backprop through matmul (dW, dx, db)", 4, 4, "Transpose-and-multiply rules for the linear layer."),
    (5, "Backprop Ninja", "Backprop through broadcasting (sum over dims)", 4, 4, "A broadcast forward becomes a sum backward."),
    (5, "Backprop Ninja", "Manual backprop through tanh", 3, 2, "1 - t**2 times the incoming grad."),
    (5, "Backprop Ninja", "Manual backprop through BatchNorm", 4, 5, "Every example couples to every other via mean/var. THE hard one."),
    (5, "Backprop Ninja", "Manual backprop through embedding", 3, 3, "Scatter-add grads back into the used rows."),

    # ---- Lecture 6: WaveNet ----
    (6, "WaveNet", "Hierarchical / tree-like architecture", 3, 3, "Fuse pairs progressively instead of all at once."),
    (6, "WaveNet", "Dilated causal convolution (concept)", 3, 4, "Same tree, done efficiently with strided conv."),
    (6, "WaveNet", "torch.nn module design (Linear/BN/Tanh)", 4, 2, "Package layers as reusable objects with parameters()."),
    (6, "WaveNet", "BatchNorm1d over 3D input (the dim bug)", 3, 4, "Getting the normalized dimension wrong is a silent killer."),
    (6, "WaveNet", "Dev workflow / shape bookkeeping", 4, 2, "Reading docs, tracking tensor shapes constantly."),

    # ---- Lecture 7: GPT / Transformer ----
    (7, "GPT / Transformer", "Self-attention", 5, 5, "Tokens decide who to listen to. Hardest to intuit."),
    (7, "GPT / Transformer", "Query / Key / Value", 5, 5, "Three projections: what I want, what I offer, what I pass on."),
    (7, "GPT / Transformer", "Scaled dot-product (why /sqrt(d))", 4, 4, "Keep the softmax from saturating at init."),
    (7, "GPT / Transformer", "Causal masking", 4, 3, "You may only look at the past, never the future."),
    (7, "GPT / Transformer", "Multi-head attention", 5, 4, "Several attention channels in parallel."),
    (7, "GPT / Transformer", "Positional embeddings", 4, 3, "Attention is order-blind; inject position."),
    (7, "GPT / Transformer", "Feed-forward block", 3, 2, "Per-token compute after tokens have talked."),
    (7, "GPT / Transformer", "Residual (skip) connections", 5, 3, "A gradient superhighway around each block."),
    (7, "GPT / Transformer", "LayerNorm", 4, 3, "Normalize per-token, not per-batch."),
    (7, "GPT / Transformer", "Dropout", 3, 2, "Randomly mute units to regularize."),
    (7, "GPT / Transformer", "Decoder-only design", 3, 3, "Why GPT drops the encoder half."),

    # ---- Lecture 8: Tokenizer / BPE ----
    (8, "GPT Tokenizer (BPE)", "Byte Pair Encoding training", 4, 3, "Greedily merge the most frequent pair, repeat."),
    (8, "GPT Tokenizer (BPE)", "Unicode / UTF-8 bytes", 4, 3, "Text is bytes; that's the real alphabet."),
    (8, "GPT Tokenizer (BPE)", "encode / decode", 4, 2, "The two functions a trained tokenizer exposes."),
    (8, "GPT Tokenizer (BPE)", "Merges / vocab building", 3, 2, "The learned merge list defines the vocabulary."),
    (8, "GPT Tokenizer (BPE)", "Regex splitting (GPT-2 pattern)", 2, 3, "Pre-split text so merges don't cross word boundaries."),
    (8, "GPT Tokenizer (BPE)", "Special tokens", 2, 2, "<|endoftext|> and friends, spliced into the vocab."),
    (8, "GPT Tokenizer (BPE)", "Why tokenization breaks LLMs", 4, 2, "Spelling, math, non-English quirks trace back here."),
]

# brighter palette that pops on a dark background
LECTURE_COLORS = {
    1: "#ff5c7a", 2: "#ffa94d", 3: "#51e08a", 4: "#5c9dff",
    5: "#c77dff", 6: "#2dd4bf", 7: "#ff6ec7", 8: "#e0b072",
}
LEC_NAMES = {1:"1 · micrograd",2:"2 · bigram",3:"3 · MLP",4:"4 · BN / activations",
             5:"5 · backprop ninja",6:"6 · WaveNet",7:"7 · GPT",8:"8 · tokenizer"}

def imp_color(v):  # green scale for dark theme (bg, fg)
    return {1:("#1f3a28","#8ff0b0"), 2:("#245e39","#c9ffdc"), 3:("#2e8b57","#eafff2"),
            4:("#37b96e","#04220f"), 5:("#42e08a","#04220f")}[v]

def hard_color(v):  # red/amber scale for dark theme
    return {1:("#3a2426","#ffb3b3"), 2:("#5e2a2f","#ffcccc"), 3:("#a13b3b","#ffe3e3"),
            4:("#e0533f","#2a0b06"), 5:("#ff5c3d","#2a0b06")}[v]

# ---------- scatter (importance x, hardness y) ----------
W, H = 860, 620
padL, padR, padT, padB = 74, 250, 54, 64
plotW = W - padL - padR
plotH = H - padT - padB

def xpix(imp):  return padL + (imp - 1) / 4 * plotW
def ypix(hard): return padT + (5 - hard) / 4 * plotH  # hardness 5 at top

cells = defaultdict(list)
for row in DATA:
    cells[(row[3], row[4])].append(row)

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">')
svg.append('<defs>'
           '<radialGradient id="glow" cx="50%" cy="50%" r="50%">'
           '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.25"/>'
           '<stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>'
           '</radialGradient></defs>')

midx = xpix(3.5); midy = ypix(3.5)
# quadrant fills (subtle on dark)
svg.append(f'<rect x="{midx:.0f}" y="{padT}" width="{padL+plotW-midx:.0f}" height="{midy-padT:.0f}" fill="#ffd54f" opacity="0.10"/>')
svg.append(f'<rect x="{midx:.0f}" y="{midy:.0f}" width="{padL+plotW-midx:.0f}" height="{padT+plotH-midy:.0f}" fill="#42e08a" opacity="0.08"/>')
svg.append(f'<rect x="{padL}" y="{padT}" width="{midx-padL:.0f}" height="{midy-padT:.0f}" fill="#ff5c3d" opacity="0.07"/>')
svg.append(f'<line x1="{midx:.0f}" y1="{padT}" x2="{midx:.0f}" y2="{padT+plotH}" stroke="#ffd54f" stroke-dasharray="4 5" opacity="0.35"/>')
svg.append(f'<line x1="{padL}" y1="{midy:.0f}" x2="{padL+plotW}" y2="{midy:.0f}" stroke="#ffd54f" stroke-dasharray="4 5" opacity="0.35"/>')

svg.append(f'<text x="{midx+10:.0f}" y="{padT+18:.0f}" font-size="12" fill="#ffd970" font-weight="700">HARD &amp; ESSENTIAL — invest here</text>')
svg.append(f'<text x="{midx+10:.0f}" y="{padT+plotH-8:.0f}" font-size="12" fill="#6be6a0" font-weight="700">Easy &amp; essential — quick wins</text>')
svg.append(f'<text x="{padL+8:.0f}" y="{padT+18:.0f}" font-size="12" fill="#ff8a75" font-weight="700">Hard but niche</text>')
svg.append(f'<text x="{padL+8:.0f}" y="{padT+plotH-8:.0f}" font-size="12" fill="#7a8699">Easy &amp; niche</text>')

for v in range(1, 6):
    gx = xpix(v); gy = ypix(v)
    svg.append(f'<line x1="{gx:.0f}" y1="{padT}" x2="{gx:.0f}" y2="{padT+plotH}" stroke="#2a3350"/>')
    svg.append(f'<line x1="{padL}" y1="{gy:.0f}" x2="{padL+plotW}" y2="{gy:.0f}" stroke="#2a3350"/>')
    svg.append(f'<text x="{gx:.0f}" y="{padT+plotH+24:.0f}" font-size="12" fill="#8b96ad" text-anchor="middle">{v}</text>')
    svg.append(f'<text x="{padL-16:.0f}" y="{gy+4:.0f}" font-size="12" fill="#8b96ad" text-anchor="end">{v}</text>')

svg.append(f'<line x1="{padL}" y1="{padT+plotH}" x2="{padL+plotW}" y2="{padT+plotH}" stroke="#4a5578" stroke-width="1.5"/>')
svg.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{padT+plotH}" stroke="#4a5578" stroke-width="1.5"/>')
svg.append(f'<text x="{padL+plotW/2:.0f}" y="{padT+plotH+50:.0f}" font-size="14" fill="#c6cede" text-anchor="middle" font-weight="700">Importance  \u2192</text>')
svg.append(f'<text x="20" y="{padT+plotH/2:.0f}" font-size="14" fill="#c6cede" text-anchor="middle" font-weight="700" transform="rotate(-90 20 {padT+plotH/2:.0f})">Hardness to grok  \u2192</text>')

def esc(s): return _html.escape(str(s), quote=True)

for (imp, hard), items in cells.items():
    cx, cy = xpix(imp), ypix(hard)
    n = len(items)
    for i, row in enumerate(items):
        if n == 1:
            ox = oy = 0
        else:
            ang = 2 * math.pi * i / n
            rad = 13 + (i % 2) * 9
            ox = rad * math.cos(ang); oy = rad * math.sin(ang)
        col = LECTURE_COLORS[row[0]]
        svg.append(
            f'<circle class="dot" cx="{cx+ox:.1f}" cy="{cy+oy:.1f}" r="7" fill="{col}" '
            f'stroke="#0d1326" stroke-width="1.5" '
            f'data-lec="{row[0]}" data-lecname="{esc(LEC_NAMES[row[0]])}" '
            f'data-concept="{esc(row[2])}" data-imp="{imp}" data-hard="{hard}" '
            f'data-note="{esc(row[5])}" data-color="{col}"/>'
        )
svg.append('</svg>')
scatter_svg = "\n".join(svg)

# legend as HTML (crisper than SVG text)
legend_html = "".join(
    f'<span class="leg"><i style="background:{LECTURE_COLORS[ln]}"></i>{esc(name)}</span>'
    for ln, name in LEC_NAMES.items()
)

# ---------- table ----------
by_lec = defaultdict(list)
for row in DATA:
    by_lec[(row[0], row[1])].append(row)

def badge(v, kind):
    bg, fg = (imp_color(v) if kind == "imp" else hard_color(v))
    dots = "\u25cf" * v + "\u25cb" * (5 - v)
    return f'<span class="badge" style="background:{bg};color:{fg}">{v}<span class="dots">{dots}</span></span>'

rows_html = []
for (ln, title) in sorted(by_lec):
    col = LECTURE_COLORS[ln]
    rows_html.append(
        f'<tr class="lechead"><td colspan="4" style="border-left:5px solid {col}">'
        f'<span class="ltag" style="background:{col}">L{ln}</span> {esc(title)}</td></tr>'
    )
    for row in sorted(by_lec[(ln, title)], key=lambda r: (-r[4], -r[3])):
        _, _, concept, imp, hard, note = row
        rows_html.append(
            "<tr>"
            f'<td class="concept">{esc(concept)}</td>'
            f'<td class="rate">{badge(imp,"imp")}</td>'
            f'<td class="rate">{badge(hard,"hard")}</td>'
            f'<td class="note">{esc(note)}</td>'
            "</tr>"
        )
table_html = "\n".join(rows_html)

top = sorted([r for r in DATA if r[3] >= 4 and r[4] >= 4], key=lambda r: (-r[4], -r[3]))
top_html = "".join(
    f'<li><span class="ltag" style="background:{LECTURE_COLORS[r[0]]}">L{r[0]}</span>'
    f'<b>{esc(r[2])}</b><span class="mini">imp {r[3]} · hard {r[4]}</span><br>'
    f'<span class="tnote">{esc(r[5])}</span></li>'
    for r in top
)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NN Zero-to-Hero — Concept Map</title>
<style>
  :root {{
    --bg:#0b1020; --bg2:#0e1428; --panel:#141c33; --panel2:#182142;
    --line:#243056; --text:#e8ecf6; --muted:#8b96ad; --accent:#ffd54f;
  }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    margin:0; color:var(--text);
    background:radial-gradient(1200px 700px at 15% -5%, #16213f 0%, transparent 60%),
               radial-gradient(1000px 600px at 100% 0%, #241a3e 0%, transparent 55%),
               linear-gradient(160deg,#0b1020 0%, #0a0e1c 100%);
    background-attachment:fixed; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:40px 24px 90px; }}
  h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.5px;
    background:linear-gradient(90deg,#8fd3ff,#c77dff 55%,#ff6ec7);
    -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .sub {{ color:var(--muted); margin:0 0 30px; font-size:15.5px; max-width:760px; line-height:1.55; }}
  .card {{ background:linear-gradient(180deg,var(--panel),var(--bg2));
    border:1px solid var(--line); border-radius:16px; padding:22px 24px; margin-bottom:28px;
    box-shadow:0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03); }}
  h2 {{ margin:0 0 14px; font-size:20px; letter-spacing:-.3px; }}
  .scale {{ font-size:14px; color:#c6cede; line-height:1.8; }}
  .scale b {{ color:#fff; }}
  .scale .g {{ color:#6be6a0; font-weight:700; }} .scale .r {{ color:#ff8a75; font-weight:700; }}
  svg {{ width:100%; height:auto; display:block; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px 20px; margin-top:14px; padding-top:16px; border-top:1px solid var(--line); }}
  .leg {{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:#c6cede; }}
  .leg i {{ width:12px; height:12px; border-radius:50%; display:inline-block; box-shadow:0 0 8px rgba(255,255,255,.15); }}
  .hint {{ font-size:13px; color:var(--muted); margin:14px 0 0; }}
  table {{ border-collapse:separate; border-spacing:0; width:100%; font-size:14px; }}
  td {{ padding:10px 12px; border-bottom:1px solid #1c2540; vertical-align:top; }}
  tr.lechead td {{ background:rgba(255,255,255,.03); font-size:15px; padding:13px 12px; letter-spacing:.2px; }}
  tr:not(.lechead):hover td {{ background:rgba(120,150,255,.06); }}
  .concept {{ font-weight:600; width:280px; color:#f2f5ff; }}
  .rate {{ width:120px; white-space:nowrap; }}
  .note {{ color:var(--muted); }}
  .badge {{ display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:20px;
    font-weight:800; font-size:13px; box-shadow:0 2px 8px rgba(0,0,0,.3); }}
  .badge .dots {{ font-size:8px; letter-spacing:1.5px; opacity:.9; }}
  .ltag {{ display:inline-block; color:#0d1326; font-weight:800; font-size:11px;
    padding:2px 8px; border-radius:8px; margin-right:9px; }}
  ul.top {{ list-style:none; padding:0; margin:0; display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  ul.top li {{ background:rgba(255,213,79,.06); border:1px solid rgba(255,213,79,.18);
    border-radius:12px; padding:13px 15px; line-height:1.5; }}
  ul.top b {{ color:#fff; font-size:14.5px; }}
  .mini {{ float:right; font-size:11.5px; color:var(--accent); font-weight:700; }}
  .tnote {{ color:var(--muted); font-size:13px; }}
  @media (max-width:720px) {{ ul.top {{ grid-template-columns:1fr; }} .concept {{ width:auto; }} }}
  /* --- interactive dots --- */
  .dot {{ cursor:pointer; transition:r .12s ease, stroke-width .12s ease; }}
  .dot:hover {{ stroke:#fff; stroke-width:2.5; }}
  /* --- custom tooltip --- */
  #tip {{ position:fixed; pointer-events:none; z-index:50; opacity:0; transform:translateY(4px);
    transition:opacity .12s ease, transform .12s ease; max-width:300px;
    background:linear-gradient(180deg,#1b2340,#141c33); background:#141c33;
    border:1px solid #33406e; border-radius:13px; padding:13px 15px;
    box-shadow:0 14px 44px rgba(0,0,0,.6); }}
  #tip.on {{ opacity:1; transform:translateY(0); }}
  #tip .th {{ font-weight:800; font-size:14.5px; color:#fff; margin-bottom:4px; line-height:1.35; }}
  #tip .tl {{ display:inline-block; color:#0d1326; font-weight:800; font-size:10.5px; padding:2px 7px; border-radius:7px; margin-bottom:8px; }}
  #tip .tr {{ display:flex; gap:16px; margin:8px 0 9px; font-size:12px; }}
  #tip .tr b {{ display:block; font-size:17px; }}
  #tip .tr .lbl {{ color:var(--muted); text-transform:uppercase; letter-spacing:.6px; font-size:9.5px; }}
  #tip .tn {{ color:#c6cede; font-size:12.5px; line-height:1.5; border-top:1px solid #26315a; padding-top:8px; }}
  #tip .imp b {{ color:#5be89a; }} #tip .hard b {{ color:#ff7a5c; }}
</style></head>
<body><div class="wrap">
  <h1>Neural Networks: Zero to Hero — Concept Map</h1>
  <p class="sub">Every concept across Lectures 1&ndash;8, rated on two independent axes: how <b>important</b> it is for an ML researcher, and how <b>hard</b> it is to truly grok. Hover any dot in the chart for its full card.</p>

  <div class="card">
    <div class="scale">
      <span class="g">Importance (1&ndash;5)</span> — how central this idea is to understanding modern deep learning &amp; building things yourself.<br>
      <span class="r">Hardness (1&ndash;5)</span> — how hard the intuition is to actually lock in (not how tedious to code).
    </div>
  </div>

  <div class="card">
    <h2>The landscape</h2>
    {scatter_svg}
    <div class="legend">{legend_html}</div>
    <p class="hint">Top-right (gold) = hard <i>and</i> essential: where your grokking effort pays off most. Hover a dot for details.</p>
  </div>

  <div class="card">
    <h2>Focus shortlist — the "hard &amp; essential" quadrant</h2>
    <ul class="top">{top_html}</ul>
  </div>

  <div class="card">
    <h2>Full table <span style="font-size:13px;color:var(--muted);font-weight:400">(hardest-first within each lecture)</span></h2>
    <table>
      <tr style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;">
        <td>Concept</td><td>Importance</td><td>Hardness</td><td>One-line intuition</td>
      </tr>
      {table_html}
    </table>
  </div>
</div>

<div id="tip">
  <span class="tl" id="tip-lec"></span>
  <div class="th" id="tip-title"></div>
  <div class="tr">
    <div class="imp"><span class="lbl">Importance</span><b id="tip-imp"></b></div>
    <div class="hard"><span class="lbl">Hardness</span><b id="tip-hard"></b></div>
  </div>
  <div class="tn" id="tip-note"></div>
</div>

<script>
(function() {{
  var tip = document.getElementById('tip');
  var elLec = document.getElementById('tip-lec'), elTitle = document.getElementById('tip-title'),
      elImp = document.getElementById('tip-imp'), elHard = document.getElementById('tip-hard'),
      elNote = document.getElementById('tip-note');
  var stars = function(n) {{ return '\u2605'.repeat(n) + '\u2606'.repeat(5 - n); }};
  document.querySelectorAll('.dot').forEach(function(d) {{
    d.addEventListener('mouseenter', function() {{
      var col = d.getAttribute('data-color');
      elLec.textContent = d.getAttribute('data-lecname'); elLec.style.background = col;
      elTitle.textContent = d.getAttribute('data-concept');
      elImp.textContent = stars(+d.getAttribute('data-imp'));
      elHard.textContent = stars(+d.getAttribute('data-hard'));
      elNote.textContent = d.getAttribute('data-note');
      tip.style.borderColor = col;
      tip.classList.add('on');
    }});
    d.addEventListener('mouseleave', function() {{ tip.classList.remove('on'); }});
  }});
  document.addEventListener('mousemove', function(e) {{
    if (!tip.classList.contains('on')) return;
    var w = tip.offsetWidth, h = tip.offsetHeight, pad = 16;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w + 8 > window.innerWidth)  x = e.clientX - w - pad;
    if (y + h + 8 > window.innerHeight) y = e.clientY - h - pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }});
}})();
</script>
</body></html>"""

OUT = "llm_output/concepts_importance_vs_hardness.html"
with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, f"({len(DATA)} concepts)")
