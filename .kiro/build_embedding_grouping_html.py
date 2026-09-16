"""
Generator: why the embedding groups similar letters (lecture 3 grok).

Pure-stdlib (math + random + string building). Run with `py`:
    py .kiro/build_embedding_grouping_html.py

Reads the real names.txt, computes each character's distributional fingerprint
(what letters come before + after it), lays the 27 chars out in 2D purely from
that similarity via metric MDS (SMACOF / Guttman transform), and emits inline
SVG into one HTML file. The point: vowels cluster from DATA ALONE, before any
neural net is trained -- the learned embedding just rediscovers this map.
"""

import math
import os
import random

random.seed(7)

# ----------------------------------------------------------------------------
# 1. load names.txt (try a few known locations)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CANDIDATES = [
    os.path.join(ROOT, "lectures", "makemore", "names.txt"),
    os.path.join(ROOT, "my_implementaion", "lecture_2", "names.txt"),
    os.path.join(ROOT, "my_implementaion", "names.txt"),
]
names_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if names_path is None:
    raise SystemExit("names.txt not found in known locations")
with open(names_path) as f:
    words = [w.strip() for w in f if w.strip()]

# ----------------------------------------------------------------------------
# 2. vocab: 0='.', 1..26 = a..z
# ----------------------------------------------------------------------------
letters = sorted(set("".join(words)))          # a..z
itos = ["."] + letters
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)                                   # 27
VOWELS = set("aeiou")

# ----------------------------------------------------------------------------
# 3. bigram counts, both directions
#    Nnext[i][j] = count of j occurring AFTER i
#    Nprev[i][j] = count of j occurring BEFORE i
# ----------------------------------------------------------------------------
Nnext = [[0.0] * V for _ in range(V)]
Nprev = [[0.0] * V for _ in range(V)]
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        ia, ib = stoi[a], stoi[b]
        Nnext[ia][ib] += 1
        Nprev[ib][ia] += 1


def row_normalize(row):
    s = sum(row)
    if s == 0:
        return row[:]
    return [x / s for x in row]


Pnext = [row_normalize(r) for r in Nnext]       # P(next | char)
Pprev = [row_normalize(r) for r in Nprev]       # P(prev | char)

# distributional fingerprint = concat of forward + backward distribution (54-dim)
fingerprint = [Pnext[i] + Pprev[i] for i in range(V)]


def l2norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v[:]


fp_unit = [l2norm(v) for v in fingerprint]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# target (dis)similarity distance between two L2-normalized vectors:
# euclidean dist = sqrt(2 - 2*cos)  -> proper metric in [0, 2]
def target_dist(i, j):
    c = cosine(fp_unit[i], fp_unit[j])
    c = max(-1.0, min(1.0, c))
    return math.sqrt(max(0.0, 2.0 - 2.0 * c))


D = [[target_dist(i, j) for j in range(V)] for i in range(V)]

# ----------------------------------------------------------------------------
# 4. metric MDS via SMACOF (Guttman transform) -> 2D coords
# ----------------------------------------------------------------------------
X = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(V)]


def dist2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def center(X):
    cx = sum(p[0] for p in X) / len(X)
    cy = sum(p[1] for p in X) / len(X)
    for p in X:
        p[0] -= cx
        p[1] -= cy


for _ in range(400):
    # build B(X)
    d = [[dist2d(X[i], X[j]) for j in range(V)] for i in range(V)]
    B = [[0.0] * V for _ in range(V)]
    for i in range(V):
        s = 0.0
        for j in range(V):
            if i != j and d[i][j] > 1e-9:
                B[i][j] = -D[i][j] / d[i][j]
                s += B[i][j]
        B[i][i] = -s
    # X_new = (1/n) B X
    Xn = [[0.0, 0.0] for _ in range(V)]
    for i in range(V):
        ax = ay = 0.0
        for k in range(V):
            ax += B[i][k] * X[k][0]
            ay += B[i][k] * X[k][1]
        Xn[i][0] = ax / V
        Xn[i][1] = ay / V
    X = Xn
    center(X)

# scale coords into an SVG viewport
xs = [p[0] for p in X]
ys = [p[1] for p in X]
minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)
W_MAP, H_MAP = 620, 480
PAD = 50


def sx(x):
    return PAD + (x - minx) / (maxx - minx) * (W_MAP - 2 * PAD)


def sy(y):
    return PAD + (y - miny) / (maxy - miny) * (H_MAP - 2 * PAD)


# ----------------------------------------------------------------------------
# 5. SVG builders
# ----------------------------------------------------------------------------
def color_for(ch):
    if ch == ".":
        return "#888"
    return "#e4572e" if ch in VOWELS else "#3d7bd6"


def svg_map():
    parts = [f'<svg viewBox="0 0 {W_MAP} {H_MAP}" width="100%" '
             'style="max-width:680px" font-family="ui-monospace,monospace">']
    parts.append(f'<rect x="0" y="0" width="{W_MAP}" height="{H_MAP}" '
                 'fill="#0f1117" rx="12"/>')
    # dashed ring around the vowel cluster (centroid + spread, in data coords)
    hx = sx(vc[0])
    hy = sy(vc[1])
    hr = (vspread / (maxx - minx) * (W_MAP - 2 * PAD)) * 1.9 + 22
    parts.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="{hr:.1f}" '
                 'fill="#e4572e" fill-opacity="0.06" stroke="#e4572e" '
                 'stroke-opacity="0.5" stroke-dasharray="5 5"/>')
    parts.append(f'<text x="{hx:.1f}" y="{hy - hr - 8:.1f}" fill="#e4572e" '
                 'font-size="12" text-anchor="middle">vowel cluster</text>')
    # faint links between very-similar chars
    for i in range(V):
        for j in range(i + 1, V):
            if D[i][j] < 0.55:  # strongly similar company
                parts.append(
                    f'<line x1="{sx(X[i][0]):.1f}" y1="{sy(X[i][1]):.1f}" '
                    f'x2="{sx(X[j][0]):.1f}" y2="{sy(X[j][1]):.1f}" '
                    'stroke="#e4572e" stroke-opacity="0.18" stroke-width="1"/>')
    for i in range(V):
        ch = itos[i]
        cx, cy = sx(X[i][0]), sy(X[i][1])
        col = color_for(ch)
        r = 15 if ch in VOWELS else 12
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
                     f'fill="{col}" fill-opacity="0.22" stroke="{col}" '
                     'stroke-width="1.5"/>')
        label = "•" if ch == "." else ch
        fw = "700" if ch in VOWELS else "500"
        parts.append(f'<text x="{cx:.1f}" y="{cy + 5:.1f}" fill="{col}" '
                     f'font-size="16" font-weight="{fw}" '
                     f'text-anchor="middle">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def svg_barchart(ci, title):
    """Next-char distribution for char index ci as a row of 27 bars."""
    w, h = 340, 90
    bw = (w - 30) / V
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:360px" '
             'font-family="ui-monospace,monospace">']
    row = Pnext[ci]
    mx = max(row) or 1.0
    for j in range(V):
        bh = row[j] / mx * (h - 30)
        x = 15 + j * bw
        col = color_for(itos[j])
        parts.append(f'<rect x="{x:.1f}" y="{h - 15 - bh:.1f}" '
                     f'width="{bw * 0.8:.1f}" height="{bh:.1f}" fill="{col}" '
                     'fill-opacity="0.85"/>')
    parts.append(f'<text x="15" y="12" fill="#cbd3e1" font-size="12" '
                 f'font-weight="700">{title}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# --- honest evidence: nearest neighbors of each vowel + summary stats ---
def nearest(i, k=4):
    sims = sorted(((cosine(fp_unit[i], fp_unit[j]), itos[j])
                   for j in range(1, V) if j != i), reverse=True)
    return sims[:k]


vow_idx = [stoi[c] for c in "aeiou"]
within = [cosine(fp_unit[i], fp_unit[j])
          for a, i in enumerate(vow_idx) for j in vow_idx[a + 1:]]
allpairs = [cosine(fp_unit[i], fp_unit[j])
            for i in range(1, V) for j in range(i + 1, V)]
avg_within = sum(within) / len(within)
avg_all = sum(allpairs) / len(allpairs)

nn_rows = ""
for c in "aeiou":
    nbrs = nearest(stoi[c])
    cells = " ".join(
        f'<span class="{"vowel" if ch in VOWELS or ch=="y" else "cons"}">{ch}</span>'
        f'<span style="color:#8b93a3">({s:.2f})</span>'
        for s, ch in nbrs)
    nn_rows += f"<tr><td class='vowel' style='font-size:17px'>{c}</td><td>{cells}</td></tr>"

# most-different pairs (great contrast: vowel vs rare consonant)
pairs = sorted((D[i][j], itos[i], itos[j])
               for i in range(1, V) for j in range(i + 1, V))
diff_rows = "".join(
    f"<tr><td><b>{a}</b> &amp; <b>{b}</b></td><td>{d:.2f}</td></tr>"
    for d, a, b in reversed(pairs[-6:]))

# how tight are vowels in the final 2D map vs everyone?
vpos = [X[i] for i in vow_idx]
vc = [sum(p[0] for p in vpos) / 5, sum(p[1] for p in vpos) / 5]
vspread = sum(dist2d(p, vc) for p in vpos) / 5
gc = [sum(p[0] for p in X) / V, sum(p[1] for p in X) / V]
gspread = sum(dist2d(p, gc) for p in X) / V
print(f"avg cosine within vowels {avg_within:.3f} vs all {avg_all:.3f}")
print(f"2D spread vowels {vspread:.2f} vs all {gspread:.2f}")

# ----------------------------------------------------------------------------
# 6. static analogy SVG: seats + one translator
# ----------------------------------------------------------------------------
ANALOGY_SVG = '''
<svg viewBox="0 0 640 300" width="100%" style="max-width:680px"
     font-family="ui-monospace,monospace">
  <rect x="0" y="0" width="640" height="300" fill="#0f1117" rx="12"/>
  <text x="20" y="30" fill="#cbd3e1" font-size="14" font-weight="700">
    ONE translator (shared network), smooth rule: nearby seats &#8594; same shout</text>

  <!-- translator -->
  <circle cx="540" cy="160" r="34" fill="#f2c14e" fill-opacity="0.25"
          stroke="#f2c14e" stroke-width="2"/>
  <text x="540" y="150" fill="#f2c14e" font-size="13" text-anchor="middle">shared</text>
  <text x="540" y="167" fill="#f2c14e" font-size="13" text-anchor="middle">net</text>
  <text x="540" y="184" fill="#f2c14e" font-size="13" text-anchor="middle">f( )</text>

  <!-- vowel cluster -->
  <ellipse cx="150" cy="110" rx="70" ry="55" fill="#e4572e" fill-opacity="0.10"
           stroke="#e4572e" stroke-dasharray="4 4"/>
  <text x="150" y="105" fill="#e4572e" font-size="18" font-weight="700" text-anchor="middle">a e</text>
  <text x="150" y="130" fill="#e4572e" font-size="18" font-weight="700" text-anchor="middle">i o u</text>
  <text x="150" y="185" fill="#e4572e" font-size="12" text-anchor="middle">same company &#8594; herded together</text>

  <!-- lone consonant -->
  <circle cx="330" cy="230" r="22" fill="#3d7bd6" fill-opacity="0.15"
          stroke="#3d7bd6"/>
  <text x="330" y="236" fill="#3d7bd6" font-size="18" font-weight="700" text-anchor="middle">q</text>
  <text x="330" y="270" fill="#3d7bd6" font-size="12" text-anchor="middle">different shout &#8594; pushed away</text>

  <!-- arrows to translator -->
  <line x1="215" y1="120" x2="508" y2="155" stroke="#e4572e" stroke-opacity="0.5" stroke-width="1.5"/>
  <line x1="350" y1="225" x2="512" y2="172" stroke="#3d7bd6" stroke-opacity="0.5" stroke-width="1.5"/>
</svg>
'''

# ----------------------------------------------------------------------------
# 7. assemble HTML
# ----------------------------------------------------------------------------
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Why the embedding groups similar letters</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0a0b0f; color:#dfe5ee;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         line-height:1.6; }}
  .wrap {{ max-width:820px; margin:0 auto; padding:40px 22px 90px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 6px; }}
  h2 {{ font-size:21px; margin:44px 0 10px; color:#fff;
        border-left:3px solid #e4572e; padding-left:12px; }}
  .sub {{ color:#8b93a3; font-size:15px; margin-bottom:8px; }}
  .card {{ background:#12141c; border:1px solid #1e2230; border-radius:14px;
           padding:20px 22px; margin:16px 0; }}
  .key {{ background:#1a1410; border:1px solid #e4572e44; border-left:4px solid #e4572e; }}
  code {{ background:#1e2230; padding:2px 6px; border-radius:5px;
          font-size:13.5px; color:#f2c14e; }}
  .fig {{ text-align:center; margin:10px 0 4px; }}
  .cap {{ color:#8b93a3; font-size:13.5px; text-align:center; margin:4px 0 0; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  td, th {{ border-bottom:1px solid #232838; padding:5px 8px; text-align:left; }}
  .vowel {{ color:#e4572e; font-weight:700; }}
  .cons {{ color:#3d7bd6; font-weight:700; }}
  .pill {{ display:inline-block; background:#1e2230; border-radius:20px;
           padding:3px 12px; font-size:13px; margin:2px 4px 2px 0; }}
  a {{ color:#f2c14e; }}
  @media(max-width:640px){{ .grid{{grid-template-columns:1fr}} }}
</style></head>
<body><div class="wrap">

<h1>Why the embedding groups similar letters</h1>
<p class="sub">Lecture 3 &middot; the "magic" Karpathy skipped &mdash; computed from your
real <code>names.txt</code> ({len(words):,} names)</p>

<div class="card key">
<b>The headline:</b> the grouping is not a bonus feature &mdash; it is the
<b>lowest-loss configuration</b>, forced by two facts: (A) there is only
<b>one</b> shared network downstream, so a letter's embedding <i>position</i> is its
entire identity; (B) that network is <b>smooth</b>, so nearby embeddings produce
nearly identical predictions. Letters that must be predicted the same way get
squeezed together; letters that must differ get pushed apart.
<br><br>Measured on your data: average distributional similarity
<b>within the vowels is {avg_within:.2f}</b> vs <b>{avg_all:.2f} across all letter
pairs</b> &mdash; and every single vowel's nearest neighbors are <i>other vowels</i>
(see below). That structure is in the raw statistics <b>before any embedding is
trained</b>; the neural net just rediscovers it.
</div>

<h2>The proof: the map exists in the data, before any training</h2>
<p>Below, each of the 27 characters is placed in 2D <b>purely from its
distributional fingerprint</b> &mdash; which letters come before and after it in real
names. No neural net, no embedding table, no gradient descent. Just: <i>letters that
keep the same company sit close.</i> The
<span class="vowel">vowels</span> fall into one cluster on their own. When you train
the actual embedding, it <b>rediscovers this same map</b>, because it is driven by
the same signal.</p>
<div class="fig">{svg_map()}</div>
<p class="cap"><span class="vowel">red = vowels</span> &nbsp;
<span class="cons">blue = consonants</span> &nbsp; grey &bull; = start/end token.
Faint red lines connect the most distributionally-similar pairs. Layout: metric MDS
on cosine similarity of before+after letter distributions.</p>

<h2>Fact A &mdash; one shared network, so position = identity</h2>
<p>Your <code>W1, b1, W2, b2</code> are the same for every character. There is no
per-letter logic anywhere in the model. So the only thing the network can know about a
character is <b>where its embedding vector sits</b>. Change the seat, change the
meaning &mdash; that's the whole channel.</p>

<h2>Fact B &mdash; the network is smooth</h2>
<p>The MLP is a continuous, differentiable function: nudge an input embedding a little,
the output prediction changes a little. So <b>nearby embeddings &#8594; nearly identical
predictions</b>. To make the model predict <i>differently</i> for two letters, their
embeddings <i>must</i> be far apart. This is the lever gradient descent pulls.</p>

<h2>The squeeze (why GD herds them)</h2>
<div class="fig">{ANALOGY_SVG}</div>
<p><b>Seating-room analogy.</b> The embedding space is seats; the shared network is one
translator who reads your seat and shouts a prediction. One translator for everyone
(Fact A), smooth rule &mdash; nearby seats get the same shout (Fact B). Training nudges
each letter's seat to make its shout more correct. Vowels need the same shout, so they
get herded into one corner; <span class="cons">q</span> needs a different shout, so it
drifts away. Mapping: seat &#8594; <code>C[i]</code> &nbsp;|&nbsp; translator &#8594;
shared MLP &nbsp;|&nbsp; shout &#8594; softmax prediction &nbsp;|&nbsp; nudging seats
&#8594; gradient descent on embedding rows.</p>

<h2>The evidence, letter by letter</h2>
<p>&quot;Keeping the same company&quot; is literal. Here is the next-character
distribution after three vowels &mdash; notice how alike they look &mdash; versus a
consonant, which looks nothing like them:</p>
<div class="grid">
  <div>{svg_barchart(stoi['a'], "after  a")}</div>
  <div>{svg_barchart(stoi['e'], "after  e")}</div>
  <div>{svg_barchart(stoi['o'], "after  o")}</div>
  <div>{svg_barchart(stoi['q'], "after  q")}</div>
</div>
<p class="cap">Each bar = P(next letter). The vowels share a shape; that shared shape is
what pulls their embeddings together.</p>

<div class="grid" style="margin-top:18px">
  <div class="card"><b>Nearest neighbors by company</b><br>
    <span class="cap" style="text-align:left">each vowel's closest letters &mdash;
    they are the other vowels (+ <span class="vowel">y</span>, the semi-vowel)</span>
    <table><tr><th>letter</th><th>nearest (cosine)</th></tr>{nn_rows}</table></div>
  <div class="card"><b>Most different company</b> (largest distance)
    <table><tr><th>pair</th><th>dist</th></tr>{diff_rows}</table>
    <span class="cap" style="text-align:left">a vowel vs a rare consonant &mdash; the
    model is forced to place these far apart.</span></div>
</div>
<p class="cap" style="text-align:left">Honest footnote: some rare consonant pairs
(<b>m&amp;z</b>, <b>j&amp;k</b>) score even closer than the vowels &mdash; not a bug.
They're <b>name-initial</b> letters that share the same company (<code>.</code> before,
a vowel after), so the distributional hypothesis groups them too, just in a different
neighborhood.</p>

<h2>The origin story</h2>
<p>This is the <b>distributional hypothesis</b> &mdash; linguist J.R. Firth, 1957:
<i>&ldquo;you shall know a word by the company it keeps.&rdquo;</i> The embedding is a
machine that turns <i>company</i> into <i>position</i>.</p>
<p>The paper you are reimplementing &mdash;
<a href="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">Bengio et al.
2003, &ldquo;A Neural Probabilistic Language Model&rdquo;</a> &mdash; was the first to
<i>learn</i> these distributed representations with a neural LM, and this clustering was
its headline result. A decade later, <a
href="https://arxiv.org/abs/1301.3781">word2vec (Mikolov 2013)</a> made the same
phenomenon famous at word scale:</p>
<p><span class="pill">king &minus; man + woman &#8776; queen</span>
<span class="pill">Paris &minus; France + Italy &#8776; Rome</span></p>
<p>So &ldquo;it just does it&rdquo; is really: <i>one shared smooth network + gradient
descent</i> = distances in embedding space are forced to encode behavioral similarity.
You rediscovered, live, the reason that entire line of research exists.</p>

<p class="cap" style="margin-top:40px">Generated from {os.path.basename(names_path)} by
.kiro/build_embedding_grouping_html.py &mdash; SVG coords are real
(math-computed MDS), not hand-drawn.</p>

</div></body></html>
"""

out = os.path.join(ROOT, "llm_output", "embedding_grouping_grok.html")
with open(out, "w") as f:
    f.write(html)
print("wrote", out)
