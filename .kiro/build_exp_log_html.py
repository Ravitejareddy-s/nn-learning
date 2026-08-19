#!/usr/bin/env python3
"""Builds a self-contained teaching HTML page about exp and log for the softmax + NLL
pipeline. Pure stdlib only (math, html) -> hand-drawn SVG. No numpy / matplotlib, so it
runs under the free-threaded Python 3.13t interpreter where numpy's C-extension is broken.

Output: llm_output/exp_and_log_grok.html
"""
import math
import html as _html

BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#e8820c"
GREEN = "#2ca02c"
GRAY = "#8a8a8a"
DARK = "#333333"
FAINT = "#e6e6e6"


def fmt(v):
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v - round(v)) < 1e-9:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    if abs(v) >= 0.001:
        return f"{v:.3f}"
    return f"{v:.1e}"


def esc(s):
    return _html.escape(str(s))


class Plot:
    def __init__(self, w, h, xr, yr, ml=64, mr=24, mt=22, mb=44, bg="#ffffff"):
        self.w, self.h = w, h
        self.x0, self.x1 = xr
        self.y0, self.y1 = yr
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.pw = w - ml - mr
        self.ph = h - mt - mb
        self.bg = bg
        self.parts = []

    def sx(self, x):
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * self.pw

    def sy(self, y):
        return self.mt + (1 - (y - self.y0) / (self.y1 - self.y0)) * self.ph

    def add(self, s):
        self.parts.append(s)

    def box(self):
        self.add(f'<rect x="{self.ml}" y="{self.mt}" width="{self.pw}" height="{self.ph}" '
                 f'fill="{self.bg}" stroke="#cccccc" stroke-width="1"/>')

    def xticks(self, vals, fmtf=fmt, grid=True):
        for v in vals:
            X = self.sx(v)
            if grid:
                self.add(f'<line x1="{X:.1f}" y1="{self.mt}" x2="{X:.1f}" y2="{self.mt+self.ph}" '
                         f'stroke="{FAINT}" stroke-width="1"/>')
            self.add(f'<line x1="{X:.1f}" y1="{self.mt+self.ph}" x2="{X:.1f}" y2="{self.mt+self.ph+5}" '
                     f'stroke="{DARK}" stroke-width="1"/>')
            self.add(f'<text x="{X:.1f}" y="{self.mt+self.ph+18}" font-size="12" fill="{DARK}" '
                     f'text-anchor="middle">{esc(fmtf(v))}</text>')

    def yticks(self, vals, fmtf=fmt, grid=True):
        for v in vals:
            Y = self.sy(v)
            if grid:
                self.add(f'<line x1="{self.ml}" y1="{Y:.1f}" x2="{self.ml+self.pw}" y2="{Y:.1f}" '
                         f'stroke="{FAINT}" stroke-width="1"/>')
            self.add(f'<line x1="{self.ml-5}" y1="{Y:.1f}" x2="{self.ml}" y2="{Y:.1f}" '
                     f'stroke="{DARK}" stroke-width="1"/>')
            self.add(f'<text x="{self.ml-9}" y="{Y+4:.1f}" font-size="12" fill="{DARK}" '
                     f'text-anchor="end">{esc(fmtf(v))}</text>')

    def axis0(self):
        if self.y0 <= 0 <= self.y1:
            Y = self.sy(0)
            self.add(f'<line x1="{self.ml}" y1="{Y:.1f}" x2="{self.ml+self.pw}" y2="{Y:.1f}" '
                     f'stroke="#999999" stroke-width="1.3"/>')
        if self.x0 <= 0 <= self.x1:
            X = self.sx(0)
            self.add(f'<line x1="{X:.1f}" y1="{self.mt}" x2="{X:.1f}" y2="{self.mt+self.ph}" '
                     f'stroke="#999999" stroke-width="1.3"/>')

    def xlabel(self, s):
        self.add(f'<text x="{self.ml+self.pw/2:.1f}" y="{self.h-6}" font-size="12.5" '
                 f'fill="{DARK}" text-anchor="middle">{esc(s)}</text>')

    def ylabel(self, s):
        x = 15
        y = self.mt + self.ph / 2
        self.add(f'<text x="{x}" y="{y:.1f}" font-size="12.5" fill="{DARK}" '
                 f'text-anchor="middle" transform="rotate(-90 {x} {y:.1f})">{esc(s)}</text>')

    def curve(self, fn, xa, xb, color, width=2.4, n=260, dash=None):
        pts = []
        for i in range(n + 1):
            x = xa + (xb - xa) * i / n
            try:
                y = fn(x)
            except ValueError:
                continue
            if y < self.y0 - (self.y1 - self.y0) or y > self.y1 + (self.y1 - self.y0):
                pass
            pts.append(f"{self.sx(x):.1f},{self.sy(y):.1f}")
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def poly(self, xys, color, width=2.4, dash=None, fill="none"):
        pts = " ".join(f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in xys)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{pts}" fill="{fill}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def dot(self, x, y, color, r=4.5):
        self.add(f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" '
                 f'fill="{color}" stroke="#ffffff" stroke-width="1.3"/>')

    def seg(self, x1, y1, x2, y2, color, width=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{self.sx(x1):.1f}" y1="{self.sy(y1):.1f}" '
                 f'x2="{self.sx(x2):.1f}" y2="{self.sy(y2):.1f}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def vspan(self, xa, xb, color, opacity=0.12):
        X = self.sx(xa)
        W = self.sx(xb) - X
        self.add(f'<rect x="{X:.1f}" y="{self.mt}" width="{W:.1f}" height="{self.ph}" '
                 f'fill="{color}" opacity="{opacity}"/>')

    def bar(self, x_center, y_top, half_w, color, y_base=0.0, opacity=1.0):
        xa = self.sx(x_center - half_w)
        xb = self.sx(x_center + half_w)
        yt = self.sy(y_top)
        yb = self.sy(y_base)
        y = min(yt, yb)
        hgt = abs(yb - yt)
        self.add(f'<rect x="{xa:.1f}" y="{y:.1f}" width="{xb-xa:.1f}" height="{hgt:.1f}" '
                 f'fill="{color}" opacity="{opacity}" rx="2"/>')

    def txt(self, x, y, s, color=DARK, size=12.5, anchor="middle", weight="normal"):
        self.add(f'<text x="{self.sx(x):.1f}" y="{self.sy(y):.1f}" font-size="{size}" '
                 f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def txt_px(self, X, Y, s, color=DARK, size=12.5, anchor="middle", weight="normal"):
        self.add(f'<text x="{X:.1f}" y="{Y:.1f}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def svg(self, title=None):
        head = ""
        if title:
            head = (f'<text x="{self.w/2:.1f}" y="15" font-size="13.5" fill="{DARK}" '
                    f'text-anchor="middle" font-weight="600">{esc(title)}</text>')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'style="max-width:100%;height:auto;font-family:system-ui,Segoe UI,Arial,sans-serif" '
                f'xmlns="http://www.w3.org/2000/svg"><rect width="{self.w}" height="{self.h}" '
                f'fill="#ffffff"/>{head}{"".join(self.parts)}</svg>')


# ---------------------------------------------------------------- figures

def fig_exp():
    p = Plot(820, 380, (-2.2, 2.4), (-0.6, 9.2), mt=30)
    p.box()
    p.xticks([-2, -1, 0, 1, 2])
    p.yticks([0, 2, 4, 6, 8])
    p.axis0()
    p.curve(math.exp, -2.2, 2.2, BLUE, 2.8)
    marks = [-1, 0, 1, 2]
    ys = [math.exp(m) for m in marks]
    for m, y in zip(marks, ys):
        p.seg(m, 0, m, y, GRAY, 1, "3,3")
        p.seg(p.x0, y, m, y, GRAY, 1, "3,3")
        p.dot(m, y, BLUE)
    # growing gaps on the left axis
    for i in range(len(marks) - 1):
        ymid = (ys[i] + ys[i + 1]) / 2
        gap = ys[i + 1] - ys[i]
        p.txt_px(p.ml + 8, p.sy(ymid) + 4, f"+{gap:.2f}", RED, 12, "start", "600")
    p.txt_px(p.sx(1.15), p.sy(7.6), "each +1 in x  =  x2.72 in height", DARK, 12.5, "start", "600")
    p.txt_px(p.sx(-2.0), p.sy(0.6), "never dips to 0 or below", GREEN, 12, "start")
    p.xlabel("x  (a logit / score)")
    p.ylabel("e^x  (a count)")
    return p.svg("e^x : always positive, always rising, gaps grow (but the ratio is constant)")


def fig_log():
    p = Plot(820, 380, (-0.2, 6.2), (-3.4, 2.2), mt=30)
    p.box()
    p.xticks([0, 1, 2, 3, 4, 5, 6])
    p.yticks([-3, -2, -1, 0, 1, 2])
    p.axis0()
    p.curve(math.log, 0.05, 6.2, RED, 2.8)
    marks = [0.5, 1, 2, 4]
    ys = [math.log(m) for m in marks]
    for m, y in zip(marks, ys):
        p.seg(m, p.y0, m, y, GRAY, 1, "3,3")
        p.seg(p.x0, y, m, y, GRAY, 1, "3,3")
        p.dot(m, y, RED)
    p.dot(1, 0, RED, 5.5)
    p.txt_px(p.sx(1.1), p.sy(0.15), "log(1) = 0", DARK, 12, "start", "600")
    p.txt_px(p.sx(2.4), p.sy(-2.4), "each x2 in x  =  +0.69 in height", DARK, 12.5, "start", "600")
    p.txt_px(p.sx(0.15), p.sy(-3.0), "drops toward -inf as x -> 0", DARK, 11.5, "start")
    p.xlabel("x  (a probability or count)")
    p.ylabel("log(x)")
    return p.svg("log(x) : the mirror of e^x. Turns 'x2' into '+0.69'  (multiply -> add)")


def fig_inverse():
    p = Plot(820, 430, (-3.2, 6.2), (-3.2, 6.2), mt=30)
    p.box()
    p.xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    p.yticks([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    p.axis0()
    # y = x mirror line
    p.seg(-3.2, -3.2, 6.2, 6.2, GRAY, 1.3, "5,4")
    p.txt_px(p.sx(5.1), p.sy(5.6), "mirror line y = x", GRAY, 12, "middle")
    p.curve(math.exp, -3.2, 1.83, BLUE, 2.6)
    p.curve(math.log, 0.05, 6.2, RED, 2.6)
    p.txt_px(p.sx(1.35), p.sy(5.4), "e^x", BLUE, 15, "middle", "700")
    p.txt_px(p.sx(5.3), p.sy(1.9), "log(x)", RED, 15, "middle", "700")
    # a matched pair of points reflected across y=x
    p.dot(1, math.e, BLUE)
    p.dot(math.e, 1, RED)
    p.seg(1, math.e, math.e, 1, GREEN, 1.2, "2,3")
    p.xlabel("x")
    p.ylabel("output")
    return p.svg("e^x and log undo each other: reflections across y = x  (nothing is lost)")


def fig_mountain():
    # top: L(w) bump; bottom: log L(w). shared x, same peak at w=2
    W, H = 820, 470
    peak = 2.0
    sig = 0.9
    amp = 0.05  # keep L small (like a real likelihood) so logs are negative

    def L(w):
        return amp * math.exp(-((w - peak) ** 2) / (2 * sig * sig))

    def logL(w):
        return math.log(L(w))

    xa, xb = -1.0, 5.0
    # top panel
    top = Plot(W, 210, (xa, xb), (0, amp * 1.15), ml=70, mr=24, mt=28, mb=26)
    top.box()
    top.xticks([-1, 0, 1, 2, 3, 4, 5], grid=False)
    top.yticks([0, 0.02, 0.04])
    top.curve(L, xa, xb, BLUE, 2.8)
    top.vspan(peak - 0.012, peak + 0.012, DARK, 0.0)
    top.seg(peak, 0, peak, amp * 1.13, DARK, 1.6, "5,4")
    top.dot(peak, L(peak), BLUE, 5)
    top.txt_px(top.sx(peak) + 8, top.sy(L(peak)) - 6, "best w", DARK, 12.5, "start", "700")
    top.ylabel("likelihood  L(w)")
    top_svg = top.svg("How good is the model as we slide one weight w  (higher = better)")

    bot = Plot(W, 235, (xa, xb), (logL(xa) - 0.4, logL(peak) + 0.6), ml=70, mr=24, mt=16, mb=42)
    bot.box()
    bot.xticks([-1, 0, 1, 2, 3, 4, 5])
    yt = [-9, -7, -5, -3]
    bot.yticks([v for v in yt if bot.y0 <= v <= bot.y1])
    bot.curve(logL, xa, xb, RED, 2.8)
    bot.seg(peak, bot.y0, peak, logL(peak), DARK, 1.6, "5,4")
    bot.dot(peak, logL(peak), RED, 5)
    bot.txt_px(bot.sx(peak) + 8, bot.sy(logL(peak)) - 6, "same best w", DARK, 12.5, "start", "700")
    bot.xlabel("w  (one weight)")
    bot.ylabel("log L(w)")
    bot_svg = bot.svg("Take the log of the height above. Shape changes wildly -- the peak does NOT move.")
    return top_svg + bot_svg


def _barpanel(title, labels, values, colors, w=390, h=300, ymax=None, val_fmt=fmt,
              ylabel="", baseline0=True):
    n = len(values)
    ymax = ymax if ymax is not None else max(values) * 1.25
    ymin = 0 if baseline0 else min(0, min(values) * 1.2)
    p = Plot(w, h, (-0.5, n - 0.5), (ymin, ymax), ml=52, mr=16, mt=30, mb=40)
    p.box()
    p.axis0()
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        p.bar(i, v, 0.34, c)
        ty = v + (ymax - ymin) * 0.03 if v >= 0 else v - (ymax - ymin) * 0.06
        p.txt(i, ty, val_fmt(v), DARK, 12, "middle", "600")
        p.txt_px(p.sx(i), p.mt + p.ph + 17, lab, DARK, 12, "middle")
    if ylabel:
        p.ylabel(ylabel)
    return p.svg(title)


def fig_softmax():
    labels = ["A", "B", "C", "D"]
    logits = [2.0, 0.5, -1.0, -0.5]
    counts = [math.exp(v) for v in logits]
    s = sum(counts)
    probs = [c / s for c in counts]
    cols = [BLUE, "#5b9bd5", "#9dc3e6", "#c9ddf0"]
    a = _barpanel("1. logits from the net (can be negative)", labels, logits, cols,
                  ymax=2.6, baseline0=False, ylabel="score")
    b = _barpanel("2. e^logit = counts (all positive, order kept)", labels, counts, cols,
                  ymax=max(counts) * 1.25, ylabel="count")
    c = _barpanel("3. counts / sum = probabilities (sum = 1)", labels, probs, cols,
                  ymax=max(probs) * 1.25, ylabel="probability",
                  val_fmt=lambda v: f"{v*100:.0f}%")
    return (f'<div class="row">{a}{b}{c}</div>'
            f'<p class="cap">Tallest bar stays A the whole way through: exp never reorders. '
            f'The negative logits (C, D) become small but strictly positive counts -- so no '
            f'probability is ever 0, and no log ever blows up.</p>')


def fig_product_vs_sum():
    probs = [0.4, 0.1, 0.25, 0.05, 0.3, 0.08]
    run_prod = []
    acc = 1.0
    for p_ in probs:
        acc *= p_
        run_prod.append(acc)
    run_sum = []
    acc = 0.0
    for p_ in probs:
        acc += math.log(p_)
        run_sum.append(acc)
    n = len(probs)

    left = Plot(390, 300, (0.5, n + 0.5), (0, 0.45), ml=58, mr=16, mt=30, mb=42)
    left.box()
    left.xticks(list(range(1, n + 1)), grid=False)
    left.yticks([0, 0.1, 0.2, 0.3, 0.4])
    left.poly([(i + 1, v) for i, v in enumerate(run_prod)], BLUE, 2.6)
    for i, v in enumerate(run_prod):
        left.dot(i + 1, v, BLUE, 3.6)
    left.txt_px(left.sx(4.4), left.sy(0.16), "crashes to a speck", BLUE, 12, "start", "600")
    left.xlabel("characters multiplied in")
    left.ylabel("running PRODUCT")
    left_svg = left.svg("Likelihood = p1 x p2 x p3 x ...  (underflows fast)")

    right = Plot(390, 300, (0.5, n + 0.5), (min(run_sum) - 1, 0.5), ml=58, mr=16, mt=30, mb=42)
    right.box()
    right.xticks(list(range(1, n + 1)), grid=False)
    right.yticks([0, -3, -6, -9, -12])
    right.axis0()
    right.poly([(i + 1, v) for i, v in enumerate(run_sum)], RED, 2.6)
    for i, v in enumerate(run_sum):
        right.dot(i + 1, v, RED, 3.6)
    right.txt_px(right.sx(3.4), right.sy(-2.2), "stays readable; just adds up", RED, 12, "start", "600")
    right.xlabel("characters added in")
    right.ylabel("running SUM of logs")
    right_svg = right.svg("log-likelihood = log p1 + log p2 + ...  (multiply -> add)")
    return f'<div class="row">{left_svg}{right_svg}</div>'


def fig_surprise():
    ps = [0.5, 0.1, 0.01, 0.001]
    labels = ["50%", "10%", "1%", "0.1%"]
    vals = [-math.log(p_) for p_ in ps]
    cols = [GREEN, "#7bbf4f", ORANGE, RED]
    p = Plot(560, 300, (-0.5, len(ps) - 0.5), (0, 8), ml=54, mr=18, mt=30, mb=40)
    p.box()
    p.yticks([0, 2, 4, 6, 8])
    for i, (lab, v, c) in enumerate(zip(labels, vals, cols)):
        p.bar(i, v, 0.32, c)
        p.txt(i, v + 0.28, f"{v:.2f}", DARK, 12, "middle", "600")
        p.txt_px(p.sx(i), p.mt + p.ph + 17, lab, DARK, 12, "middle")
    for i in range(len(ps) - 1):
        ymid = (vals[i] + vals[i + 1]) / 2
        p.txt_px(p.sx(i + 0.5), p.sy(ymid), "+2.30", DARK, 11.5, "middle", "600")
    p.ylabel("surprise  = -log(p)")
    return p.svg("Every 10x rarer event adds the SAME chunk of surprise (+2.30)")


# ---------------------------------------------------------------- page

def build():
    css = """
    :root{--ink:#23252b;--soft:#5b6270;--line:#e3e6ea;--accent:#1f77b4;}
    *{box-sizing:border-box}
    body{font-family:system-ui,'Segoe UI',Arial,sans-serif;line-height:1.62;color:var(--ink);
         max-width:940px;margin:0 auto;padding:34px 22px 90px;background:#fbfbfc}
    h1{font-size:27px;line-height:1.25;margin:0 0 6px}
    h2{font-size:20px;margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--line)}
    h3{font-size:16px;margin:22px 0 4px;color:var(--soft)}
    p{margin:10px 0}
    .lede{color:var(--soft);font-size:15px;margin-top:0}
    .fig{margin:20px 0 6px;text-align:center}
    .row{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:18px 0 6px}
    .cap{color:var(--soft);font-size:14px;text-align:center;max-width:760px;margin:6px auto 0}
    .callout{border-left:4px solid var(--accent);background:#f2f7fc;padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}
    .keep{border-left:4px solid #2ca02c;background:#f1f8f0}
    .origin{border-left:4px solid #e8820c;background:#fdf5ec}
    .worry{border-left:4px solid #d62728;background:#fcf1f1}
    .callout b{color:var(--ink)}
    code{background:#eef0f3;padding:1px 6px;border-radius:5px;font-size:.92em}
    .big{font-size:16px}
    ul{margin:8px 0 8px 2px;padding-left:22px}
    li{margin:5px 0}
    .tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;
         text-transform:uppercase;color:#fff;background:var(--accent);padding:2px 9px;border-radius:20px;margin-bottom:4px}
    .tag.g{background:#2ca02c}.tag.o{background:#e8820c}.tag.r{background:#d62728}
    """

    parts = []
    A = parts.append
    A(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Grokking exp and log</title><style>{css}</style></head><body>")

    A("<h1>exp and log: why changing the numbers doesn't change the answer</h1>")
    A("<p class='lede'>Your exact worry, first: &ldquo;exp and log <i>rescale</i> the value of x. "
      "If the numbers change, how can the result be trusted?&rdquo; This page answers that, "
      "starting from the shape of each function and ending at why they fit softmax + loss perfectly.</p>")

    A("<div class='callout worry'><span class='tag r'>The whole resolution in one line</span>"
      "<p class='big' style='margin:2px 0 0'>Both functions are <b>strictly increasing relabelings</b> "
      "of the number line. A relabeling that never reorders anything keeps the two things we actually "
      "use &mdash; <b>which value is bigger</b>, and <b>where the best answer sits</b> &mdash; and only "
      "changes raw magnitudes, which we never use. Change a ruler's markings however you like; the "
      "tallest kid is still the tallest.</p></div>")

    A("<p>Hold that sentence. Everything below is just making it concrete and then cashing it in twice: "
      "once for <code>exp</code> (building probabilities) and once for <code>log</code> (measuring them).</p>")

    # ---- exp
    A("<h2>1. The shape of e^x</h2>")
    A("<p>Feed <code>e^x</code> any real number and read off four properties straight from the curve:</p>")
    A("<ul>"
      "<li><b>Always positive.</b> The whole real line (negatives included) maps into (0, infinity). "
      "It never touches zero, never goes negative.</li>"
      "<li><b>Always rising.</b> Bigger x gives bigger output, no exceptions &mdash; that's the "
      "&ldquo;never reorders&rdquo; property.</li>"
      "<li><b>Equal steps in x give a constant <i>ratio</i>, not a constant gap.</b> Each <code>+1</code> "
      "in x multiplies the output by <code>e</code> (~2.72). So the gaps grow, but the multiply factor is "
      "fixed. <i>This</i> is the &ldquo;same distance&rdquo; you were sensing &mdash; it's a constant "
      "ratio, not a constant gap.</li>"
      "<li><b>Its slope equals its own height.</b> That's why it steepens forever, and why it's a curve "
      "rather than a line.</li></ul>")
    A(f"<div class='fig'>{fig_exp()}</div>")

    A("<div class='callout origin'><span class='tag o'>where e comes from</span>"
      "<p style='margin:2px 0 0'>Jacob Bernoulli hit <code>e</code> in the 1680s studying compound "
      "interest: interest on your interest on your interest is repeated multiplication, and compounding "
      "it continuously converges to <code>e</code> ~ 2.71828. That's the same &ldquo;each step multiplies&rdquo; "
      "behaviour you see in the curve.</p></div>")

    # ---- log
    A("<h2>2. The shape of log(x)</h2>")
    A("<p><code>log</code> is <code>e^x</code> seen in a mirror. It takes a positive number and asks "
      "&ldquo;<i>e</i> to what power gives this?&rdquo; Read its properties:</p>")
    A("<ul>"
      "<li><b>Input positive, output any real.</b> It maps (0, infinity) onto the whole number line &mdash; "
      "exactly the reverse of exp.</li>"
      "<li><b>log(1) = 0</b>, and it dives toward minus-infinity as x approaches 0.</li>"
      "<li><b>Always rising</b> &mdash; again, never reorders.</li>"
      "<li><b>It turns multiply into add.</b> <code>log(a x b) = log a + log b</code>. Each doubling of x "
      "adds a fixed <code>0.69</code>; each x10 adds a fixed <code>2.30</code>.</li></ul>")
    A(f"<div class='fig'>{fig_log()}</div>")

    A("<div class='callout origin'><span class='tag o'>where log comes from</span>"
      "<p style='margin:2px 0 0'>John Napier published logarithms in 1614 for one reason: to turn brutal "
      "hand-multiplication of huge astronomical numbers into simple addition (via printed log tables and "
      "later the slide rule). &ldquo;Multiply becomes add&rdquo; wasn't a side effect &mdash; it was the "
      "entire point. Keep that; it's exactly why log shows up in the loss.</p></div>")

    A("<h2>3. They are inverses &mdash; a lossless relabel</h2>")
    A("<p>Because <code>log(e^x) = x</code> and <code>e^(log y) = y</code>, the two curves are reflections "
      "across the line <code>y = x</code>. That's the deep reason your worry dissolves: going through exp "
      "and coming back through log lands you <b>exactly</b> where you started. No information is lost. "
      "It's a faithful relabel of the axis, like Celsius&nbsp;&harr;&nbsp;Fahrenheit, not a distortion of "
      "the facts.</p>")
    A(f"<div class='fig'>{fig_inverse()}</div>")

    # ---- the resolver
    A("<h2>4. The picture that answers your question</h2>")
    A("<p>Here is the crux. Imagine sliding one weight <code>w</code> and plotting how good the model is "
      "(top). It rises to a best value, then falls. Now take the <code>log</code> of that height (bottom). "
      "The curve looks totally different &mdash; but the <b>peak sits at the exact same w</b>.</p>")
    A(f"<div class='fig'>{fig_mountain()}</div>")
    A("<div class='callout'><p style='margin:0' class='big'>Stretching the height axis <b>cannot move the "
      "summit</b>. Since training only ever asks &ldquo;<i>which</i> w is best?&rdquo;, and a monotonic "
      "relabel never moves the best point, optimizing the log gives the identical answer as optimizing the "
      "original &mdash; while being far easier to compute with. The scale changed; the answer did not.</p></div>")

    # ---- softmax
    A("<h2>5. Cashing in exp: building probabilities (softmax)</h2>")
    A("<p>The net spits out <b>logits</b> &mdash; arbitrary scores, some negative. We need a probability "
      "distribution: all positive, summing to 1. Watch what each requirement demands, and notice exp is "
      "the tool that satisfies both at once:</p>")
    A("<ul>"
      "<li><b>Must be positive</b> &rarr; exp maps every score (even negative) to a positive number.</li>"
      "<li><b>Must keep the ranking the net learned</b> &rarr; exp is monotonic, so the biggest logit "
      "stays the biggest probability. It doesn't invent an order; it preserves the one the net chose.</li>"
      "<li><b>Must sum to 1</b> &rarr; divide by the total.</li></ul>")
    A(fig_softmax())
    A("<p>What exp <i>decides</i> is only <b>how much</b> more likely a higher score is: a logit one unit "
      "bigger becomes <code>e</code>&times; more count. That's the single modelling choice &mdash; convert "
      "score <i>gaps</i> into probability <i>ratios</i> &mdash; and it's the natural one precisely because "
      "exp turns additive scores into multiplicative weights.</p>")

    # ---- loss
    A("<h2>6. Cashing in log: measuring them (the loss)</h2>")
    A("<p>To score the model on a name, multiply the probabilities it gave each correct next letter. That "
      "product is the <b>likelihood</b>. But a product of many numbers below 1 collapses toward zero almost "
      "instantly (left) &mdash; ugly to read and prone to underflow. Take logs and the product becomes a "
      "tidy running sum (right, Napier's trick):</p>")
    A(fig_product_vs_sum())
    A("<p>Two payoffs, both already proven above: the sum is numerically sane, and &mdash; because log is "
      "monotonic (&sect;4) &mdash; the weights that maximize the product are the same weights that maximize "
      "the sum of logs. We then flip the sign so that <i>low = good</i> (a loss) and average over the "
      "dataset. That is the <b>average negative log-likelihood</b>.</p>")
    A("<h3>Why log also gives a clean notion of &ldquo;surprise&rdquo;</h3>")
    A("<p><code>-log(p)</code> is how surprised the model is by an outcome. Because log turns multiply into "
      "add, every time an event gets 10&times; rarer, the surprise goes up by the <b>same</b> fixed amount "
      "(2.30). Surprise stacks by addition &mdash; which is exactly what a loss you sum over a dataset needs.</p>")
    A(f"<div class='fig'>{fig_surprise()}</div>")

    # ---- close
    A("<h2>7. Why they're a matched pair here</h2>")
    A("<p><code>exp</code> turns scores into positive, correctly-ranked weights you can normalize into "
      "probabilities. <code>log</code> turns those probabilities back into an additive, well-behaved score "
      "to optimize. They're inverses, so the round trip is consistent &mdash; and every step only ever "
      "leans on <i>ordering</i> and <i>positivity</i>, never on the raw magnitudes that changed.</p>")

    A("<div class='callout origin'><span class='tag o'>bonus origin</span>"
      "<p style='margin:2px 0 0'>&ldquo;Softmax&rdquo; is lifted from physics: the Boltzmann distribution "
      "<code>e^(-E/kT)</code> gives the probability a system sits in a state of energy E. Your logits play "
      "the role of (negative) energies. Same exp-then-normalize shape, invented for a completely different "
      "reason a century earlier.</p></div>")

    A("<div class='callout keep'><span class='tag g'>keep this</span>"
      "<p style='margin:2px 0 0'><b>exp and log rescale the numbers but never reorder them and never move "
      "the best point.</b> Softmax needs only positivity + ordering; the loss needs only the location of "
      "the best weights. All of that survives the rescale &mdash; so changing the scale costs nothing, and "
      "buys positivity (exp) and add-instead-of-multiply (log).</p></div>")

    A("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = "llm_output/exp_and_log_grok.html"
    html_str = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}  ({len(html_str)} bytes)")
