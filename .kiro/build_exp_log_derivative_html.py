#!/usr/bin/env python3
"""Builds a self-contained teaching HTML page about the DERIVATIVES of exp and log,
framed in the user's own 'wiggle the input, how much does the output wiggle?' language
(the micrograd / lecture 1 backprop intuition).

Pure stdlib only (math, html) -> hand-drawn SVG. No numpy / matplotlib, so it runs under
the free-threaded Python 3.13t interpreter. Run with:  py .kiro/build_exp_log_derivative_html.py

Output: llm_output/exp_log_derivative_grok.html
"""
import math
import html as _html

BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#e8820c"
GREEN = "#2ca02c"
PURPLE = "#7d3ac1"
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
    def __init__(self, w, h, xr, yr, ml=64, mr=24, mt=30, mb=44, bg="#ffffff"):
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

    def curve(self, fn, xa, xb, color, width=2.6, n=300, dash=None):
        pts = []
        for i in range(n + 1):
            x = xa + (xb - xa) * i / n
            try:
                y = fn(x)
            except ValueError:
                continue
            pts.append(f"{self.sx(x):.1f},{self.sy(y):.1f}")
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def dot(self, x, y, color, r=4.5):
        self.add(f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" '
                 f'fill="{color}" stroke="#ffffff" stroke-width="1.3"/>')

    def seg(self, x1, y1, x2, y2, color, width=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{self.sx(x1):.1f}" y1="{self.sy(y1):.1f}" '
                 f'x2="{self.sx(x2):.1f}" y2="{self.sy(y2):.1f}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def rise_run(self, x0, y0, dx, slope, color, run_color=GRAY, lbl=None):
        """Draw the little right-triangle: horizontal run dx, vertical rise = slope*dx,
        starting at (x0,y0). Labels the vertical rise."""
        y1 = y0 + slope * dx
        # run (horizontal, along the input step)
        self.seg(x0, y0, x0 + dx, y0, run_color, 1.6)
        # rise (vertical, the output wiggle)
        self.seg(x0 + dx, y0, x0 + dx, y1, color, 3.0)
        # tangent hypotenuse (the local slope of the curve)
        self.seg(x0, y0, x0 + dx, y1, color, 1.4, "4,3")
        self.dot(x0, y0, color, 4)
        if lbl:
            self.add(f'<text x="{self.sx(x0+dx)+7:.1f}" y="{self.sy((y0+y1)/2)+4:.1f}" '
                     f'font-size="12" fill="{color}" text-anchor="start" '
                     f'font-weight="700">{esc(lbl)}</text>')

    def txt(self, x, y, s, color=DARK, size=12.5, anchor="middle", weight="normal"):
        self.add(f'<text x="{self.sx(x):.1f}" y="{self.sy(y):.1f}" font-size="{size}" '
                 f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def txt_px(self, X, Y, s, color=DARK, size=12.5, anchor="middle", weight="normal"):
        self.add(f'<text x="{X:.1f}" y="{Y:.1f}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def svg(self, title=None):
        head = ""
        if title:
            head = (f'<text x="{self.w/2:.1f}" y="16" font-size="13.5" fill="{DARK}" '
                    f'text-anchor="middle" font-weight="600">{esc(title)}</text>')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'style="max-width:100%;height:auto;font-family:system-ui,Segoe UI,Arial,sans-serif" '
                f'xmlns="http://www.w3.org/2000/svg"><rect width="{self.w}" height="{self.h}" '
                f'fill="#ffffff"/>{head}{"".join(self.parts)}</svg>')


# ---------------------------------------------------------------- figures

def fig_recap():
    """The two rules he already owns, drawn as wiggle triangles on straight lines."""
    W = 400
    # sum: y = x + 1, slope 1
    a = Plot(W, 300, (-0.4, 4.2), (-0.4, 5.2), mt=30, mb=40)
    a.box(); a.xticks([0, 1, 2, 3, 4]); a.yticks([0, 1, 2, 3, 4, 5]); a.axis0()
    a.curve(lambda x: x + 1, -0.4, 4.0, BLUE, 2.6)
    dx = 0.7
    for x0 in (1, 2.6):
        a.rise_run(x0, x0 + 1, dx, 1.0, RED, lbl=f"+{dx:.1f}")
    a.txt_px(a.sx(0.15), a.sy(4.7), "wiggle in = wiggle out  ->  slope 1", DARK, 12, "start", "600")
    a.xlabel("x"); a.ylabel("x + 1")
    A = a.svg("sum:  d/dx (x + 1) = 1   (the step passes straight through)")

    # multiply: y = 3x, slope 3
    b = Plot(W, 300, (-0.4, 4.2), (-0.8, 11), mt=30, mb=40)
    b.box(); b.xticks([0, 1, 2, 3, 4]); b.yticks([0, 3, 6, 9]); b.axis0()
    b.curve(lambda x: 3 * x, -0.2, 3.4, BLUE, 2.6)
    for x0 in (0.7, 2.0):
        b.rise_run(x0, 3 * x0, dx, 3.0, RED, lbl=f"+{3*dx:.1f}")
    b.txt_px(b.sx(0.15), b.sy(10.3), "1 step in  ->  3 steps out  ->  slope 3", DARK, 12, "start", "600")
    b.xlabel("x"); b.ylabel("3 * x")
    B = b.svg("multiply:  d/dx (3x) = 3   (each input step is worth 3)")
    return f'<div class="row">{A}{B}</div>'


def fig_exp_wiggle():
    p = Plot(820, 400, (-0.5, 2.75), (-0.6, 10.4), mt=30)
    p.box()
    p.xticks([0, 1, 2])
    p.yticks([0, 1, 2, 4, math.e, math.e ** 2], fmtf=lambda v: {1: "1", math.e: "e", math.e ** 2: "e^2"}.get(v, fmt(v)))
    p.axis0()
    p.curve(math.exp, -0.5, 2.55, BLUE, 2.8)
    dx = 0.28
    for x0 in (0.0, 1.0, 2.0):
        h = math.exp(x0)
        rise = h * dx
        p.rise_run(x0, h, dx, h, RED, lbl=f"wiggle = {rise:.2f}")
        # faint marker of the height it equals
        p.seg(p.x0, h, x0, h, GRAY, 1, "2,3")
    p.txt_px(p.sx(0.05), p.sy(9.6), "same input step everywhere; output wiggle = the current height", DARK, 12.5, "start", "700")
    p.xlabel("x   (same size step at each dot)")
    p.ylabel("e^x")
    return p.svg("e^x :  output wiggle GROWS with height  ->  slope = the value itself")


def fig_log_wiggle():
    p = Plot(820, 400, (0.0, 8.6), (-1.5, 2.6), mt=30)
    p.box()
    p.xticks([0.5, 1, 2, 4, 8], fmtf=lambda v: fmt(v))
    p.yticks([-1, 0, 1, 2])
    p.axis0()
    p.curve(math.log, 0.06, 8.5, RED, 2.8)
    dx = 0.7
    for x0 in (0.5, 1.0, 2.0, 4.0):
        y0 = math.log(x0)
        slope = 1.0 / x0
        rise = slope * dx
        p.rise_run(x0, y0, dx, slope, BLUE, lbl=f"{rise:.2f}")
    p.txt_px(p.sx(2.6), p.sy(-1.05), "same input step; output wiggle = step / x  =  the % change in x", DARK, 12.5, "start", "700")
    p.xlabel("x   (same size step at each dot)")
    p.ylabel("log(x)")
    return p.svg("log(x) :  output wiggle SHRINKS as x grows  ->  slope = 1 / x")


def fig_log_percent():
    """Equal ratios in x -> equal gaps in log. Why 'percent change' is the right lens."""
    p = Plot(820, 320, (0.0, 9.0), (-0.4, 2.4), mt=30, mb=44)
    p.box()
    p.xticks([1, 2, 4, 8])
    p.yticks([0, 0.69, 1.39, 2.08], fmtf=lambda v: fmt(v))
    p.axis0()
    p.curve(math.log, 0.06, 8.8, RED, 2.6)
    xs = [1, 2, 4, 8]
    ys = [math.log(x) for x in xs]
    for x, y in zip(xs, ys):
        p.seg(x, 0, x, y, GRAY, 1, "3,3")
        p.seg(p.x0, y, x, y, GRAY, 1, "3,3")
        p.dot(x, y, RED)
    for i in range(len(xs) - 1):
        ymid = (ys[i] + ys[i + 1]) / 2
        p.txt_px(p.ml + 8, p.sy(ymid) + 4, "+0.69", GREEN, 12, "start", "700")
        xmid = (xs[i] + xs[i + 1]) / 2
        p.txt_px(p.sx(xmid), p.mt + p.ph + 30, "x2", PURPLE, 12, "middle", "700")
    p.xlabel("x   (each jump is x2  =  +100%)")
    p.ylabel("log(x)")
    return p.svg("Equal MULTIPLIES in x  ->  equal ADDS in log. log measures percentage, not amount.")


def fig_mirror_slopes():
    p = Plot(820, 440, (-2.6, 5.6), (-2.6, 5.6), mt=30)
    p.box()
    p.xticks([-2, -1, 0, 1, 2, 3, 4, 5])
    p.yticks([-2, -1, 0, 1, 2, 3, 4, 5])
    p.axis0()
    p.seg(-2.6, -2.6, 5.6, 5.6, GRAY, 1.3, "5,4")
    p.txt_px(p.sx(4.7), p.sy(5.15), "mirror  y = x", GRAY, 12, "middle")
    p.curve(math.exp, -2.6, 1.55, BLUE, 2.6)
    p.curve(math.log, 0.06, 5.6, RED, 2.6)
    p.txt_px(p.sx(1.15), p.sy(4.7), "e^x", BLUE, 15, "middle", "700")
    p.txt_px(p.sx(4.9), p.sy(1.35), "log(x)", RED, 15, "middle", "700")

    # a matched pair reflected across y=x: (a, e^a) on exp  <->  (e^a, a) on log
    a = 1.0
    h = math.exp(a)  # = e ~ 2.718, the slope of exp there
    # exp point + tangent slope h
    dx = 0.55
    p.seg(a, h, a + dx, h, GRAY, 1.4)
    p.seg(a + dx, h, a + dx, h + h * dx, BLUE, 3.0)
    p.dot(a, h, BLUE, 5)
    p.txt_px(p.sx(a) - 6, p.sy(h) + 4, f"slope = {h:.2f}", BLUE, 12, "end", "700")
    # log point (h, a): slope = 1/h
    p.seg(h, a, h + dx, a, GRAY, 1.4)
    p.seg(h + dx, a, h + dx, a + (1 / h) * dx, RED, 3.0)
    p.dot(h, a, RED, 5)
    p.txt_px(p.sx(h) + 8, p.sy(a) + 4, f"slope = 1/{h:.2f} = {1/h:.2f}", RED, 12, "start", "700")
    # connector
    p.seg(a, h, h, a, GREEN, 1.2, "2,3")
    p.xlabel("x")
    p.ylabel("output")
    return p.svg("Mirror across y = x swaps rise and run  ->  slope flips to 1/slope")


# ---------------------------------------------------------------- page

def build():
    css = """
    :root{--ink:#23252b;--soft:#5b6270;--line:#e3e6ea;--accent:#1f77b4;}
    *{box-sizing:border-box}
    body{font-family:system-ui,'Segoe UI',Arial,sans-serif;line-height:1.62;color:var(--ink);
         max-width:940px;margin:0 auto;padding:34px 22px 90px;background:#fbfbfc}
    h1{font-size:26px;line-height:1.25;margin:0 0 6px}
    h2{font-size:20px;margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--line)}
    h3{font-size:16px;margin:22px 0 4px;color:var(--soft)}
    p{margin:10px 0}
    .lede{color:var(--soft);font-size:15px;margin-top:0}
    .fig{margin:20px 0 6px;text-align:center}
    .row{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:18px 0 6px}
    .cap{color:var(--soft);font-size:14px;text-align:center;max-width:780px;margin:6px auto 0}
    .callout{border-left:4px solid var(--accent);background:#f2f7fc;padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}
    .keep{border-left:4px solid #2ca02c;background:#f1f8f0}
    .origin{border-left:4px solid #e8820c;background:#fdf5ec}
    .callout b{color:var(--ink)}
    code{background:#eef0f3;padding:1px 6px;border-radius:5px;font-size:.92em}
    .big{font-size:16px}
    ul{margin:8px 0 8px 2px;padding-left:22px} li{margin:5px 0}
    table{border-collapse:collapse;margin:16px auto;font-size:14.5px}
    th,td{border:1px solid var(--line);padding:8px 14px;text-align:left}
    th{background:#f2f7fc}
    .tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;
         text-transform:uppercase;color:#fff;background:var(--accent);padding:2px 9px;border-radius:20px;margin-bottom:4px}
    .tag.g{background:#2ca02c}.tag.o{background:#e8820c}
    """
    P = []
    A = P.append
    A(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Grokking the derivative of exp and log</title><style>{css}</style></head><body>")

    A("<h1>The derivative of exp and log, in your wiggle language</h1>")
    A("<p class='lede'>Same question you used for + and &times;: <b>nudge the input by a tiny step, "
      "how much does the output wiggle?</b> The ratio (out-wiggle / in-wiggle) is the derivative. "
      "Let's carry that straight into e^x and log.</p>")

    A("<h2>0. The two you already own</h2>")
    A("<p>For a <b>sum</b>, the step passes straight through: nudge x, the output moves the same amount, "
      "so the slope is 1. For a <b>multiply</b>, each input step is worth &lsquo;the other factor&rsquo; "
      "of output, so <code>3x</code> has slope 3. Notice both slopes are <i>constant</i> &mdash; straight "
      "lines. exp and log are curves, so their slope will <i>change</i> as you move along.</p>")
    A(fig_recap())

    A("<h2>1. e^x : the output wiggles by its own height</h2>")
    A("<p>Take the multiply idea and make the multiplier the output itself. Ask &lsquo;nudge x a little, "
      "how much does e^x move?&rsquo; Answer: by an amount equal to <b>how tall e^x already is</b>. "
      "That is the whole meaning of <code>d/dx(e^x) = e^x</code>.</p>")
    A(f"<div class='fig'>{fig_exp_wiggle()}</div>")
    A("<p class='cap'>Same size input step at every dot. The output jump is tiny down low and huge up high "
      "&mdash; because the jump equals the current height.</p>")
    A("<div class='callout'><p style='margin:0' class='big'><b>Percentage lens:</b> a step in x doesn't add "
      "a fixed <i>amount</i>, it adds a fixed <i>percentage</i>. Where e^x = 5 a nudge grows it by 5&times;step; "
      "where e^x = 100 the same nudge grows it by 100&times;step &mdash; same %, twenty times the wiggle. So "
      "&lsquo;how much it wiggles&rsquo; = &lsquo;how big it is&rsquo;.</p></div>")
    A("<div class='callout origin'><span class='tag o'>physical anchor</span>"
      "<p style='margin:2px 0 0'>A bank balance on continuous interest. $100 earns 20&times; more dollars "
      "per second than $5 &mdash; same <i>rate</i>, bigger absolute growth because there's more to grow. "
      "<code>e</code> is the base tuned so that rate is exactly &lsquo;1 per unit of x&rsquo;, which makes "
      "the growth speed numerically equal the balance. (Bernoulli's compound-interest limit "
      "<code>(1 + 1/n)^n &rarr; e</code>.)</p></div>")

    A("<h2>2. log(x) : the output wiggles by the % change in x  (= 1/x)</h2>")
    A("<p>Flip it. exp says &lsquo;<b>add</b> to x &rarr; <b>multiply</b> the output&rsquo;. log is the "
      "mirror: &lsquo;<b>multiply</b> x &rarr; <b>add</b> to the output&rsquo;. So log reacts to the "
      "<b>percentage</b> change in x, not the absolute change. Nudge x by Δx; the fraction that step "
      "represents is <code>Δx / x</code>, and that's how much log moves. Divide the step by x &rarr; "
      "slope <code>= 1/x</code>.</p>")
    A(f"<div class='fig'>{fig_log_wiggle()}</div>")
    A("<p class='cap'>Same size input step again. Down at x = 0.5 the step is a big % of x, so log jumps a "
      "lot. Up at x = 4 it's a tiny %, so log barely moves. Small x = twitchy, big x = sluggish.</p>")
    A("<p>Here's the &lsquo;percentage&rsquo; claim made literal: equal <i>multiplies</i> in x become equal "
      "<i>adds</i> in log.</p>")
    A(f"<div class='fig'>{fig_log_percent()}</div>")
    A("<div class='callout origin'><span class='tag o'>where this shows up</span>"
      "<p style='margin:2px 0 0'>Every &lsquo;log scale&rsquo; you've met &mdash; decibels, Richter, pH, "
      "stock returns &mdash; exists for this exact reason: they turn <i>percentage / multiplicative</i> "
      "change into <i>equal steps</i> of distance. Napier built logs in 1614 to turn multiplication into "
      "addition, and this 1/x slope is the calculus fingerprint of that trick.</p></div>")

    A("<h2>3. Why they're reciprocals: the mirror</h2>")
    A("<p>exp's wiggle <b>grows</b> with size; log's wiggle <b>shrinks</b> with size &mdash; and the two "
      "slopes are reciprocals (<code>h</code> vs <code>1/h</code>). That's not a coincidence: log is exp "
      "reflected across <code>y = x</code>. Reflecting a ramp swaps its rise and its run, so a slope of "
      "<code>h</code> becomes a slope of <code>1/h</code>.</p>")
    A(f"<div class='fig'>{fig_mirror_slopes()}</div>")
    A("<p class='cap'>The blue point sits on e^x with slope e (&asymp; 2.72). Reflect it across y = x and you "
      "land on the red log curve, where the slope is 1/e (&asymp; 0.37). Rise and run traded places.</p>")

    A("<h2>4. Keep this</h2>")
    A("<table>"
      "<tr><th>function</th><th>wiggle question</th><th>output wiggle per input step</th><th>derivative</th></tr>"
      "<tr><td>x + c</td><td>step passes through</td><td>1</td><td>1</td></tr>"
      "<tr><td>a &middot; x</td><td>each step worth a</td><td>a</td><td>a</td></tr>"
      "<tr><td>e^x</td><td>grows with height</td><td>the current value</td><td>e^x</td></tr>"
      "<tr><td>log(x)</td><td>the % change in x</td><td>step / x</td><td>1 / x</td></tr>"
      "</table>")
    A("<div class='callout keep'><span class='tag g'>one-liners</span>"
      "<p style='margin:2px 0 0'><b>e^x:</b> the output wiggles by its own current height (so it steepens "
      "forever). <b>log(x):</b> the output wiggles by the <i>percentage</i> change in x, which is step / x, "
      "so it flattens as x grows. They're mirror images, which is why one slope is the other flipped.</p></div>")

    A("</body></html>")
    return "".join(P)


if __name__ == "__main__":
    out = "llm_output/exp_log_derivative_grok.html"
    html_str = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}  ({len(html_str)} bytes)")
