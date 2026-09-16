#!/usr/bin/env python3
"""Builds a self-contained teaching HTML page about mean / variance / standard deviation
and what (hpreact - mean)/std does in batch normalization (Lecture 4, ~45:00).

Pure stdlib only (math, html) -> hand-drawn but numerically-exact SVG. No numpy/matplotlib,
so it runs under the free-threaded python3.13t. Run with:  py .kiro/build_batchnorm_std_html.py

Output: llm_output/batchnorm_mean_std_grok.html
"""
import math
import html as _html

# --- dark theme palette ---
BLUE = "#5aa9ee"
RED = "#ff6b6b"
ORANGE = "#f0a54a"
GREEN = "#5ec46f"
PURPLE = "#b28cf0"
GRAY = "#9aa2af"        # muted secondary text / ticks
DARK = "#dfe3ea"        # primary "ink" -> light on dark
FAINT = "#2b303a"       # faint grid lines
PAGE = "#0f1116"        # svg / page background
PANEL = "#191d25"       # plot-area background (slightly lighter card)
LINE = "#3a404c"        # borders


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


def std_pop(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var, math.sqrt(var)


class Plot:
    def __init__(self, w, h, xr, yr, ml=64, mr=24, mt=22, mb=44, bg=PANEL):
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
                 f'fill="{self.bg}" stroke="{LINE}" stroke-width="1"/>')

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
                     f'stroke="#5b616d" stroke-width="1.3"/>')
        if self.x0 <= 0 <= self.x1:
            X = self.sx(0)
            self.add(f'<line x1="{X:.1f}" y1="{self.mt}" x2="{X:.1f}" y2="{self.mt+self.ph}" '
                     f'stroke="#5b616d" stroke-width="1.3"/>')

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
            pts.append(f"{self.sx(x):.1f},{self.sy(y):.1f}")
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def dot(self, x, y, color, r=4.5):
        self.add(f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" '
                 f'fill="{color}" stroke="{PAGE}" stroke-width="1.3"/>')

    def seg(self, x1, y1, x2, y2, color, width=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{self.sx(x1):.1f}" y1="{self.sy(y1):.1f}" '
                 f'x2="{self.sx(x2):.1f}" y2="{self.sy(y2):.1f}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>')

    def arrow(self, x1, y1, x2, y2, color, width=2.0):
        # simple arrow from (x1,y1)->(x2,y2) in data coords
        X1, Y1, X2, Y2 = self.sx(x1), self.sy(y1), self.sx(x2), self.sy(y2)
        ang = math.atan2(Y2 - Y1, X2 - X1)
        ah = 7
        a1 = ang + math.radians(150)
        a2 = ang - math.radians(150)
        self.add(f'<line x1="{X1:.1f}" y1="{Y1:.1f}" x2="{X2:.1f}" y2="{Y2:.1f}" '
                 f'stroke="{color}" stroke-width="{width}"/>')
        self.add(f'<polyline points="{X2+ah*math.cos(a1):.1f},{Y2+ah*math.sin(a1):.1f} '
                 f'{X2:.1f},{Y2:.1f} {X2+ah*math.cos(a2):.1f},{Y2+ah*math.sin(a2):.1f}" '
                 f'fill="none" stroke="{color}" stroke-width="{width}"/>')

    def vspan(self, xa, xb, color, opacity=0.12):
        X = self.sx(xa)
        W = self.sx(xb) - X
        self.add(f'<rect x="{X:.1f}" y="{self.mt}" width="{W:.1f}" height="{self.ph}" '
                 f'fill="{color}" opacity="{opacity}"/>')

    def rect_data(self, xa, ya, xb, yb, color, opacity=1.0, stroke="none", sw=1.0):
        X = min(self.sx(xa), self.sx(xb))
        Y = min(self.sy(ya), self.sy(yb))
        W = abs(self.sx(xb) - self.sx(xa))
        H = abs(self.sy(yb) - self.sy(ya))
        self.add(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{W:.1f}" height="{H:.1f}" '
                 f'fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')

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
                f'fill="{PAGE}"/>{head}{"".join(self.parts)}</svg>')


# ================================================================ figures

SCORES = [30, 40, 50, 60, 70]
M_S, V_S, STD_S = std_pop(SCORES)   # 50, 200, 14.142...


def fig_mean_balance():
    p = Plot(820, 150, (20, 80), (-1, 1), ml=30, mr=30, mt=30, mb=40)
    Y = 0
    # number line
    p.seg(20, Y, 80, Y, DARK, 1.6)
    for v in [20, 30, 40, 50, 60, 70, 80]:
        p.seg(v, -0.08, v, 0.08, GRAY, 1.2)
        p.txt(v, -0.45, str(v), DARK, 12)
    for v in SCORES:
        p.dot(v, Y, BLUE, 6)
    # fulcrum at mean
    Xm = p.sx(M_S)
    Ym = p.sy(Y)
    p.add(f'<polygon points="{Xm:.1f},{Ym+3:.1f} {Xm-9:.1f},{Ym+22:.1f} {Xm+9:.1f},{Ym+22:.1f}" '
          f'fill="{ORANGE}"/>')
    p.txt(M_S, 0.62, "mean = 50", ORANGE, 13, "middle", "700")
    p.txt_px(p.sx(M_S), p.sy(-0.75) + 18, "the balance point: dots left of it pull down, dots right pull up, and they cancel",
             GRAY, 11.5, "middle")
    return p.svg("MEAN: the balance point of the values")


def fig_deviations():
    p = Plot(820, 210, (20, 80), (-1.2, 1.2), ml=30, mr=30, mt=30, mb=36)
    Y = 0
    p.seg(20, Y, 80, Y, DARK, 1.6)
    for v in [20, 30, 40, 50, 60, 70, 80]:
        p.seg(v, -0.06, v, 0.06, GRAY, 1.2)
        p.txt(v, -0.5, str(v), DARK, 11.5)
    # mean line
    p.seg(M_S, -0.95, M_S, 0.95, ORANGE, 1.6, "5,4")
    p.txt(M_S, 1.02, "mean 50", ORANGE, 12, "middle", "700")
    for v in SCORES:
        dev = v - M_S
        col = RED if dev < 0 else (GREEN if dev > 0 else GRAY)
        if dev != 0:
            p.arrow(M_S, 0, v, 0, col, 2.2)
        p.dot(v, 0, BLUE, 6)
        p.txt(v, 0.32, f"{int(dev):+d}" if dev else "0", col, 12.5, "middle", "700")
    p.txt_px(p.sx(50), p.sy(-0.78), "raw deviations:  (-20) + (-10) + 0 + (+10) + (+20)  =  0    <- they cancel, so averaging them is useless",
             DARK, 12, "middle", "600")
    return p.svg("STEP 1: how far is each point from the mean?  (a 'deviation')")


def fig_squares():
    # show squares with area = deviation^2, then variance = average area, std = side of avg square
    devs = [v - M_S for v in SCORES]
    sqs = [d * d for d in devs]
    W, H = 820, 340
    p = Plot(W, H, (0, 5), (0, 460), ml=58, mr=20, mt=30, mb=52)
    p.box()
    p.yticks([0, 100, 200, 300, 400])
    # bars = squared deviations
    cols = [RED, RED, GRAY, GREEN, GREEN]
    for i, (d, sq, c) in enumerate(zip(devs, sqs, cols)):
        xc = i + 0.5
        p.bar(xc, sq, 0.32, c, opacity=0.55)
        p.txt(xc, sq + 18, f"{int(d):+d}^2 = {sq:.0f}", DARK, 11.5, "middle", "700")
        p.txt_px(p.sx(xc), p.mt + p.ph + 18, f"pt {SCORES[i]}", DARK, 11.5, "middle")
    # variance line
    var = sum(sqs) / len(sqs)
    p.seg(0, var, 5, var, ORANGE, 2.0, "6,4")
    p.txt_px(p.sx(4.98), p.sy(var) - 6, f"variance = average area = {var:.0f}", ORANGE, 12.5, "end", "700")
    p.xlabel("each bar's height = (deviation) squared   ->   variance is their average   ->   std = sqrt(variance) = 14.14")
    return p.svg("STEP 2: square each deviation, average them (=variance), sqrt back to a length (=std)")


def fig_dartboard():
    def board(cx, cy, pts, title, sub, col):
        s = []
        for r, o in [(70, 0.10), (48, 0.14), (26, 0.20)]:
            s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" opacity="{o}" '
                     f'stroke="{LINE}" stroke-width="1"/>')
        s.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="none" stroke="{DARK}" stroke-width="1.4"/>')
        s.append(f'<line x1="{cx-84}" y1="{cy}" x2="{cx+84}" y2="{cy}" stroke="{FAINT}"/>')
        s.append(f'<line x1="{cx}" y1="{cy-84}" x2="{cx}" y2="{cy+84}" stroke="{FAINT}"/>')
        for dx, dy in pts:
            s.append(f'<circle cx="{cx+dx}" cy="{cy+dy}" r="4.6" fill="{col}" '
                     f'stroke="{PAGE}" stroke-width="1.2"/>')
        s.append(f'<text x="{cx}" y="{cy-96}" font-size="14" fill="{DARK}" text-anchor="middle" '
                 f'font-weight="700">{esc(title)}</text>')
        s.append(f'<text x="{cx}" y="{cy+108}" font-size="12.5" fill="{col}" text-anchor="middle" '
                 f'font-weight="700">{esc(sub)}</text>')
        return "".join(s)

    tight = [(-6, 4), (8, -5), (2, 9), (-9, -3), (5, 6), (-2, -8)]
    wide = [(-52, 20), (44, -38), (10, 55), (-60, -30), (38, 44), (-20, -58)]
    W, H = 700, 260
    svg = [f'<svg viewBox="0 0 {W} {H}" width="{W}" '
           f'style="max-width:100%;height:auto;font-family:system-ui,Segoe UI,Arial,sans-serif" '
           f'xmlns="http://www.w3.org/2000/svg"><rect width="{W}" height="{H}" fill="{PAGE}"/>']
    svg.append(f'<text x="{W/2}" y="16" font-size="13.5" fill="{DARK}" text-anchor="middle" '
               f'font-weight="600">STD is just &ldquo;how tight is the grouping?&rdquo;</text>')
    svg.append(board(175, 130, tight, "small std", "tight cluster -> small typical distance", GREEN))
    svg.append(board(525, 130, wide, "large std", "scattered -> large typical distance", RED))
    svg.append("</svg>")
    return "".join(svg)


def fig_bell():
    p = Plot(820, 340, (-3.6, 3.6), (0, 0.46), ml=44, mr=24, mt=30, mb=46)
    p.box()

    def g(x):
        return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

    # shaded bands
    def band(a, b, col, op):
        xs = [a + (b - a) * i / 60 for i in range(61)]
        pts = [f"{p.sx(a):.1f},{p.sy(0):.1f}"]
        pts += [f"{p.sx(x):.1f},{p.sy(g(x)):.1f}" for x in xs]
        pts += [f"{p.sx(b):.1f},{p.sy(0):.1f}"]
        p.add(f'<polygon points="{" ".join(pts)}" fill="{col}" opacity="{op}"/>')
    band(-2, 2, BLUE, 0.14)
    band(-1, 1, BLUE, 0.22)
    p.curve(g, -3.5, 3.5, BLUE, 2.6)
    p.axis0()
    for s in [-3, -2, -1, 0, 1, 2, 3]:
        X = p.sx(s)
        p.seg(s, 0, s, -0.012, DARK, 1.2)
        lab = "mean" if s == 0 else (f"{s:+d} std")
        p.txt_px(X, p.sy(0) + 18, lab, DARK, 11.5, "middle")
    p.txt(0, 0.20, "68%", DARK, 13, "middle", "700")
    p.txt(0, 0.055, "95%", DARK, 12.5, "middle", "700")
    p.txt_px(p.sx(0), p.sy(0.43), "'unit gaussian' = mean 0, std 1.  This bell is exactly what BatchNorm forces each neuron into.",
             GRAY, 11.5, "middle")
    return p.svg("Why std is THE ruler: 68% of a bell sits within 1 std, 95% within 2 std")


def fig_matrix():
    # 5 examples x 3 neurons grid, showing dim=0 collapse per column
    data = [
        [30, 5, -2],
        [40, 5, 2],
        [50, 5, -1],
        [60, 5, 1],
        [70, 5, 0],
    ]
    ncol = ["nA", "nB", "nC"]
    means = [sum(r[j] for r in data) / len(data) for j in range(3)]
    stds = [std_pop([r[j] for r in data])[2] for j in range(3)]

    cw, ch = 78, 34
    x0, y0 = 150, 60          # top-left of data cells
    W, H = 760, 430
    s = [f'<svg viewBox="0 0 {W} {H}" width="{W}" '
         f'style="max-width:100%;height:auto;font-family:system-ui,Segoe UI,Arial,sans-serif" '
         f'xmlns="http://www.w3.org/2000/svg"><rect width="{W}" height="{H}" fill="{PAGE}"/>']
    s.append(f'<text x="{W/2}" y="20" font-size="13.5" fill="{DARK}" text-anchor="middle" '
             f'font-weight="600">hpreact is 32x200: rows = examples, columns = neurons  '
             f'(shown here 5x3)</text>')

    # highlight column nA (j=0)
    s.append(f'<rect x="{x0-3}" y="{y0-3}" width="{cw+6}" height="{ch*5+6}" '
             f'fill="{BLUE}" opacity="0.12" stroke="{BLUE}" stroke-width="1.5" rx="4"/>')
    # highlight row ex2 (i=2) for the dim=1 contrast
    ry = y0 + 2 * ch
    s.append(f'<rect x="{x0-3}" y="{ry-3}" width="{cw*3+6}" height="{ch+6}" '
             f'fill="{RED}" opacity="0.10" stroke="{RED}" stroke-width="1.2" rx="4" '
             f'stroke-dasharray="5,4"/>')

    # column headers
    for j, nm in enumerate(ncol):
        cx = x0 + j * cw + cw / 2
        s.append(f'<text x="{cx:.0f}" y="{y0-12}" font-size="12.5" fill="{DARK}" '
                 f'text-anchor="middle" font-weight="700">{nm}</text>')
    # row headers + cells
    for i, row in enumerate(data):
        cy = y0 + i * ch
        s.append(f'<text x="{x0-14}" y="{cy+ch/2+4:.0f}" font-size="12" fill="{DARK}" '
                 f'text-anchor="end">ex{i}</text>')
        for j, v in enumerate(row):
            cx = x0 + j * cw
            s.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="none" '
                     f'stroke="{LINE}" stroke-width="1"/>')
            s.append(f'<text x="{cx+cw/2:.0f}" y="{cy+ch/2+4:.0f}" font-size="12.5" '
                     f'fill="{DARK}" text-anchor="middle">{v}</text>')

    # down arrow under nA
    ax = x0 + cw / 2
    ytop = y0 + 5 * ch + 6
    s.append(f'<line x1="{ax:.0f}" y1="{ytop}" x2="{ax:.0f}" y2="{ytop+26}" stroke="{BLUE}" '
             f'stroke-width="2.4"/>')
    s.append(f'<polyline points="{ax-6:.0f},{ytop+18} {ax:.0f},{ytop+27} {ax+6:.0f},{ytop+18}" '
             f'fill="none" stroke="{BLUE}" stroke-width="2.4"/>')
    s.append(f'<text x="{ax+14:.0f}" y="{ytop+20}" font-size="12" fill="{BLUE}" '
             f'text-anchor="start" font-weight="700">mean(dim=0): collapse the ROWS</text>')

    # results row
    resy = ytop + 40
    s.append(f'<text x="{x0-14}" y="{resy+ch/2+4:.0f}" font-size="11.5" fill="{ORANGE}" '
             f'text-anchor="end" font-weight="700">mean</text>')
    s.append(f'<text x="{x0-14}" y="{resy+ch+ch/2+4:.0f}" font-size="11.5" fill="{ORANGE}" '
             f'text-anchor="end" font-weight="700">std</text>')
    for j in range(3):
        cx = x0 + j * cw
        s.append(f'<rect x="{cx}" y="{resy}" width="{cw}" height="{ch}" fill="{ORANGE}" '
                 f'opacity="0.10" stroke="{ORANGE}" stroke-width="1"/>')
        s.append(f'<text x="{cx+cw/2:.0f}" y="{resy+ch/2+4:.0f}" font-size="12.5" fill="{DARK}" '
                 f'text-anchor="middle" font-weight="600">{fmt(means[j])}</text>')
        s.append(f'<rect x="{cx}" y="{resy+ch}" width="{cw}" height="{ch}" fill="{ORANGE}" '
                 f'opacity="0.10" stroke="{ORANGE}" stroke-width="1"/>')
        s.append(f'<text x="{cx+cw/2:.0f}" y="{resy+ch+ch/2+4:.0f}" font-size="12.5" fill="{DARK}" '
                 f'text-anchor="middle" font-weight="600">{fmt(stds[j])}</text>')
    s.append(f'<text x="{x0+3*cw+16}" y="{resy+ch/2+4:.0f}" font-size="12" fill="{ORANGE}" '
             f'text-anchor="start" font-weight="700">one pair per NEURON  (shape 1x3)</text>')

    # dim=1 note on the right of the red row
    s.append(f'<text x="{x0+3*cw+16}" y="{ry+ch/2+4:.0f}" font-size="11.5" fill="{RED}" '
             f'text-anchor="start" font-weight="700">dim=1 (across a row) would</text>')
    s.append(f'<text x="{x0+3*cw+16}" y="{ry+ch/2+20:.0f}" font-size="11.5" fill="{RED}" '
             f'text-anchor="start">average nA,nB,nC together -> meaningless</text>')

    s.append("</svg>")
    return "".join(s)


def _strip(vals, xr, ticks, title, mean_lbl, extra=None):
    p = Plot(760, 120, xr, (-1, 1), ml=30, mr=30, mt=28, mb=34)
    Y = 0
    p.seg(xr[0], Y, xr[1], Y, DARK, 1.5)
    for t in ticks:
        p.seg(t, -0.08, t, 0.08, GRAY, 1.1)
        p.txt(t, -0.55, fmt(t), DARK, 11)
    m = sum(vals) / len(vals)
    p.seg(m, -0.9, m, 0.9, ORANGE, 1.5, "5,4")
    p.txt(m, 1.0, mean_lbl, ORANGE, 12, "middle", "700")
    for v in vals:
        p.dot(v, 0, BLUE, 6)
    if extra:
        p.txt_px(p.sx((xr[0]+xr[1])/2), p.sy(-0.78), extra, GRAY, 11.5, "middle")
    return p.svg(title)


def fig_zscore():
    raw = SCORES
    centered = [v - M_S for v in raw]
    scaled = [(v - M_S) / STD_S for v in raw]
    a = _strip(raw, (20, 80), [30, 40, 50, 60, 70],
               "1. raw neuron values  (mean 50, std 14.14)", "mean 50",
               "centered at 50, spread of ~14")
    b = _strip(centered, (-30, 30), [-20, -10, 0, 10, 20],
               "2. subtract the mean  ->  now centered at 0  (spread still 14.14)", "mean 0",
               "same shape, just slid left by 50")
    c = _strip(scaled, (-2.2, 2.2), [-2, -1, 0, 1, 2],
               "3. divide by std  ->  spread squeezed to 1  (UNIT GAUSSIAN: mean 0, std 1)", "mean 0",
               "now every value is measured in 'std units': -1.41, -0.71, 0, +0.71, +1.41")
    return a + b + c


def fig_tanh():
    def th(x):
        return math.tanh(x)
    W, H = 820, 360
    p = Plot(W, H, (-8, 8), (-1.25, 1.25), ml=48, mr=24, mt=30, mb=44)
    p.box()
    p.xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
    p.yticks([-1, -0.5, 0, 0.5, 1])
    p.axis0()
    # saturated zones
    p.vspan(-8, -2.2, RED, 0.10)
    p.vspan(2.2, 8, RED, 0.10)
    p.vspan(-2.2, 2.2, GREEN, 0.10)
    p.curve(th, -8, 8, BLUE, 2.8)
    p.txt_px(p.sx(-5.1), p.sy(-1.12), "saturated: slope~0, no gradient", RED, 11.5, "middle", "700")
    p.txt_px(p.sx(5.1), p.sy(1.12), "saturated", RED, 11.5, "middle", "700")
    p.txt_px(p.sx(0), p.sy(1.12), "active zone", GREEN, 11.5, "middle", "700")
    # raw wide inputs (std ~14 would be off-chart; show std~4 spread landing in tails)
    for v in [-6.5, -4.5, 5, 6.8]:
        p.dot(v, th(v), RED, 5)
    # unit-gaussian inputs land in the middle
    for v in [-1.4, -0.7, 0.3, 1.1]:
        p.dot(v, th(v), GREEN, 5)
    p.xlabel("hpreact going into tanh")
    return p.svg("Why we bother: unit-gaussian inputs (green) land in tanh's active zone; wide raw inputs (red) saturate")


# ================================================================ page

def build():
    css = """
    :root{--ink:#dfe3ea;--soft:#9aa2af;--line:#2b303a;--accent:#5aa9ee;--bg:#0f1116;--card:#181b22;}
    *{box-sizing:border-box}
    body{font-family:system-ui,'Segoe UI',Arial,sans-serif;line-height:1.62;color:var(--ink);
         max-width:940px;margin:0 auto;padding:34px 22px 90px;background:var(--bg)}
    h1{font-size:26px;line-height:1.25;margin:0 0 6px;color:#f2f4f8}
    h2{font-size:20px;margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--line);color:#f2f4f8}
    h3{font-size:16px;margin:22px 0 4px;color:var(--soft)}
    p{margin:10px 0}
    .lede{color:var(--soft);font-size:15px;margin-top:0}
    .fig{margin:18px 0 6px;text-align:center}
    .row{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:18px 0 6px}
    .cap{color:var(--soft);font-size:14px;text-align:center;max-width:780px;margin:6px auto 0}
    .callout{border-left:4px solid var(--accent);background:#132030;padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}
    .keep{border-left:4px solid #5ec46f;background:#13251a}
    .origin{border-left:4px solid #f0a54a;background:#291f12}
    .worry{border-left:4px solid #ff6b6b;background:#2a1618}
    .callout b{color:#f2f4f8}
    code{background:#262b34;color:#e6e9ef;padding:1px 6px;border-radius:5px;font-size:.92em}
    .big{font-size:16px}
    ul{margin:8px 0 8px 2px;padding-left:22px}
    li{margin:5px 0}
    table.map{border-collapse:collapse;margin:14px auto;font-size:14px}
    table.map td,table.map th{border:1px solid var(--line);padding:7px 13px;text-align:left}
    table.map th{background:#20252e}
    .tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;
         text-transform:uppercase;color:#0f1116;background:var(--accent);padding:2px 9px;border-radius:20px;margin-bottom:4px}
    .tag.g{background:#5ec46f}.tag.o{background:#f0a54a}.tag.r{background:#ff6b6b}.tag.p{background:#b28cf0}
    """
    P = []
    A = P.append
    A(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Mean, std and (x-mean)/std in BatchNorm</title><style>{css}</style></head><body>")

    A("<h1>Standard deviation &amp; <code>(hpreact&nbsp;&minus;&nbsp;mean)&nbsp;/&nbsp;std</code>: what BatchNorm is doing at 45:00</h1>")
    A("<p class='lede'>You already own the mean. This page builds std from nothing, explains why "
      "Karpathy takes it <i>down the batch</i> (dim=0), and shows what the whole line does &mdash; "
      "each idea with a picture, exact numbers computed in Python.</p>")

    A("<div class='callout worry'><span class='tag r'>the whole thing in one breath</span>"
      "<p class='big' style='margin:2px 0 0'><b>Standard deviation = the typical distance of the points "
      "from their average.</b> Karpathy computes mean &amp; std <b>per neuron</b> (one number per column, "
      "over the 32 examples) because a neuron is a <i>feature</i>, and its health is a property measured "
      "across many examples. Then <code>(hpreact &minus; mean)/std</code> rewrites every firing value as "
      "&ldquo;how many typical-distances am I from my neuron's average&rdquo; &mdash; which forces each "
      "neuron to <b>mean&nbsp;0, std&nbsp;1</b>, the shape tanh likes best.</p></div>")

    # ---- mean
    A("<h2>0. Mean, just to set the stage</h2>")
    A("<p>The mean is the <b>balance point</b> of the values. If the dots were weights on a ruler, the "
      "mean is where you'd put the fulcrum so it doesn't tip. Toy neuron firing on 5 examples: "
      "<code>30, 40, 50, 60, 70</code> &rarr; mean <b>50</b>.</p>")
    A(f"<div class='fig'>{fig_mean_balance()}</div>")

    # ---- variance / std
    A("<h2>1. Standard deviation: &ldquo;how spread out&rdquo;, built step by step</h2>")
    A("<p>Mean tells you the center. It says nothing about whether the points hug that center or fly all "
      "over. <code>30,40,50,60,70</code> and <code>48,49,50,51,52</code> have the <i>same</i> mean but "
      "feel completely different. Std is the number that captures that difference. Build it in two moves.</p>")

    A("<h3>Move 1 &mdash; measure each point's distance from the mean</h3>")
    A(f"<div class='fig'>{fig_deviations()}</div>")
    A("<p>Natural first idea: just average these distances. It <b>fails</b> &mdash; the points below the "
      "mean give negative distances, the ones above give positive, and they cancel to exactly 0 (they "
      "must, that's what &ldquo;balance point&rdquo; means). So raw distances can't measure spread.</p>")

    A("<h3>Move 2 &mdash; square, average, square-root</h3>")
    A("<p>Fix the cancelling by <b>squaring</b> each distance (a square is always positive, and it "
      "punishes far-away points harder). Average the squares &mdash; that average is the <b>variance</b> "
      "(here 200). Variance is now in &ldquo;squared units,&rdquo; so finally take the <b>square root</b> "
      "to get back to the original scale: that's the <b>standard deviation</b> = "
      f"<b>{STD_S:.2f}</b>.</p>")
    A(f"<div class='fig'>{fig_squares()}</div>")
    A("<div class='callout'><p style='margin:0'>Read std in plain words: &ldquo;the points of this neuron "
      "sit, <b>typically about 14 away</b> from their average of 50.&rdquo; Small std = tight cluster; "
      "big std = scattered. That's the entire idea.</p></div>")
    A(f"<div class='fig'>{fig_dartboard()}</div>")

    A("<div class='callout origin'><span class='tag o'>where it comes from</span>"
      "<p style='margin:2px 0 0'>Karl Pearson coined the very name <b>&ldquo;standard deviation&rdquo;</b> "
      "in 1893 &mdash; before that people used clunkier terms like &ldquo;mean error.&rdquo; And the choice "
      "to <i>square</i> (instead of taking absolute values) traces to Gauss: squared error is what makes "
      "the math close cleanly and is exactly what produces the bell curve below. Squaring wasn't arbitrary; "
      "it's the choice that made the whole theory of the normal distribution work.</p></div>")

    A("<h3>Why std is the natural ruler: the bell curve</h3>")
    A("<p>For a bell-shaped (gaussian) spread, std has a crisp meaning: about <b>68%</b> of the mass lands "
      "within 1 std of the mean, about <b>95%</b> within 2 std. So &ldquo;2 std away&rdquo; means the same "
      "level of rare no matter the raw scale. <b>&ldquo;Unit gaussian&rdquo; = mean 0, std 1</b> &mdash; "
      "that phrase Karpathy keeps saying is just <i>this</i> bell, recentred and rescaled.</p>")
    A(f"<div class='fig'>{fig_bell()}</div>")

    # ---- dim=0
    A("<h2>2. Why the mean/std is taken <code>dim=0</code> (down the batch)</h2>")
    A("<p>Your instinct: &ldquo;we usually summarize a <i>record</i>.&rdquo; Here's the flip that fixes it "
      "&mdash; in almost every dataset you actually summarize a <b>feature down the rows</b>, not a record "
      "across its features. To normalize people's heights you take the mean/std of the <i>height column</i> "
      "over all people; you'd never average one person's height, weight and age together. <b>A neuron is "
      "that feature/column.</b> So going down the column (dim=0) is the normal move, and going across a row "
      "would be the strange one.</p>")
    A(f"<div class='fig'>{fig_matrix()}</div>")
    A("<p><code>hpreact.mean(0, keepdim=True)</code> collapses the <b>row</b> dimension: for each of the 200 "
      "neurons it averages that neuron over all 32 examples, giving a <b>1&times;200</b> row of per-neuron "
      "means (and likewise for std). <code>keepdim=True</code> keeps the shape 1&times;200 so it broadcasts "
      "cleanly back over the 32 rows when we subtract.</p>")
    A("<div class='callout'><span class='tag'>the analogy, mapped</span>"
      "<table class='map'>"
      "<tr><th>picture</th><th>the model</th><th>your notebook</th></tr>"
      "<tr><td>a student</td><td>one neuron</td><td>a column of <code>hpreact</code></td></tr>"
      "<tr><td>one test they sat</td><td>one training example</td><td>a row of <code>hpreact</code></td></tr>"
      "<tr><td>the student's avg &amp; consistency across all their tests</td>"
      "<td>that neuron's mean &amp; std over the batch</td><td><code>.mean(0)</code>, <code>.std(0)</code></td></tr>"
      "</table>"
      "<p style='margin:6px 0 0'>Averaging <i>across a row</i> (dim=1) would be &ldquo;for one test, average "
      "this student, that student, and a third together&rdquo; &mdash; mixing unrelated neurons, which means "
      "nothing.</p></div>")
    A("<div class='callout p' style='border-left-color:#b28cf0;background:#1e1830'>"
      "<span class='tag p'>the subtle catch (bonus)</span>"
      "<p style='margin:2px 0 0'>Because each neuron's mean/std is computed <i>from the batch</i>, one "
      "example's normalized value now depends on <i>which other examples</i> happen to share its batch. That "
      "coupling is unusual (and is why BatchNorm needs a separate running mean/std at test time, which "
      "Karpathy handles right after this). It also acts as a mild regularizer. File it away &mdash; it's the "
      "root of the train/test wrinkle coming up.</p></div>")

    # ---- z-score
    A("<h2>3. What <code>(hpreact &minus; mean) / std</code> actually gives</h2>")
    A("<p>Two moves on each neuron's column, and both are things you now understand:</p>")
    A("<ul>"
      "<li><b>&minus; mean</b>: slide the values so their center sits at 0. Now each number reads as "
      "&ldquo;how far above/below this neuron's average.&rdquo;</li>"
      "<li><b>&divide; std</b>: rescale so the typical distance becomes exactly 1. Now each number reads as "
      "&ldquo;how many <i>typical distances</i> (std units) from average&rdquo; &mdash; this is called a "
      "<b>z-score</b>.</li></ul>")
    A(f"<div class='fig'>{fig_zscore()}</div>")
    A("<p>The output column always comes out at <b>mean 0, std 1</b> &mdash; a unit gaussian &mdash; no "
      "matter how big or shifted the raw values were. It's the same idea as <b>grading on a curve</b>: a "
      "raw mark of 85 tells you little, but &ldquo;you were 1.4 std above the class average&rdquo; places "
      "you exactly. z-scores are marks re-expressed in <i>surprise units</i>.</p>")

    A("<h3>Why force that shape at all? Look at tanh</h3>")
    A("<p>This closes the loop with the first half of the lecture. If <code>hpreact</code> is spread too "
      "wide, most values land in tanh's flat tails (output stuck near &plusmn;1, slope ~0, <b>no gradient "
      "flows</b> &mdash; the &ldquo;saturated tanh&rdquo; problem). Squeezing every neuron to unit gaussian "
      "drops the values into tanh's active middle band, where the curve actually bends and gradients "
      "survive.</p>")
    A(f"<div class='fig'>{fig_tanh()}</div>")

    A("<div class='callout keep'><span class='tag g'>keep this</span>"
      "<p style='margin:2px 0 0'><b>std = typical distance from the mean</b> (square the distances so they "
      "don't cancel, average = variance, sqrt back = std). <b>dim=0 = per-neuron</b>, because a neuron is a "
      "feature and features are summarized down the rows. <b><code>(x&minus;mean)/std</code> = z-score</b>, "
      "which re-expresses each value in std-units and lands every neuron at mean&nbsp;0, std&nbsp;1 &mdash; "
      "the shape that keeps tanh awake. The learnable <code>bngain</code>/<code>bnbias</code> Karpathy adds "
      "right after just let the network scale/shift away from that default if it wants to.</p></div>")

    A("</body></html>")
    return "".join(P)


if __name__ == "__main__":
    out = "llm_output/batchnorm_mean_std_grok.html"
    html_str = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}  ({len(html_str)} bytes)")
