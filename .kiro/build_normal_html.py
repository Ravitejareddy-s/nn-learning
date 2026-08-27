#!/usr/bin/env python3
"""Builds a self-contained teaching HTML page about MEAN and STD (the normal / Gaussian
distribution) in the context of torch.randn for weight init. Pure stdlib only, reuses the
SVG Plot toolkit from build_exp_log_html.py. Runs under python3.13t (no numpy/matplotlib).

Output: llm_output/mean_and_std_grok.html
"""
import math
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_exp_log_html import Plot, fmt, esc, BLUE, RED, ORANGE, GREEN, GRAY, DARK, FAINT  # noqa: E402

PURPLE = "#7a4fbf"
TEAL = "#1b9e8f"

SQRT2PI = math.sqrt(2 * math.pi)


def npdf(x, mu=0.0, sd=1.0):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * SQRT2PI)


# ---------------------------------------------------------------- figures

def fig_bell_bands():
    p = Plot(820, 400, (-4, 4), (-0.02, 0.46), mt=30, mb=46)
    p.box()
    p.xticks([-3, -2, -1, 0, 1, 2, 3])
    p.yticks([0, 0.1, 0.2, 0.3, 0.4])
    # shaded +-1 std region (the 68% chunk)
    p.vspan(-1, 1, GREEN, 0.16)
    # the bell
    p.curve(lambda x: npdf(x), -4, 4, BLUE, 2.8)
    # mean line
    p.seg(0, 0, 0, npdf(0), DARK, 1.8, "5,4")
    p.dot(0, npdf(0), BLUE, 5)
    p.txt_px(p.sx(0), p.sy(npdf(0)) - 10, "mean = 0  (center)", DARK, 12.5, "middle", "700")
    # std markers
    for s in (-3, -2, -1, 1, 2, 3):
        p.seg(s, 0, s, npdf(s), GRAY, 1.2, "3,3")
    # 1-std double arrow along the baseline
    yb = 0.03
    p.seg(0, yb, 1, yb, RED, 1.6)
    p.txt_px(p.sx(0.5), p.sy(yb) - 6, "1 std = 1", RED, 12, "middle", "700")
    # band labels
    p.txt_px(p.sx(0), p.sy(0.20), "68%", GREEN, 15, "middle", "800")
    p.txt_px(p.sx(0), p.sy(0.145), "within +-1", GREEN, 11.5, "middle", "700")
    p.txt_px(p.sx(-1.5), p.sy(0.075), "95% within +-2", DARK, 11.5, "middle", "600")
    p.txt_px(p.sx(2.55), p.sy(0.045), "99.7% within +-3", DARK, 11, "middle", "600")
    p.xlabel("value  (how far from the mean, measured in std's)")
    p.ylabel("how often")
    return p.svg("The bell: mean says WHERE the pile sits, std says HOW WIDE it is")


def _bell_panel(title, curves, xr=(-4, 4), yr=(-0.02, 0.72), legend=None):
    p = Plot(400, 300, xr, yr, ml=44, mr=14, mt=30, mb=40)
    p.box()
    p.xticks([-3, 0, 3], grid=False)
    p.yticks([0, 0.2, 0.4, 0.6])
    p.axis0()
    for (mu, sd, col) in curves:
        p.curve(lambda x, mu=mu, sd=sd: npdf(x, mu, sd), xr[0], xr[1], col, 2.5)
        p.seg(mu, 0, mu, npdf(mu, mu, sd), col, 1.1, "3,3")
    if legend:
        for i, (txt, col) in enumerate(legend):
            yy = yr[1] - 0.05 - i * (yr[1] - yr[0]) * 0.09
            p.add(f'<text x="{p.sx(xr[0]) + 8:.1f}" y="{p.sy(yy):.1f}" font-size="11.5" '
                  f'fill="{col}" font-weight="700">{esc(txt)}</text>')
    return p.svg(title)


def fig_two_knobs():
    left = _bell_panel(
        "Knob 1: MEAN slides the pile sideways",
        [(-1.6, 1.0, RED), (0.0, 1.0, BLUE), (1.6, 1.0, GREEN)],
        legend=[("mean -1.6", RED), ("mean 0", BLUE), ("mean +1.6", GREEN)],
    )
    right = _bell_panel(
        "Knob 2: STD sets the width",
        [(0.0, 0.55, RED), (0.0, 1.0, BLUE), (0.0, 1.7, GREEN)],
        legend=[("std 0.55 (narrow, tall)", RED), ("std 1.0", BLUE), ("std 1.7 (wide, flat)", GREEN)],
    )
    return f'<div class="row">{left}{right}</div>'


def fig_randn_vs_rand():
    # left: randn (normal, mean 0, std 1)
    L = Plot(400, 300, (-4, 4), (-0.05, 0.9), ml=42, mr=14, mt=30, mb=40)
    L.box()
    L.xticks([-3, -2, -1, 0, 1, 2, 3], grid=False)
    L.yticks([0, 0.4, 0.8])
    L.axis0()
    L.vspan(-4, 0, RED, 0.06)
    L.curve(lambda x: npdf(x), -4, 4, BLUE, 2.8)
    L.seg(0, 0, 0, npdf(0), DARK, 1.4, "4,4")
    L.txt_px(L.sx(-2.0), L.sy(0.5), "negatives", RED, 11.5, "middle", "700")
    L.txt_px(L.sx(-2.0), L.sy(0.43), "welcome", RED, 11.5, "middle", "700")
    L.txt_px(L.sx(0), L.sy(0.78), "centered at 0", DARK, 12, "middle", "700")
    left = L.svg("torch.randn : bell, mean 0, std 1")

    # right: rand (uniform [0,1))
    R = Plot(400, 300, (-4, 4), (-0.05, 0.9), ml=42, mr=14, mt=30, mb=40)
    R.box()
    R.xticks([-3, -2, -1, 0, 1, 2, 3], grid=False)
    R.yticks([0, 0.4, 0.8])
    R.axis0()
    # flat block height 1 over [0,1)
    R.add(f'<rect x="{R.sx(0):.1f}" y="{R.sy(1.0):.1f}" width="{R.sx(1)-R.sx(0):.1f}" '
          f'height="{R.sy(0)-R.sy(1.0):.1f}" fill="{ORANGE}" opacity="0.5" rx="2"/>')
    R.seg(0, 0, 0, 1.0, ORANGE, 2.2)
    R.seg(1, 0, 1, 1.0, ORANGE, 2.2)
    R.seg(0, 1.0, 1, 1.0, ORANGE, 2.2)
    R.seg(0.5, 0, 0.5, 1.0, DARK, 1.4, "4,4")
    R.txt_px(R.sx(0.5), R.sy(0.78), "mean 0.5", DARK, 12, "middle", "700")
    R.txt_px(R.sx(-2.2), R.sy(0.4), "nothing", GRAY, 11.5, "middle", "700")
    R.txt_px(R.sx(-2.2), R.sy(0.33), "ever < 0", GRAY, 11.5, "middle", "700")
    right = R.svg("torch.rand : flat block on [0, 1)")
    return f'<div class="row">{left}{right}</div>'


def fig_samples():
    random.seed(3)
    N = 27 * 27  # exactly what randn(27,27) produces
    xs = [random.gauss(0.0, 1.0) for _ in range(N)]
    lo, hi, nb = -3.5, 3.5, 28
    bw = (hi - lo) / nb
    counts = [0] * nb
    for v in xs:
        b = int((v - lo) / bw)
        if 0 <= b < nb:
            counts[b] += 1
    ymax = max(counts) * 1.28
    p = Plot(820, 380, (lo, hi), (-ymax * 0.04, ymax), ml=52, mr=20, mt=30, mb=46)
    p.box()
    p.xticks([-3, -2, -1, 0, 1, 2, 3])
    p.yticks([0, 20, 40, 60, 80])
    for i, c in enumerate(counts):
        xc = lo + (i + 0.5) * bw
        p.bar(xc, c, bw * 0.46, BLUE, opacity=0.55)
    # overlay the ideal bell scaled to counts
    p.curve(lambda x: N * npdf(x) * bw, lo, hi, RED, 2.6)
    p.txt_px(p.sx(1.7), p.sy(ymax * 0.7), "the ideal bell", RED, 12.5, "start", "700")
    p.txt_px(p.sx(-3.3), p.sy(ymax * 0.9), "729 actual draws", BLUE, 12.5, "start", "700")
    p.xlabel("value")
    p.ylabel("count in bin")
    return p.svg("What torch.randn(27, 27) literally hands you: 729 numbers piled into a bell")


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
    .callout b{color:var(--ink)}
    code{background:#eef0f3;padding:1px 6px;border-radius:5px;font-size:.92em}
    .big{font-size:16px}
    ul{margin:8px 0 8px 2px;padding-left:22px}
    li{margin:5px 0}
    .tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;
         text-transform:uppercase;color:#fff;background:var(--accent);padding:2px 9px;border-radius:20px;margin-bottom:4px}
    .tag.g{background:#2ca02c}.tag.o{background:#e8820c}
    """
    parts = []
    A = parts.append
    A(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Mean and std</title><style>{css}</style></head><body>")

    A("<h1>mean and std: the two numbers that describe a pile of random values</h1>")
    A("<p class='lede'>You asked what <code>mean</code> and <code>std</code> mean in "
      "<code>torch.randn</code> (mean 0, std 1). Short version, then the pictures.</p>")

    A("<div class='callout'><span class='tag'>the whole thing in one line</span>"
      "<p class='big' style='margin:2px 0 0'><b>Mean = where the pile is centered. "
      "Std (standard deviation) = how wide the pile is.</b> That's the entire vocabulary. "
      "randn = &ldquo;pile centered at 0, typical spread of 1.&rdquo;</p></div>")

    A("<p>Picture throwing darts at a number line while aiming at a target. The <b>mean</b> "
      "is where you aim; the <b>std</b> is how shaky your hand is (the typical distance a dart "
      "lands from that aim point). Aim at 0 with a shake of 1, and you get <code>randn</code>.</p>")

    A("<h2>1. The bell, and the 68 / 95 / 99.7 rule</h2>")
    A("<p>A normal distribution always makes the same bell shape. Because the shape is fixed, "
      "the percentages are fixed too &mdash; worth memorizing, they come up everywhere:</p>")
    A("<ul>"
      "<li>~<b>68%</b> of the values land within <b>1 std</b> of the mean &rarr; here, in [-1, 1]</li>"
      "<li>~<b>95%</b> land within <b>2 std</b> &rarr; in [-2, 2]</li>"
      "<li>~<b>99.7%</b> land within <b>3 std</b> &rarr; in [-3, 3]</li></ul>")
    A(f"<div class='fig'>{fig_bell_bands()}</div>")
    A("<p class='cap'>So a value past +-3 is genuinely rare (~0.3%). The std is the natural "
      "ruler here: distance is most useful measured in &ldquo;number of std's from the mean.&rdquo;</p>")

    A("<h2>2. The two knobs, one at a time</h2>")
    A("<p>Mean and std are independent dials. Moving one never touches the other:</p>")
    A(fig_two_knobs())
    A("<p class='cap'>Left: change the <b>mean</b> and the identical bell just slides sideways. "
      "Right: change the <b>std</b> and the bell squeezes taller/narrower or spreads shorter/wider "
      "&mdash; but stays centered. (Total area is always 1, so narrower must mean taller.)</p>")

    A("<h2>3. Why weights use randn, not rand</h2>")
    A("<p>This is why I steered you to <code>randn</code>. Compare the two:</p>")
    A(fig_randn_vs_rand())
    A("<div class='callout'><p style='margin:0'><b>Weights need to start small, centered at 0, and "
      "free to be negative.</b> <code>randn</code> gives exactly that. <code>rand</code> is a flat "
      "block on [0, 1): every value equally likely, mean stuck at 0.5, and <b>never negative</b>. "
      "But your logits must be able to go negative (negative logit &rarr; small probability after "
      "exp), so starting all-positive and off-center is the wrong footing.</p></div>")

    A("<h2>4. What randn(27, 27) actually gives you</h2>")
    A("<p>Not an abstraction &mdash; it's 729 real numbers (27x27). Draw them and drop each into a "
      "bin, and the pile fills in the bell you were promised:</p>")
    A(f"<div class='fig'>{fig_samples()}</div>")
    A("<p class='cap'>Most land near 0, a handful stray past +-2, and the ideal bell (red) is the "
      "shape they're scattering around. More draws = smoother fit to the curve.</p>")

    A("<h2>5. Where the bell comes from</h2>")
    A("<div class='callout origin'><span class='tag o'>origin story</span>"
      "<p style='margin:2px 0 0'>Carl Friedrich Gauss reached for this curve around 1809 to model the "
      "<b>errors</b> in astronomical measurements &mdash; small errors common, large errors rare, "
      "symmetric around the true value &mdash; which is why it's called the <b>Gaussian</b>. The "
      "physical picture that makes it click is Francis Galton's <b>bean machine</b>: drop beads through "
      "a triangle of pegs, each peg bounces a bead left or right at random, and the beads always pile up "
      "into this same bell at the bottom. Many small independent nudges adding up &rarr; a bell. That's "
      "the <b>Central Limit Theorem</b>, and it's why the normal distribution shows up all over nature "
      "and all over neural nets.</p></div>")

    A("<div class='callout keep'><span class='tag g'>keep this</span>"
      "<p style='margin:2px 0 0'><b>mean = center, std = width.</b> <code>randn</code> = a bell at 0 "
      "with width 1, so ~68% of values sit in [-1, 1] and negatives are as likely as positives &mdash; "
      "the right starting pile for weights. <code>rand</code> = a flat, all-positive block, wrong for "
      "weights. Coming up in lecture 4, tuning that std so activations stay healthy through a deep net "
      "is the entire topic &mdash; you'll be turning this exact knob on purpose.</p></div>")

    A("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = "llm_output/mean_and_std_grok.html"
    html_str = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}  ({len(html_str)} bytes)")
