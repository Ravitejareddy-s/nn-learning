#!/usr/bin/env python3
"""Builds a self-contained teaching HTML page about STEP SIZE (learning rate): how to pick
it, how to tell you're overshooting, and how to know when to stop iterating. Pure stdlib,
reuses the SVG Plot toolkit from build_exp_log_html.py (runs anywhere, no numpy/matplotlib).

All curves are numerically exact: gradient descent on the toy loss L(w) = w^2
(gradient dL/dw = 2w), so the update is  w <- w - lr*2w = w*(1 - 2*lr).
That single factor r = 1 - 2*lr explains everything:
    |r| < 1  -> converges     r > 0 -> monotone      r < 0 -> overshoots the bottom (zig-zag)
    |r| = 1  -> bounces forever            |r| > 1 -> diverges (blows up)

Output: llm_output/learning_rate_grok.html
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_exp_log_html import Plot, esc, BLUE, RED, ORANGE, GREEN, GRAY, DARK, FAINT  # noqa: E402

LIGHT = "#cbd5e1"   # the bowl
PURPLE = "#7a4fbf"


# ---------------------------------------------------------------- math

def traj(w0, lr, steps, cap=1e12):
    """Exact GD path on L(w)=w^2 from w0. Returns list of w values."""
    ws = [w0]
    w = w0
    for _ in range(steps):
        w = w - lr * (2 * w)
        ws.append(w)
        if abs(w) > cap:
            break
    return ws


def traj_sched(w0, lrs):
    """GD path where lrs is a per-step list of learning rates (for decay)."""
    ws = [w0]
    w = w0
    for lr in lrs:
        w = w - lr * (2 * w)
        ws.append(w)
    return ws


def arrow(p, x1, y1, x2, y2, color, width=2.2, head=9.0):
    p.seg(x1, y1, x2, y2, color, width)
    X1, Y1, X2, Y2 = p.sx(x1), p.sy(y1), p.sx(x2), p.sy(y2)
    ang = math.atan2(Y2 - Y1, X2 - X1)
    a = 0.5
    xa, ya = X2 - head * math.cos(ang - a), Y2 - head * math.sin(ang - a)
    xb, yb = X2 - head * math.cos(ang + a), Y2 - head * math.sin(ang + a)
    p.add(f'<polygon points="{X2:.1f},{Y2:.1f} {xa:.1f},{ya:.1f} {xb:.1f},{yb:.1f}" fill="{color}"/>')


# ---------------------------------------------------------------- figures

def fig_landscape():
    p = Plot(820, 380, (-11, 11), (-9, 95), ml=54, mt=30, mb=46)
    p.box()
    p.xticks([-10, -5, 0, 5, 10])
    p.yticks([0, 30, 60, 90])
    p.axis0()
    p.curve(lambda x: x * x, -10, 10, LIGHT, 3.2)
    # the ball on the hillside
    wb = 6.5
    p.dot(wb, wb * wb, BLUE, 7)
    p.txt_px(p.sx(wb), p.sy(wb * wb) - 13, "you are here", BLUE, 12.5, "middle", "700")
    # gradient points uphill (to the right / up the wall)
    arrow(p, wb, wb * wb, wb + 2.6, wb * wb + 2 * wb * 2.6, RED, 2.4)
    p.txt_px(p.sx(wb + 2.7), p.sy(wb * wb + 2 * wb * 2.6) - 6, "gradient -> uphill", RED, 12, "start", "700")
    # we step downhill (negative gradient)
    arrow(p, wb, wb * wb, wb - 3.0, wb * wb - 2 * wb * 3.0, GREEN, 2.6)
    p.txt_px(p.sx(wb - 3.1), p.sy(wb * wb - 2 * wb * 3.0) + 16, "step = -lr x gradient", GREEN, 12, "middle", "700")
    # the minimum
    p.dot(0, 0, DARK, 5)
    p.txt_px(p.sx(0), p.sy(0) - 12, "minimum (lowest loss)", DARK, 12, "middle", "700")
    p.xlabel("w  (a weight)   -- the whole 27x27 matrix is this picture in many dimensions at once")
    p.ylabel("loss")
    return p.svg("The loss is a landscape. Gradient descent = roll downhill. Step size = how far you hop each time.")


def bowl(title, lr, steps, xr, yr, yt, note, color, escape=False):
    p = Plot(390, 300, xr, yr, ml=42, mr=12, mt=44, mb=38)
    p.box()
    p.xticks([x for x in (-10, -5, 0, 5, 10, 15) if xr[0] < x < xr[1]], grid=False)
    p.yticks(yt)
    p.axis0()
    p.curve(lambda x: x * x, xr[0], xr[1], LIGHT, 2.8)
    ws = traj(9, lr, steps)
    pts = [(w, w * w) for w in ws]
    # connecting hops (clamp for drawing so it never spills the frame)
    def clamp(pt):
        x, y = pt
        return (max(xr[0], min(xr[1], x)), max(yr[0], min(yr[1], y)))
    drawn = [clamp(pt) for pt in pts]
    p.poly(drawn, color, 1.7)
    for i, (w, L) in enumerate(pts):
        if yr[0] <= L <= yr[1] and xr[0] <= w <= xr[1]:
            p.dot(w, L, color, 3.6)
    # start marker
    p.dot(9, 81, DARK, 4) if (yr[1] >= 81) else None
    if yr[1] >= 81:
        p.txt_px(p.sx(9), p.sy(81) - 9, "start", DARK, 11, "middle", "700")
    if escape:
        # up arrow at the last in-frame point
        arrow(p, drawn[-1][0], yr[1] * 0.72, drawn[-1][0], yr[1] * 0.95, RED, 2.6)
        p.txt_px(p.sx(0), p.sy(yr[1] * 0.9), "-> infinity", RED, 13, "middle", "800")
    r = 1 - 2 * lr
    p.txt_px((p.ml + p.pw / 2), 40, f"lr = {lr}   (multiplier r = 1-2lr = {r:+.2f})", DARK, 11.5, "middle", "700")
    p.txt_px((p.ml + p.pw / 2), p.h - 6, note, color, 12, "middle", "700")
    return p.svg(title)


def fig_bowls():
    a = bowl("TOO SMALL: crawls", 0.03, 12, (-11, 11), (-6, 92), [0, 30, 60, 90],
             "barely moves - wastes runs", ORANGE)
    b = bowl("JUST RIGHT: smooth descent", 0.25, 8, (-11, 11), (-6, 92), [0, 30, 60, 90],
             "drops straight to the bottom", GREEN)
    c = bowl("EDGE: overshoots but survives", 0.9, 8, (-11, 11), (-6, 92), [0, 30, 60, 90],
             "zig-zags across, |r|<1 so still converges", BLUE)
    d = bowl("TOO BIG: blows up", 1.05, 6, (-16, 16), (-12, 200), [0, 60, 120, 180],
             "each hop lands HIGHER - diverges", RED, escape=True)
    return (f'<div class="row">{a}{b}</div><div class="row">{c}{d}</div>')


def fig_loss_curves():
    p = Plot(820, 400, (0, 16), (-8, 185), ml=54, mt=32, mb=46)
    p.box()
    p.xticks([0, 4, 8, 12, 16])
    p.yticks([0, 40, 80, 120, 160])
    series = [
        (0.03, ORANGE, "lr too small"),
        (0.25, GREEN, "lr just right"),
        (0.9, BLUE, "lr on the edge"),
        (1.05, RED, "lr too big"),
    ]
    for lr, col, _ in series:
        ws = traj(9, lr, 16)
        xys = [(i, min(185, w * w)) for i, w in enumerate(ws)]
        p.poly(xys, col, 2.4)
        for i, (x, y) in enumerate(xys):
            if y < 185:
                p.dot(x, y, col, 3.0)
    # legend
    for i, (lr, col, lab) in enumerate(series):
        yy = 172 - i * 15
        p.seg(0.4, yy, 1.4, yy, col, 3)
        p.txt_px(p.sx(1.7), p.sy(yy) + 4, f"{lab}  (lr={lr})", col, 12, "start", "700")
    p.txt_px(p.sx(13.5), p.sy(178), "-> infinity (NaN)", RED, 12, "middle", "800")
    p.txt_px(p.sx(11.5), p.sy(8), "the plateau: stop here", GREEN, 12, "middle", "700")
    p.xlabel("iteration (run number)")
    p.ylabel("loss")
    return p.svg("The one diagnostic you actually watch: loss vs iteration. The shape tells you the lr.")


def fig_sweep():
    p = Plot(820, 400, (0, 1.35), (-8, 185), ml=54, mt=32, mb=48)
    p.box()
    p.xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    p.yticks([0, 40, 80, 120, 160])
    # blow-up region
    p.vspan(1.0, 1.35, RED, 0.10)
    # sweet spot band
    p.vspan(0.2, 0.85, GREEN, 0.12)
    xys = []
    lr = 0.01
    while lr <= 1.35:
        ws = traj(9, lr, 15)
        L = min(185, ws[-1] ** 2)
        xys.append((lr, L))
        lr += 0.01
    p.poly(xys, PURPLE, 2.8)
    p.txt_px(p.sx(0.08), p.sy(120), "too slow:", GRAY, 12, "start", "700")
    p.txt_px(p.sx(0.08), p.sy(107), "loss still high", GRAY, 12, "start", "700")
    p.txt_px(p.sx(0.52), p.sy(28), "sweet spot", GREEN, 13, "middle", "800")
    p.txt_px(p.sx(0.52), p.sy(15), "biggest lr that still goes down", GREEN, 11.5, "middle", "600")
    p.txt_px(p.sx(1.17), p.sy(150), "blow up", RED, 13, "middle", "800")
    p.seg(1.0, -8, 1.0, 185, RED, 1.4, "5,4")
    p.xlabel("learning rate you tried")
    p.ylabel("loss after a few runs")
    return p.svg("How you actually PICK it: sweep the lr, plot the result, take the edge of the cliff.")


def fig_decay():
    p = Plot(820, 380, (0, 14), (-6, 92), ml=54, mt=32, mb=46)
    p.box()
    p.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    p.yticks([0, 30, 60, 90])
    # constant aggressive lr = 0.9 -> sawtooth that lingers
    ws_c = traj(9, 0.9, 14)
    p.poly([(i, w * w) for i, w in enumerate(ws_c)], BLUE, 2.2)
    for i, w in enumerate(ws_c):
        p.dot(i, w * w, BLUE, 3.0)
    # decayed: big early (0.9) then shrink to 0.3 -> settles clean
    lrs = [0.9, 0.9, 0.9] + [0.3] * 11
    ws_d = traj_sched(9, lrs)
    p.poly([(i, w * w) for i, w in enumerate(ws_d)], GREEN, 2.6)
    for i, w in enumerate(ws_d):
        p.dot(i, w * w, GREEN, 3.0)
    p.seg(0.4, 84, 1.4, 84, BLUE, 3)
    p.txt_px(p.sx(1.7), p.sy(84) + 4, "constant big lr: keeps bouncing, never settles", BLUE, 12, "start", "700")
    p.seg(0.4, 74, 1.4, 74, GREEN, 3)
    p.txt_px(p.sx(1.7), p.sy(74) + 4, "decay lr: aggressive early, precise late -> settles", GREEN, 12, "start", "700")
    p.xlabel("iteration")
    p.ylabel("loss")
    return p.svg("Best of both: start with a big step to cover ground, shrink it to land cleanly (lr decay).")


# ---------------------------------------------------------------- page

def build():
    css = """
    :root{--ink:#23252b;--soft:#5b6270;--line:#e3e6ea;--accent:#1f77b4;}
    *{box-sizing:border-box}
    body{font-family:system-ui,'Segoe UI',Arial,sans-serif;line-height:1.62;color:var(--ink);
         max-width:940px;margin:0 auto;padding:34px 22px 90px;background:#fbfbfc}
    h1{font-size:27px;line-height:1.25;margin:0 0 6px}
    h2{font-size:20px;margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--line)}
    p{margin:10px 0}
    .lede{color:var(--soft);font-size:15px;margin-top:0}
    .fig{margin:20px 0 6px;text-align:center}
    .row{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:14px 0 6px}
    .cap{color:var(--soft);font-size:14px;text-align:center;max-width:780px;margin:6px auto 0}
    .callout{border-left:4px solid var(--accent);background:#f2f7fc;padding:12px 16px;margin:18px 0;border-radius:0 8px 8px 0}
    .keep{border-left:4px solid #2ca02c;background:#f1f8f0}
    .warn{border-left:4px solid #d62728;background:#fdf1f1}
    .callout b{color:var(--ink)}
    code{background:#eef0f3;padding:1px 6px;border-radius:5px;font-size:.92em}
    .big{font-size:16px}
    ul{margin:8px 0 8px 2px;padding-left:22px}
    li{margin:6px 0}
    table{border-collapse:collapse;margin:14px auto;font-size:14px}
    th,td{border:1px solid var(--line);padding:7px 12px;text-align:left}
    th{background:#eef2f7}
    .tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
         color:#fff;background:var(--accent);padding:2px 9px;border-radius:20px;margin-bottom:4px}
    .tag.g{background:#2ca02c}.tag.r{background:#d62728}
    """
    P = []
    A = P.append
    A(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Step size / learning rate</title><style>{css}</style></head><body>")

    A("<h1>Step size: how big a hop, how do you know you didn't overshoot, and when do you stop?</h1>")
    A("<p class='lede'>You asked the three questions that <i>are</i> the whole topic of learning-rate "
      "tuning. Short answer first, then the pictures &mdash; all drawn from real gradient descent on a "
      "toy bowl-shaped loss.</p>")

    A("<div class='callout'><span class='tag'>the whole thing in one line</span>"
      "<p class='big' style='margin:2px 0 0'>You <b>don't</b> know the right step size up front &mdash; "
      "you <b>probe</b> for it and <b>watch the loss curve</b>. Too small &rarr; loss crawls. Too big "
      "&rarr; loss goes <i>up</i> and blows up. You pick the <b>biggest step that still goes down "
      "smoothly</b>, and you <b>stop when the curve flattens</b>.</p></div>")

    A("<h2>1. The picture behind all of it: loss is a landscape</h2>")
    A("<p>Your loss is a number that depends on the weights. Plot it and you get a <b>landscape</b>. "
      "For the bigram net it's a smooth bowl (one lowest point). The <b>gradient</b> at your spot points "
      "<i>uphill</i>; you take a step <i>downhill</i> (against it). <b>The learning rate is how long that "
      "step is.</b> That's literally all it is.</p>")
    A(f"<div class='fig'>{fig_landscape()}</div>")
    A("<p class='cap'>The gradient only tells you the <b>direction</b> (which way is down) and roughly how "
      "steep &mdash; it does <b>not</b> tell you how far to walk. Picking the distance is on you. That distance "
      "is the learning rate.</p>")

    A("<h2>2. Four step sizes on the exact same bowl</h2>")
    A("<p>Same start (w = 9), same bowl, only the step size changes. Watch the ball. The update here is "
      "<code>w &larr; w - lr&middot;(2w)</code>, i.e. <code>w</code> gets multiplied by "
      "<code>r = 1 - 2&middot;lr</code> every step &mdash; that one number decides the fate:</p>")
    A(fig_bowls())
    A("<p class='cap'>Orange: step so tiny it inches down &mdash; safe but wastes runs (this was your "
      "<code>lr=0.1</code>). Green: lands near the bottom in a few hops. Blue: step overshoots the bottom "
      "and lands on the <i>other</i> wall, but a bit lower each time, so it still spirals in. Red: the step "
      "overshoots so hard it lands <i>higher</i> than it started &mdash; every hop is worse, the loss "
      "explodes to infinity / NaN. <b>That is &ldquo;blowing up.&rdquo;</b></p>")

    A("<div class='callout'><p style='margin:0'><b>How you know you overshot:</b> the loss <i>increased</i> "
      "after a step. A step can overshoot the bottom (land on the far wall) and still be fine <i>as long as "
      "it lands lower</i> &mdash; that's the blue case. It's only fatal when each hop lands higher than the "
      "last (red). So the test isn't &ldquo;did I cross the bottom?&rdquo; &mdash; it's &ldquo;is the loss "
      "still going down?&rdquo;</p></div>")

    A("<h2>3. The one thing you actually watch: the loss curve</h2>")
    A("<p>You never see the landscape directly (it lives in 729 dimensions for your net). What you "
      "<i>do</i> see is <b>loss vs iteration</b>, and its <b>shape</b> diagnoses the step size for you:</p>")
    A(f"<div class='fig'>{fig_loss_curves()}</div>")
    A("<table>"
      "<tr><th>Curve shape</th><th>Diagnosis</th><th>Do this</th></tr>"
      "<tr><td>Slopes down gently, still high</td><td>step too small</td><td>increase lr</td></tr>"
      "<tr><td>Drops fast, then flattens</td><td>just right</td><td>stop when flat</td></tr>"
      "<tr><td>Jagged / sawtooth but trending down</td><td>on the edge</td><td>lower lr a bit</td></tr>"
      "<tr><td>Rises, or jumps to NaN</td><td>too big (overshoot)</td><td>cut lr hard</td></tr>"
      "</table>")

    A("<h2>4. So how do you PICK the number? Sweep it.</h2>")
    A("<p>The honest professional answer: you don't compute it, you <b>search</b> for it. Try a range of "
      "learning rates, run each for a handful of steps, and plot the resulting loss against the lr you used. "
      "This is Karpathy's <b>learning-rate finder</b> (you'll build exactly this in lecture 3):</p>")
    A(f"<div class='fig'>{fig_sweep()}</div>")
    A("<p class='cap'>Loss is high for tiny lr (too slow to get anywhere), dips into a <b>sweet-spot valley</b>, "
      "then <b>shoots up off a cliff</b> once the step is too big. You pick the lr at the <b>edge of the "
      "cliff</b> &mdash; the largest one that's still going down. Big enough to be fast, not so big it "
      "diverges.</p>")

    A("<h2>5. &ldquo;How many runs without blowing up?&rdquo;</h2>")
    A("<p>Two separate worries hide in that question:</p>"
      "<ul>"
      "<li><b>Blowing up</b> is <i>not</i> caused by too many iterations &mdash; it's caused by too big a "
      "<i>step</i>. With a good lr you can run as long as you like and it never explodes; it just flattens.</li>"
      "<li><b>How many iterations</b> = run until the loss curve <b>plateaus</b>. When each new run barely "
      "moves the number, more runs won't help &mdash; you've hit the floor for this lr. That's your stop "
      "signal, not a fixed count.</li>"
      "</ul>")
    A("<p>And the trick that gets you both fast <i>and</i> precise: <b>learning-rate decay</b> &mdash; start "
      "with a big step to cover ground, then shrink it to settle in cleanly (like taking big strides across a "
      "field, then tiny careful steps once you're right over the drain):</p>")
    A(f"<div class='fig'>{fig_decay()}</div>")
    A("<p class='cap'>Blue (constant big lr) keeps bouncing around the bottom forever. Green shrinks its step "
      "partway through and drops straight onto the minimum. Karpathy does exactly this in lecture 3 (10x "
      "smaller lr near the end).</p>")

    A("<h2>6. Back to your bigram net: why lr=50 works there but 1.0 blows up here</h2>")
    A("<div class='warn'><span class='tag r'>the mind-bender</span>"
      "<p style='margin:2px 0 0'>In this toy bowl, lr just past <b>1.0</b> already explodes. In your bigram "
      "net, lr=<b>50</b> is perfect. How? Because the <b>gradient scale is totally different.</b> Here the "
      "gradient is <code>2w</code> (big). In your net the loss is an <b>average over 228k examples</b> of "
      "softmax probabilities, so <code>W.grad</code> is <i>tiny</i> &mdash; a step of <code>lr &middot; grad</code> "
      "needs a big <code>lr</code> just to move an appreciable amount. <b>This is the punchline: there is no "
      "universal &ldquo;good&rdquo; learning rate.</b> The right value depends on how big the gradients happen to "
      "be for <i>your</i> loss, which is exactly why you probe instead of memorizing a number. Your "
      "<code>lr=0.1</code> wasn't &ldquo;wrong&rdquo; &mdash; it was just far too small <i>for this problem's "
      "gradient scale</i>, so it sat in the orange &ldquo;crawls&rdquo; regime.</p></div>")

    A("<div class='callout keep'><span class='tag g'>keep this</span>"
      "<p style='margin:2px 0 0'>Step size is a <b>search, not a formula.</b> Watch <b>loss vs iteration</b>: "
      "crawling &rarr; bigger lr; flattening &rarr; stop; rising/NaN &rarr; overshoot, cut lr. Pick the "
      "<b>biggest lr that still descends</b> (edge of the cliff in the sweep), optionally <b>decay</b> it to "
      "land cleanly. Blowing up comes from too-big <b>steps</b>, not too many <b>runs</b>. And because the "
      "&ldquo;right&rdquo; lr rides on your loss's gradient scale, 0.1 vs 50 can both be correct &mdash; for "
      "different problems.</p></div>")

    A("<p class='cap' style='margin-top:26px'>Coming up: lecture 3 builds the lr-sweep by hand; lecture 4 "
      "introduces the <b>Adam</b> optimizer, which auto-scales the step per weight so you barely hand-tune it "
      "at all &mdash; it's basically this whole page, automated.</p>")

    A("</body></html>")
    return "".join(P)


if __name__ == "__main__":
    out = "llm_output/learning_rate_grok.html"
    html_str = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}  ({len(html_str)} bytes)")
