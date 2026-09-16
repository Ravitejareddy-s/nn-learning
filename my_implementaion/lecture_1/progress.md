# Lecture 1 (micrograd) — progress

Agent-maintained. Tracks concept-by-concept what's strong vs shaky for this lecture.

## Mental model he uses for derivatives
- Owns the "wiggle" framing: derivative = nudge the input a tiny step, how much does the
  output wiggle? ratio (out-wiggle / in-wiggle) = derivative.
- **Strong:** sum (step passes through → slope 1) and multiply (each input step worth "the
  other factor" → slope = other factor). Can do these by hand and reasons them intuitively.

## exp / log derivatives — covered [this session]
- Was shaky on intuiting d(e^x)=e^x and d(log x)=1/x. Taught via his own wiggle language:
  - e^x: output wiggle = its own current height (percentage lens: a step adds a fixed % not
    a fixed amount → absolute wiggle scales with size). Anchor: continuous-interest balance.
  - log x: output wiggle = the % change in x = Δx/x → slope 1/x; shrinks as x grows. Anchor:
    equal ×2 in x → equal +0.69 in log; log scales (dB, Richter, pH).
  - Mirror across y=x makes the two slopes reciprocal (h vs 1/h).
- Visual deliverable: `llm_output/exp_log_derivative_grok.html`
  (generator: `.kiro/build_exp_log_derivative_html.py`, pure-stdlib SVG).
- Check next time: can he re-derive 1/x from "divide the step by x to turn it into a % step"
  without the picture? If yes, it's solid.

## Env note
- `py` is not on the non-interactive shell PATH here; `python3.13t` / `python3.13` absent.
  Working interpreter for stdlib SVG scripts: `/opt/homebrew/bin/python3` (3.14).
