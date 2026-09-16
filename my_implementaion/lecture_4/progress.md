# Lecture 4 — Activations & Gradients, BatchNorm — progress

## Where he is
- Watching the video, reached **~45:00** (batch normalization, the standardize line).
- No implementation started yet for this lecture; conceptual grounding first (his usual: grok before code).

## Concept status
- **Mean** — strong. Understands it as the average / balance point.
- **Variance / std** — was the snag (taught this session). Built from scratch: deviation ->
  square (so they don't cancel) -> average = variance -> sqrt = std = "typical distance from mean".
  Toy used: `30,40,50,60,70` -> mean 50, var 200, std 14.14.
- **dim=0 vs "record level"** — was the confusion. His instinct was "we summarize a record".
  Correction that landed: a neuron is a **feature/column**, and features are summarized DOWN the
  rows (like normalizing a height column over all people); dim=0 is the normal move, dim=1 (across
  a row) would mix unrelated neurons. Analogy: neuron=student, example=one test, dim=0 = a student's
  avg/consistency across all their tests.
- **(hpreact - mean)/std** — taught as the **z-score**: subtract mean (center at 0), divide by std
  (spread -> 1) => unit gaussian (mean 0, std 1). Analogy: grading on a curve / "surprise units".
- Tied back to tanh saturation (first half of lecture): unit-gaussian inputs land in tanh's active
  middle; wide inputs saturate (no gradient).

## Still ahead / flagged for him
- **bngain / bnbias** (scale & shift) — mentioned as the next thing after the standardize line;
  lets the net move away from the forced unit-gaussian. Not yet deep-dived.
- **Train/test wrinkle** — batch coupling (an example's norm depends on its batchmates) => needs
  running mean/std at test time. Karpathy covers it right after 45:00. Flagged as bonus this session.

## Artifacts made this session
- `llm_output/batchnorm_mean_std_grok.html` — full visual walkthrough (mean, variance/std build,
  dartboard, bell 68-95, the 5x3 matrix dim=0, z-score 3-stage strip, tanh saturation).
- Generator: `.kiro/build_batchnorm_std_html.py` (pure stdlib SVG).

## ENV GOTCHA (this session)
- `py` was **not on PATH** in this shell (`command not found`). `python`/`python3` bare also absent
  as `python`. Working interpreters found: `/usr/bin/python3` and the venv `.venv/bin/python`.
  Ran the stdlib generator with `.venv/bin/python .kiro/build_batchnorm_std_html.py` — fine since
  it's stdlib-only. (If `py` stops resolving generally, fall back to `.venv/bin/python`.)
