# Lecture 2 — makemore (bigrams + neural net) — Progress

_Practice notebook: `bigram.ipynb`. Tracks concept-by-concept mastery and session history
for lecture 2. Read this at the start of a lecture-2 chat; update on request. Not
auto-loaded — see `.kiro/steering/teaching.md`._

## His mental model of the lecture
He splits lecture 2 into two sections:
- **Section 1 — counts route:** build the 27×27 count matrix → row-normalize into a
  probability matrix → sample names → score/evaluate (likelihood → NLL loss).
- **Section 2 — neural net route:** recast the exact same thing as one linear layer +
  softmax, trained by gradient descent; it converges to the same matrix.

## Concept status

### Strong (anchor to these)
- Counting bigrams, building the count matrix, row-normalizing to probabilities.
  Implemented himself.
- Sampling with `torch.multinomial`, walking until the `.` end token. Implemented himself.
- **Why we multiply probabilities to score a name** (hard-won this session): AND = multiply
  via "fraction of a fraction"; uses *conditional* probs (chain rule), not independent
  ones; the tree/walk picture. Solid now.
- Dependent vs independent events — coins where the first flip decides which coin you flip
  next → maps to "the previous letter picks which row you read."
- Terminology: likelihood, log likelihood, negative log likelihood, average NLL, MLE.
  ("relative probability" is NOT a real term — corrected.)
- **Softmax = exp-then-normalize, and why `exp` not clamp/ReLU** (hard-won this session):
  clamping negatives to 0 can make a target's prob exactly 0 → `-log(0)` = ∞/NaN; and ReLU's
  slope is 0 below zero (dead gradient), so "the grad will just pull it up" fails. `exp` is
  always positive and smooth, so gradient always flows. logits = log-counts.
- **`zero_grad` is a per-tensor reset, not graph propagation.** Only `backward` traverses the
  graph; you must reset the *weights'* grads because `backward` accumulates (`+=`). (He now
  models `gradzero` as an explicit walk-the-tree-and-zero function — valid, because the walk
  reaches the weight nodes.) Corrected his "same graph, so it auto-zeros" idea.
- **exp/log deep intuition** (from the `exp_and_log_grok.html` detour): both are
  strictly-increasing *relabelings* → they preserve ordering and the location of the max, so
  softmax (needs positivity + order) and log-likelihood (needs the argmax) are unaffected.
  They're inverses → lossless. `e^x` is a curve (equal x-steps → ×e; slope = its own height).
  "log of a bump peaks at the same w" = why maximizing log-likelihood ≡ maximizing likelihood.

### Solid (understood conceptually, not yet coded)
- Why the raw product underflows; `log` turns × into +; log is monotonic (so optimizing it
  is valid); negate → loss; average → comparable across lengths. Full chain down to
  "average negative log likelihood ≈ 2.45."
- Score the model on REAL data, not on its own generated names (self-scoring is circular;
  the degenerate "always aaaaaa" model scores its own output ~1.0).
- The length effect is a decode-time issue (beam search / length normalization), NOT a
  training or sampling bias; bigram counts are pooled and length-blind.
- Neural-net forward pass: integer → one-hot → `@ W` → logits → softmax → probability row.
  First letter picks the row, second letter is the column you read off. Logits are
  relative; the softmax output is a true probability. One-hot @ W just selects a row of W;
  `softmax(W)` converges to the counts matrix.

### Shaky (reteach / watch)
- Gradient intuition: he first guessed a product loss makes descent *overstep*; it actually
  makes gradients *vanish* (flat landscape → stuck). Reinforce when we hit the training loop.
- Probability vs likelihood (same number, different viewpoint) — just introduced, let it
  settle.
- The "exp/log change the value, so how can the result be right?" instinct — resolved via
  the monotonic-relabel + "same peak on the mountain" picture (`llm_output/exp_and_log_grok.html`).
  Verify it holds when softmax/NLL show up in code.
- Keep arithmetic in percentages, not fractions.

## Implementation status (`bigram.ipynb`)
- Done: counts, probability matrix, sampling.
- NOT yet done: scoring a name / dataset NLL loss. This was the immediate next step.
  Reminder: include the ending `i.` bigram — he initially dropped the end token when
  listing "ravi" as `.r ra av vi`.
- Section 2 (neural net): pseudocode written and validated this session (both blockers
  fixed — see log). Not yet coded; ready to implement.

## Next steps
1. Implement scoring: for real names, sum `log(prob[i][j])` over their bigrams, negate,
   average over all bigrams — expect ~2.45.
2. Implement Section 2 (**active focus** — pseudocode validated this session): one-hot →
   `W` → softmax → NLL, train with gradient descent, confirm it lands on ~2.45 and produces
   the same samples as the counts route.
3. Not yet covered from the lecture: model smoothing (fake counts) and its regularization
   equivalent in the net; sampling from the net.

## Session log
- 2026-07-15: Deep dive on the scoring half of Section 1 — why multiply (chain rule /
  conditional probs / tree), coin analogies (independent vs dependent), the
  product→log→NLL→average chain, the self-scoring pitfall, length bias, and the net forward
  pass + terminology. He grokked the multiply→log→NLL chain and the net's row/column read.
  No scoring code written yet — he wants to grok all of lecture 2 before implementing.
- 2026-07-23: Reviewed his hand-written pseudocode for Section 2 before coding. Found and
  fixed three blockers: (1) softmax — he had clamp-negatives-to-0 + normalize; corrected to
  `exp` (why: prob-0/∞ blowup + ReLU dead gradient, vs exp always positive & smooth);
  (2) NLL indexing — he re-ran the net to grab the logit at the target; corrected to index
  the *normalized* prob row; (3) `gradzero` — corrected the "zeroing loss auto-zeros weights
  because same graph" idea (zeroing is per-tensor; only backward traverses; backward `+=`
  accumulates, so the weights' grads must be reset). He re-expressed gradzero as an explicit
  tree-walk-and-zero, which is valid. His revised pseudocode is now correct on all three.
  Long detour to grok `exp`/`log`: his block was "if they change the scale, how is the result
  right?" — resolved with monotonic relabeling (order + argmax preserved), inverses
  (lossless), and the "same peak" visual. Built `llm_output/exp_and_log_grok.html` (pure-
  stdlib inline SVG, since numpy/matplotlib are broken under python3.13t). HTML is now his
  preferred explainer format. Green-lit to implement Section 2; no code written yet.
