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
- Keep arithmetic in percentages, not fractions.

## Implementation status (`bigram.ipynb`)
- Done: counts, probability matrix, sampling.
- NOT yet done: scoring a name / dataset NLL loss. This was the immediate next step.
  Reminder: include the ending `i.` bigram — he initially dropped the end token when
  listing "ravi" as `.r ra av vi`.
- Section 2 (neural net): not started.

## Next steps
1. Implement scoring: for real names, sum `log(prob[i][j])` over their bigrams, negate,
   average over all bigrams — expect ~2.45.
2. Implement Section 2: one-hot → `W` → softmax → NLL, train with gradient descent, confirm
   it lands on ~2.45 and produces the same samples as the counts route.
3. Not yet covered from the lecture: model smoothing (fake counts) and its regularization
   equivalent in the net; sampling from the net.

## Session log
- 2026-07-15: Deep dive on the scoring half of Section 1 — why multiply (chain rule /
  conditional probs / tree), coin analogies (independent vs dependent), the
  product→log→NLL→average chain, the self-scoring pitfall, length bias, and the net forward
  pass + terminology. He grokked the multiply→log→NLL chain and the net's row/column read.
  No scoring code written yet — he wants to grok all of lecture 2 before implementing.
