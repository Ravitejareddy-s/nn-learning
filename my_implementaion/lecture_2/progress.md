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
- **Neural-net forward pass + NLL — now IMPLEMENTED and validated himself** (`bigram_with_nn.ipynb`).
  integer → `F.one_hot(..., num_classes=27).float()` → `@ W` → softmax (`exp` then /sum) → index
  the true letter's prob `avg[out[i]]` → `-log` → accumulate → `/N`. Ran it: loss = **3.6873**,
  i.e. just above the uniform floor `log(27)≈3.30` — the correct random-init signature (a bug
  would give NaN or ~8). Coded entirely from nudges. Confirms the net's row/column read in code.

### Solid (understood conceptually, not yet coded)
- Why the raw product underflows; `log` turns × into +; log is monotonic (so optimizing it
  is valid); negate → loss; average → comparable across lengths. Full chain down to
  "average negative log likelihood ≈ 2.45."
- Score the model on REAL data, not on its own generated names (self-scoring is circular;
  the degenerate "always aaaaaa" model scores its own output ~1.0).
- The length effect is a decode-time issue (beam search / length normalization), NOT a
  training or sampling bias; bigram counts are pooled and length-blind.
- One-hot @ W just selects a row of W; after training, `softmax(W)` should converge to the
  counts matrix. Not yet verified in code — needs the training loop (see next steps).

### Shaky (reteach / watch)
- Gradient intuition: he first guessed a product loss makes descent *overstep*; it actually
  makes gradients *vanish* (flat landscape → stuck). Reinforce when we hit the training loop.
- **Learning-rate feel (new, watch):** his training loop used `lr=0.1` for 10000 iters and
  plateaued at ~2.57, which he read as "converged." It wasn't — the step was just too small
  (small gradients × small lr ≈ no motion). At `lr=50` it hits ~2.46 in 100–200 iters (I
  verified: counts floor 2.4540; lr=50/200it → 2.4624). This is the exact spot Karpathy bumps
  0.1→50. Lesson to reinforce: a flat-looking loss can mean "step too small," not "done" —
  check by cranking lr and seeing if it drops further.
- Probability vs likelihood (same number, different viewpoint) — just introduced, let it
  settle.
- The "exp/log change the value, so how can the result be right?" instinct — resolved via
  the monotonic-relabel + "same peak on the mountain" picture (`llm_output/exp_and_log_grok.html`).
  Verify it holds when softmax/NLL show up in code.
- **`square` vs `exp` slip (coding):** his first forward-pass draft used square-then-normalize
  as the positivity function instead of `exp`. Fixed via the monotonicity anchor — square isn't
  order-preserving (`square(-5)=25 > square(2)=4`, so a strong-negative logit would win), whereas
  `exp` always keeps bigger-logit → bigger-prob. Watch for it recurring.
- **Notebook stale-state trap:** got confused by a stale `loss` output (a `[27]` tensor tagged
  `SqueezeBackward1`) left over from an un-rerun cell. Habit to build: check the execution-count
  and re-run before trusting an output.
- **Summing vs averaging the loss:** first version did `loss += pick` but forgot the final `/N`.
  The average is what makes it comparable to the ~2.45 / `log(27)` benchmarks.

## Implementation status

**Section 1 (`bigram.ipynb`):**
- Done: counts, probability matrix, sampling.
- NOT yet done: scoring a name / dataset NLL loss (counts route). Reminder: include the ending
  `i.` bigram — he initially dropped the end token when listing "ravi" as `.r ra av vi`.

**Section 2 (`bigram_with_nn.ipynb`):**
- DONE + validated: forward pass + average NLL (loss 3.6873 at init, the correct random floor).
- DONE himself: **vectorized** the forward pass — one-hot the whole `inp` list, single
  `onehot @ weights`, row-softmax via `exp.sum(1, keepdim=True)`, and the **two-index gather**
  `probs[torch.arange(len(probs)), out]` for target probs, `.mean()`. Nailed it from nudges;
  the gather he wrote himself.
- DONE himself: **gradient-descent loop** — `loss.backward()` → `with torch.no_grad(): weights -=
  weights.grad * lr` → `weights.grad.zero_()`. Correct use of `no_grad` + in-place leaf update.
- DONE himself: **sampling from the trained net** — start at `.` (0), one-hot → forward → softmax
  → `torch.multinomial`, walk until it emits 0. Produces name-like samples.
- OPEN — **lr too small:** loop uses `lr=0.1`, so after 10000 iters it only reached 2.5757.
  Fix is one number: `lr=50` → ~2.46 in ~200 iters (matches the counts floor 2.4540). See the
  shaky "learning-rate feel" item. He's going to make this change himself.
- NOT yet coded: (a) **regularization** `+ alpha*(weights**2).mean()` (alpha~0.01) = the NN twin
  of add-fake-counts smoothing; (b) the **equivalence check** `softmax(weights, dim=1)` vs the
  counts-route `P` matrix (should nearly coincide after lr=50 training) — proves the net
  rediscovers the counts.

## PyTorch syntax he now knows (don't re-explain unless asked)
`torch.tensor` vs `torch.Tensor` (data vs shape — the footgun that gave him `[0,0,0,0,0]`),
`F.one_hot(x, num_classes=)`, `.to(torch.float32)`, `torch.randn(...)` + `requires_grad=True`,
`torch.exp`, `torch.log` (natural, float-only), `.shape`/`.ndim`, `reshape(1,-1)`/`unsqueeze`/
`squeeze`, `@` matmul, `torch.sum`, `sum(dim=, keepdim=)`, the **two-index gather**
`probs[torch.arange(N), out]`, `.mean()`, `.backward()`, `torch.no_grad()` + in-place `weights -=`,
`weights.grad.zero_()`, and `torch.multinomial` for sampling from the net.
Not yet introduced: `torch.softmax` as a one-call op (he hand-rolls exp/normalize — fine),
L2/regularization term.

## Next steps
1. **Fix the learning rate** (active): change `lr=0.1`→`50` in the loop, rerun ~200 iters,
   confirm loss lands ~2.46 (≈ the counts floor 2.4540). Reinforces the "flat loss can mean
   step-too-small, not done" lesson.
2. **Equivalence check:** compare `torch.softmax(weights, dim=1)` (the NN's per-row next-char
   distribution) against the counts-route `P` matrix — rows should nearly coincide. The visual
   proof that gradient descent rediscovered the counts. (He asked what this meant on 2026-09-05;
   mid-explanation — pick up here.)
3. **Regularization:** add `+ alpha*(weights**2).mean()` (alpha~0.01) to the loss = the NN twin
   of add-fake-counts smoothing; watch predictions get more uniform as alpha grows.
4. **Scoring in the counts route (Section 1, `bigram.ipynb`)** still pending — expect ~2.45.
   Reminder: include the ending `i.` bigram.

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
- 2026-08-26: **First PyTorch coding session** (his first time touching the library). He
  implemented Section 2's forward pass + average NLL end-to-end in `bigram_with_nn.ipynb`,
  entirely from nudges, and validated it (loss 3.6873 vs the `log(27)≈3.30` floor). Fed him
  unguessable syntax one piece at a time: `torch.tensor` vs `torch.Tensor` (shape-vs-data
  footgun), `F.one_hot(num_classes=27)`, `torch.randn` + `requires_grad=True`, `torch.exp`,
  `torch.log` (natural, needs float), `.shape`/`.ndim`, `reshape`/`unsqueeze`/`squeeze`.
  Caught and fixed: square-vs-exp slip (monotonicity), wrong loss index `avg[index]`→
  `avg[out[index]]`, stale-notebook-cell confusion, sum-vs-average. Explained `num_classes`
  via the "your net is a 27-way classifier" framing (class ← classification). Built
  `llm_output/mean_and_std_grok.html` (mean/std, the normal distribution, and why `randn` not
  `rand` for weights — same pure-stdlib inline-SVG recipe as the exp/log page; Gauss +
  Galton-board origin story). Left off ready to vectorize the loop, then wire up gradient
  descent.
- 2026-09-05: **Review session — Section 2 is functionally complete.** Since 2026-08-26 he
  independently finished all three "next steps": vectorized forward pass (single `onehot @
  weights`, hand-wrote the two-index gather), the gradient-descent loop (`backward` →
  `no_grad` in-place update → `grad.zero_()`), and sampling from the trained net (multinomial
  walk). All structurally correct — no bugs. One real issue: `lr=0.1` left the loss plateaued
  at 2.5757, which he mistook for converged. Verified by rerunning his exact code: counts
  floor 2.4540; his lr=0.1 barely moves; lr=50/100it → 2.4729; lr=50/200it → 2.4624. Told him
  to change the one number himself. Flagged the two remaining lecture-2 pieces: L2
  regularization (= smoothing twin) and the `softmax(weights)` vs `P` equivalence check. He
  asked what the equivalence check means — explained it with a live side-by-side (his counts
  `P` row for 'a' vs `softmax(W)` after lr=50, matched to 2 decimals). He then asked the deep
  lr questions ("how do you decide step size / know you're not overshooting / how many runs
  without blowing up"). Built `llm_output/learning_rate_grok.html` (pure-stdlib inline SVG,
  reusing the `build_exp_log_html.py` Plot toolkit; generator `.kiro/build_learning_rate_html.py`).
  It uses exact GD on the toy bowl `L(w)=w²` (update factor `r=1-2·lr`) to show the four regimes
  (crawl / smooth / overshoot-but-converge / blow-up), the loss-vs-iteration diagnostic table,
  the lr-sweep finder (lecture-3 preview), lr decay, and the key grok: lr=0.1 vs 50 are both
  "right" for different gradient scales → there's no universal lr, hence you probe. Forward
  pointers left in: lr-finder (lec 3) and Adam (lec 4).
