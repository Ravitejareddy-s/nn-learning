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
- DONE + validated himself: one-hot all inputs, `W = torch.randn(27,27,requires_grad=True)`,
  full forward pass + average NLL. Runs; loss = **3.6873** (correct random-init value).
- Current code loops in Python over all ~228k bigrams — correct but far too slow to train.
  Immediate next step is to vectorize into a single `X @ W` over the whole `[N,27]` matrix.
  I already gave the nudges: stack one-hots (or `F.one_hot(torch.tensor(inp),27).float()`),
  one matmul, row-softmax with `sum(dim=1, keepdim=True)`, then the **two-index gather** for
  target probs (`probs[torch.arange(N), out]`) — I told him I'd hand over the gather syntax
  when he reaches it, so don't pre-empt it.
- NOT yet coded: the training loop (`loss.backward()` → update → reset grad); sampling from the net.

## PyTorch syntax he now knows (don't re-explain unless asked)
`torch.tensor` vs `torch.Tensor` (data vs shape — the footgun that gave him `[0,0,0,0,0]`),
`F.one_hot(x, num_classes=)`, `.to(torch.float32)`, `torch.randn(...)` + `requires_grad=True`,
`torch.exp`, `torch.log` (natural, float-only), `.shape`/`.ndim`, `reshape(1,-1)`/`unsqueeze`/
`squeeze`, `@` matmul, `torch.sum`.
Not yet introduced: `sum(dim=, keepdim=)`, the two-index gather, `.backward()`, `torch.no_grad()`,
`.mean()`, and `torch.multinomial` in the net-sampling context.

## Next steps
1. **Vectorize the forward pass** (active focus): replace the per-bigram loop with `X @ W` on the
   full `[N,27]` matrix; row-softmax via `counts.sum(dim=1, keepdim=True)`; pull target probs with
   the two-index gather `probs[torch.arange(N), out]`; `.mean()`. Hand him the gather syntax when
   he hits it, not before.
2. **Gradient-descent loop:** `loss.backward()` → `W.data += -lr * W.grad` (under `torch.no_grad()`)
   → reset `W.grad`. Confirm loss falls toward ~2.45 and samples match the counts route. The
   gradient-vanishing intuition (shaky item) becomes concrete here.
3. Scoring in the counts route (Section 1) still pending — expect ~2.45.
4. Not yet covered: smoothing (fake counts) ↔ L2 regularization on the net; sampling from the
   trained net.

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
