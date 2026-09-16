# Lecture 3 — makemore Part 2: MLP — Progress

_Tracks concept-by-concept mastery and session history for lecture 3 (the Bengio-2003 MLP
character LM). Read this at the start of a lecture-3 chat; update on request. Not
auto-loaded — see `.kiro/steering/teaching.md`. Practice folder: `my_implementaion/lecture_3/`._

## Status: NOT STARTED (scaffold created 2026-09-05 at the lecture-2 → 3 handoff)

## Handoff from lecture 2 (anchor to these)
He finished lecture 2 strong and **built the bridge to this lecture himself**:
- Implemented the bigram neural net end-to-end (one-hot `@ W` → softmax → avg NLL → GD loop →
  sampling), all from nudges. Converges to ~2.46 with lr=50 (counts floor 2.4540).
- **Then built a trigram out of curiosity** and hit the **27ⁿ exponential wall** head-on:
  he represented the 2-char context as a one-hot over 628 observed pairs, weights (628,27).
  It works (counts floor 1.92, beats bigram) but he saw two problems live: (a) can't generalize
  to unseen context-pairs (his sampler literally breaks on `pair not in vocab`), (b) input blows
  up as 27ⁿ. **That wall is exactly the motivation for this lecture** — use it as the on-ramp.
- Rock-solid concepts to lean on: softmax = exp-then-normalize, NLL/MLE, the GD loop
  (`backward` → `no_grad` update → `zero_()`), learning-rate intuition (he has
  `llm_output/learning_rate_grok.html`: crawl/overshoot/blow-up, the lr-sweep finder, decay),
  and the **`one_hot @ W` = "select a row of W"** trick — which IS the embedding-table lookup,
  just reused. Point this out: an embedding table is the same operation he already knows.

## Agreed learning workflow (his choice, 2026-09-05)
**Generate-first hybrid** (not video-first): (1) he tries to *invent* the fix to the 27ⁿ wall
himself with nudges; (2) short grok session with me to lock the skeleton; (3) THEN watch the
Karpathy video as confirmation + nuance (esp. the ML-methodology half); (4) implement from
nudges. Rationale: he stops at the first snag, learns by proposing hypotheses, and groks before
coding — so passive video-first preloads Karpathy's frame and sticks worse for him. Struggle →
grok → video → implement.

## Concepts this lecture will cover (map, fill in as we go)
- **Embedding table** — each char → a short learned vector (e.g. 2–10 dims). = his `one_hot @ W`
  lookup, so input grows *linearly* (block_size × emb_dim) not 27ⁿ, and similar chars share
  structure → generalizes to unseen contexts. (Directly answers his trigram wall.)
- **Concatenation** of the context embeddings into one input vector.
- **Hidden layer + tanh** — why a nonlinearity (a stack of linear layers collapses to one).
- **Train / dev(val) / test split** — why, and what each is for (overfitting vs generalization).
- **Minibatches** — why we don't use all 228k every step; noisier but far faster gradient.
- **Learning-rate finder** — the sweep he already previewed in `learning_rate_grok.html`;
  here he builds it by hand. Plus **lr decay** near the end.
- **Over/underfitting, model capacity, hyperparameters** — the ML-craft half of the lecture.

## Watch-for (from lecture 2 habits)
- **Notebook stale-state / cell re-run accumulation**: re-running a training cell keeps training
  the same `W` (bit him twice — the trigram 2.47 was stacked 80-iter runs). Re-init before
  measuring.
- `square` vs `exp` monotonicity slip; sum-vs-mean on the loss. Both fixed in lec 2, watch for
  recurrence.
- Give unguessable PyTorch APIs straight (name + one-line + tiny numeric example); keep nudging
  on anything derivable (shapes, indexing, the algorithm). New APIs likely needed: embedding
  indexing `C[X]`, `.view()`/reshape for concatenation, `torch.randn` layers, cross-entropy
  (`F.cross_entropy` — introduce the why: numerically-stable fused softmax+NLL), minibatch
  indexing with `torch.randint`.

## Next steps
1. Challenge him: "feed 3–5 previous chars without the input exploding" → let him reach embeddings.
2. Grok the MLP skeleton (embed → concat → tanh hidden → logits → softmax/cross-entropy).
3. Watch the Karpathy video (lecture 3).
4. Implement from nudges; verify loss beats the trigram and samples improve.

## Session log
- 2026-09-05: Scaffold created at the lecture-2 → 3 transition. He chose the generate-first
  workflow. Entry point queued: invent the fix to the 27ⁿ wall. No lecture-3 code yet.
