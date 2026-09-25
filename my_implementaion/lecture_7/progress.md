# Lecture 7 — Let's build GPT — progress

> Agent-maintained. Concept-by-concept status + where he got stuck/broke through.
> Read at the start of a lecture-7 session; update when he asks.

## Plan he chose
Implement GPT by **growing a model that trains end-to-end**, not by assembling isolated
parts at the end. He is skipping train/val split and other "add-ons" for speed, but wants
**every real GPT detail**. Working file: `my_implementaion/lecture_7/gpt.ipynb`.
Toy dims while prototyping: `T=4, B=10, C=4, heads=2` (so `head_size=2`).

## Architecture — where he is on the ladder
Target: `embed → stack of (MHA+FFN) blocks → final LN → linear→logits → loss / sample`.

DONE (built + understood):
1. input → tokens ✓
2. token embedding + positional embedding (added) ✓
3. single-head attention ✓
4. **multi-head attention** ✓ — per-head q/k/v in `ModuleList`s, `head_size=C//heads`,
   concat over `dim=-1` → back to `C`, **output projection `Linear(C,C)`** after concat.
5. **feedforward** ✓ — `Sequential(Linear(C,4C), ReLU, Linear(4C,C))`, per-token.
6. **Block** ✓ — `class block(nn.Module)` with pre-norm + residuals in `forward`:
   `x = x + attention(ln1(x))`, then `x = x + feedf(ln2(x))`. `ln1,ln2 = nn.LayerNorm(C)`.
7. **Stacked N blocks** ✓ — `nn.Sequential(*[block() for _ in range(n)])`; verified each
   block has independent weights (`block1.0/1/2.*` in named_parameters).

NOT done yet:
- **Final LayerNorm + final linear (C→vocab_size) → logits.** He understands placement:
  LN **before** the linear, never after (see "sharp corrections" below). Not yet coded.
- **Consolidate into `forward(idx, targets=None)`** returning logits (+ loss).
- **Loss** — cross-entropy on reshaped logits `(B*T, vocab)` vs targets `(B*T,)`.
- **Training loop** — real `get_batch` (fresh batches; current `inp2` is one hand-made
  batch), AdamW, `zero_grad → backward → step`.
- **Sampling / generate** — crop to block_size, last-step logits, softmax, multinomial,
  append.

## OPEN BUGS / cleanups to fix next session (told him, not yet fixed)
1. **`class GPT(block)` is wrong — must be `class GPT(nn.Module)`.** Inheriting from `block`
   makes `super().__init__()` create a whole extra UNUSED block's worth of params (the top
   set in named_parameters with no `block1.` prefix). Dead weight; also `gpt(idx)` would run
   the inherited `block.forward` on raw token ids and crash.
2. **`block_size` arg actually means number of layers → rename to `n_layer`.** `block_size`
   is the reserved term for context length (= `T`). Collision will bite at sampling.
3. **Naming mess:** class `block`, attribute `block1`, method `block` — three near-identical
   names. Suggest `class Block`, `self.blocks`, drop the wrapper method (just `self.blocks(x)`).
4. **Hardcoded `T` (deferred, agreed to fix at sampling):** `torch.arange(T)`, `tril(T,T)`
   use global `T`. Derive `B,T,C = x.shape` inside methods. Breaks at generation (context
   length varies); harmless during training (fixed-length batches).

## Sharp conceptual unlocks this session (all now solid)
- **Dot product, deep:** = length(a)×shadow(b on a); "add vs multiply" resolved (the `+` is
  bookkeeping for tilted axes). dot vs distance: distance also works, dot chosen for
  efficiency; `‖q−k‖²=‖q‖²+‖k‖²−2q·k`. Historical rival was **additive attention**, not
  distance. Visual: `llm_output/dot_product_projection_grok.html`.
- **√head_size scaling** = the **softmax** fix (keeps attention scores from saturating to
  one-hot at init). Visual: `llm_output/attention_scaling_softmax_grok.html`.
- **softmax dim bug:** on 3D `(B,T,T)` normalize the KEY axis = last = `dim=-1` (he'd used
  `dim=1`). Rule he now has: each query-row must sum to 1.
- **qk is not symmetric** — q,k are different projections, so `q(i)·k(j) ≠ q(j)·k(i)`; dot
  commutativity ≠ matrix symmetry. Attention is directional.
- **Residual = ADD, multi-head = CONCAT** (he had them swapped). Add keeps width fixed +
  gives identity highway; concat over depth explodes width. Anchored to token+pos embedding
  (also an add) to show adding ≠ destructive.
- **Each head reads the FULL input** (`Linear(C, head_size)`), not a slice; the split is on
  the OUTPUT (concat), not the input.
- **Output projection after concat** mixes across heads + rotates into add-compatible
  coordinates for the residual.
- **LayerNorm mechanics:** `(x-mean)/sqrt(var+eps)` then learnable per-feature `gamma*·+beta`
  (element-wise, NOT a linear layer). Normalizes over feature axis (last dim), per token.
  `nn.LayerNorm(C)` needs the size to allocate gamma/beta; `elementwise_affine=False` = pure
  math. Output is mean0/std1, NOT clamped to [-1,1].
- **BIG one (last topic): LayerNorm is NOT about softmax.** He kept fusing it with the
  √head_size/softmax story. Separated: √head_size = softmax fix; **LayerNorm = general
  activation-scale conditioning** for the next learnable layer + un-drifting the residual
  stream (used in nets with no softmax at all). Pre-FFN LN proves it (no softmax after it).
  Final LN goes **before** the final linear (condition its input), never after (logits must
  stay free; softmax is shift- but not scale-invariant, so normalizing logits distorts the
  distribution; CE wants raw logits).

## Teaching notes that worked
- Interactive HTML+SVG visuals land well. Built this lecture in `llm_output/`:
  `dot_product_projection_grok.html`, `attention_scaling_softmax_grok.html`.
- He drives the code from nudges; hand unguessable APIs straight (name + 1-line + tiny
  example), keep nudging on derivable shapes/logic. He wrote all the module code himself.
- Recurring pattern: he **fuses two adjacent mechanisms** (√head_size vs LayerNorm, concat
  vs add, heads vs residuals, batchnorm vs layernorm) and reasons from the merge. Fastest
  fix = explicitly name and separate the two tools and their distinct jobs.
