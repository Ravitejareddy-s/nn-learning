# Lecture 7 — Let's build GPT — progress

> Agent-maintained. Concept-by-concept status + where he got stuck/broke through.
> Read at the start of a lecture-7 session; update when he asks.

## Plan he chose
Implement GPT by **growing a model that trains end-to-end**, not by assembling
isolated parts at the end. Started by building/understanding the **attention head first**
on a toy scratchpad, before wiring the real trainable pipeline.

## Done + solid (understood AND implemented himself, from nudges)
- **Bag-of-words average** as a lower-triangular matmul (v1→v2→v3). Understood that the
  causal "look only at the past" lives in a **separate (T,T) weight matrix**, not in the
  data tensor `(B,T,C)`.
- **masked_fill(-inf) + softmax** to get the averaging weights. Hit and understood the
  `exp(0)=1` pitfall (why zeros must become `-inf`, not 0).
- **Single self-attention head**: `q,k,v = x@W`; `qk = q@k.transpose(-2,-1)`; causal mask;
  `softmax(dim=-1)`; `out = we@v`. Built the whole forward pass himself.
- Bug he hit and fixed: **softmax `dim=1` vs `dim=-1`** once the tensor became 3D `(B,T,T)`.
  Rule he now has: normalize the key axis = last axis.
- **qk is not symmetric** because q and k are *different* projections
  (`q(t)·k(h) ≠ q(h)·k(t)`); dot-product commutativity ≠ matrix symmetry. Groked.
- **√head_size scaling** (`* head_size**-0.5`): understood WHY (q·k variance ~ head_size →
  softmax saturates to one-hot at init → gradients starve) and implemented it. Distinguished
  it from LayerNorm (fixed constant vs data-dependent measure-and-normalize + learnable).

## Dot-product deep dive (he wanted to fully grok before proceeding)
- Geometric meaning: **dot = length(a) × shadow(b on a)**. Resolved his "why add vs
  multiply": the `+` is only bookkeeping for tilted axes — align `a` to x-axis and the
  y-term vanishes, leaving one multiply.
- **dot vs distance**: distance also works for attention (net adapts); dot chosen for
  efficiency (single matmul) + signed/bilinear. Identity: `‖q−k‖² = ‖q‖²+‖k‖²−2q·k`.
  Corrected his belief that distance breaks in high-D (it doesn't). Historical rival to
  dot-product attention was **additive attention**, not distance.

## Shaky / new / not yet done
- **Scientific notation** (`e-06`, `e+01`) was new to him — now explained.
- **No training loop yet** for GPT. Current notebook is a scratchpad: random toy data,
  `C=2`, `T=4`, `B=10`, projections are fixed `torch.randn` (nothing learns).
- Not yet built: positional embeddings, multi-head, feedforward, residual + LayerNorm,
  scale-up.

## NEXT (agreed ladder — re-train after each rung, loss = oracle)
1. **Skeleton**: bigram model that trains + generates on `input.txt` (real get_batch,
   token embedding→logits, cross-entropy, training loop, generate). Baseline loss.
2. Graft the **attention head** in as a block + **positional embeddings**. Loss should drop.
3. **Multi-head** (parallel heads, concat).
4. **Feedforward** layer.
5. **Residual connections + LayerNorm** (enables depth).
6. **Scale up** (layers/heads/block_size, dropout).
Move projections to learnable `nn.Linear` inside an `nn.Module` at step 1–2.

## Teaching notes (what landed for him this lecture)
- Interactive **HTML+SVG visuals** worked very well. Built in `llm_output/`:
  `dot_product_projection_grok.html`, `attention_scaling_softmax_grok.html`.
- He drives the code from nudges; hand unguessable APIs straight (name + 1-line + tiny
  numeric example), keep nudging on derivable shapes/logic.
