# Lecture 10 — Let's reproduce GPT-2 (124M) — progress

> Agent-maintained. Concept-by-concept status + where he got stuck/broke through.
> Read at the start of a lecture-10 session; update when he asks.
> NOTE: repo transcript numbering: file "10" = reproduce GPT-2 (this lecture).
> He calls the tokenizer "lecture 9"; this is "the next one / GPT-2". Code folder:
> `my_implementaion/lecture_10/`. He is doing GPT-2 BEFORE the tokenizer (lecture 9),
> and will circle back to the tokenizer afterward.

## Session mode
Grok-first, Socratic, no spilled answers — same as his standard. High-level map first,
then re-derive each piece himself. THEN watch video, THEN implement. "Assume he
understands up to lecture 8" (full lecture-7 GPT is his anchor).

## IMPORTANT framing for this lecture (two flavors)
- Sections 0–1 = ARCHITECTURE + training basics. Natural extension of his lecture-7 GPT;
  highly re-derivable from first principles.
- Sections 2–3 = SYSTEMS / PERFORMANCE (make it fast on GPUs) + optimization recipe.
  More "hardware reality + a trick with a reason" than pure derivation. Flag this so he
  knows what KIND of understanding to expect (still valuable for a researcher).

## The 4 sections (Karpathy's own structure) — the map
0. Intro + explore checkpoint: GPT-2 2019, miniseries 124M→1558M. 124M = 12 layers,
   768 channels (d_model), 12 heads, vocab 50257, context 1024.
1. Implement GPT-2 nn.Module. Deltas vs original Transformer / his L7 GPT:
   - LEARNED positional embeddings (wpe), not sinusoidal.
   - PRE-norm blocks: x = x + attn(ln1(x)); x = x + mlp(ln2(x)). Clean residual stream.
   - MLP: Linear(4x) → GELU(tanh approx) → Linear.
   - CausalSelfAttention batched (qkv in one matmul), multi-head.
   - HF weight naming (wte/wpe/h/ln_f/lm_head); load HF weights (transpose Conv1D).
   - Training basics: (B,T)→logits, cross-entropy (init loss ≈ -ln(1/50257) ≈ 10.82),
     overfit one batch, dataloader lite.
   - WEIGHT TYING: wte == lm_head weight (saves ~30% params). 
   - INIT: std 0.02; residual projections scaled by (2*n_layer)^-0.5 to tame residual
     stream variance growth.
2. Make it fast: baseline ~1000ms → TF32 (333) → bf16 (300) → torch.compile (130)
   → flash attention (96) → vocab 50257→50304 nice number (93). Core idea for compile
   + flash: the bottleneck is MEMORY I/O, not FLOPs.
3. Hyperparams (from GPT-3 paper) + distributed: AdamW betas(0.9,0.95) eps 1e-8,
   grad clip norm 1.0, warmup+cosine LR decay to 10%, weight decay 0.1 on 2D params only,
   FusedAdamW. GRADIENT ACCUMULATION (simulate 0.5M-token batch; scale loss by 1/N —
   because CE is a mean). DDP: data-parallel across GPUs, gradient all-reduce/average.
   Data: FineWeb-Edu (WebText/CommonCrawl not released). Eval: HellaSwag (multiple-choice
   by lowest avg loss). Results: beats GPT-2 124M overnight (~$10, ~1hr on 8xA100).

## Best first anchor
His lecture-7 GPT forward pass skeleton → find the DELTAS that make it GPT-2.
First re-derivations worth having him reach: pre-norm vs post-norm (gradient flow /
clean residual highway), why learned pos emb is fine, why weight tying makes sense,
the init-scaling-for-residual-stream argument.

## Status
- Just starting. Map about to be given; Socratic derivation next.

## Teaching notes carried in
- Reads partially / stops at first snag → front-load, self-contained, short.
- Percentages, toy numbers, physical pictures. Origin stories land.
- He drives logic; hand unguessable APIs one at a time (name + 1-line + tiny numeric ex).
- Visual/spatial → HTML+SVG in llm_output/ via `py` (stdlib only; NOT python3.13t).
  Candidate visuals: residual stream + pre/post-norm, precision formats (fp32/tf32/bf16
  bit layout), flash-attention memory-tiling, LR warmup+cosine curve, grad-accum.
- Transcript is ~4h; I have the full chapter arc + read intro. Will pull each section's
  transcript lines before drilling into its specifics (context budget).
