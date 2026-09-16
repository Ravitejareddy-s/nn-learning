# Neural Networks: Zero to Hero — Lecture Coverage Map (by timestamp)

> A compact, timestamp-indexed map of each lecture in Karpathy's *Neural Networks: Zero to
> Hero*. For each time range it says what he does — explains theory vs. starts coding a given
> piece. Assumes the reader already knows the underlying math; this only records *what is
> covered when*, so a model has full context of the course without the videos.

---

## Lecture 1 — Intro to neural networks and backpropagation: building micrograd
(~2h26m; builds a scalar-valued autograd engine and trains a small MLP)

- **00:00–00:25** — Intro: goal is to define and train a neural net from scratch and see
  everything under the hood.
- **00:25–08:08** — Theory + demo: what micrograd is (an autograd engine implementing
  backprop), why it's scalar-valued for pedagogy, and how tensors are just efficiency. The
  whole thing is ~150 lines.
- **08:08–19:09** — Theory: what a derivative actually is (slope/sensitivity, limit
  definition), estimating it numerically by nudging; first with one input, then a function of
  multiple inputs.
- **19:09–32:10** — Coding: the core `Value` object — wrapping a scalar, tracking children +
  the op that produced it, and visualizing the expression graph.
- **32:10–51:10** — Theory + by-hand work: manual backprop through a simple expression;
  introduces the chain rule, local vs. global gradient, and that `+` routes / `*` scales.
- **51:10–52:52** — Preview: one optimization step (nudging inputs along the gradient).
- **52:52–01:09:02** — Theory + by-hand work: manual backprop through a single neuron
  (weighted sum + bias → tanh); derives tanh's local derivative `1−tanh²`.
- **01:09:02–01:22:28** — Coding: automate backward — per-op `_backward` closures, then a
  topological sort to run the whole graph's backward pass in order.
- **01:22:28–01:27:05** — Coding + theory: fix the "node used more than once" bug by
  accumulating gradients (`+=`), via the multivariate chain rule.
- **01:27:05–01:39:31** — Coding: add more ops (exp, pow, div, sub, handling constants /
  right-side ops) by breaking tanh into atoms; shows abstraction level of an op is arbitrary.
- **01:39:31–01:43:55** — Comparison: redo the same neuron in PyTorch; micrograd mirrors its
  `.data`/`.grad`/`.backward()` API, scalars being the single-element-tensor special case.
- **01:43:55–01:51:04** — Coding: the tiny NN library — Neuron, Layer, MLP (stacked layers),
  mirroring PyTorch's module design.
- **01:51:04–01:57:56** — Coding + theory: a tiny dataset and the loss function (mean-squared
  error), and why the loss is arranged so minimizing it means good predictions.
- **01:57:56–02:01:12** — Coding: collect all parameters (weights + biases) via
  `parameters()`.
- **02:01:12–02:14:03** — Coding + theory: manual gradient descent and the training loop —
  update sign (step against the gradient), learning-rate sensitivity, and the classic
  `zero_grad` bug.
- **02:14:03–02:16:46** — Theory summary: what neural nets are, and how this scales to modern
  nets (batching, cross-entropy, better optimizers, LR decay, other activations, regularization).
- **02:16:46–02:24:39** — Walkthrough: the full micrograd repo, then finding PyTorch's real
  tanh backward kernel and its `autograd.Function` extension API.
- **02:24:39–end** — Conclusion + outtakes.


---

## Lecture 2 — Building makemore: bigram language model
(~1h57m; builds a character-level bigram model two ways — counting normalized bigram
frequencies, and as a single-layer neural net trained by gradient descent — showing they
converge to the same result.)

- **00:00–03:03** — Intro: what makemore is, character-level language modeling, and the
  roadmap from bigram up through transformers.
- **03:03–09:24** — Coding: load the names dataset, build a word list, extract consecutive
  character bigrams with zip, and add start/end tokens around each word.
- **09:24–18:19** — Coding: count bigram occurrences (dict, then a 2D count tensor N), build
  char↔int lookup tables, index/accumulate counts, and visualize the count matrix.
- **18:19–24:02** — Coding: collapse the two special tokens into a single `.` token at index
  0, shrink the matrix to 27×27.
- **24:02–36:17** — Coding: turn a count row into a probability distribution, sample with
  torch.multinomial using a seeded generator, loop to generate names; compare trained vs.
  uniform samples.
- **36:17–50:14** — Theory + Coding: precompute the full probability matrix P; vectorized row
  normalization and broadcasting rules (the keepdim subtlety, in-place ops).
- **50:14–01:02:57** — Theory: evaluate model quality via likelihood → log likelihood →
  negative log likelihood → average NLL as the loss; zero-probability bigrams give infinite
  loss, fixed with model smoothing (add fake counts).
- **01:02:57–01:05:26** — Theory: recast bigram modeling as a neural net — input char →
  weights → next-char distribution, tuned by gradient descent.
- **01:05:26–01:13:53** — Coding: build the training set as integer input/label tensors,
  one-hot encode inputs and cast to float.
- **01:13:53–01:26:17** — Theory + Coding: implement the single linear layer as a matrix
  multiply of one-hot inputs by a 27×27 weight matrix; interpret outputs as logits (log
  counts) → exponentiate → normalize (softmax); note everything is differentiable.
- **01:26:17–01:38:36** — Coding: vectorized NLL loss via tensor indexing, then the backward
  pass and update (requires_grad, zero grads, backward, nudge W by −lr·grad).
- **01:38:36–01:47:49** — Coding: full training loop, scale from one word to all ~228k
  bigrams, tune the learning rate, reaching the same ~2.45 loss as the counting method.
- **01:47:49–01:54:31** — Theory: one-hot × W just selects a row of W, so the (exponentiated)
  W is equivalent to the counts table — reached by gradient descent instead of counting;
  smoothing is equivalent to L2 regularization pushing W toward zero (uniform predictions).
- **01:54:31–end** — Coding + Summary: sample from the neural net (identical results to
  counting), recap both approaches, and why the gradient-based one scales toward transformers.

---

## Lecture 3 — Building makemore Part 2: MLP (Bengio et al. 2003)
(~1h15m; a character-level MLP language model following Bengio 2003, plus core ML workflow —
embeddings, minibatches, learning-rate search, train/val/test splits.)

- **00:00–01:48** — Intro: why the bigram approach fails with more context — the count table
  grows exponentially with context length and becomes too sparse.
- **01:48–09:03** — Walkthrough: the Bengio et al. 2003 paper — embedding each token into a
  low-dimensional space so similar tokens sit nearby, enabling generalization to unseen
  contexts; the input-embeddings → hidden → softmax-output architecture.
- **09:03–12:19** — Coding: (re-)build the dataset with a configurable block size (context
  length), sliding a padded context window over each word to make (context → next-char)
  examples.
- **12:19–18:35** — Coding: the embedding lookup table C; show that indexing rows is
  equivalent to one-hot × C, and that PyTorch indexing embeds the whole (32×3) input at once.
- **18:35–29:15** — Coding + Theory: the hidden layer — concatenating the per-character
  embeddings, then the torch.Tensor internals (storage, views, strides) that make `.view()`
  a free reshape, plus broadcasting the bias correctly through the tanh.
- **29:15–32:17** — Coding: the output layer (logits), softmax to probabilities, and NLL loss;
  summary of the full ~3400-parameter network.
- **32:49–37:56** — Theory: replace the hand-rolled loss with F.cross_entropy — why it's
  preferred: fused kernels, a simpler/cheaper backward pass, and numerical stability via
  subtracting the max logit (avoiding exp overflow).
- **37:56–41:25** — Coding: the training loop; overfit a single batch (32 examples) to near-
  zero loss, and note why exact zero is impossible (same context maps to different targets).
- **41:25–45:40** — Coding + Theory: train on the full dataset (~228k examples); introduce
  minibatches — approximate but far cheaper gradients, so more steps beat fewer exact steps.
- **45:40–53:20** — Theory + Coding: finding a good learning rate — sweep exponentially spaced
  LRs, plot loss vs. LR, pick the valley; then use LR decay (10× lower) near the end.
- **53:20–01:00:49** — Theory + Coding: train/val/test splits (≈80/10/10) and why — model
  capacity lets nets memorize the training set, so tune hyperparameters on val and touch test
  only rarely; observe train≈val here means underfitting.
- **01:00:49–01:07:16** — Coding + Comparison: scale up the hidden layer (100→300), diagnose
  slow/noisy convergence, then visualize the learned 2D character embeddings (vowels cluster,
  `.` and `q` are outliers) suggesting the 2D embedding is now the bottleneck.
- **01:07:16–01:13:24** — Coding + Comparison: enlarge the embedding to 10 dims; plot log-loss
  (nicer than the hockey-stick), reach ~2.17 val loss, and watch train/val begin to diverge
  (onset of overfitting); list the knobs available to beat it.
- **01:13:24–end** — Coding: sample from the trained model (embed context → logits → softmax →
  multinomial, rolling the context window); names now sound more name-like. Plus a Colab note.

---

## Lecture 4 — Building makemore Part 3: Activations, Gradients & BatchNorm
(~1h56m; scrutinizing forward-pass activations and backward-pass gradients in an MLP, fixing
initialization, and introducing Batch Normalization.)

- **00:00–01:22** — Intro: why staying at the MLP level matters — understanding activation and
  gradient behavior is the key to why deeper/recurrent nets are hard to optimize.
- **01:22–04:19** — Walkthrough: cleaned-up starter code — dataset splits, refactored MLP with
  named hyperparameters, a split-evaluation helper under `torch.no_grad`, and sampling.
- **04:19–12:59** — Theory + Coding: fixing the initial loss — a well-configured net has a
  predictable init loss (~3.29 vs 27); confidently-wrong logits; scale down output weights,
  zero the output bias; the vanished hockey-stick loss curve.
- **12:59–20:00** — Theory: the saturated tanh — activation/pre-activation histograms, tanh
  tails killing gradients (1−t²), dead neurons, and the same issue in sigmoid/ReLU/ELU.
- **20:00–27:53** — Coding: squash W1/B1 to de-saturate tanh, check the saturation map, and
  the loss improvement; note why shallow nets are forgiving.
- **27:53–40:40** — Theory: Kaiming init — variance analysis of matmul, dividing by √(fan-in),
  the gain factor (√2 for ReLU, 5/3 for tanh), and the modern innovations (residuals,
  normalization, Adam) that reduce init sensitivity.
- **40:40–47:00** — Theory + Coding: batch normalization — normalizing pre-activations to unit
  Gaussian, why it's differentiable, and adding learnable gain/bias for scale and shift.
- **47:00–53:00** — Coding + Theory: placing the BatchNorm line, the coupling of examples
  across a batch, and how that jitter acts as a regularizer that's hard to remove.
- **53:00–01:00:00** — Coding: the inference-time problem — calibrating batch mean/std over the
  training set, then folding it into a running (EMA) estimate during training.
- **01:00:00–01:03:07** — Theory: the epsilon guard against divide-by-zero, and why the
  preceding layer's bias is spurious under BatchNorm.
- **01:03:07–01:04:50** — Summary: recap of the BatchNorm layer (trainable gain/bias, running
  buffers, center-then-scale-shift, inference behavior).
- **01:04:50–01:14:10** — Walkthrough: ResNet-50 — the conv→batchnorm→ReLU motif, bottleneck
  blocks, `bias=False` before norm layers, and a tour of PyTorch's Linear/BatchNorm1d params.
- **01:14:10–01:18:35** — Summary: big-picture recap — activation/gradient statistics matter,
  init scaling, and why deep nets need normalization layers.
- **01:18:35–01:26:51** — Coding: PyTorch-ifying — Linear, BatchNorm1d, and Tanh as stackable
  modules, a 6-layer MLP, the `.training` flag, parameters vs. buffers, retaining grads.
- **01:26:51–01:32:07** — Coding + Theory: diagnostic viz #1 & #2 — forward-pass activation
  histograms (percent saturation, how the 5/3 gain stabilizes std) and backward-pass gradient
  histograms (layers should share roughly equal gradient scale).
- **01:32:07–01:36:15** — Theory: the fully-linear case — removing tanh gives a shrinking/
  diffusion asymmetry (correct gain is 1); why nonlinearities are needed (linear stacks
  collapse to one layer).
- **01:36:15–01:46:04** — Coding + Theory: viz #3 & #4 — weight gradient-to-data ratios (the
  output layer trains ~10× too fast at init) and the update-to-data ratio over time (the ~1e-3
  heuristic) as a learning-rate/init diagnostic.
- **01:46:04–01:51:34** — Comparison: reintroduce BatchNorm — robustness to gain and to
  dropping fan-in normalization, but the learning rate must be retuned.
- **01:51:34–01:56:00** — Summary: the three goals (introduce BatchNorm, PyTorch-ify into
  modules, teach diagnostic tools), what was left undone, and that context length is now the
  bottleneck.

---

## Lecture 5 — Building makemore Part 4: Becoming a Backprop Ninja
(~1h55m; manually backpropagating through a 2-layer MLP with BatchNorm, without calling
loss.backward().)

- **00:00–07:26** — Intro: why manual backprop matters (backprop as a leaky abstraction,
  subtle real-world gradient bugs), plus a historical note on hand-written backward passes.
- **07:26–13:01** — Walkthrough: starter setup — small-random bias init to unmask gradient
  bugs, the fully expanded forward pass in intermediate tensors, the gradient-comparison
  utility, and a preview of the four exercises.
- **13:01–20:56** — Coding: backprop into logprobs from the mean-of-NLL loss (only the plucked
  correct-label entries get −1/n), then through log into probs.
- **20:56–33:14** — Coding: through the probability normalization — the broadcasting↔sum
  duality, the power rule for the reciprocal, accumulating the two branches into counts, and
  through exp.
- **33:14–41:44** — Coding + Theory: through the max-subtraction (numerical-stability) step —
  why its gradient is ~zero, and scattering it back via one-hot at argmax positions.
- **41:44–53:36** — Theory + Coding: backprop through linear layer 2 — deriving matmul backward
  from a small worked example (dA = dD·Bᵀ, dB = Aᵀ·dD, dC = column sum) via shape-matching,
  then through tanh (1−h²).
- **53:36–01:05:17** — Coding: the BatchNorm internals piece by piece — gain/bias (sum over
  batch), the inverse-std and variance (power rule), the squared-diff, and the branches into
  the centered diff, establishing the sum↔broadcast duality throughout.
- **01:05:17–01:26:31** — Theory + Coding: a Bessel's-correction digression (biased 1/n vs
  unbiased 1/(n−1) variance and the train/test mismatch), then finishing backprop through the
  mean, layer 1, the view/reshape, and the embedding-index loop into C.
- **01:26:31–01:36:37** — Theory + Coding: exercise 2 — analytically deriving the cross-entropy
  gradient w.r.t. logits (softmax probs with −1 at the correct label, scaled by 1/n),
  collapsing the whole atomic chain into one short expression; intuition as a push/pull force
  that sums to zero per row.
- **01:36:37–01:50:02** — Theory + Coding: exercise 3 — deriving the single-formula BatchNorm
  backward pass on paper (including the term that vanishes because μ is the batch mean), then
  implementing it in one broadcast line.
- **01:50:02–01:55:23** — Coding + Outro: exercise 4 — assembling the full ~20-line manual
  backward pass into the training loop, verifying against PyTorch, dropping loss.backward(),
  and confirming matching loss and samples; preview of RNNs/LSTMs.

---

## Lecture 6 — Building makemore Part 5: WaveNet
(~56 min; complexify the MLP into a hierarchical, tree-like WaveNet-style model that fuses
context progressively instead of crushing it in one layer.)

- **00:00–01:40** — Intro: motivation to take more context and fuse it gradually rather than in
  a single hidden layer; preview of the WaveNet hierarchical architecture.
- **01:40–06:56** — Walkthrough: starter code carried from part 3 — data setup, the
  Linear/BatchNorm1d/Tanh modules mimicking torch.nn, baseline val loss 2.10, sampled names.
- **06:56–09:16** — Coding: fix the noisy loss plot by reshaping the loss list into rows and
  averaging, revealing a clean curve and the LR-decay settling effect.
- **09:16–17:11** — Coding: PyTorch-ify — add Embedding, FlattenConsecutive, and a Sequential
  container, collapse the forward pass; trace and fix a BatchNorm training-mode bug.
- **17:11–19:55** — Theory + Coding: overview of WaveNet's progressive fusion
  (characters→bigrams→4-grams up a tree; dilated causal convolution framed as an efficiency
  detail), then bump context from 3 to 8 characters.
- **19:55–37:41** — Coding + Comparison: re-run the flat baseline at block size 8 (val loss
  2.10→2.02), then implement WaveNet — inspect tensor shapes, exploit batched matmul on the
  last dim, rework FlattenConsecutive to group pairs, and stack three hierarchical layers.
- **37:41–46:07** — Coding + Comparison: first WaveNet run shows no gain, exposing a
  BatchNorm1d bug (it kept per-position stats); fix the reduction dims, retrain, val loss
  2.029→2.022.
- **46:07–47:44** — Coding + Summary: scale up (bigger embeddings and hidden units, ~76k
  params) to val loss 1.993, with honest caveats — no experimental harness, guess-and-check
  tuning, and the gated/residual/skip parts of the WaveNet paper left unimplemented.
- **47:44–51:34** — Theory: how this relates to convolutions — sliding the tree over the
  sequence turns many independent forward passes into one, hiding the loop in CUDA kernels and
  reusing shared nodes.
- **51:34–56:20** — Summary: the layer/container work amounts to re-implementing torch.nn; the
  real development process (living in docs, shape gymnastics, notebook↔repo); what's ahead
  (real convolutions, residual/skip connections, an experiment harness, RNNs/Transformers);
  and a challenge to beat val loss 1.993.

---

## Lecture 7 — Let's build GPT from scratch, in code, spelled out
(~1h56m; builds a decoder-only Transformer character-level language model on Tiny Shakespeare,
from a bigram baseline up through full self-attention.)

- **00:00–07:52** — Intro: ChatGPT as a probabilistic language model, the "Attention is All
  You Need" Transformer, nanoGPT, and the goal of building a char-level GPT on Tiny Shakespeare.
- **07:52–14:27** — Coding: load the text, build the sorted 65-char vocabulary and a char-level
  tokenizer; contrast with subword tokenizers (SentencePiece, tiktoken/BPE); encode and split
  train/val.
- **14:27–22:11** — Coding + Theory: the data loader — block size, why context is
  block_size+1, the examples packed in a chunk, and the batch dimension.
- **22:11–38:00** — Coding: the bigram baseline as an nn.Module — token embedding → logits,
  cross-entropy loss (expected ~4.17), the generate loop (softmax + multinomial), and training
  with AdamW down to ~2.5.
- **38:00–42:13** — Walkthrough: port the notebook to a script — hyperparameters, device
  handling, the averaged estimate_loss under no_grad, train/eval mode.
- **42:13–51:54** — Theory + Coding: the math trick of self-attention — averaging past context
  (bag-of-words) with nested loops, then reframing it as a matmul with a normalized lower-
  triangular matrix (v1 → v2 vectorized).
- **51:54–01:00:18** — Coding: v3 — rewrite the weights with masked_fill(−inf) + softmax
  (triangular weights as data-dependent affinities); cleanup with n_embed and a separate LM
  head.
- **01:00:18–01:02:00** — Coding: positional encoding — add a position embedding table, sum
  token + position embeddings.
- **01:02:00–01:11:38** — Theory + Coding: the crux — a single self-attention head:
  query/key/value projections, affinities = q·kᵀ, masking, softmax, aggregating values.
- **01:11:38–01:16:56** — Theory: attention as communication over a directed graph; it has no
  notion of space (hence positional encoding); no communication across the batch; encoder
  (unmasked) vs. decoder (triangular-masked, autoregressive) blocks.
- **01:16:56–01:19:11** — Theory: self- vs. cross-attention (where Q,K,V come from), and scaled
  attention — dividing by √(head_size) to keep variance ~1 so softmax stays diffuse at init.
- **01:19:11–01:21:59** — Coding: package the Head module (register_buffer for tril), insert one
  attention block, crop context to block_size in generate; retrain to ~2.4.
- **01:21:59–01:26:48** — Coding: multi-head attention (parallel smaller heads, concatenated) →
  ~2.28; then the per-token feedforward MLP so tokens "think" after communicating → ~2.24.
- **01:26:48–01:32:51** — Theory + Coding: residual/skip connections — the Block interleaving
  communication and computation, the gradient "super-highway" via addition, projection layers,
  and 4× feedforward expansion → ~2.08.
- **01:32:51–01:37:49** — Theory + Coding: LayerNorm vs. BatchNorm (normalize rows, no running
  buffers), the pre-norm formulation, adding LayerNorms and a final LayerNorm → ~2.06.
- **01:37:49–01:42:39** — Coding: scale up — n_layer/n_head variables, dropout, bigger
  hyperparameters (batch 64, block 256, n_embed 384, 6 heads, 6 layers) → val loss ~1.48.
- **01:42:39–01:46:22** — Theory: encoder vs. decoder vs. encoder-decoder Transformers — why
  ours is decoder-only, and how translation adds an encoder with cross-attention.
- **01:46:22–01:48:53** — Walkthrough: nanoGPT — train.py/model.py, multi-head attention as a
  single batched 4D op, GELU.
- **01:48:53–01:56:18** — Theory + Summary: back to ChatGPT — pretraining vs. fine-tuning,
  scale comparison to GPT-3 (175B params, 300B tokens), and the RLHF pipeline (SFT → reward
  model → PPO); recap and pointers to further fine-tuning.

---

## Transcript 08 — State of GPT (talk)
(~41 min; how GPT assistants are trained end-to-end, then how to prompt and apply them
effectively. This is a talk, not a build-along, so all ranges are explanation.)

- **00:00–02:04** — Explanation: talk split into training then using GPT assistants; the four-
  stage pipeline (pretraining → supervised finetuning → reward modeling → RLHF), with
  pretraining ~99% of compute and the rest cheap finetuning.
- **02:04–05:00** — Explanation: data collection and the LLaMA-style data mixture; tokenization
  as lossless text→integer translation via BPE; GPT-3 vs LLaMA orders of magnitude, and why
  parameter count alone doesn't rank models (LLaMA 65B/1.4T beats GPT-3 175B/300B).
- **05:00–08:13** — Explanation: what pretraining does — B×T batches, documents packed with
  end-of-text delimiters, next-token prediction as the signal at every position; a Shakespeare
  example of outputs becoming coherent over training.
- **08:13–11:37** — Explanation: base models learn powerful general representations; the GPT-1→
  GPT-2 shift to few-shot prompting; the evolutionary tree and availability (GPT-4 base
  unreleased, GPT-2 weights public, LLaMA best available); base models are completers, not
  assistants.
- **11:37–16:55** — Explanation: supervised finetuning (small, high-quality prompt/response
  pairs from contractors → deployable assistant); reward modeling (rank multiple completions,
  train a binary comparator on a reward-readout token); RLHF (score completions and weight the
  LM loss by reward).
- **16:55–19:41** — Explanation: why RLHF (empirically preferred; the generate-vs-judge
  asymmetry) and its cost — mode collapse / entropy loss, so base models are still better for
  high-diversity generation.
- **19:41–20:26** — Explanation: the assistant landscape via Elo ratings (GPT-4, Claude, then
  SFT-only models like Vicuna/Koala).
- **20:26–24:23** — Explanation: human internal monologue (tool use, reflection, iterative
  rephrasing) vs. a GPT — a token simulator with equal compute per token, no self-knowledge or
  self-correction by default, but vast factual knowledge and near-perfect working memory (the
  context window).
- **24:23–27:15** — Explanation: prompting to close the cognitive gap — "tokens to think"
  (few-shot show-your-work, "let's think step by step"), and self-consistency / ensembling
  (sample multiple times, majority vote) because models can't recover from a bad token.
- **27:15–32:13** — Explanation: asking for reflection (System 1 vs System 2), Tree of Thought,
  recreating System 2 with Python glue (ToT tree search, the AlphaGo/MCTS parallel, ReAct,
  AutoGPT); "LLMs want to imitate, not succeed" — condition on quality without going out of
  distribution.
- **32:13–36:17** — Explanation: tool use / plugins (tell the model what it's bad at, e.g.
  arithmetic → calculators/code/search); retrieval-augmented generation (chunk, embed, vector
  store, stuff into context); constraint prompting for templated/JSON output; finetuning
  basics (parameter-efficient methods like LoRA, low-precision inference).
- **36:17–39:12** — Explanation: finetuning caveats (SFT achievable, RLHF unstable research);
  default recommendations — first maximize performance (GPT-4, detailed prompts, retrieved
  context, few-shot, tools, prompt chains/reflection), then optimize cost with smaller models.
- **39:12–41:30** — Explanation: limitations (bias, hallucination, reasoning errors, knowledge
  cutoffs, prompt-injection/jailbreak/data-poisoning) → use in low-stakes settings with human
  oversight as copilots; closing demo.

---

## Lecture 8 — Let's build the GPT Tokenizer
(~2h13m; building byte-pair encoding tokenizers from scratch and dissecting how tokenization
causes many LLM quirks.)

- **00:00–05:50** — Intro: why tokenization matters, the naive char-level tokenizer, GPT-2/
  Llama 2 vocab sizes, and a preview of tokenization-caused failures (spelling, arithmetic,
  non-English, trailing whitespace, SolidGoldMagikarp).
- **05:50–14:56** — Walkthrough: a live tokenizer UI showing arbitrary number splits, case/
  space sensitivity, non-English token bloat, Python-whitespace waste, and GPT-2 vs GPT-4
  token-count differences.
- **14:56–22:47** — Theory: Python strings as Unicode code points; why raw code points are a
  poor vocabulary; Unicode encodings (UTF-8/16/32) and why UTF-8 wins; why raw bytes give
  too-long sequences for a 256 vocab.
- **22:47–27:02** — Theory: daydreaming about tokenization-free byte-level models; then the
  byte-pair-encoding algorithm on a toy vocab (repeatedly merge the most frequent pair).
- **27:02–39:20** — Coding: encode text to UTF-8 bytes, count consecutive pairs (get_stats),
  the merge function, and the training while-loop over a longer text — vocab size as a
  hyperparameter, the merges dictionary, and compression ratio.
- **39:20–42:47** — Theory: the tokenizer is a separate stage from the LLM with its own
  training set; the training mix (language/code) determines merge density and sequence length.
- **42:47–57:36** — Coding: the decode function (id→bytes→string, with the invalid-UTF-8
  errors="replace" fix) and the encode function (apply merges in insertion order), verifying
  round-trips.
- **57:36–01:14:59** — Walkthrough + Comparison: the GPT-2 regex split pattern that blocks
  unwanted merges across letters/numbers/punctuation/whitespace; apostrophe/case quirks; and
  the tiktoken library with GPT-2 vs GPT-4 pattern differences (case-insensitive, merged
  whitespace, 3-digit chunks).
- **01:14:59–01:18:26** — Walkthrough: OpenAI's released GPT-2 encoder, mapping their
  encoder/vocab.bpe to our vocab/merges, and the spurious byte-encoder layer.
- **01:18:26–01:25:28** — Theory: special tokens (endoftext, chat tokens like im_start/im_end),
  how tiktoken injects them outside BPE, and the model surgery needed to add them.
- **01:25:28–01:28:42** — Walkthrough: the minbpe exercise/repo — the four-step build to a
  GPT-4 tokenizer and comparing learned merges against GPT-4's.
- **01:28:42–01:43:27** — Comparison: SentencePiece (Llama 2/Mistral) — merges on code points
  with byte-fallback rather than bytes-first; a config walkthrough and its historical quirks
  (add_dummy_prefix, unk token).
- **01:43:27–01:49:58** — Theory: where vocab_size matters in the Transformer (embedding table
  and LM head), the tradeoffs bounding it, extending vocab on a pretrained model via surgery,
  and gist tokens for prompt compression.
- **01:49:58–01:51:41** — Theory: multimodal tokenization — chunking images/video/audio into
  (hard or soft) tokens so the same Transformer applies (VQ, Sora visual patches).
- **01:51:41–02:10:20** — Walkthrough: revisiting each tokenization quirk with explanations —
  spelling/reversing failures, non-English bloat, arithmetic, Python inefficiency, the
  endoftext attack surface, trailing-whitespace out-of-distribution behavior, partial/unstable
  tokens, and the SolidGoldMagikarp untrained-embedding phenomenon.
- **02:10:20–end** — Summary: recommendations — reuse GPT-4 tokens via tiktoken if possible,
  use SentencePiece BPE (with caution) for custom training; wish for a trainable tiktoken.

---

## Transcript 10 — Let's reproduce GPT-2 (124M)
(~4h; end-to-end reproduction of the 124M-parameter GPT-2, from loading OpenAI's checkpoint to
training a from-scratch model that matches it, with a heavy focus on training speed and scale.)

- **00:00–03:39** — Intro: the GPT-2 sizes, why 124M, scaling laws, cost/time to reproduce
  (~$10, ~1hr), and why the GPT-3 paper is referenced for hyperparameters.
- **03:39–13:47** — Walkthrough: load OpenAI's GPT-2 124M via HuggingFace, inspect the state-
  dict shapes, visualize positional embeddings (sinusoidal structure, undertraining), and
  sample coherent text as the target.
- **13:47–31:00** — Coding + Theory: build the GPT-2 nn.Module (decoder-only, pre-norm + final
  LN) — clean residual stream, attention-as-reduce vs MLP-as-map, GELU (tanh approx) vs ReLU,
  multi-head attention; then load HuggingFace weights (skip buffers, transpose TF weights) via
  a from_pretrained classmethod.
- **31:00–41:47** — Coding: the forward pass (token + position embeddings → blocks → LM head)
  and the sampling loop (eval mode, device, tiktoken prefix, top-k=50 sampling).
- **41:47–52:53** — Coding: random-init garbage; device auto-detection (CPU/CUDA/MPS); data
  batching from tiny Shakespeare — reshape the token stream into (B,T) inputs and shifted
  targets.
- **52:53–01:06:14** — Coding + Theory: cross-entropy loss (sanity-check ~10.82 at init);
  overfit a single batch with AdamW toward zero loss; a lightweight looping data loader.
- **01:06:14–01:22:18** — Theory + Coding: weight tying between the token embedding and LM head
  (shared pointer, semantic motivation, saves ~30% of params); initialization per the GPT-2
  source (std 0.02, zero biases, 1/√N residual scaling to control activation variance).
- **01:22:18–01:39:38** — Theory + Coding: SECTION 2 speedups — A100 capabilities, float32
  default and lower-precision formats, tensor cores, memory-bandwidth limits; enable TF32
  (~3× via a one-liner, memory-bound so not the full 8×) with a timing harness.
- **01:39:38–02:00:18** — Theory + Coding: bfloat16 mixed precision via autocast (~300ms), then
  torch.compile (kernel fusion, fewer GPU round-trips, ~130ms).
- **02:00:18–02:14:55** — Theory + Coding: flash attention (online-softmax kernel avoiding the
  big attention matrix, ~96ms), and padding vocab 50257→50304 to nice powers of two (~93ms).
- **02:14:55–02:26:21** — Coding: SECTION 3 — GPT-3 hyperparameters (AdamW betas/eps, global
  grad-norm clipping at 1.0) and the LR scheduler (linear warmup + cosine decay to a minimum).
- **02:26:21–02:46:52** — Coding: selective weight decay (2D params), fused AdamW (~90ms), and
  gradient accumulation to reach GPT-3's large effective batch size with correct loss
  normalization.
- **02:46:52–03:10:21** — Coding: distributed data parallel (DDP) across 8 GPUs — process
  ranks, sharded data loader, gradient sync on the final micro-step, aggregated logging.
- **03:10:21–03:28:23** — Walkthrough + Coding: the GPT-2/GPT-3 datasets and choosing
  FineWeb-EDU (sharding/preprocessing), plus a validation split and periodic sampling.
- **03:28:23–03:43:05** — Coding: the HellaSwag eval harness (multiple-choice via lowest
  average loss) and kicking off the full training run.
- **03:43:05–03:59:39** — Comparison + Walkthrough: SECTION 4 results — surpassing GPT-2 124M on
  val loss and HellaSwag, comparing against GPT-3 numbers; shoutout to llm.c (raw C/CUDA).
- **03:59:39–end** — Summary: recap of the full reproduction and pointer to the build-nanogpt
  repo.
