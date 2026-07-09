# Karpathy's Neural Networks: Zero to Hero — Concept Priority Map

## How to read this

Each concept is rated on two axes:
- **Importance** (1-10): How foundational is this for everything else?
- **Difficulty** (1-10): How hard is it to truly grok?
- **Category**: 🧠 AXIOM (must deeply internalize) | 🔧 DERIVABLE (understand once, re-derive when needed) | 📦 TOOL (just know it exists and what it does)

---

## 🧠 AXIOMS — These are the load-bearing walls. Grok them or nothing else makes sense.

### 1. Backpropagation = recursive chain rule (Lecture 1)
- **Importance: 10 | Difficulty: 6**
- The gradient of the loss w.r.t. any parameter is computed by multiplying local gradients along the path from loss back to that parameter. That's it. Every single thing in deep learning rests on this.
- You MUST be able to: given any computation graph, manually trace the gradient backward through it.
- Key insight: it's just `d(loss)/d(x) = d(loss)/d(y) * d(y)/d(x)` applied recursively.

### 2. What a gradient actually means (Lecture 1)
- **Importance: 10 | Difficulty: 3**
- The gradient tells you: "if I nudge this parameter a tiny bit, how does the loss change?" Positive gradient = increasing the parameter increases the loss. That's the ONLY thing it means.
- This is the foundation of all optimization. If you don't have this intuition cold, nothing else will click.

### 3. The training loop: forward → loss → backward → update (Lectures 1-3)
- **Importance: 10 | Difficulty: 3**
- Forward pass computes predictions. Loss measures how bad they are. Backward pass computes gradients. Update nudges parameters to reduce loss. Repeat.
- This is the heartbeat of ALL neural network training. GPT, diffusion models, everything.

### 4. Softmax + Cross-Entropy Loss (Lectures 2, 3, 7)
- **Importance: 9 | Difficulty: 5**
- Softmax converts raw logits into a probability distribution. Cross-entropy measures how far your predicted distribution is from the true one.
- You must understand: why exponentiate? why normalize? why negative log probability IS the loss?
- Key: `loss = -log(p_correct)`. If you assign high probability to the right answer, loss is low. If you're confidently wrong, loss explodes.

### 5. Self-Attention mechanism (Lecture 7)
- **Importance: 10 | Difficulty: 8**
- Every token produces a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what do I give if attended to"). Attention weights = softmax(Q·Kᵀ / √d_k). Output = weighted sum of Values.
- This is THE core innovation of the Transformer. Without understanding this, you cannot understand GPT, BERT, or any modern LLM.
- Key insight: it's a communication mechanism. Tokens "talk" to each other through attention. The masking makes it causal (can only look backward).

### 6. Embeddings as learned representations (Lectures 2, 3, 7)
- **Importance: 9 | Difficulty: 4**
- An embedding table is just a matrix where row i is the learned vector for token i. Indexing into it is equivalent to one-hot × matrix.
- Words/characters that behave similarly end up near each other in embedding space. This is how the network generalizes.
- This concept extends everywhere: word2vec, token embeddings in GPT, image patch embeddings in ViT.

### 7. The gradient flow problem: vanishing/exploding gradients (Lectures 4, 5)
- **Importance: 9 | Difficulty: 6**
- In deep networks, gradients get multiplied many times. If each multiplication shrinks them → vanishing (network stops learning). If each multiplication grows them → exploding (training blows up).
- This is WHY we need careful initialization, batch normalization, residual connections, and why certain architectures work and others don't.
- Saturated tanh/sigmoid neurons kill gradients. Dead ReLU neurons permanently stop learning.

### 8. Residual connections (Lecture 7)
- **Importance: 9 | Difficulty: 3**
- `output = x + F(x)` instead of `output = F(x)`. The gradient flows directly through the `+` (which is just a gradient distributor), creating a "gradient highway."
- This is what makes training very deep networks possible. Without it, Transformers wouldn't work.
- Simple to understand, profound in impact.

---

## 🔧 DERIVABLE — Understand the idea deeply once. You can re-derive the details when needed.

### 9. Batch Normalization (Lecture 4)
- **Importance: 7 | Difficulty: 7**
- Normalizes activations across the batch to have zero mean and unit variance, then applies learnable scale and shift.
- WHY it works: keeps activations in the "active" region of nonlinearities, prevents internal covariate shift.
- Quirks to know: different behavior at train vs eval time, couples examples in a batch, has running statistics.
- You don't need to memorize the backward pass formula, but you should understand what it does and why.

### 10. Layer Normalization (Lecture 7)
- **Importance: 7 | Difficulty: 4**
- Same idea as BatchNorm but normalizes across features instead of across the batch. No train/eval mode difference.
- Used in Transformers instead of BatchNorm. Simpler, no batch coupling.

### 11. Kaiming/He initialization (Lecture 4)
- **Importance: 6 | Difficulty: 5**
- Scale your initial weights by `1/√(fan_in)` to keep activation variance stable through layers.
- The principle matters more than the formula: you want activations to neither shrink nor explode as they pass through layers.

### 12. Multi-Head Attention (Lecture 7)
- **Importance: 8 | Difficulty: 5**
- Run multiple attention heads in parallel, each with smaller dimension, then concatenate. Different heads learn different types of relationships.
- Once you understand single-head attention (Axiom #5), multi-head is just "do it N times in parallel with smaller dimensions."

### 13. Positional Encoding (Lecture 7)
- **Importance: 7 | Difficulty: 4**
- Attention has no notion of position — it operates on sets. Positional encoding injects position information.
- In the original Transformer: sinusoidal functions. In GPT: learned position embeddings. The key insight is WHY you need it, not the specific formula.

### 14. The Transformer Block architecture (Lecture 7)
- **Importance: 8 | Difficulty: 5**
- LayerNorm → Multi-Head Attention → Residual → LayerNorm → FeedForward → Residual. Stack N of these.
- Once you understand each component (attention, residual, layernorm, feedforward), the block is just assembly.

### 15. Byte Pair Encoding tokenization (Lecture 9)
- **Importance: 7 | Difficulty: 5**
- Start with bytes. Repeatedly find the most common pair, merge it into a new token. This builds a vocabulary of subword units.
- Important because tokenization affects EVERYTHING: arithmetic ability, multilingual performance, code handling.
- You should understand the algorithm well enough to implement it, but you don't need to memorize it.

### 16. Manual backprop through tensor operations (Lecture 5)
- **Importance: 7 | Difficulty: 8**
- Manually deriving gradients through matmul, batchnorm, cross-entropy at the tensor level.
- Extremely valuable exercise for building intuition, but in practice you'll use autograd.
- Do it once thoroughly. After that, the intuition stays with you.

### 17. Language modeling as next-token prediction (Lectures 2, 7)
- **Importance: 8 | Difficulty: 2**
- Given context, predict the next token. That's it. This simple objective, scaled up, gives you GPT.
- The entire autoregressive framework: train by predicting next token, generate by sampling from the predicted distribution and feeding it back.

### 18. Hierarchical feature fusion / WaveNet architecture (Lecture 6)
- **Importance: 5 | Difficulty: 5**
- Instead of crushing all context into one layer, progressively fuse pairs → bigrams → 4-grams in a tree structure.
- The specific architecture matters less than the principle: deep networks should process information gradually, not all at once.

---

## 📦 TOOL — Know what these are and when to use them. Don't spend time internalizing deeply.

### 19. torch.Tensor internals: views, storage, strides (Lecture 3)
- **Importance: 4 | Difficulty: 4**
- `.view()` reshapes without copying memory. Useful to know for efficiency, but just a PyTorch API detail.

### 20. Topological sort for backprop ordering (Lecture 1)
- **Importance: 3 | Difficulty: 3**
- Needed to implement autograd from scratch. In practice, frameworks handle this. Understand the concept (process nodes in reverse dependency order), don't memorize the algorithm.

### 21. Specific nonlinearities: tanh vs ReLU vs LeakyReLU (Lectures 1, 4)
- **Importance: 4 | Difficulty: 2**
- They all squash/threshold. tanh squashes to [-1,1], ReLU clips negatives to 0, LeakyReLU lets a small gradient through.
- Know they exist, know ReLU/GELU are standard now. Don't agonize over which one.

### 22. Learning rate schedules and decay (Lectures 3, 6)
- **Importance: 5 | Difficulty: 2**
- Start with higher LR, decay over time. Practical tuning detail. The concept is simple: big steps early, small steps to fine-tune.

### 23. Train/val/test splits (Lecture 3)
- **Importance: 5 | Difficulty: 1**
- Train on train set, tune hyperparameters on val set, report final numbers on test set. Standard ML practice.

### 24. Minibatch SGD (Lecture 3)
- **Importance: 5 | Difficulty: 2**
- Don't compute gradients on the entire dataset — sample a random batch. Noisy but fast. Standard practice.

### 25. Regularization: L2, dropout (Lectures 1, 7)
- **Importance: 5 | Difficulty: 3**
- Techniques to prevent overfitting. L2 penalizes large weights. Dropout randomly zeros activations during training.
- Know they exist and roughly why. Details are derivable.

### 26. Encoder vs Decoder Transformers (Lecture 7)
- **Importance: 5 | Difficulty: 3**
- Encoder: bidirectional attention (BERT). Decoder: causal/masked attention (GPT). Encoder-Decoder: both (original Transformer for translation).
- Just a taxonomy. The attention mechanism is the same, the masking pattern differs.

### 27. Graphviz visualization of computation graphs (Lecture 1)
- **Importance: 1 | Difficulty: 1**
- Just a debugging/visualization tool. Skip.

### 28. Bigram model with counts (Lecture 2)
- **Importance: 3 | Difficulty: 1**
- Count character pairs, normalize to probabilities. Simplest possible language model. Good for building intuition about what a language model IS, but you'll never use this in practice.

### 29. Zero-grad before backward (Lecture 1)
- **Importance: 4 | Difficulty: 1**
- Gradients accumulate by default in PyTorch. You must zero them before each backward pass. A practical gotcha, not a deep concept.

### 30. Sentencepiece vs tiktoken vs character-level tokenizers (Lecture 9)
- **Importance: 4 | Difficulty: 2**
- Different tokenizer implementations. Know they exist, know BPE is the core algorithm. Don't memorize API differences.

---

## Summary: Your study priority order

If you want to grok this in the shortest time:

1. **First pass (the non-negotiables):** Axioms 1-8. These are the skeleton. Spend 60% of your time here.
2. **Second pass (the muscle):** Derivables 9-17. These flesh out the skeleton. Spend 30% of your time here.
3. **Third pass (the skin):** Tools 19-30. Skim these. Spend 10% of your time here. You'll pick them up naturally as you implement.

The single most important thing: **be able to implement a Transformer from scratch** (Lecture 7). If you can do that, you've internalized most of what matters.
