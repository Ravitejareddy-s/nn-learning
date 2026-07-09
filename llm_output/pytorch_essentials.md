# PyTorch Essentials — Things to Memorize

These are the operations you'll use constantly throughout the Karpathy series. Internalize these and you'll spend less time fighting the framework and more time understanding the neural nets.

---

## 1. Tensor Creation

```python
torch.tensor([1, 2, 3])              # from a Python list
torch.zeros(3, 5)                     # 3x5 of zeros
torch.ones(3, 5)                      # 3x5 of ones
torch.randn(3, 5)                     # 3x5 random normal (mean=0, std=1)
torch.arange(0, 10)                   # [0, 1, 2, ..., 9]
torch.linspace(0, 1, steps=5)         # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.empty(3, 5)                     # uninitialized (garbage values, fast)
```

**Key gotcha:** `torch.tensor` infers dtype from input. Pass floats if you want float tensor:
```python
torch.tensor([1, 2, 3])        # int64 (bad for neural nets)
torch.tensor([1.0, 2.0, 3.0])  # float32 (good)
```

---

## 2. Tensor Properties

```python
t.shape          # dimensions, e.g. torch.Size([3, 5])
t.dtype          # data type, e.g. torch.float32
t.device         # cpu or cuda
t.ndim           # number of dimensions (same as len(t.shape))
t.numel()        # total number of elements
```

---

## 3. Indexing & Slicing (same as NumPy)

```python
t[0]             # first row
t[:, 0]          # first column
t[1, 3]          # element at row 1, col 3
t[0:5]           # first 5 rows
t[:, 1:3]        # columns 1 and 2
t[[0, 2, 4]]     # rows 0, 2, 4 (fancy indexing)
t[t > 0]         # boolean masking — all elements > 0
```

---

## 4. Reshaping

```python
t.view(5, 3)         # reshape (must be contiguous in memory)
t.reshape(5, 3)      # reshape (works always, may copy)
t.unsqueeze(0)       # add dimension: (3,5) → (1,3,5)
t.squeeze()          # remove all size-1 dimensions
t.T                  # transpose (2D only)
t.permute(1, 0, 2)   # reorder dimensions (general transpose)
t.flatten()          # collapse to 1D
```

**Key insight:** `-1` means "figure it out":
```python
t.view(-1, 5)   # whatever rows needed to make 5 columns
t.view(15, -1)  # 15 rows, figure out columns
```

---

## 5. Math Operations

```python
# Element-wise
a + b, a - b, a * b, a / b, a ** 2

# Matrix multiply (THE most important one for neural nets)
a @ b                    # matrix multiply
torch.matmul(a, b)      # same thing

# Reductions
t.sum()                  # sum all elements
t.sum(dim=0)             # sum along rows (collapse rows)
t.sum(dim=1)             # sum along columns (collapse columns)
t.mean(), t.max(), t.min(), t.std()

# Keepdim (critical for broadcasting)
t.sum(dim=1, keepdim=True)   # shape stays (N, 1) instead of (N,)
```

---

## 6. Broadcasting Rules

When shapes don't match, PyTorch auto-expands dimensions **from the right**:
```python
# (3, 5) + (5,)    → works! (5,) becomes (1, 5) then (3, 5)
# (3, 5) + (3, 1)  → works! (3, 1) expands to (3, 5)
# (3, 5) + (3,)    → ERROR! 5 ≠ 3, can't broadcast
```

**Rule:** Dimensions must either be equal or one of them must be 1.

This is why `keepdim=True` matters:
```python
probs = counts / counts.sum(dim=1, keepdim=True)  # (27,27) / (27,1) → works
probs = counts / counts.sum(dim=1)                 # (27,27) / (27,) → ERROR
```

---

## 7. Common Neural Net Operations

```python
# Softmax (turn logits into probabilities)
probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
# or
probs = torch.softmax(logits, dim=1)

# Log softmax + NLL loss (what cross_entropy does internally)
loss = torch.nn.functional.cross_entropy(logits, targets)

# One-hot encoding
F.one_hot(torch.tensor([0, 3, 2]), num_classes=5)

# Embedding lookup (just fancy indexing)
C = torch.randn(27, 10)    # embedding table
C[5]                        # embedding for index 5
C[[1, 5, 13]]              # embeddings for multiple indices
```

---

## 8. Autograd (Backpropagation)

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()          # compute gradients
x.grad                # dy/dx = 2*x + 3 = 7.0

# For parameters in training loop:
loss.backward()       # compute all gradients
with torch.no_grad(): # don't track these operations
    p -= lr * p.grad  # update parameters
p.grad.zero_()        # MUST zero gradients before next backward()
```

**Key gotcha:** Gradients ACCUMULATE. Always zero them before the next `.backward()`.

---

## 9. Copy vs Reference (the bug you just hit)

```python
b = a           # SAME tensor, two names (modifying b modifies a)
b = a.clone()   # independent copy (safe)
b = a.detach()  # shares data but detached from computation graph
```

---

## 10. Random Number Generation (Reproducibility)

```python
torch.manual_seed(42)           # set seed for reproducibility
g = torch.Generator().manual_seed(2147483647)
torch.randn(3, 5, generator=g)  # reproducible random with specific generator
```

---

## 11. Useful Utilities

```python
torch.set_printoptions(sci_mode=False)   # no scientific notation
torch.multinomial(probs, num_samples=1)  # sample from distribution
torch.cat([a, b], dim=0)                 # concatenate tensors
torch.stack([a, b])                      # stack into new dimension
(a == b).sum()                           # count matches
torch.where(condition, x, y)             # conditional select
```

---

## 12. In-Place Operations (the underscore convention)

```python
t.zero_()        # fill with zeros IN PLACE
t.add_(5)        # add 5 IN PLACE
t.uniform_()     # fill with uniform random IN PLACE
p.grad.zero_()   # zero gradients IN PLACE
```

Anything ending in `_` modifies the tensor directly instead of returning a new one.

---

## 13. Device Management (GPU)

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t = t.to(device)
t = torch.zeros(3, 5, device=device)  # create directly on GPU
```

---

## Quick Reference: Shapes You'll See Constantly

| What | Typical Shape | Example |
|------|--------------|---------|
| Batch of inputs | `(B, C)` | (32, 27) — 32 examples, 27 features |
| Embedding table | `(vocab_size, embed_dim)` | (27, 10) |
| Weight matrix | `(fan_in, fan_out)` | (10, 200) |
| Bias vector | `(fan_out,)` | (200,) |
| Batch of logits | `(B, vocab_size)` | (32, 27) |

---

## The Pattern You'll Write 100 Times

```python
# Forward pass
logits = x @ W + b
loss = torch.nn.functional.cross_entropy(logits, targets)

# Backward pass
for p in parameters:
    p.grad = None        # zero gradients (modern way, slightly faster than .zero_())
loss.backward()

# Update
with torch.no_grad():
    for p in parameters:
        p -= lr * p.grad
```

This is literally the entire training loop. Everything else is just making the forward pass fancier.
