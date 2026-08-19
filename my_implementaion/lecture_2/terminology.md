# Lecture 2 — Key Terminology

Quick-reference glossary for makemore lecture 2 (bigrams + the neural-net version).
Plain definitions, anchored to the notebook where useful.

## Setup & data
- **Character-level language model** — predicts the next *character* in a sequence (not the
  next word). makemore builds new name-like strings one letter at a time.
- **Bigram** — a pair of adjacent characters (`.r`, `ra`, `av`, ...). The bigram model
  predicts the next char from *only* the one before it.
- **Token** — a unit the model reads and predicts. Here each character is a token, plus one
  special `.` token marking the start/end of a name.
- **Vocabulary** — the set of distinct tokens: 26 letters + `.` = **27**.
- **`stoi` / `itos`** — lookup tables: string→index (`stoi['a']=1`) and index→string
  (`itos[1]='a'`). Needed because tensors are indexed by integers, not letters.

## The counting route
- **Count matrix (`N`)** — a 27×27 grid where `N[i][j]` = how many times char `j` follows
  char `i` in the data. **Row = current char, column = next char.**
- **Normalizing** — dividing a row of counts by its sum so it becomes probabilities.
- **Probability distribution** — a normalized row: positive numbers that sum to 1, giving
  P(next char | current char).
- **Broadcasting** — the torch rule for stretching a smaller-shaped tensor to match a bigger
  one in an elementwise op (e.g. dividing a 27×27 matrix by a 27×1 column of row-sums). Get
  the direction wrong and you silently normalize the wrong axis.
- **Multinomial sampling (`torch.multinomial`)** — given a probability row, draw a random
  index according to those probabilities (a weighted die roll).

## Evaluating a model
- **Likelihood** — the product of the probabilities the model gave to every actual bigram in
  the data. Higher = better fit.
- **Log likelihood** — the log of the likelihood; turns the giant product into a sum. It's
  monotonic, so maximizing it is the same as maximizing the likelihood.
- **Negative log likelihood (NLL)** — minus the log likelihood, so that **lower = better** (a
  loss). For one bigram it's `-log(prob the model gave the correct next char)`.
- **Average NLL (the loss)** — NLL averaged over all bigrams; comparable across datasets and
  name lengths. A trained bigram model lands around **2.45**.
- **Maximum likelihood estimation (MLE)** — the principle of choosing parameters that make
  the observed data as likely as possible. Minimizing average NLL is exactly this.
- **Model smoothing** — adding fake counts (e.g. +1) to every cell so nothing has probability
  0 (which would give `-log(0) = ∞`). More fake counts → smoother, more uniform model.

## The neural-net route
- **One-hot encoding** — represent an integer index as a length-27 vector of all 0s with a
  single 1 at that index. This is how a character is fed into the net. Multiplying a one-hot
  by `W` just **plucks out one row of `W`**.
- **Weights (`W`)** — the net's learnable parameters; here a 27×27 matrix, no bias, no
  nonlinearity. After training, `W` holds the **log-counts** (so `exp(W)` matches `N`).
- **Linear layer** — the operation `one_hot @ W` that produces logits; a matrix multiply
  with no activation.
- **Logits** — the raw, unnormalized scores the net outputs. Any real number (+ or −).
  Interpreted as **log-counts**.
- **Softmax** — turns logits into probabilities in two steps: **exponentiate** (`exp`, makes
  them positive = counts) then **normalize** (divide by the sum). Output is positive and
  sums to 1.
- **Forward pass** — running inputs through the net to get outputs (logits → probabilities)
  and the loss.
- **Backward pass / backpropagation** — computing the gradient of the loss with respect to
  every parameter by walking the compute graph backward.

## Training
- **Parameters** — the numbers the model learns (here, the entries of `W`).
- **Loss function** — a single number scoring how bad the model is; here the average NLL.
  Training minimizes it.
- **Gradient** — for each parameter, how much the loss changes if you nudge that parameter up
  a hair. Its sign tells you which way to move.
- **Gradient descent** — repeatedly nudging every parameter a small step *against* its
  gradient to lower the loss.
- **Learning rate** — the size of that step (e.g. 0.1). Too small = slow; too big = overshoot.
- **Regularization** — a penalty added to the loss (e.g. the mean of `W**2`) that pushes
  weights toward 0. It's the neural-net equivalent of model smoothing: stronger
  regularization → more uniform predictions.
