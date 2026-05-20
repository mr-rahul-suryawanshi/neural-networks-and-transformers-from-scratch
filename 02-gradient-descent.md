# Chapter 2: Gradient Descent
### How Neural Networks Learn

> *"Training a neural network is not programming. It is optimisation. You define what 'good' looks like, and the algorithm finds its way there."*

---

## The Core Problem

Chapter 1 left us with 13,002 parameters — all set to random values. A random network produces garbage outputs. Here is the precise question we need to answer:

> **Given 13,002 knobs, all currently set randomly, how do we know which direction to turn each one and by how much, to make the network's predictions improve?**

The Setup: 13,002 Wrong Knobs - Picture a recording studio mixing board - 13,002 dials. All currently set to random positions. The music sounds terrible.

Your job: adjust every dial until the music sounds right. But you're blindfolded and can only hear whether the overall sound got better or worse after each adjustment.
That is training a Neural Network.

This is not a simple problem. Consider what "turn them randomly" would look like:

```
13,002 parameters
Each can be any real number
Space of possible configurations = ℝ^13,002

Brute-force search: completely infeasible
Random adjustment: statistically guaranteed to fail
```

We need something mathematically principled. That something is **Gradient Descent.**

- The "Sound Quality" is the Loss — a single number measuring how wrong the network currently is.
- The "Dial Adjustments" are Gradient Descent — the algorithm for deciding which direction to turn each dial.

---

## The Loss Function: One Number That Summarizes "How Wrong Are We?"

Before we can improve, we need to measure how bad the network currently is. This measurement is called the **Loss Function** (also called Cost Function).

After the network makes a prediction, we compare it to the right answer and compute a single number: The Loss.

### Intuition

The network sees an image of "3" and outputs:

```
Network said: "I think this is a 2" (with 80% confidence)
Correct answer: It's a 3

Loss = HIGH   ← very wrong, big penalty

---

Network said: "I think this is a 3" (with 92% confidence)
Correct answer: It's a 3

Loss = LOW    ← nearly right, small penalty

---

Neuron 0 ("0"): 0.04
Neuron 1 ("1"): 0.02
Neuron 2 ("2"): 0.31   ← network thinks "maybe a 2?"
Neuron 3 ("3"): 0.21   ← correct answer, should be 1.0
Neuron 4 ("4"): 0.18
...
```

The network was wrong. We need a single number that captures *How* wrong — so we can compare different parameter settings and know which is better. We want the loss to be as small as possible across all training examples.  **Simply, That's the entire goal of training.**


### Mean Squared Error

- The correct output (one-hot encoded): `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
- The actual output: `[0.04, 0.02, 0.31, 0.21, 0.18, ...]`

```
Loss = sum of (correct - actual)² for each output neuron

= (0 - 0.04)² + (0 - 0.02)² + (0 - 0.31)² + (1 - 0.21)² + ...
= 0.0016 + 0.0004 + 0.0961 + 0.6241 + ...
= high number → network is very wrong
```

**High loss = network is badly wrong. Low loss = network is close to right.**

The total cost across ALL training examples (60,000 images for MNIST):

```
Total Cost = average loss across all training examples
           = (1/N) × sum of losses for each example
```

**This Cost Function is what we are trying to minimise.** It is a function of all 13,002 parameters — every weight and every bias. Change any parameter and the cost changes.

---

## The Loss Landscape: Geometry of Learning

Here is the central mental model of gradient descent.

### With 2 Parameters (Visualisable)

Imagine a network with only 2 parameters: `w₁` and `w₂`. The cost function `C(w₁, w₂)` creates a **surface** in 3D space:

```
         C (height = error)
         │
         │        hill (high error)
         │   ╭──────╮
         │  ╱        ╲      ╭──╮
         │ ╱           ╲   ╱    ╲
         │╱     valley   ╲╱      ╲
         └─────────────────────────── w₁
        ╱
      w₂

Goal: find the lowest valley (minimum cost)
```

- **Peaks** = terrible parameter values (high error)
- **Valleys** = good parameter values (low error)  
- **The global minimum** = the best possible parameters

### With 13,002 Parameters (Reality)

The landscape exists in a **13,002-dimensional space**. You cannot visualise this. Nobody can. But the mathematics works the same way regardless of dimensionality — and that is the power of calculus.

> **Training a Neural Network = navigating a high-dimensional loss landscape to find a low valley, is why depth matters**

---

## Gradient Descent: The Algorithm

### The Blindfolded Hiker Analogy

Imagine you're standing on a hilly landscape, Blindfolded and you want to reach the lowest point — a valley. You can't see anything. But you can feel the slope beneath your feet.
You feel the ground tilting down to your left. So you take a step left.
The gradient is the slope you feel under your feet — but in the space of all 13,002 parameters.
For each dial on our mixing board, the gradient answers:

If I turn this dial slightly higher, does the loss go up or down? And by how much?

- If turning dial #47 up makes things worse → the gradient says "turn it DOWN"
- If turning dial #831 up makes things better → the gradient says "keep going UP"
- If turning dial #5,241 barely changes anything → "this dial doesn't matter much right now"

Your strategy:
1. Feel which direction is downhill
2. Take a step in that direction
3. Repeat

**That is gradient descent.** Gradient descent is just: Feel the slope, Step downhill, Repeat.

### The Gradient: Which Way Is Downhill?

The **Gradient** is a vector — one number per parameter — that answers:

> "If I slightly increase this parameter, does the cost go up or down and by how much?"


```
Gradient = [∂C/∂w₁, ∂C/∂w₂, ..., ∂C/∂w₁₃₀₀₂, ∂C/∂b₁, ...]
```

- **Positive component** → increasing this parameter increases cost → Decrease it
- **Negative component** → increasing this parameter decreases cost → Increase it  
- **Near zero** → this parameter barely affects cost right now

### The Update Rule

Step opposite to the gradient:

```
For each weight: new_weight = old_weight - (learning_rate × gradient)

w ← w - η · (∂C/∂w)
```

Where `η` (eta) is the **Learning Rate** — a small positive number controlling step size.

**A concrete example:**

```
Current weight:     w = 3.2
Gradient at w:      ∂C/∂w = +0.8   (cost increases when w increases)
Learning rate:      η = 0.1

New weight: w = 3.2 - (0.1 × 0.8) = 3.2 - 0.08 = 3.12
```

We moved slightly in the direction that reduces cost. Repeat for all 13,002 parameters simultaneously, across hundreds of thousands of steps → **The Network Learns.**

---

## The Learning Rate: The Most Critical Hyperparameter

```
w ← w - η · (∂C/∂w)
```

`η` looks simple. It is not.

### Too Large → Divergence

```
Imagine descending a valley:

  Step size = 10 metres
  Valley width = 5 metres

  Step 1: Overshoot the valley, land on the other side
  Step 2: Overshoot back, land even higher
  Step 3: Loss is now larger than when we started

Symptom: loss bounces wildly, then becomes NaN (undefined)
```

### Too Small → Slow Convergence

```
  Step size = 0.001 metres per step
  1 million steps later: barely moved

Symptom: loss decreases but agonisingly slowly
         training may not finish in a reasonable timeframe
```

### Just Right → Steady Convergence

```
  Loss curve: smooth, consistent downward slope
  Converges to a good solution in reasonable time
```

**The learning rate is so consequential that choosing it wrong is the #1 reason models fail to train.** Not the architecture. Not the data. The learning rate.

### Modern Solutions

| Technique | What It Does |
|-----------|-------------|
| **Learning rate schedules** | Start large, decay over time |
| **Warmup** | Start tiny → increase → decay |
| **Adam optimizer** | Automatically adapts learning rate per parameter |
| **Grid search** | Try `{0.1, 0.01, 0.001, 0.0001}` empirically |

---

## Three Variants of Gradient Descent

Computing the exact gradient requires evaluating all N training examples per step. For MNIST (N = 60,000) this is manageable. For GPT (N = hundreds of billions of tokens) it is not. Three variants address this:

### 1. Batch Gradient Descent (Full)

```
Use ALL N examples per gradient update

+ Most accurate gradient estimate
+ Smooth, predictable loss curve
- Extremely slow per update at large N
- Doesn't fit in memory for large datasets
- Gets stuck in sharp local minima
```

### 2. Stochastic Gradient Descent (SGD)

```
Use exactly 1 random example per update

+ Very fast updates
+ Noise can help escape local minima
- Gradient estimate is very noisy
- Erratic path toward minimum
- GPU sits mostly idle (no parallelism)
```

### 3. Mini-Batch Gradient Descent ← Used In Practice

```
Use B random examples per update (typically B = 32 to 256)

+ Good gradient estimate (noise averages across the batch)
+ GPU fully utilised (batch processed in parallel)
+ Fast enough for practical training
+ Noise level tunable via batch size
+ The Goldilocks balance

Going from 60,000 examples to 32 means ~1,875 fewer
examples per gradient step — a significant compute saving.
```

> **Every modern neural network — including GPT, BERT, Stable Diffusion — is trained with mini-batch gradient descent or an improved variant. This is the workhorse algorithm of all deep learning.**

### Batch Size Decision Framework

```
Batch size 1 (pure SGD):
  - Each update uses only 1 example
  - Gradient measurement is: [very accurate / very noisy] ← pick one
  - Updates per second: [many / few] ← pick one
  - Risk: the loss curve will look [smooth / jagged] ← pick one
  - Real problem: GPU is [fully utilized / mostly idle] ← pick one
    (because GPUs are designed for parallel computation, processing 1 example at a time wastes 99% of GPU capacity)

Batch size 50,000 (near-full batch):
  - Gradient measurement is: [very accurate / very noisy] ← pick one
  - Updates per second: [many / few] ← pick one
  - Risk: might converge to [sharp / flat] minima
    (large batches tend to find sharp minima that generalise poorly)
  - Memory: may [fit / not fit] in GPU RAM

Batch size 64-256 (mini-batch):
  + Balance - Good gradient estimate (noise averages out)
  + GPU utilization fully utilised
  + Many reliable updates per hour
  + Slight noise helps escape bad local minima
  → The correct default choice
```

---

## The Loss Landscape: Deeper Truths

### Local vs Global Minima

```
Global minimum: the absolute lowest point in the entire landscape
Local minimum:  a valley surrounded by walls but not the lowest overall

         │   ╭─╮           ╭──────────╮
    C    │  ╱   ╲         ╱            ╲
         │ ╱  L   ╲   ╭──╯   G (global)╰───
         │╱ (local) ╲╱
         └──────────────────────────── params
```

**Will gradient descent find the global minimum?**

No. It finds a local minimum and cannot see whether a deeper valley exists elsewhere.

**The surprising empirical truth for large networks:**

For networks with billions of parameters, the loss landscape is so high-dimensional that most local minima are almost as good as the global minimum. The network almost always escapes saddle points due to the noise in mini-batch gradients. This is one of the empirically observed mysteries of deep learning that theory doesn't fully explain yet.

### Saddle Points

```
A saddle point: flat in some directions, curved in others
Gradient = 0, looks like a minimum but isn't
Mini-batch noise helps escape these
```

---

## Beyond Vanilla Gradient Descent

### Momentum

Adds "velocity" — like a ball rolling downhill accumulating speed:

```
velocity = momentum_coefficient × velocity - learning_rate × gradient
weight   = weight + velocity
```

- Accelerates through flat regions
- Dampens oscillations in narrow valleys  
- Escapes shallow local minima

### Adam (Adaptive Moment Estimation) — The Industry Standard

Adam computes per-parameter adaptive learning rates. Parameters with consistently large gradients get smaller effective steps. Parameters with small gradients get larger effective steps.

```python
# In practice:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Adam is the default optimizer for most neural networks.** When you see `Adam(lr=3e-4)` in code, the model is using adaptive per-parameter learning rates computed from the first and second moments of the gradient history.

---

## The Complete Training Loop - What Actually Happens
```
START with random weights (chaos)
    ↓
Show the network 32 random images (mini-batch)
    ↓
Network makes predictions (probably wrong)
    ↓
Compute loss (how wrong?)
    ↓
Compute gradient (which direction is downhill for each weight?)
    ↓
Update all 13,002 weights (step downhill)
    ↓
Repeat with next 32 images
    ↓
Do this hundreds of thousands of times
    ↓
ARRIVE at weights that produce low loss = trained network

```
```python
# Pseudocode — the actual training loop

model = NeuralNetwork(layers=[784, 16, 16, 10])
optimizer = Adam(learning_rate=0.001)

for epoch in range(num_epochs):             # Multiple passes over dataset
    for batch in dataloader(batch_size=32): # Mini-batches
        
        # 1. FORWARD PASS: compute predictions
        predictions = model.forward(batch.inputs)
        
        # 2. COMPUTE LOSS: measure how wrong we are
        loss = mean_squared_error(predictions, batch.labels)
        
        # 3. BACKWARD PASS: compute gradients ( will see in next Chapter 3)
        gradients = loss.backward()
        
        # 4. UPDATE: gradient descent step
        optimizer.step(gradients)
        
        # 5. ZERO GRADIENTS: reset for next batch
        optimizer.zero_grad()
```

Step 3 — `loss.backward()` — computes the gradient for every single weight in the network. **How?** That is backpropagation — the subject of Chapters 3 and 4.

---

## Production Scale: Gradient Descent on Large Models

When training GPT-scale models, the same algorithm runs with these engineering additions:

| Challenge | Engineering Solution |
|-----------|---------------------|
| Model too large for one GPU | **Model parallelism**: split layers across GPUs |
| Batch computation too slow | **Data parallelism**: each GPU processes different batches, gradients averaged |
| Memory constraint | **Gradient checkpointing**: recompute activations instead of storing all of them |
| Training instability | **Gradient clipping**: cap gradient magnitude to prevent explosion |
| Slow convergence | **Warmup + cosine decay schedule** |

GPT-3's training infrastructure ran across thousands of A100 GPUs. Every concept above — batch size, learning rate, optimizer — operated at a scale that required months of engineering to make feasible. The **Algorithm** is what you just learned. The **Engineering** is what made it possible.

---

## Key Takeaways

```
✓ Loss function = a single number measuring how wrong the network is

✓ Gradient = vector pointing in direction of steepest ascent (one component per parameter)

✓ Gradient descent = step opposite the gradient repeatedly (navigate toward lower loss)

✓ Learning rate = step size; getting it wrong is the #1 training failure

✓ Mini-batch SGD = the practical standard (balance between gradient accuracy and compute speed)

✓ Adam = adaptive per-parameter learning rates; the default optimizer

✓ Training loop = forward pass → loss → backward pass → update → repeat

✓ The algorithm is simple. The scale is what makes it expensive.
```

---

## What's Next

Gradient descent **uses** gradients to update weights. But how are those gradients actually computed?

For 13,002 weights, computing the gradient naively (perturb each weight individually, measure loss change) would require 26,004 forward passes per update step. That is completely infeasible.

**Chapter 3** introduces **Backpropagation** — the algorithm that computes the gradient for all 13,002 weights in just **two passes** (one forward, one backward), regardless of how many parameters the network has.

It is arguably the most important algorithm in Modern AI.

---

*[← Chapter 1](01-what-is-a-neural-network.md) | [Chapter 3: Backpropagation →](03-backpropagation-intuition.md)*
