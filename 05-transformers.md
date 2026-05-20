# Chapter 5: Transformers
### The Architecture Behind GPT, BERT and Claude

> *"The transformer didn't improve on RNNs. It made RNNs obsolete."*

---

## Why Transformers Were Invented: The RNN's Three Fatal Flaws

Before transformers, the dominant architecture for language tasks was the **Recurrent Neural Network (RNN).** RNNs processed text sequentially — one word at a time, left to right — maintaining a hidden state that carried context forward.

To understand why transformers matter, you need to understand what RNNs got wrong.

### Fatal Flaw 1: Sequential Processing Cannot Be Parallelised

```
RNN processing "The cat sat on the mat":

Step 1: Process "The"     → update hidden state
Step 2: Process "cat"     → update hidden state (must wait for step 1)
Step 3: Process "sat"     → update hidden state (must wait for step 2)
Step 4: Process "on"      → update hidden state (must wait for step 3)
Step 5: Process "the"     → update hidden state (must wait for step 4)
Step 6: Process "mat"     → final output

A GPU has 10,000 parallel cores.
RNN uses 1 at a time.
GPU utilisation: ~0.01%
```

**Real-world consequence:** Training takes 10-100× longer than necessary. Iteration speed collapses. At scale, this becomes a business-critical engineering failure — models that could train in a week instead take months.

### Fatal Flaw 2: Vanishing Gradients Over Long Sequences

We saw vanishing gradients in Chapter 4 — they kill gradients in deep networks. RNNs suffer the same problem, but across **time steps** instead of layers.

For a document of 10,000 words, the gradient must flow from step 10,000 back to step 1 through 10,000 chain rule multiplications. It vanishes completely.

**Real-world consequence:** RNNs cannot learn long-range dependencies. In legal contracts, a definition on page 1 may govern a clause on page 35. An RNN cannot connect these. For long documents, critical context is permanently lost.

```
"The trophy didn't fit in the suitcase because it was too big."

What does "it" refer to? → The trophy.

For an RNN: by the time it reaches "it", the memory of "trophy"
has been partially overwritten by everything in between.
The connection is unreliable.
```

### Fatal Flaw 3: Fixed-Size Memory Bottleneck

The hidden state is a fixed-size vector that must compress all relevant context from the entire preceding sequence. For a 10,000-word document, one fixed vector carries everything. This is an impossible compression constraint.

**Real-world consequence:** Context gets lost. The model cannot reliably use information from far back in the input, regardless of how important it was.

---

## The Transformer's Solution — One Sentence

> **Instead of reading sequentially and compressing context into a fixed memory, transformers look at ALL tokens simultaneously and let every token directly attend to every other token.**

That's the entire innovation. Everything else is engineering around this core idea.

This eliminates all three RNN failure modes:
- All tokens processed in parallel → full GPU utilisation
- Every token directly connected to every other → no long-range gradient problem
- No compression bottleneck → full sequence accessible at every layer

---

## The Transformer — Big Picture First

let's establish what a transformer is at the highest level.

INPUT:  A sequence of tokens (words/subwords)
        "The cat sat on the mat"
         [1]  [2] [3]  [4] [5]  [6]

OUTPUT: Depends on the task:
        Translation → different language sequence
        Generation  → next token prediction
        Classification → single label

CORE MECHANISM:
Each token looks at ALL other tokens simultaneously and decides: "How relevant is every other token to MY meaning?"
This is ATTENTION.

The transformer processes the entire sequence at once — not one word at a time. This is what enables GPU parallelisation.

---

## Tokens: What the Transformer Actually Sees
let's establish what the input actually looks like. Text doesn't enter a transformer directly. It is first converted to **Tokens.**

```
"Hello world"     → [15339, 995]
"Transformer"     → [8291, 15099]   (may split into subwords)
```

Each token is represented as a **high-dimensional embedding vector** — typically 512 to 4096 numbers depending on the model. This vector encodes the token's meaning based on how it appeared across billions of training examples.

```
"king"   →  [0.2, -0.5, 0.8, 1.2, ..., 0.3]   (768 numbers)
"queen"  →  [0.1, -0.6, 0.7, 1.1, ..., 0.4]   (768 numbers)
"cat"    →  [0.9,  0.3, -0.2, 0.1, ..., 0.7]  (768 numbers)
```

"King" and "queen" are numerically close in this space. "Cat" is numerically distant. The geometry of the embedding space encodes semantic relationships.

The **input to the transformer** is a matrix: one row per token, all rows processed simultaneously.

---

## Positional Encoding: Solving the Order Problem

If we feed all tokens simultaneously, the transformer has no idea which word came first.

"Cat bit dog" and "Dog bit cat" contain the same three tokens — but mean opposite things. Order matters.

RNNs got order for free through sequential processing. Transformers, processing in parallel, need order information added explicitly.

The solution: **Positional encoding** adds a unique position signal to each token's embedding before it enters the transformer:

```
Token 1 "The":    embedding + position_signal_1
Token 2 "cat":    embedding + position_signal_2
Token 3 "sat":    embedding + position_signal_3
```

The position signal is a unique pattern of numbers for each position. It doesn't interfere with meaning — it marks "this token is at position N."

After positional encoding, each token vector carries:
1. **What** it means (the embedding)
2. **Where** it is in the sequence (the position signal)

The transformer can now process all tokens simultaneously AND respect their order.

---

## The Attention Mechanism: The Heart of Everything

This is the most important concept in all of modern AI.

### The Core Question Attention Answers

For every token in the sequence, attention asks:

> **"To understand MY meaning in this specific context, how much should I attend to every other token?"**

This isn't a fixed relationship. This is **context-dependent** — the same word attends differently in different contexts. "Bank" attends to "river" in one sentence and to "money" in another. The network learned these patterns from training data, not from rules.

### The Library System Analogy

```
You walk into a library with a QUERY: "books about machine learning"

Each book has a KEY on its spine — a summary of its contents:
  Book A: "machine learning, neural networks, Python"
  Book B: "cooking, Italian cuisine, recipes"
  Book C: "deep learning, transformers, AI"

You match your QUERY against each KEY:
  Query vs Book A: high relevance → high score
  Query vs Book B: no relevance  → low score
  Query vs Book C: high relevance → high score

You retrieve the VALUE — the actual content of relevant books.
```

The Three-Player System: Query, Key, Value
In attention:
- **Query (Q):** What this token is looking for — "What context do I need?"
- **Key (K):** What each token offers — "Here's what I contain"
- **Value (V):** The actual information a token contributes when selected

For every token:
1. Its Query is matched against every other token's Key
2. Relevance scores are computed for every pair
3. Scores are converted to attention weights (sum to 1.0)
4. The token receives a weighted combination of all Values

```
Token "it" in "The trophy didn't fit because it was too big":

"it"'s Query matched against all Keys:
  "The"     → weight 0.02
  "trophy"  → weight 0.61  ← highest relevance
  "didn't"  → weight 0.04
  "fit"     → weight 0.12
  "because" → weight 0.03
  "it"      → weight 0.08
  "was"     → weight 0.05
  "too"     → weight 0.03
  "big"     → weight 0.02

Weighted sum of Values → "it" absorbs 61% of its information from "trophy"
Result: "it" now has a representation that encodes its reference to "trophy"
```
The network learned these attention patterns through training. Nobody programmed "attend to noun antecedents." The weights Q, K, V were learned through backpropagation on billions of examples.

---

## The Attention Score — How Relevance is Computed

### Step 1: Compute Raw Scores
```
Take "it"s Query vector
Take "trophy"s Key vector
Compute how similar they are (dot product — a measure of alignment)
Result: a single number — the raw attention score
```
High score = these two tokens are highly relevant to each other.
Low score = these two tokens have little to say to each other.

### Step 2: Scale
Raw scores can get very large for high-dimensional vectors, which causes training instability. So we divide by a scaling factor (square root of the dimension size).
We turn down the volume so the scores stay in a reasonable range.

### Step 3: Softmax — Convert to Percentages
We take all the raw scores for one token (one score per other token) and convert them to weights that sum to 1.0.
```
Raw scores:  [0.1, 3.2, 0.4, 0.8, 0.2, 0.6, 0.3, 0.1, 0.1]
After softmax: [0.02, 0.61, 0.04, 0.12, 0.03, 0.08, 0.05, 0.02, 0.02]
               (these sum to 1.0 — like attention percentages)
```
The softmax does something important: it makes attention a competition. If one token becomes highly relevant, the others' weights get pushed down. Attention focuses — just like human attention focuses.

### Step 4: Weighted Sum of Values
Multiply each token's Value vector by its attention weight. Sum them up. This is the new, context-enriched representation of our token.
```
New representation of "it" =
  0.02 × Value("The") +
  0.61 × Value("trophy") +   ← "trophy" contributes 61% of the information
  0.04 × Value("didn't") +
  ...
```

"It" has absorbed information from "trophy" proportional to how relevant they were to each other. This is how transformers resolve ambiguity, track references and understand context.

---

## Multi-Head Attention: Multiple Simultaneous Perspectives

The word "bank" in "I went to the bank by the river to deposit money" — what should it attend to?

For understanding location: attend to "river"
For understanding action: attend to "deposit"
For understanding object: attend to "money"

One attention head computes one attention pattern. One attention mechanism might focus on location but miss the financial context. 

A sentence has multiple simultaneous relationship types:
```
"She gave him the book she wrote"

Syntactic:     "gave" → verb with subject "She" and object "him"
Coreference:   second "she" → same entity as first "She"
Semantic:      "book" → "wrote" → produced by someone
Positional:    "him" → "the" → adjacent article-noun relationship
```

No single attention pattern can optimally capture all of these simultaneously.

**Multi-head attention runs H independent attention mechanisms in parallel**, each learning to look for different relationship types:

```
Head 1: W_Q¹, W_K¹, W_V¹  → learns syntactic relationships (subject-verb-object)
Head 2: W_Q², W_K², W_V²  → learns coreference (pronouns → antecedents)
Head 3: W_Q³, W_K³, W_V³  → learns semantic similarity (related concepts)
Head 4: Learns positional relationships (nearby words)
...
Head H: W_Qᴴ, W_Kᴴ, W_Vᴴ → learns whatever reduces loss

(The network decides what each head specialises in — not us - the engineer )
```

All heads run simultaneously (full GPU parallelism). Their outputs are concatenated and projected back to the original embedding dimension.

GPT-3 uses **96 attention heads.** Each processes the entire sequence simultaneously. This is a central reason why transformers capture complex linguistic patterns so effectively.

---

## The Feed-Forward Layer: The Other Half

Attention gets all the attention. But there's a second major component in every transformer layer that's often underexplained.
After the attention mechanism produces a context-enriched representation for each token, a **feed-forward neural network** processes each token independently.
```
Transformer Layer =
  [Multi-Head Attention]   ← tokens communicate with each other
           +
  [Feed-Forward Network]   ← each token individually processed
```

**The split serves two distinct functions:**

- **Attention** = communication — tokens share information with each other
- **Feed-forward** = computation — each token processes the information it received

> Attention is the meeting room where tokens discuss context.
> Feed-forward is the office where each token individually processes what it learned.

Interpretability research suggests the feed-forward layers are where much of the factual knowledge in LLMs is stored. More on this in Chapter 7.

---

## Residual Connections and Layer Normalisation - Keeping It Stable

Two engineering additions that make deep transformers trainable at scale.

### Residual Connections (Skip Connections)

```
Output = LayerOutput + Input
```

**Why it matters for backpropagation:** Remember vanishing gradients? In a deep transformer (GPT-3 has 96 layers), gradients would vanish long before reaching the early layers.
The original input is added directly to each layer's output, creating a gradient highway that bypasses each layer.

```
Without residuals: gradient flows through 96 layers of chain rule → vanishes
With residuals:    gradient can skip directly through shortcut paths → survives

GPT-3 has 96 layers. Without residuals, it could not be trained.
```

This is borrowed from ResNet (2015) — originally developed for computer vision. The transformer inherited it and it's now universal in deep architectures.

### Layer Normalisation

After each sub-layer (attention and feed-forward), the activations are normalised to maintain consistent scale. Prevents activations from growing too large or too small as they pass through many layers, causing training instability. keeping training stable throughout.

---

## The Complete Transformer Layer

```
INPUT: Matrix of token vectors [sequence_length × embedding_dimension]

STEP 1 — MULTI-HEAD ATTENTION:
  Every token attends to every other token
  H heads compute in parallel with different W_Q, W_K, W_V
  Outputs concatenated and projected via W_O
  → Each token has context from the entire sequence

STEP 2 — ADD & NORMALISE (first residual):
  Output = Attention_output + Input
  Apply layer normalisation
  → Gradient highway preserved, activations stable

STEP 3 — FEED-FORWARD NETWORK:
  Each token processed independently
  Two-layer network per position
  → Tokens process information received from attention

STEP 4 — ADD & NORMALISE (second residual):
  Output = FF_output + Previous
  Apply layer normalisation

OUTPUT: Refined token representations [same shape as input]
        Each token now encodes context from the full sequence
```

This entire block = one transformer layer. Stack 96 of them (GPT-3) and you have **a large language model.**

---

## Three Variants: Encoder, Decoder, Encoder-Decoder

| Variant | Examples | How Attention Works | Best For |
|---------|----------|--------------------|----|
| **Encoder-only** | BERT, RoBERTa | Bidirectional: each token attends to all others (left AND right) | Understanding, sentiment, search, embeddings, classification |
| **Decoder-only** | GPT, Claude, Llama | Causal: each token attends only to previous tokens (no future) | Generation, completion, chat |
| **Encoder-Decoder** | T5, BART | Encoder: bidirectional; Decoder: causal + cross-attention to encoder | Translation, summarisation, seq-to-seq |

### Architecture Selection as an Engineering Decision

```
Building semantic search?          → Encoder-only (BERT family)
Building a chatbot?                → Decoder-only (GPT/Claude family)
Building a translation pipeline?   → Encoder-Decoder (T5 family)
Building a RAG system?             → Encoder for retrieval + Decoder for generation

Wrong choice = Fundamental architecture mismatch
This decision is made before writing a line of application code.
```
---

## Masked Attention: How Decoders Stay Honest

During training, decoder models see complete sentences but must predict the next word without seeing future words.

**Causal masking** prevents cheating by setting future attention scores to negative infinity before softmax:

```
Attention matrix for "The capital of France":

            The  capital   of   France
The       [ 0.8,   -∞,    -∞,    -∞  ]  ← sees only itself
capital   [ 0.5,  1.0,    -∞,    -∞  ]  ← sees The + itself
of        [ 0.3,  0.6,   0.9,    -∞  ]  ← sees The, capital, itself
France    [ 0.2,  0.7,   0.4,   1.0  ]  ← sees everything
```

After softmax, -∞ becomes exactly 0. Future tokens contribute nothing.

Result: All positions processed simultaneously (training efficiency) while each position only sees its causal past (no cheating). Both requirements satisfied with one elegant mechanism.

During inference (actual generation), there are no future tokens — the model generates one token, that becomes context for the next and so on.
This is **autoregressive generation** — the process by which ChatGPT produces text one token at a time.

---

## System Design - Production Scale: Transformers at Infrastructure Level

### The Quadratic Attention Cost

```
Attention computes relationships between every token pair:

N tokens → N² pairs

N = 1,000:    1,000,000 pairs     → manageable
N = 10,000:   100,000,000 pairs   → expensive
N = 100,000:  10,000,000,000 pairs → very expensive
```

This **Quadratic Scaling** is the central constraint on context window size. GPT-4's 128K context window required significant engineering to make computationally feasible.
Claude Sonnet 4.5 features a standard context window of 200k tokens,
Gemini AI models feature massive context windows ranging from 1 million to 2 million tokens.

### KV Cache: The Most Important Inference Optimisation

When generating token 500, the transformer needs Keys and Values from all 499 previous tokens to compute attention.

```
Without KV Cache:
  Generating token 500 → recompute K, V for tokens 1-499 + 500
  Work grows linearly with sequence length per new token

With KV Cache:
  Generating token 500 → load cached K, V for tokens 1-499
  Compute K, V only for token 500
  Work = constant per new token
```

KV caching is the difference between practical real-time inference and an unusable system. Every production LLM deployment uses it.

| Challenge | Solution |
|-----------|---------|
| High generation latency | KV caching, speculative decoding |
| Large memory footprint | Quantisation (INT8, INT4), sliding window attention |
| High cost | Model parallelism, continuous batching |
| Slow throughput | Batching multiple user requests |

---

## Key Takeaways

```
Transformer = Parallel attention-based sequence processing

✓ Core innovation over RNNs: sequential (no GPU parallelism), vanishing gradients over long sequences, fixed memory bottleneck

✓ Transformers: process all tokens simultaneously, Every token directly attends to every other token


Key components:
  Token Embeddings    → convert words to vectors
  Positional Encoding → inject order information into parallel processing
  Attention (Q, K, V): every token computes relevance to every other and receives a weighted blend of context
  Multi-Head Attention → H simultaneous attention patterns each learning different relationship types
  Feed-Forward Layer  → per-token independent processing, where factual knowledge is largely stored
  Residual Connections → gradient highways for deep training
  Layer Normalisation → stability across 96+ layers


Three variants:
  Encoder-only   → understanding  (BERT) → use for embeddings/search
  Decoder-only   → generation     (GPT)  → use for chat/completion
  Enc-Decoder    → transformation (T5)   → use for translation/summarisation

Masked attention: decoder training without future token leakage

Production reality:
  Quadratic attention cost → context window constraints
  KV cache → essential for inference efficiency
  Quantisation → cost reduction at scale
 
```

---

*[← Chapter 4](04-backpropagation-calculus.md) | [Chapter 6: Attention Step by Step →](06-attention-step-by-step.md)*
