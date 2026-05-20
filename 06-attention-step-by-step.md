# Chapter 6: Attention in Transformers — Step by Step
### The Complete Mechanical Walkthrough

> *"One equation. Five steps. The foundation of all modern AI."*

---

## Why Go Deeper Into Attention?

Chapter 5 explained what attention does. Chapter 6 explains exactly how it does it — every computation, every step, every design decision.

This matters for three reasons:

1. **RAG system design:** Vector similarity search (the core of RAG) uses the same dot-product mathematics as attention. Understanding attention deeply means understanding retrieval deeply.
2. **Debugging transformer behaviour:** When models produce unexpected outputs, understanding attention patterns is the diagnostic tool.

---

## Setup: Two Tokens, Four Dimensions

To make this concrete without drowning in scale, we'll work with two tokens and four-dimensional vectors. Real models use 512 to 4096 dimensions — the principle is identical.

```
Token "capital":  embedding = [0.5, 0.2, 0.8, 0.1]
Token "France":   embedding = [0.3, 0.9, 0.1, 0.7]
```
These are the embeddings — the starting representations before attention.
Goal: compute how much "capital" should attend to "France" and vice versa.

---

## Where Q, K, V Come From: Learned Projections

The most commonly missed point in attention explanations:

**Q, K and V are NOT the token embeddings themselves.** They are learned transformations of the embeddings.

Each attention head has three learned weight matrices: `W_Q`, `W_K`, `W_V`. These are learned through backpropagation — just like any other network weights.

W_Q​ — the Query projection matrix
W_K​ — the Key projection matrix
W_V​ — the Value projection matrix

For each token, we compute three derived vectors:

```
Query  = token_embedding × W_Q
Key    = token_embedding × W_K
Value  = token_embedding × W_V
```

**Why three separate projections?**

Q, K and V serve fundamentally different roles:

```
QUERY  → "What am I looking for?"
         Asking: which other tokens are relevant to understanding me?
         Optimised to: match against Keys efficiently

KEY    → "What do I offer?"
         Advertising: here's what I contain, match your query against me
         Optimised to: be matched against Queries

VALUE  → "What do I actually contribute?"
         Providing: if you select me, here's the information you receive
         Optimised to: carry useful information to the recipient
```

The separation allows the network to learn:

How to ask questions (Q weights)
How to label content (K weights)
What information to transfer (V weights)

Independently. Each optimised for its specific role.

If Q and K were the same projection, the network loses the ability to independently optimise what it asks for versus what it offers. The separation is what allows each head to specialise. If you collapsed Q and K into the same matrix, heads could no longer develop distinct "asking" and "answering" behaviours — they'd be forced to be symmetric, losing significant representational flexibility.

---

## Computing Attention — The Five Steps of Attention: Complete Walkthrough

### Step 1: Compute Q, K, V for Every Token

After multiplying embeddings by projection matrices (learned during training):

```
"capital":
  Q_capital = [0.6, 0.1, 0.9, 0.2]   ← what "capital" is looking for
  K_capital = [0.4, 0.3, 0.7, 0.5]   ← what "capital" offers
  V_capital = [0.8, 0.2, 0.5, 0.3]   ← what "capital" contributes

"France":
  Q_France  = [0.2, 0.8, 0.3, 0.6]   ← what "France" is looking for
  K_France  = [0.1, 0.7, 0.2, 0.9]   ← what "France" offers
  V_France  = [0.5, 0.9, 0.1, 0.8]   ← what "France" contributes
```

---

### Step 2: Compute Attention Scores (Dot Products)

How relevant is "France" to "capital"?

**Dot product** = a measure of alignment between two vectors. 
Two vectors are "similar" if their numbers point in the same direction. The dot product measures this similarity — high value means high alignment, near zero means unrelated.

```
Score(capital→France) = Q_capital · K_France
  = (0.6×0.1) + (0.1×0.7) + (0.9×0.2) + (0.2×0.9)
  = 0.06 + 0.07 + 0.18 + 0.18
  = 0.49   ← reasonably high alignment

Score(capital→capital) = Q_capital · K_capital
  = (0.6×0.4) + (0.1×0.3) + (0.9×0.7) + (0.2×0.5)
  = 0.24 + 0.03 + 0.63 + 0.10
  = 1.00   ← self-attention, typically high
```

For a full 6-token sentence, this produces a full attention score matrix:

```
            The  capital  of  France  is   ?
The       [ 0.8,   0.2,  0.1,  0.3,  0.7, 0.4 ]
capital   [ 0.1,   1.0,  0.3,  0.8,  0.5, 0.2 ]
of        [ 0.2,   0.3,  0.9,  0.4,  0.3, 0.1 ]
France    [ 0.3,   0.7,  0.2,  1.0,  0.4, 0.3 ]
is        [ 0.6,   0.3,  0.2,  0.4,  0.9, 0.3 ]
?         [ 0.5,   0.4,  0.1,  0.3,  0.2, 0.8 ]

Each row = one token's scores across the entire sequence
```

---

### Step 3: Scale (Why This Step Exists ? - Preventing Overconfident Scores ) 

Raw dot products become very large in high-dimensional vectors 512 or 4096 dimensions). 
Large scores → after softmax, attention collapses to a single dominant token ( one token gets nearly 100% of attention ) → gradient vanishes during training.

**Fix:** Divide every score by √(dimension of Key vectors): 


```
For dimension = 4:   √4 = 2
Score 1.00 → 1.00 / 2 = 0.50

For dimension = 512: √512 ≈ 22.6
Keeps scores in a stable range regardless of vector size
```

This is the "scaled" in "Scaled Dot-Product Attention." One line, critical purpose: training stability in high dimensions.

---

### Step 4: Softmax — Converting Scores to Attention Weights

Scaled scores for "capital" attending to all tokens: `[0.05, 0.50, 0.15, 0.40, 0.25, 0.10]`

After softmax:

```
Before: [0.05, 0.50, 0.15, 0.40, 0.25, 0.10]
After:  [0.05, 0.30, 0.10, 0.26, 0.16, 0.08]
          The   cap   of   Fr    is    ?
         (sum = 1.00 — these are attention percentages)
```

**What softmax does that raw scores cannot:**

1. Converts any numbers to a probability distribution (sum = 1.0)
2. **Amplifies differences:** a score of 0.50 vs 0.40 becomes a larger gap after softmax — attention focuses rather than spreading evenly

This makes attention a **winner-take-more** competition. The most relevant token receives disproportionately more attention weight. The model focuses — as human attention focuses.

---

### Step 5: Weighted Sum of Values — The Final Output

Combine Value vectors using attention weights as proportions:

```
New representation of "capital" =
  0.05 × V_The    +
  0.30 × V_capital +   ← 30% from self
  0.10 × V_of     +
  0.26 × V_France  +   ← 26% from France
  0.16 × V_is     +
  0.08 × V_?

= one vector blending information from all tokens,
  weighted by their relevance to "capital"
```

**Before attention:** "capital" = a generic embedding for the word
**After attention:** "capital" = a context-specific representation that knows it's being asked about France, in a question, in this particular sentence

This transformation happens for every token simultaneously. That is one full attention head forward pass.

---

## The Compact Formula

Everything above in one line:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) × V
```

Reading left to right in English:

```
QKᵀ          → dot product of every Query against every Key
               (relevance scores for all token pairs)

÷ √d_k       → scale down to prevent large values

softmax(...)  → convert scores to attention percentages

× V           → blend Values proportional to relevance
```

**This is the entire attention mechanism. Everything else in the transformer is engineering around it.**

---

## Multi-Head Attention: Full Precision

### Why Multiple Heads?

One attention head computes one set of Q, K, V projections and one attention pattern — one type of relationship. Language has multiple simultaneous relationship types: syntax, coreference, semantics, position.

"The bank by the river where I deposit my savings"

Syntactic relationships:  "bank" → "deposit" (verb governs object)
Semantic relationships:   "river" → "bank" (disambiguates to riverbank)
Long-range reference:     "I" → "my" (same person)
Positional proximity:     "by" → "river" (adjacent modifiers)

One head cannot optimally capture all of these simultaneously — the weight matrices would be fighting each other.

Solution: Run H heads in parallel, each learning different projection matrices.
```
Head 1: W_Q¹, W_K¹, W_V¹  →  Attention pattern 1 (maybe syntax)
Head 2: W_Q², W_K², W_V²  →  Attention pattern 2 (maybe coreference)
Head 3: W_Q³, W_K³, W_V³  →  Attention pattern 3 (maybe semantics)
...
Head H: W_Qᴴ, W_Kᴴ, W_Vᴴ →  Attention pattern H
```
All heads run simultaneously on the same input (GPU parallelism)

How Outputs Are Combined
Each head produces an output vector for each token. These are:

Concatenated side by side into a longer vector
Projected back to the original embedding dimension through another learned weight matrix WOW_O
WO​

```
Head 1 output for "capital": [0.3, 0.7, 0.2, 0.8]   (dim = 4)
Head 2 output for "capital": [0.5, 0.1, 0.9, 0.4]   (dim = 4)
Head 3 output for "capital": [0.8, 0.6, 0.3, 0.2]   (dim = 4)

Concatenated: [0.3, 0.7, 0.2, 0.8, 0.5, 0.1, 0.9, 0.4, 0.8, 0.6, 0.3, 0.2]  (dim = 12)

× W_O (projection matrix):

Final output for "capital": [0.6, 0.4, 0.7, 0.5]   (back to dim = 4)
```
The projection matrix W_O learns how to best combine information from all heads into a single unified representation.

**Multi-head attention runs H heads in parallel**, each with independent W_Q, W_K, W_V:

```python
# Conceptually:
head_outputs = []
for i in range(num_heads):
    Q_i = token_embeddings @ W_Q[i]
    K_i = token_embeddings @ W_K[i]
    V_i = token_embeddings @ W_V[i]
    head_output = attention(Q_i, K_i, V_i)
    head_outputs.append(head_output)

# Concatenate and project:
multi_head_output = concatenate(head_outputs) @ W_O
```

### What Heads Actually Specialise In

In 2019, researchers at Anthropic and elsewhere used interpretability tools to visualise what attention heads actually learn, found consistent patterns:

```
"Previous token heads"  → attend to the immediately preceding token
"Duplicate token heads" → attend to other positions of the same token
"Subject-verb heads"    → verbs attend to their grammatical subject
"Coreference heads"     → pronouns attend to their antecedents
"Positional heads"      → attend based on relative distance
```
**What this means architecturally:**
Nobody programmed these behaviours. They **emerged** from training on language prediction. The model invented its own internal organisation. This is why mechanistic interpretability research exists — we are reverse-engineering what the network discovered on its own.

---

## Cross-Attention — How Encoders and Decoders Talk

In cross-attention:

Queries (Q) come from the decoder (what the decoder is currently generating)
Keys (K) and Values (V) come from the encoder output (the full encoded input)

```
Translation: "The cat" → "Le chat"

Encoder processes "The cat" → rich representations for each token

Decoder generating "Le":
  Q = Query from decoder (what does "Le" need to understand?)
  K = Keys from encoder output ("The" key, "cat" key)
  V = Values from encoder output
  
  → "Le" attends to "The" → learns it's a definite article
  → Decoder integrates this into generating "Le"

Decoder generating "chat":
  Q = Query from decoder
  → "chat" attends strongly to "cat" → learns the noun to translate
```
Cross-attention is the bridge between understanding the input (encoder) and generating the output (decoder). The decoder never directly sees the input tokens — only their encoded representations, accessed through this attention mechanism.
  
---

## Attention as a Retrieval System: The RAG Connection

**Attention and vector database search are the same operation, differently applied:**

```
ATTENTION (inside transformer):
  Query  → "What does this token need to understand itself?"
  Keys   → "What does each other token contain?"
  Values → "What information does each token provide?"
  Retrieval: soft — weighted combination of ALL values
  Used for: context understanding within a sequence

VECTOR DATABASE SEARCH (RAG):
  Query embedding  → "What is the user asking?"
  Document keys    → "What does each document contain?"
  Document content → "What information does each document provide?"
  Retrieval: hard — return top-k most similar documents
  Used for: finding relevant external knowledge
```

**The mathematics is identical — dot product similarity scoring:**

Compute similarity between query and keys (dot product)
Select most relevant entries (top-k instead of softmax)
Return their values (document content instead of value vectors)

This is why understanding attention makes you better at building RAG systems.

**When you design a RAG pipeline and choose:**

Embedding model (affects quality of K vectors)
Similarity metric (dot product vs cosine — same attention math)
Top-k retrieval (hard selection vs attention's soft selection)

**Engineering and architectural decisions in RAG directly mirror attention design choices:**

| RAG Decision | Attention Equivalent |
|-------------|---------------------|
| Choice of embedding model | Quality of K vectors |
| Similarity metric (cosine vs dot product) | Attention score computation |
| Top-k retrieval | Hard threshold vs softmax's soft weighting |
| Chunk size | Granularity of V (Value) information |

Understanding attention at this level makes you a better RAG engineer — the concepts transfer directly.

---

## Key Takeaways

```
✓ Q, K, V are separate learned projections — not the raw embeddings
  Separation enables independent optimisation of "asking" vs "answering"

✓ Five steps of attention:
  1. Compute Q, K, V via learned projections
  2. Dot product of every Q against every K → relevance scores
  3. Scale by √d_k → prevent large values, maintain training stability
  4. Softmax → convert scores to attention percentages (sum = 1.0)
  5. Weighted sum of V → context-enriched token representation

✓ Multi-head: H heads in parallel, each specialising in different
  relationship types (syntax, coreference, semantics...)

✓ Head specialisation emerges from training — not engineered

✓ Attention = soft retrieval = same mathematics as vector search
  Understanding attention = understanding RAG at the architectural level
```

---

*[← Chapter 5](05-transformers.md) | [Chapter 7: How LLMs Store Facts →](07-how-llms-store-facts.md)*

---
---
