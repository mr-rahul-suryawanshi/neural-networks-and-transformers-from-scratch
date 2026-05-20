# Chapter 7: How LLMs Store Facts
### Where Does GPT "Know" That Paris Is the Capital of France?

> *"There is no database. There is no lookup table. There is only a very large collection of numbers that, when the right input arrives, reconstructs the right answer."*

---

## The Question That Changes Everything

When GPT answers "Paris" to "What is the capital of France?" — where exactly in the network does that knowledge live?

This is not academic curiosity. The answer determines:
- Why LLMs hallucinate and when
- When RAG helps and when it doesn't
- What Fine-Tuning actually changes
- Why Larger models know more
- How to Design AI systems that are reliably accurate

---

## What Is a "Fact" to a Neural Network?

Before asking where facts are stored, we need to understand what a "Fact" even looks like inside a neural network.

A human stores "Paris is the capital of France" as a proposition — a discrete, retrievable piece of information.

A neural network has no propositions. No database. No rows. No keys.

**A Neural Network has only one thing: Numbers** — weight matrices that transform input vectors into output vectors.

The question "Where is the fact stored?" really means:

> **Which weights, when activated by the right input, produce a transformation that results in "Paris" having the highest output probability?**

That's the precise engineering question, And The answer is really surprising. And distributed.

---

## The Residual Stream: The Highway Through the Network

Before locating where facts live, we need one concept: **The Residual Stream.**

Remember Residual Connections, Each transformer layer adds its output to its input.

```
Output = LayerOutput + Input
```
Think of this differently. Across all 96 layers of GPT-3, there is a single **Information Highway** running from input to output —  **The Residual Stream.**

```
TOKEN "capital" FLOWING THROUGH GPT-3:

After input embedding:  [0.2, -0.5, 0.8, ..., 0.3]   (12,288 numbers)
After attention layer 1: [0.3, -0.4, 0.9, ..., 0.5]  (same 12,288 numbers)
After feed-forward 1:   [0.5, -0.2, 1.1, ..., 0.8]   (still 12,288 numbers)
After attention layer 2: [0.7, -0.1, 1.3, ..., 1.0]  (still 12,288 numbers)
                     ... (96 layers later)
Final representation:   [2.1,  0.8, 3.2, ..., 1.9]   (still 12,288 numbers)
```

The token vector flows through the network. Each layer **reads from and writes to it**, adding information and refining the representation.

**Think of it as a document passed through a team of 96 editors.** Each editor reads the current draft and annotates it. By the final layer, the document contains contributions from all editors. The final state is the accumulated knowledge of all layers.

---

## Two Types of Layers, Two Different Jobs

### Attention Layers: Moving Information Between Tokens

```
Job:    Let tokens communicate with each other
        Read the residual streams of ALL tokens
        Write contextual information back into each token's stream

Example:
  Processing "it" in "The trophy didn't fit because it was too big"
  → Attention reads "trophy"'s residual stream
  → Writes trophy-related information into "it"'s residual stream
  → "it" now carries trophy context

What it stores: RELATIONSHIPS and CONTEXT
What it moves: Information across positions in the sequence
```

### Feed-Forward Layers: Knowledge Retrieval ( Looking Up and Writing Facts ) 

```
Job:    Process each token's residual stream independently
        Read the current state of ONE token's stream
        Write additional information back into that stream

Example:
  Processing "capital" after attention has established "France" context
  → Feed-forward reads: "capital" + "France context"
  → Recognises the pattern: "capital of France"
  → Writes: Paris-related information into the residual stream

What it stores: FACTS, ASSOCIATIONS, WORLD KNOWLEDGE
What it retrieves: Knowledge patterns learned from training data
```

**The architectural split in one line:**

> Attention = the communication system. Feed-forward = the knowledge base.

This claim is supported by interpretability research: when researchers surgically remove specific feed-forward neurons, specific factual associations disappear while general language ability remains intact.

**Most of the factual knowledge in an LLM lives in the feed-forward layers.**

---

## Feed-Forward Layers as Key-Value Memory

A 2021 paper (*"Transformer Feed-Forward Layers Are Key-Value Memories"*, Geva et al.) demonstrated:

> Each neuron in the feed-forward layer's intermediate (expanded) layer acts like a key-value memory entry.

```
Feed-forward layer structure:
  Input (12,288 dim)
    ↓ First linear layer: expand (12,288 → 49,152)
    ↓ Activation function (ReLU/GELU)
    ↓ Second linear layer: compress (49,152 → 12,288)
  Output (12,288 dim, added to residual stream)
```
**Why expand then compress?** The expansion creates a high-dimensional intermediate space where pattern matching happens. Each dimension in that expanded space can be thought of as a "detector" — active when a specific input pattern is recognised.

The 49,152-dimensional intermediate space is where pattern matching happens.

```
EACH NEURON IN THE INTERMEDIATE LAYER:

Key   (first layer weights): pattern detector
  "Activate when you see [capital] + [European country context]"

Value (second layer weights): information to write
  "If activated, write Paris-related associations into residual stream"

The first layer's weights  = Keys (pattern detectors)
The second layer's weights = Values (what to write when pattern detected)

```

**Sound familiar? This is the same Q-K-V intuition as attention** — but operating as a static **lookup table** rather than a dynamic context-dependent system.

- Attention QKV: dynamic, depends on current input
- Feed-forward KV: static, fixed in weights from training
  
The feed-forward layer is a learned pattern-matching memory system. When the right input pattern activates the right neurons, the corresponding factual information is retrieved and written into the residual stream.

---

## Tracing "Paris" Through the Network

Let's trace exactly what happens When GPT processes "The capital of France is ___":

**In Attention Layers (1-6, roughly):**

```
Layer 1-3 attention:
"capital" attends to "France" → France context flows into "capital"'s stream

Layer 4-6 attention:
"is" attends to "capital" + "France" → question-answering pattern established

By layer 6, 
Final token position carries: [question] + [capital] + [France context]
```

**In Feed-Forward Layers:**

```
FF Layer 7 reads: [question pattern] + [capital] + [France]

Neurons that activate:
  Neuron 4,821: "capital + European country" pattern detected
  Neuron 12,043: "France political geography" detected
  Neuron 7,392: "geography lookup" detected

These neurons write into residual stream:
  Paris-related associations strengthened
  Other European capitals suppressed
  "Paris" vector representation amplified

Feed-Forward Layer 8:
  Refines further → confidence in "Paris" increases

Feed-Forward Layer 9:
  Additional Paris associations written in
  (coordinates, language, culture — all suppressed for this query)
  (capital city function — amplified)
```

**At Output:**

```
Final residual stream projected onto vocabulary (~50,000 tokens):
  "Paris"    → probability 0.94
  "Lyon"     → probability 0.02
  "Berlin"   → probability 0.01
  ...


```
**The fact wasn't retrieved from a database. It was reconstructed by pattern-matching neurons firing in sequence across multiple feed-forward layers.**

---

## Distributed Storage: The Uncomfortable Truth

Here's what makes LLM knowledge fundamentally different from a database:
> **Facts are not stored in one place. They are distributed across thousands of weights across multiple layers.**

The knowledge that "Paris is the capital of France" is encoded across:
- Weights in attention heads that connect "France" + "capital"
- Neurons in feed-forward layers that activate on France-related patterns
- The embedding vectors of "Paris", "France", "capital"
- Output projection weights that boost "Paris" probability in the right context

**Delete any one:** the fact degrades but may not disappear.
**Damage multiple:** the model starts hallucinating.

```
DATABASE:                              LLM:
┌──────────────────────┐              ┌──────────────────────────────┐
│ France  → Paris      │              │ Fact is smeared across       │
│ Germany → Berlin     │              │ billions of weights,         │
│ Italy   → Rome       │              │ reconstructed probabilistically│
└──────────────────────┘              │ at inference time            │
                                      └──────────────────────────────┘
Update a fact: one SQL row            Update a fact: retrain the model
Query: exact match                    Query: probabilistic reconstruction
Guarantee: certain                    Guarantee: none
```

**This is the root cause of hallucination. The model doesn't retrieve facts — it reconstructs them. Sometimes the reconstruction goes wrong.**

---

## The Superposition Hypothesis: More Facts Than Neurons

One of the most fascinating findings from interpretability research (Anthropic, 2022):
**The problem:** GPT-3 has roughly 50,000 neurons per feed-forward layer. But it appears to know millions of facts. How?

**The answer — Superposition** (Anthropic, 2022):

> Neurons don't store one feature each. Multiple features are encoded in overlapping combinations across neurons, like signals superimposed on the same channel.

```
Idealised (one concept per neuron):
  Neuron 1 → "France facts"
  Neuron 2 → "Germany facts"
  Neuron 3 → "capital cities"

Real (superposition):
  Neuron 1 → 0.7 × "France" + 0.3 × "Roman history" + 0.1 × "wine"
  Neuron 2 → 0.5 × "Germany" + 0.4 × "France" + 0.2 × "WWI"
  Neuron 3 → 0.8 × "capitals" + 0.3 × "Germany" + 0.4 × "population"
```

The network stores **far more information than it has neurons** through overlapping combinations — a form of lossy data compression that is extraordinarily efficient.

The engineering consequence:
**When two superimposed patterns interfere during reconstruction:**
The model produces a blend of both, or flips between them unpredictably. This is hallucination at the mechanical level.

---

## Why LLMs Hallucinate: The Engineering Explanation

Armed with everything above, we can now give a precise engineering explanation of hallucination:

### Cause 1: Incomplete Pattern Match

```

Input: "Who won the FIFA World Cup in 2027?"

Model: No strong learned pattern exists for "2027 World Cup winner"
No neurons strongly activate for "FIFA World Cup + 2027" (the event has not happened yet)

But: The pattern [FIFA World Cup  winner] + [historically strong football countries] activates likely continuations like Brazil, Germany, Argentina

Result: Model outputs "Brazil won the 2027 FIFA World Cup" with high confidence, incorrectly

Mechanism: partial pattern match producing confident but wrong output
```

Hallucination happens when the model has enough patterns to sound correct, but not enough grounded knowledge to actually be correct.

### Cause 2: Superposition Interference

```
Input: "Who invented the telephone?"

Model: Strong pattern exists linking: telephone → Alexander Graham Bell
But: Another overlapping pattern also activates: famous inventor → electricity → Thomas Edison

During generation: Both inventor-related patterns partially activate together

Result: Model may generate: "Thomas Edison invented the telephone"
```

### Cause 3: Training Frequency Imbalance

```
"What is the capital of France?"
  → Seen millions of times in training
  → Strong, well-reinforced pattern
  → Rarely hallucinates

"What was the third prime minister of Bhutan?"
  → Rare in training data
  → Weak, poorly reinforced pattern
  → High hallucination probability
```

**Engineering rule:**

> Hallucination probability is inversely proportional to how frequently and consistently a fact appeared in training data.


Common, repeatedly-confirmed facts → reliable. 

Rare, inconsistently-sourced facts → hallucination-prone.

---

## RAG: The Architectural Solution

**RAG exists because of how LLMs store facts.**

```
LLM fact storage problems:
  - Facts distributed across weights — can't update without retraining
  - Probabilistic reconstruction — hallucination on rare facts
  - Knowledge cutoff — weights frozen after training
  - No source attribution — can't point to where fact came from
  - Superposition interference — facts blur together

RAG solves these:
  - Facts in a real database — can update without touching the model
  - Exact retrieval — document found or not found, no reconstruction
  - Real-time knowledge — retrieve from current documents
  - Source attribution — return the source document
  - No interference — retrieved context is explicit, not reconstructed
```

### BUT — RAG Is Not Free

```
RAG introduces its own failure modes:
  - Retrieval failure — wrong documents retrieved
  - Context window limits — can't retrieve everything
  - LLM still generates — hallucination can occur ON retrieved context
  - Latency — retrieval adds round-trip time
  - Cost — vector DB + embedding model + LLM = 3 systems
```  
### The Architectural Decision — When to Use RAG vs When Not To

```
USE RAG when:
  → Facts change over time (news, prices, policies)
  → Facts are rare or niche and domain-specific (internal company documents)
  → Facts must need source attribution (compliance, legal, medical)
  → Facts are long and specific (contracts, technical manuals)
  → Hallucination on this fact is unacceptable high-risk (medical, legal)
  → Knowledge must be current (training cutoff is a constraint)

DON'T USE RAG when:
  → The model reliably knows the fact (common world knowledge)
  → Latency is critical (RAG adds 100-500ms retrieval overhead)
  → The "facts" are reasoning patterns, not retrievable data
  → Cost must be minimised (3 systems vs 1)
```

This decision framework is AI engineering. Most engineers reach for RAG by default. Knowing when NOT to use it — and being able to justify that decision — is what distinguishes architects from implementers.

---

## Fine-Tuning vs RAG: The Real Trade-off

One more critical implication of how facts are stored: Fine-tuning = continuing to train the model on new data, adjusting the weights.

Since facts live primarily in feed-forward weights, fine-tuning on new data literally rewrites those key-value memory entries.
```
Before fine-tuning:
  Feed-forward neurons: trained pattern "capital of France → Paris"

After fine-tuning on company data:
  Some neurons adjusted: new patterns for company-specific knowledge
  But: other patterns degraded — the adjustment isn't surgical
```
This is called **Catastrophic Forgetting** — teaching a model new facts overwrites some old facts in the distributed weight structure.

```
Fine-tuning vs RAG — the real tradeoff:

FINE-TUNING:
  → No retrieval step — facts baked in, fast inference
  → Model "knows" the domain deeply — better reasoning
  → Understands domain-specific terminology and style
  → Expensive — requires GPU training
  → Catastrophic forgetting — adjusting weights for new facts, can overwrite old facts in the distributed representation
  → Can't update without retraining
  → Hard to audit — where did this fact come from?

RAG:
  → Update knowledge without touching model weights
  → Auditable — source document traceable
  → No forgetting — model unchanged
  → Retrieval can fail or retrieve wrong documents
  → Adds latency and infrastructure cost
  → LLM Model reasoning over retrieved content can still hallucinate

MATURE ANSWER: use both.
  Fine-tune for: domain reasoning style, terminology, task format
  RAG for: specific facts, current information, auditable claims
```

---

## The Geometry of Knowledge: Embeddings

Token embeddings (the vectors representing each word) are not random. They encode semantic relationships as geometric structure.
Part of "Paris is the capital of France" is encoded in the geometric relationships between embedding vectors:

```
Famous relationship (inherited from Word2Vec, present in transformers):

  king - man + woman ≈ queen

Vector arithmetic on embeddings produces meaningful semantic results.

Geographic pattern:
  Paris - France + Germany ≈ Berlin
  Paris - France + Japan   ≈ Tokyo

The model doesn't store individual facts — it stores patterns of relationships that generalise across many facts simultaneously.
```
**Engineering implication for RAG:** 
When you design a RAG embedding pipeline and choose an embedding model, you're choosing how faithfully this semantic geometry is captured. Good embeddings preserve these relationships — which is why semantic search finds "capital of France" documents when you query "Paris government seat."

---

## Key Takeaways

```
Storage mechanism:
✓ No database — no rows, no keys, no retrieval guarantee
  Facts are distributed across billions of weights and reconstructed probabilistically at inference time

Where facts live:
✓ Attention layers: move information between tokens (relationships and context)
  Feed-forward layers: store and retrieve factual knowledge (key-value memory interpretation)
  Embeddings         → semantic geometry (word relationships)

The residual stream:
  Highway through the network — each layer reads and writes
  Attention writes context, feed-forward writes knowledge
  Final state = accumulated contribution of all layers

Why hallucination happens:
  → Incomplete patterns → model confabulates from partial match
  → Superposition interference → more facts than neurons through overlapping encoding, Interference between superimposed patterns = hallucination
  → Low training frequency

RAG: architectural solution to LLM knowledge limits:
  Use RAG for: changing facts, rare facts, auditable facts
  Skip RAG for: stable common knowledge, latency-critical paths

Fine-tuning vs RAG:
  Fine-tune for: style, reasoning, domain depth
  RAG for: current facts, specific documents, auditability
  Mature answer: both, for different purposes
```

---

*[← Chapter 6](06-attention-step-by-step.md) | [Chapter 8: AI Images and Video →](08-ai-images-and-video.md)*
