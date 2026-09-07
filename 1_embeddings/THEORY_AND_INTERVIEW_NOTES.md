# Module 1: Vector Embeddings & Similarity Metrics in RAG

---

## 📖 Table of Contents
1. [Core Concepts & Mathematical Foundations](#-core-concepts--mathematical-foundations)
   - What are Vector Embeddings?
   - From One-Hot & Word2Vec to Contextual Bi-Encoders
   - Dense vs. Sparse Vector Representations
2. [Similarity & Distance Metrics Deep Dive](#-similarity--distance-metrics-deep-dive)
   - Cosine Similarity & Cosine Distance
   - Dot Product (Inner Product)
   - Euclidean Distance ($L_2$ Norm)
   - The Unit-Vector Normalization Theorem
3. [Embedding Models: Proprietary vs. Open Source](#-embedding-models-proprietary-vs-open-source)
   - OpenAI Embeddings (`text-embedding-3-small` / `large`, `ada-002`)
   - Open Source Models (SentenceTransformers, `all-MiniLM-L6-v2`, `bge-large`, `e5`)
   - Matryoshka Representation Learning (MRL)
   - MTEB (Massive Text Embedding Benchmark)
4. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
5. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 Core Concepts & Mathematical Foundations

### What are Vector Embeddings?
An **Embedding** is a mapping from discrete symbolic tokens (words, sentences, documents) to a continuous, dense, high-dimensional vector space ($\mathbb{R}^d$ where $d \in [384, 3072]$). 
The core objective is to position semantically similar pieces of text geometrically close to each other in vector space, allowing mathematical operations to reason about semantics.

```
                      High-Dimensional Semantic Space
                               (Dimension 2)
                                     ▲
                                     │   [Puppy] (0.65, 0.35)
                                     │      •
                                     │         • [Dog] (0.70, 0.30)
                                     │
           [Car] (-0.50, 0.20)       │
              •                      │               • [Kitten] (0.75, 0.65)
                 • [Truck]           │                  • [Cat] (0.80, 0.60)
               (-0.45, 0.15)         │
     ────────────────────────────────┼────────────────────────────────► (Dimension 1)
                                     │
                                     │
```

### Evolution of Embeddings
1. **One-Hot Encoding**: Sparse, orthogonal vectors with dimension $|V|$. No notion of similarity between words ($d(\text{cat}, \text{dog}) = d(\text{cat}, \text{airplane})$).
2. **Word2Vec / GloVe (2013-2014)**: Static dense vectors based on distributional hypothesis (*"a word is characterized by the company it keeps"*). Failed on polysemy (e.g., "bank" of a river vs. financial "bank").
3. **BERT / Cross-Encoders (2018)**: Deep contextual bidirectional representations. Very high quality, but computationally impossible for large-scale search because both query and document must be concatenated and passed through all Transformer layers simultaneously ($O(N \cdot M)$ full self-attention).
4. **SentenceTransformers / Bi-Encoders (SBERT, 2019-Present)**: Uses Siamese network architectures to encode queries and documents independently into fixed-size dense vectors. Allows pre-computing document embeddings offline and executing sub-millisecond retrieval online using Approximate Nearest Neighbor (ANN) search.

```
      BI-ENCODER (Used in Embedding Search)        CROSS-ENCODER (Used in Reranking)
    ┌────────────────┐    ┌────────────────┐            ┌────────────────────────────┐
    │     Query      │    │    Document    │            │     Query + Document       │
    └───────┬────────┘    └────────┬───────┘            └──────────────┬─────────────┘
            ▼                      ▼                                   ▼
    ┌────────────────┐    ┌────────────────┐            ┌────────────────────────────┐
    │ Encoder A (LM) │    │ Encoder B (LM) │            │     Full Transformer       │
    └───────┬────────┘    └────────┬───────┘            │   (Cross-Self-Attention)   │
            ▼                      ▼                    └──────────────┬─────────────┘
       Vector q               Vector d                                 ▼
            └──────────┬───────────┘                            Relevance Score
                       ▼                                         (Single Float)
             Cosine / Dot Product
```

---

## 📐 Similarity & Distance Metrics Deep Dive

Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

### 1. Cosine Similarity & Cosine Distance
Measures the cosine of the angle $\theta$ between two vectors, completely independent of their magnitude:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* **Range**: $[-1, 1]$ (for text embeddings usually $[0, 1]$).
* **Cosine Distance**: $1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$.
* **Best Used For**: Text embeddings where document length variations can arbitrarily scale vector magnitudes.

### 2. Dot Product (Inner Product)
Computes the sum of element-wise products:

$$\text{Dot Product}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$$

* **Range**: $(-\infty, +\infty)$.
* **Pros**: Extremely fast hardware computation on CPUs/GPUs (BLAS matrix multiplications).

### 3. Euclidean Distance ($L_2$ Distance)
Measures the geometric straight-line distance between two points:

$$L_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

* **Range**: $[0, +\infty)$ ($0$ indicates identical vectors).

### 🌟 The Unit-Vector Normalization Theorem
When vectors are normalized to unit length ($L_2$ norm = 1, such that $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):

$$L_2^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 1 + 1 - 2\cos(\theta) = 2(1 - \cos(\theta))$$

$$\text{Dot Product}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

> **Key Takeaway**: If you normalize your vectors before inserting them into a vector database, **Euclidean Distance, Cosine Similarity, and Dot Product yield mathematically identical relative rankings**, but Dot Product is computed significantly faster!

---

## ⚖️ Embedding Models: Proprietary vs. Open Source

| Model | Provider | Dimensions | Max Context | Price / 1M Tokens | MTEB Rank | Key Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`text-embedding-3-small`** | OpenAI | 1536 (or custom) | 8191 tokens | $0.02 | High | Matryoshka support, cost-effective |
| **`text-embedding-3-large`** | OpenAI | 3072 (or custom) | 8191 tokens | $0.13 | Top Tier | Best proprietary multilingual quality |
| **`all-MiniLM-L6-v2`** | HuggingFace | 384 | 256/512 tokens | Free (Local) | Medium | Ultra-fast inference on CPU (80MB model) |
| **`bge-large-en-v1.5`** | BAAI | 1024 | 512 tokens | Free (Local) | Top Tier | Industry standard for open-source RAG |
| **`nomic-embed-text-v1.5`** | Nomic | 768 | 8192 tokens | Free / Hosted | Top Tier | Long context open-source model + MRL |

### Matryoshka Representation Learning (MRL)
Named after Russian nesting dolls, MRL trains embedding models such that the most critical semantic variance is compressed into the first $k$ dimensions ($k < d$).
* For example, OpenAI's `text-embedding-3-large` produces 3072 dimensions by default.
* Using MRL, you can truncate the vector to **512 dimensions**:
  * **Storage & RAM Savings**: Reduced by **83.3%**.
  * **Search Latency**: Drastically accelerated.
  * **Accuracy Loss**: Typically less than **1.5% - 2%**.

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 1. OpenAI Embeddings with Matryoshka Dimension Truncation
openai_embed = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # MRL dimensionality reduction (default 1536)
)

# 2. Open-Source HuggingFace Embeddings (SentenceTransformers)
hf_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # Enforces unit vector normalization
)

# Embedding Query vs. Documents
doc_texts = [
    "LangChain simplifies building LLM applications.",
    "FastAPI is a modern web framework for Python."
]
query_text = "How to develop with large language models?"

# Embed documents (batch operation)
doc_vectors = hf_embed.embed_documents(doc_texts)

# Embed query (single string)
query_vector = hf_embed.embed_query(query_text)

# Compute Cosine Similarity via Dot Product (since normalized)
scores = [np.dot(query_vector, doc_vec) for doc_vec in doc_vectors]
print(f"Similarity Score with Doc 1: {scores[0]:.4f}")
print(f"Similarity Score with Doc 2: {scores[1]:.4f}")
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: Why is Cosine Similarity generally preferred over Euclidean Distance for text embeddings?
**Answer**:
Text chunks can vary drastically in length. In unnormalized vector spaces, longer documents with repetitive words naturally produce vectors with much larger magnitudes ($\|\mathbf{v}\|$), causing Euclidean distance to view them as distant even if they discuss the exact same topic.
Cosine Similarity isolates the **directional angle** between vectors and disregards vector magnitude, making similarity calculations invariant to document length.

---

### Q2: What is the mathematical relationship between Dot Product, Cosine Similarity, and Euclidean Distance for normalized vectors?
**Answer**:
For any two vectors $\mathbf{u}$ and $\mathbf{v}$ normalized to unit length ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):
1. $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos(\theta) = \cos(\theta) = \text{Cosine Similarity}$.
2. $\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2\cos(\theta) = 2(1 - \text{Cosine Similarity})$.
Therefore, optimizing for maximum Dot Product, maximum Cosine Similarity, or minimum Euclidean Distance produces identical ranking orders.

---

### Q3: What is Matryoshka Representation Learning (MRL), and why is it important for vector databases?
**Answer**:
MRL is a training technique where the embedding loss function evaluates nested sub-vectors (e.g., the first 64, 128, 256, 512, 1024 dimensions) simultaneously.
In production, storing 10 million 3072-dimensional vectors in RAM for HNSW graphs requires ~120 GB of memory. By using MRL to truncate vectors to 512 dimensions, RAM consumption drops to ~20 GB (an 83% reduction) with minimal retrieval accuracy degradation.

---

### Q4: Explain the difference between Bi-Encoders and Cross-Encoders in terms of architecture, computational complexity, and retrieval quality.
**Answer**:
- **Bi-Encoder (Siamese Network)**:
  - Encodes Query $Q$ and Document $D$ independently into fixed vectors $\mathbf{q}, \mathbf{d}$.
  - Scoring: Simple vector similarity $\mathbf{q} \cdot \mathbf{d}$.
  - Complexity: $O(1)$ similarity calculation online; documents are embedded offline.
  - Quality: Good semantic capture, but loses fine-grained token-to-token cross-attention.
- **Cross-Encoder**:
  - Concatenates $[Q; D]$ and passes the pair through all Transformer self-attention layers simultaneously.
  - Scoring: Direct probability score output by the classification head.
  - Complexity: $O((|Q|+|D|)^2 \cdot L)$ for every query-document pair (too slow for millions of docs).
  - Quality: State-of-the-art precision. Used in stage-2 **Reranking**.

---

### Q5: What is the embedding model context window limit, and what happens when a text chunk exceeds it?
**Answer**:
Embedding models have fixed context lengths (e.g., 512 tokens for BERT/`all-MiniLM`, 8192 tokens for `text-embedding-3-small` / `nomic-embed-text`).
- If a chunk exceeds the limit, the tokenizer silently truncates the excess tokens unless configured to error out.
- Any text beyond the token limit is completely lost from the embedding representation, causing zero retrieval capability for information in the truncated tail.
- **Solution**: Always enforce chunking token limits with a safety margin (e.g., set chunk size to 400 tokens for a 512-token model).

---

### Q6: What is the "Out-Of-Domain" generalization problem in embedding models?
**Answer**:
Embedding models trained on general web text (Wikipedia, Common Crawl, Reddit) often struggle on specialized proprietary domains (e.g., semiconductor datasheets, rare legal statutes, proprietary medical nomenclatures). In such domains, dense embeddings may place semantically distinct terms near each other due to lack of domain-specific training data.
- **Mitigations**:
  1. Fine-tune embedding models on domain contrastive pairs (triplet loss: Anchor, Positive, Negative).
  2. Implement **Hybrid Search** (Dense + BM25) to anchor retrieval on exact domain keywords.

---

### Q7: Why do we have separate methods `embed_documents` and `embed_query` in LangChain's embedding classes?
**Answer**:
1. **Asymmetric Embedding Models**: Many modern retrieval models (e.g., `E5`, `BGE`, `Instructor`) are asymmetric and require different instruction prefixes for queries vs. documents (e.g., BGE requires `"Represent this sentence for searching relevant passages:"` prefixed to queries, but raw text for documents).
2. **Batching Optimization**: `embed_documents` accepts a `List[str]` and optimizes network/GPU batches, whereas `embed_query` accepts a single `str`.

---

### Q8: What metrics are used on the MTEB (Massive Text Embedding Benchmark) to evaluate retrieval quality?
**Answer**:
1. **NDCG@k (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality, heavily penalizing relevant documents appearing lower in the top-k list.
2. **MRR@k (Mean Reciprocal Rank)**: Measures the reciprocal rank ($\frac{1}{\text{rank}}$) of the first relevant document.
3. **Hit Rate@k / Recall@k**: The percentage of queries where at least one relevant chunk appears in the top $k$ retrieved results.
4. **MAP (Mean Average Precision)**: Average precision across multiple recall levels.

---

### Q9: How do you choose between local open-source embedding models and cloud APIs (OpenAI/Cohere) in production?
**Answer**:
- **Choose Local (HuggingFace / Ollama / vLLM / TEI)**:
  - Strict data privacy & compliance (HIPAA, GDPR, banking air-gapped environments).
  - High query volumes (eliminates per-token API costs; amortized GPU cost is cheaper at scale).
  - Zero external network latency / no dependency on third-party SLA uptime.
- **Choose Cloud APIs (OpenAI / Cohere)**:
  - Zero infrastructure maintenance and automatic scalability.
  - Superior multilingual and long-context performance out of the box.
  - Fast time-to-market for prototypes and low-to-medium volume production apps.

---

### Q10: What is the "Curse of Dimensionality" and how does it impact vector search in RAG?
**Answer**:
As vector dimensionality $d$ increases to thousands of dimensions:
1. **Distance Concentration**: The distance between the nearest neighbor and the furthest neighbor converges to the same value ($(\text{dist}_{\max} - \text{dist}_{\min}) / \text{dist}_{\min} \to 0$), making discrimination between semantically close and distant vectors noisier.
2. **Computational & Memory Overhead**: Exact search requires $O(N \cdot d)$ operations, and graph-based indexes (HNSW) consume gigabytes of expensive RAM.
- **Countermeasures**: Dimensionality reduction (MRL/PCA), Product Quantization (PQ), and graph-based indexing (HNSW).
