# 🧠 Module 1: Vector Embeddings & Similarity Metrics for Production RAG

Welcome to **Module 1: Vector Embeddings & Similarity Metrics**. This module covers the mathematical foundations, machine learning architectures, and production strategies for translating human natural language into dense high-dimensional geometric representations ($\mathbb{R}^d$).

---

## 📑 Table of Contents
1. [Vector Embedding Concepts & Architectural Overview](#-vector-embedding-concepts--architectural-overview)
2. [Folder Structure & Notebook Roadmap](#-folder-structure--notebook-roadmap)
3. [Mathematical Foundations & Distance Metrics](#-mathematical-foundations--distance-metrics)
4. [Embedding Model Comparison Matrix](#-embedding-model-comparison-matrix)
5. [🎯 Comprehensive Technical Interview Questions & Answers](#-comprehensive-technical-interview-questions--answers)
   - [Section A: Mathematical Foundations & Geometric Intuition](#section-a-mathematical-foundations--geometric-intuition)
   - [Section B: Model Architectures, Bi-Encoders & MRL](#section-b-model-architectures-bi-encoders--mrl)
   - [Section C: Asymmetric Embeddings & Instruction Tuning](#section-c-asymmetric-embeddings--instruction-tuning)
   - [Section D: Quantization, Late Interaction & Multi-Vector Search](#section-d-quantization-late-interaction--multi-vector-search)
   - [Section E: Enterprise Scale, Fine-Tuning & Cost Optimization](#section-e-enterprise-scale-fine-tuning--cost-optimization)
6. [Code Implementation Quick Reference](#-code-implementation-quick-reference)

---

## 🏗️ Vector Embedding Concepts & Architectural Overview

Vector embeddings project discrete text tokens into a continuous semantic space where geometric proximity reflects semantic similarity:

```
                      High-Dimensional Semantic Space (2D Projection)
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

### Bi-Encoders (Retrieval) vs. Cross-Encoders (Reranking)

```
        BI-ENCODER (Embedding Search)               CROSS-ENCODER (Stage-2 Reranker)
     ┌────────────────┐   ┌────────────────┐            ┌────────────────────────────┐
     │  Query String  │   │ Document Chunk │            │     Query + Document       │
     └───────┬────────┘   └────────┬───────┘            └──────────────┬─────────────┘
             ▼                     ▼                                   ▼
     ┌────────────────┐   ┌────────────────┐            ┌────────────────────────────┐
     │ Encoder A (LM) │   │ Encoder B (LM) │            │     Full Transformer       │
     └───────┬────────┘   └────────┬───────┘            │   (Cross-Self-Attention)   │
             ▼                     ▼                    └──────────────┬─────────────┘
        Vector q              Vector d                                 ▼
             └──────────┬──────────┘                            Relevance Score
                        ▼                                        (Single Float)
               Cosine / Dot Product
```

---

## 📂 Folder Structure & Notebook Roadmap

| File / Notebook | Modality | Key Concepts & Frameworks |
| :--- | :--- | :--- |
| [`embedding.ipynb`](file:///d:/Udemy/Rag_krish%20naik/1_embeddings/embedding.ipynb) | Theory & Local Models | 2D geometric intuition, NumPy Cosine/Euclidean metrics from scratch, HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`), Normalization |
| [`openaiembeddings.ipynb`](file:///d:/Udemy/Rag_krish%20naik/1_embeddings/openaiembeddings.ipynb) | Cloud APIs & MRL | OpenAI `text-embedding-3-small` / `large`, Matryoshka Representation Learning (MRL), custom dimension truncation (512 vs 1536), batching, token limits |
| [`18-Embeddings.pdf`](file:///d:/Udemy/Rag_krish%20naik/1_embeddings/18-Embeddings.pdf) | Visual Slides | Architectural diagrams, vector space geometry, Transformer representations |
| [`THEORY_AND_INTERVIEW_NOTES.md`](file:///d:/Udemy/Rag_krish%20naik/1_embeddings/THEORY_AND_INTERVIEW_NOTES.md) | Theory Reference | Mathematical proofs, MTEB metrics, and core interview study notes |

---

## 📐 Mathematical Foundations & Distance Metrics

Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

### 1. Cosine Similarity & Cosine Distance
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$
- **Range**: $[-1, 1]$ (Text embeddings are typically in $[0, 1]$).
- **Cosine Distance**: $1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$.
- **Key Property**: Length-invariant (measures angular deviation, ignoring vector magnitude differences caused by document length).

### 2. Dot Product (Inner Product)
$$\text{Dot Product}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$$
- **Range**: $(-\infty, +\infty)$.
- **Key Property**: Heavily accelerated on modern hardware (AVX-512, CUDA tensor cores via Basic Linear Algebra Subprograms / BLAS).

### 3. Euclidean Distance ($L_2$ Distance)
$$L_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
- **Range**: $[0, +\infty)$ ($0$ represents identical vectors).

### 🌟 The Unit-Vector Normalization Theorem
When vectors are normalized to unit length ($L_2$ norm = 1, such that $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):

$$L_2^2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 1 + 1 - 2\cos(\theta) = 2(1 - \cos(\theta))$$
$$\text{Dot Product}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

> [!IMPORTANT]
> **Production Insight**: When embeddings are unit-normalized before indexing, **Cosine Similarity, Dot Product, and Euclidean Distance generate mathematically identical relative rankings**. Vector databases use Dot Product because it requires zero square roots or division operations during similarity calculations.

---

## 📊 Embedding Model Comparison Matrix

| Model | Provider / Lab | Dimensions | Max Context | Price / 1M Tokens | MTEB Rank | Key Superpower |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`text-embedding-3-small`** | OpenAI | 1536 (or 512 via MRL) | 8,191 tokens | $0.02 | High | Matryoshka support, cost-efficient default |
| **`text-embedding-3-large`** | OpenAI | 3072 (or 1024/512 MRL) | 8,191 tokens | $0.13 | Top Tier | Industry standard for proprietary multilingual RAG |
| **`bge-large-en-v1.5`** | BAAI | 1024 | 512 tokens | Free (Local) | Top Tier | Industry standard for self-hosted enterprise search |
| **`all-MiniLM-L6-v2`** | HuggingFace / SBERT | 384 | 256/512 tokens | Free (Local) | Medium | Ultra-lightweight (80 MB model), CPU-friendly |
| **`nomic-embed-text-v1.5`**| Nomic AI | 768 (or 64-512 MRL) | 8,192 tokens | Free (Local) / API | Top Tier | Long-context open-source + Matryoshka learning |
| **`embed-english-v3.0`** | Cohere | 1024 | 512 tokens | $0.10 | Top Tier | Native compression support (Int8 / binary) |
| **`voyage-3`** | Voyage AI | 1024 | 32,000 tokens | $0.12 | Top Tier | State-of-the-art retrieval on domain-specific benchmarks |

---

## 🎯 Comprehensive Technical Interview Questions & Answers

### Section A: Mathematical Foundations & Geometric Intuition

#### Q1: Why is Cosine Similarity preferred over Euclidean Distance for text embeddings when vectors are unnormalized?
**Answer:**
In natural language, the length of a text chunk can vary significantly. In unnormalized embedding spaces:
1. **Magnitude Sensitivity**: A 200-word passage explaining *Transformer Attention* and an 800-word blog post repeating the same concepts with more narrative detail will point in the same semantic direction, but the 800-word document vector will have a significantly larger magnitude ($\|\mathbf{v}\|_2$).
2. **Euclidean Distortion**: Euclidean Distance measures the absolute straight-line geometric distance $\|\mathbf{u} - \mathbf{v}\|_2$. Because of magnitude differences, Euclidean distance will classify these two passages as distant.
3. **Cosine Invariance**: Cosine Similarity divides by the norms ($\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$), isolating the directional angle $\theta$. It measures **conceptual orientation** rather than document length.

---

#### Q2: Prove that Cosine Similarity and Dot Product are identical for $L_2$-normalized vectors, and explain why vector databases leverage this.
**Answer:**
**Proof**:
Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ where $\|\mathbf{u}\| = \sqrt{\sum u_i^2} = 1$ and $\|\mathbf{v}\| = \sqrt{\sum v_i^2} = 1$:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\mathbf{u} \cdot \mathbf{v}}{1 \cdot 1} = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \text{Dot Product}$$

**Vector Database Implication**:
- Computing raw Cosine Similarity requires $2d$ multiplications, $2d$ additions, 2 square roots, and 1 floating-point division per candidate vector.
- Computing Dot Product on pre-normalized vectors requires only $d$ multiply-accumulate (MAC) operations.
- Vector databases (Chroma, FAISS, Milvus, Pinecone) normalize vectors upon ingestion, converting all top-$k$ nearest neighbor searches into blazing-fast BLAS matrix multiplications.

---

#### Q3: What is the "Curse of Dimensionality" in high-dimensional vector spaces, and how does it affect Approximate Nearest Neighbor (ANN) search?
**Answer:**
As dimensionality $d$ climbs into thousands of dimensions ($d \ge 1536$):
1. **Distance Concentration Phenomenon**: The relative distance difference between the closest neighbor and the furthest neighbor approaches zero:
   $$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$
   This causes high-dimensional vectors to appear nearly equidistant from each other, making contrastive discrimination noisier.
2. **Empty Space Explosion**: The volume of the hypercube scales exponentially ($2^d$), causing points to become sparse outliers located on the outer shell/hypersphere surface.
3. **Computational & Indexing Degradation**: Exact $k$-NN scales as $O(N \cdot d)$. Graph-based ANN algorithms (such as HNSW) suffer higher memory overhead to maintain navigational connectivity.
- **Production Solutions**: Matryoshka Representation Learning (MRL), Product Quantization (PQ), and dimensionality reduction before building graph indexes.

---

### Section B: Model Architectures, Bi-Encoders & MRL

#### Q4: What is the architectural difference between a Bi-Encoder and a Cross-Encoder, and why cannot we use Cross-Encoders for initial retrieval over millions of documents?
**Answer:**
- **Bi-Encoder (Dense Retriever)**:
  - Encodes query $q$ and document $d$ independently through a shared Transformer backbone into two fixed-size vectors $\mathbf{q} = f(q)$ and $\mathbf{d} = f(d)$.
  - Similarity is a simple dot product: $\text{score} = \mathbf{q} \cdot \mathbf{d}$.
  - **Complexity**: $O(1)$ per document at query time. Millions of document vectors $\mathbf{d}$ are pre-computed offline and indexed in an HNSW graph.
- **Cross-Encoder (Reranker)**:
  - Concatenates query and document as a single input string: `[CLS] Query [SEP] Document [SEP]` and feeds it through all Transformer self-attention layers.
  - Every query token attends to every document token simultaneously ($O((|Q| + |D|)^2)$ full cross-attention).
  - Produces a single classification score $\in [0, 1]$.
- **Why Cross-Encoders cannot be used for stage-1 retrieval**:
  - To search 1,000,000 documents, a Cross-Encoder must run 1,000,000 full Transformer forward passes for *every single user query*. At 10 ms per pass, a single query would take **almost 3 hours**.
- **Industry Standard 2-Stage Pipeline**:
  - **Stage 1 (Bi-Encoder)**: Fast vector search retrieves Top-50 candidates in < 5 ms.
  - **Stage 2 (Cross-Encoder)**: High-precision reranker scores only those 50 candidates in ~30 ms, returning Top-5 high-precision chunks to the LLM.

---

#### Q5: What is Matryoshka Representation Learning (MRL) and how does it save up to 80% in vector storage and RAM costs?
**Answer:**
**How MRL Works**:
Traditional embedding models optimize loss only at the full dimension $d$ (e.g., 3072). MRL modifies the training loss function to be a weighted sum of multiple nested sub-vector losses:
$$\mathcal{L}_{\text{total}} = \sum_{m \in \{64, 128, 256, 512, 1024, 3072\}} \mathcal{L}_{\text{contrastive}}(\mathbf{v}_{1:m})$$
This forces the model to pack the most critical semantic variance into the earliest dimensions (like Russian nesting dolls).

```
  Full Embedding (3072 dimensions)
 ┌─────────────┬──────────────────────────┬────────────────────────────────────────────────────────┐
 │   Dim 1-512 │       Dim 513-1024       │                     Dim 1025-3072                      │
 └─────────────┴──────────────────────────┴────────────────────────────────────────────────────────┘
  ▲             ▲                          ▲
  │             │                          └─ Full representation: 100% capacity
  │             └─ Medium sub-vector: 99.2% retrieval recall
  └─ Compact sub-vector: 98.5% retrieval recall (83.3% RAM & Disk savings!)
```

**Production Benefits**:
- Storing 10M vectors at 3072 dimensions in 32-bit float requires **~122.8 GB** of RAM.
- Truncating to 512 dimensions via MRL reduces memory requirements to **~20.5 GB** (an **83.3% savings**), with less than **1.5% drop in retrieval accuracy (NDCG@10)**.

---

### Section C: Asymmetric Embeddings & Instruction Tuning

#### Q6: Why do modern embedding models (e.g., BGE, E5, Instructor) require different instruction prefixes for queries versus documents?
**Answer:**
**The Asymmetric Search Problem**:
Queries and documents have fundamentally different linguistic properties:
- **Query**: Short, ambiguous, question-like (*"how to fix postgres connection timeout"*).
- **Document Chunk**: Long, explanatory, declarative passage (*"The following configuration in postgresql.conf manages connection limits..."*).

If an embedding model is trained symmetrically, it tends to match questions with other questions, rather than matching questions with answers.

**Instruction Tuning Solution**:
Models like `BGE` or `E5` prepend task-specific instructions:
- **Query side**: `"Represent this sentence for searching relevant passages: how to fix postgres timeout"`
- **Document side**: Passed as raw text without prefix.
This steers the Transformer's attention mechanism to map question semantics into the corresponding answer subspace in vector space.

> [!TIP]
> In LangChain, this is why embedding classes provide two distinct methods: `embed_query()` (which automatically attaches the instruction prefix) and `embed_documents()` (which processes raw corpus chunks in batch).

---

#### Q7: What happens when an ingested document chunk exceeds the embedding model's context window limit?
**Answer:**
Embedding models have fixed context windows (e.g., 512 tokens for `all-MiniLM` / `bge-large`, 8192 tokens for `text-embedding-3`).
- **Silent Truncation**: By default, model tokenizers truncate any tokens beyond the maximum limit ($N > \text{max\_tokens}$) without throwing an exception.
- **The Information Blindspot**: Any critical facts, numbers, or conclusions in the truncated tail are completely omitted from the vector embedding. The retriever will **never** be able to find the document based on content in that tail.
- **Production Best Practice**:
  - Always align the text splitter's `chunk_size` with the embedding model's context window.
  - Leave a 15-20% safety margin for special tokens and tokenizer variability (e.g., set chunk size to 384 tokens for a 512-token embedding model).

---

### Section D: Quantization, Late Interaction & Multi-Vector Search

#### Q8: Explain the difference between Full Precision (FP32), Int8 Scalar Quantization, and 1-Bit Binary Quantization in vector retrieval.
**Answer:**

| Precision Mode | Storage per Vector (1536-dim) | RAM Reduction | Distance Metric Used | Speed Benchmark | Recall Drop |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Float32 (FP32)** | 6,144 bytes | Baseline (1x) | Dot Product (Floating-point) | Baseline | 0% (Ground truth) |
| **Int8 Quantization** | 1,536 bytes | **4x reduction (75%)** | Int8 Dot Product | 2x-3x faster | < 1% |
| **Binary Quantization** | 192 bytes | **32x reduction (97%)**| **Hamming Distance** (XOR + POPCNT) | **20x-40x faster** | 3% - 5% (Recoverable with over-sampling & rescore) |

**Two-Stage Binary Rescoring Pattern**:
1. Index 100 million chunks as **Binary Vectors** (1-bit per dimension).
2. Query runs lightning-fast hardware **XOR + POPCNT** (Pop Count) Hamming distance search to retrieve Top-200 candidates.
3. Fetch the original FP32 vectors for only those 200 candidates and rescore them via standard Dot Product to return the final Top-10.

---

#### Q9: What is the ColBERT / ColPali Late-Interaction multi-vector representation, and how does it outperform single-vector embeddings?
**Answer:**
- **Single-Vector Bottleneck**: Standard bi-encoders compress an entire 500-word passage into a single 1536-dimensional vector. This creates an information bottleneck where subtle details, rare technical terms, and specific numbers get smoothed out.
- **ColBERT (Contextualized Late Interaction)**:
  - Generates a separate 128-dimensional embedding for **every individual token** in the query and document.
  - **MaxSim Operator**: Computes the sum of maximum cosine similarities between each query token vector and all document token vectors:
    $$\text{Score}(Q, D) = \sum_{i \in Q} \max_{j \in D} (\mathbf{q}_i \cdot \mathbf{d}_j)$$
  - Preserves token-level semantic granularity without needing heavy full cross-encoder attention.
- **ColPali (Vision Extension)**:
  - Applies late interaction directly on the visual patch tokens of document page screenshots using Vision-Language models (PaliGemma), bypassing traditional OCR errors entirely.

---

### Section E: Enterprise Scale, Fine-Tuning & Cost Optimization

#### Q10: How do you evaluate an embedding model on the MTEB (Massive Text Embedding Benchmark)? Which metrics matter most for RAG?
**Answer:**
The **MTEB** benchmark evaluates models across Retrieval, Clustering, Classification, PairClassification, Reranking, STS (Semantic Textual Similarity), and Summarization.

For RAG systems, the most critical retrieval metrics are:
1. **NDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality, penalizing relevant chunks if they appear lower in the Top-10.
2. **MRR@10 (Mean Reciprocal Rank)**: Measures how fast the system returns the first relevant chunk ($\frac{1}{\text{rank}}$).
3. **Recall@k / Hit Rate@k**: The percentage of queries where the true answer chunk is present in the Top-$k$ candidates.

---

#### Q11: When and how should you fine-tune an open-source embedding model on proprietary corporate data?
**Answer:**
**When to Fine-Tune**:
- When retrieval recall on proprietary domain acronyms, internal code repos, legal statutes, or medical terms falls below acceptable SLAs (< 75% Recall@5).
- When generic models confuse domain-specific homonyms.

**How to Fine-Tune (Contrastive Triplet Training)**:
1. **Synthetic Data Generation**: Pass 1,000 corporate documents to an LLM (e.g., GPT-4o) to generate 5-10 realistic user questions per passage $\to$ creates `(Query, Positive_Passage)` pairs.
2. **Hard Negative Mining**: Run dense retrieval using the base model to retrieve top candidates that are semantically similar but do not contain the answer $\to$ creates `(Query, Positive_Passage, Hard_Negative)` triplets.
3. **Multiple Negatives Ranking Loss (MNRL)**: Train the model using SentenceTransformers with InfoNCE / MNRL loss to pull positive passages closer while pushing hard negatives apart in vector space.

---

#### Q12: Compare self-hosting embedding models (Text Embeddings Inference / vLLM) vs. Cloud APIs (OpenAI / Cohere) in terms of Total Cost of Ownership (TCO), latency, and data privacy.
**Answer:**

```
                  ┌────────────────────────────────────────────────────────┐
                  │            EMBEDDING INFRASTRUCTURE DECISION           │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                              Do you have strict air-gapped / HIPAA
                              or zero-external-egress data privacy laws?
                                       /              \
                                     YES               NO
                                     /                  \
                     ┌───────────────────────┐   What is your daily ingestion
                     │ Self-Host Open-Source │   and query volume?
                     │ (TEI on AWS / On-Prem)│         /              \
                     └───────────────────────┘     HIGH (>50M t/day)  LOW (<5M t/day)
                                                   /                    \
                                      ┌───────────────────────┐ ┌──────────────────────┐
                                      │ Self-Hosted vLLM / TEI│ │ Managed Cloud API    │
                                      │ Lower amortized cost  │ │ (OpenAI / Cohere)    │
                                      └───────────────────────┘ └──────────────────────┘
```

- **Cloud APIs (OpenAI / Cohere / Voyage)**:
  - **Pros**: Zero GPU maintenance, instant horizontal scaling, continuous model updates.
  - **Cons**: Per-token recurring cost, network latency (50-200 ms overhead), strict rate limits (TPM/RPM).
- **Self-Hosted (HuggingFace TEI / vLLM on NVIDIA L4 or A10G)**:
  - **Pros**: Sub-5ms ultra-low latency, complete data sovereignty, fixed predictable hardware cost, batch inference speeds up to 10,000 sentences/sec.
  - **Cons**: Infrastructure management, GPU provisioning and orchestration.

---

## 💻 Code Implementation Quick Reference

```python
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 1. OpenAI Embeddings with Matryoshka Dimension Truncation (512 dims)
openai_embed = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # MRL: reduces vector size from 1536 to 512
)

# 2. Local Open-Source Embeddings with Unit Vector Normalization
hf_embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # Enforces unit L2 norm
)

# Corpus Documents & User Query
corpus = [
    "Retrieval-Augmented Generation combines search systems with generative LLMs.",
    "PostgreSQL supports vector similarity search using the pgvector extension.",
    "Deep learning models require high-performance GPU hardware for training."
]
query = "How does RAG integrate search with language models?"

# Batch embed documents and single embed query
doc_embeddings = hf_embed.embed_documents(corpus)
query_embedding = hf_embed.embed_query(query)

# Compute similarity via Dot Product (Equivalent to Cosine Similarity because vectors are normalized)
similarity_scores = [np.dot(query_embedding, doc_vec) for doc_vec in doc_embeddings]

for score, doc in sorted(zip(similarity_scores, corpus), reverse=True):
    print(f"Score: {score:.4f} | Document: {doc}")
```

---

## 🔗 Related Modules & Next Steps
- 👈 **[Module 0: Data Ingestion & Parsing](../0_DataInagestion/README.md)**: Parse and extract raw data from PDFs, DOCX, CSV, JSON, and SQL.
- 👉 **[Module 2: Vector Stores & Vector Databases](../2_vectore_store_and_vector_database/THEORY_AND_INTERVIEW_NOTES.md)**: Store and index high-dimensional embeddings using HNSW, IVF, and PQ.
- 👉 **[Module 5: Hybrid Search Strategies](../5_Hybrid_search_strategies/THEORY_AND_INTERVIEW_NOTES.md)**: Combine Dense Embeddings with Sparse BM25 and Cross-Encoder Rerankers.
