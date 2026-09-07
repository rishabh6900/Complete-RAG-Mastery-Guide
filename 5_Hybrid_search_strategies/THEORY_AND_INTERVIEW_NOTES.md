# Module 5: Hybrid Search Strategies, Reranking & MMR in RAG

---

## 📖 Table of Contents
1. [Core Concepts: Dense vs. Sparse Retrieval](#-core-concepts-dense-vs-sparse-retrieval)
   - Why Pure Semantic (Dense) Search Fails
   - Why Pure Keyword (Sparse / BM25) Search Fails
   - The Hybrid Search Paradigm
2. [Hybrid Retrieval & Fusion Algorithms](#-hybrid-retrieval--fusion-algorithms)
   - BM25 Mathematical Formulation
   - Reciprocal Rank Fusion (RRF) Deep Dive
   - Weighted Score Convex Combination
3. [Two-Stage Retrieval & Re-ranking](#-two-stage-retrieval--re-ranking)
   - Bi-Encoder vs. Cross-Encoder Mechanics
   - Cross-Encoder Re-rankers (Cohere, BGE-Reranker, MS-MARCO)
   - Latency vs. Accuracy Trade-Off Analysis
4. [Maximal Marginal Relevance (MMR) for Diversity](#-maximal-marginal-relevance-mmr-for-diversity)
   - The Context Redundancy & Clutter Problem
   - Mathematical Formulation of MMR
   - Parameter Tuning (`fetch_k`, `k`, `lambda_mult`)
5. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
6. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 Core Concepts: Dense vs. Sparse Retrieval

Modern state-of-the-art enterprise RAG systems do not rely solely on dense vector embeddings. They implement **Hybrid Retrieval** to achieve optimal recall and precision.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DENSE VS. SPARSE RETRIEVAL                         │
├──────────────────────────────────────┬──────────────────────────────────────┤
│      DENSE RETRIEVAL (Semantic)      │       SPARSE RETRIEVAL (Lexical)     │
│   (e.g., SentenceTransformers, OpenAI)│             (e.g., BM25, TF-IDF)     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Bi-Encoder Transformer models      │ • Inverted Index on exact term freq  │
│ • Captures semantic intent & concept │ • Captures exact keyword matches     │
│ • Handles synonyms & rephrasings     │ • Handles rare IDs, SKUs, acronyms   │
│ • Tolerant to typos and fuzzy phrasing│ • Strong on legal/medical statutory  │
│ ❌ Fails on exact serial numbers/code │ ❌ Fails on paraphrases/synonyms    │
│ ❌ High compute & vector DB overhead │ ❌ Zero contextual understanding     │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

```
                                 HYBRID RETRIEVAL FLOW
                                ┌──────────────────────┐
                                │      User Query      │
                                └──────────┬───────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌──────────────────────┐                      ┌──────────────────────┐
        │   Dense Retriever    │                      │   Sparse Retriever   │
        │ (FAISS / Pinecone)   │                      │        (BM25)        │
        └──────────┬───────────┘                      └──────────┬───────────┘
                   │                                             │
                   │ Top-K Candidates                            │ Top-K Candidates
                   └──────────────────────┬──────────────────────┘
                                          ▼
                             ┌──────────────────────────┐
                             │  Fusion: RRF / Weighted  │
                             └────────────┬─────────────┘
                                          ▼
                             ┌──────────────────────────┐
                             │ Stage 2: Cross-Encoder   │
                             │        Re-Ranking        │
                             └────────────┬─────────────┘
                                          ▼
                             ┌──────────────────────────┐
                             │   Top-N Relevant &       │
                             │   Diverse Chunks to LLM  │
                             └──────────────────────────┘
```

---

## 🔀 Hybrid Retrieval & Fusion Algorithms

### 1. BM25 (Best Matching 25)
BM25 ranks documents based on the matching query terms, penalizing overly frequent terms across the corpus (IDF) and normalizing for document length:

$$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

* $f(q_i, D)$: Frequency of query term $q_i$ in document $D$.
* $|D| / \text{avgdl}$: Ratio of document length to average document length in the corpus.
* $k_1$ (typically $1.2 - 2.0$): Term frequency saturation parameter.
* $b$ (typically $0.75$): Document length normalization penalty.

### 2. Reciprocal Rank Fusion (RRF)
When combining results from Dense (similarity scores $\in [0, 1]$) and Sparse (BM25 scores $\in [0, \infty)$), **raw score normalization is fragile and prone to calibration drift**.
**RRF** combines search results based solely on the **rank positions** of items across retrievers:

$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$

* $M$: Set of retrievers (Dense + Sparse).
* $\text{rank}_m(d)$: Rank of document $d$ in retriever $m$ (1-indexed: $1, 2, 3\dots$).
* $k$: Smoothing constant (industry standard: $k = 60$).

#### Why $k=60$ in RRF?
A smoothing constant of $k = 60$ prevents top-ranked items from single outlier retrievers from overwhelmingly dominating the fused score, while giving a massive boost to documents appearing near the top of **both** lists.

---

## 🎯 Two-Stage Retrieval & Re-ranking

A standard single-stage bi-encoder retriever achieves ~75-85% recall. Adding a **Stage-2 Cross-Encoder Re-ranker** elevates retrieval precision to **95%+**.

```
                           THE TWO-STAGE RETRIEVAL FUNNEL
    Corpus: 1,000,000 Chunks
               │
               ▼  Stage 1: Fast Candidate Generation (Bi-Encoder + BM25)
      Top 50-100 Chunks   (Latency: ~10-20 ms)
               │
               ▼  Stage 2: High-Precision Reranker (Cross-Encoder / Cohere)
        Top 5 Chunks      (Latency: ~50-100 ms)
               │
               ▼
           LLM Context
```

### Why Cross-Encoders are Superior at Scoring
* **Bi-Encoders**: Encode Query and Document in isolation. They compress the entire semantic essence of a 500-word document into a single static point in vector space, losing fine-grained cross-token attention.
* **Cross-Encoders**: Jointly attend across all tokens of both the Query and the Document simultaneously using full self-attention layers ($O((|Q| + |D|)^2)$), capturing subtle negation, numerical relationships, and relational logic that vector distance metrics miss.

---

## 🌈 Maximal Marginal Relevance (MMR) for Diversity

### The Redundancy Problem
In naive similarity search, if 5 chunks in the database are near-identical copies or slight variations of the same paragraph, all top-5 retrieved slots will be filled with redundant text. This wastes precious LLM context tokens and crowds out other critical pieces of information.

### Mathematical Formulation of MMR
MMR iteratively selects the next document that maximizes query relevance while penalizing similarity to already selected documents:

$$\text{MMR} = \operatorname{argmax}_{d_i \in R \setminus S} \left[ \lambda \cdot \operatorname{Sim}_1(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in S} \operatorname{Sim}_2(d_i, d_j) \right]$$

* $R$: Candidate set of documents retrieved by vector search (size `fetch_k`).
* $S$: Set of already selected documents (size reaches `k`).
* $\operatorname{Sim}_1(d_i, q)$: Similarity between candidate document $d_i$ and user query $q$.
* $\operatorname{Sim}_2(d_i, d_j)$: Similarity between candidate document $d_i$ and already selected document $d_j$.
* $\lambda \in [0, 1]$ (`lambda_mult`):
  * $\lambda = 1.0 \implies$ Pure Relevance (Identical to standard similarity search).
  * $\lambda = 0.0 \implies$ Pure Diversity (Picks maximally dissimilar chunks).
  * $\lambda = 0.5 - 0.7 \implies$ **Optimal Balance** for Production RAG.

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.documents import Document

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content="LangChain is a framework for developing applications powered by LLMs.", metadata={"id": 1}),
    Document(page_content="LangChain provides modular components like chains, agents, and vectorstores.", metadata={"id": 2}),
    Document(page_content="Pinecone is a cloud-native vector database designed for fast similarity search.", metadata={"id": 3}),
    Document(page_content="BM25 is a ranking function used by search engines to estimate relevance of documents.", metadata={"id": 4}),
    Document(page_content="Error code 0x80070005 indicates Access Denied on Windows OS.", metadata={"id": 5})
]

# 1. Setup Dense Retriever (FAISS)
faiss_db = FAISS.from_documents(documents, embeddings)
dense_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

# 2. Setup Sparse Retriever (BM25)
sparse_retriever = BM25Retriever.from_documents(documents)
sparse_retriever.k = 3

# 3. Hybrid Ensemble Retriever (RRF / Weighted Score Fusion)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4]  # 60% Dense + 40% Sparse
)
hybrid_results = hybrid_retriever.invoke("What is error code 0x80070005?")
print(f"Hybrid Top Match: {hybrid_results[0].page_content}")

# 4. MMR (Maximal Marginal Relevance) Search
mmr_retriever = faiss_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,              # Final documents returned
        "fetch_k": 5,        # Initial candidate pool evaluated
        "lambda_mult": 0.6   # Diversity factor
    }
)
mmr_results = mmr_retriever.invoke("How does LangChain work?")
print("\nMMR Results (Relevance + Diversity):")
for doc in mmr_results:
    print(f"- {doc.page_content}")
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: Why is Hybrid Search (Dense + Sparse) considered mandatory for enterprise-grade RAG systems?
**Answer**:
Dense retrieval relies on bi-encoder embeddings trained on semantic proximity. It excels at paraphrases and conceptual questions, but systematically fails on:
1. **Exact keywords, part numbers, and error codes** (e.g., `"Error 0x80040154"` or product SKU `"Sony WH-1000XM5"`).
2. **Domain-specific acronyms** not well-represented in the embedding model's pre-training vocabulary.
3. **Short queries** where semantic context is sparse.
Sparse retrieval (BM25) anchors retrieval with exact inverted index matching, while Dense retrieval captures semantic synonyms. Combining them via Hybrid Search provides high recall across all query types.

---

### Q2: What is Reciprocal Rank Fusion (RRF), and why is it preferred over raw score weighted averaging?
**Answer**:
- **Score Averaging Issue**: Dense cosine similarity scores are bounded in $[0, 1]$, while BM25 scores are unbounded $[0, \infty)$ and vary wildly depending on corpus size and term frequency. Normalizing BM25 scores linearly ($x / x_{\max}$) is unstable across dynamic queries.
- **RRF Advantage**: RRF operates strictly on the **ordinal ranks** of results ($\frac{1}{k + \text{rank}}$). It is scale-invariant, requires zero calibration across heterogeneous search backends, and naturally rewards documents that perform consistently well across multiple search strategies.

---

### Q3: Explain the architectural trade-offs between Bi-Encoder retrieval and Cross-Encoder re-ranking.
**Answer**:
- **Bi-Encoder (Stage 1 Retrieval)**:
  - Embeds query and documents separately.
  - Can pre-compute billions of document vectors offline.
  - Sub-millisecond ANN search ($O(1)$ lookup via HNSW).
  - Trades off token-level cross-attention fidelity for extreme scale and speed.
- **Cross-Encoder (Stage 2 Reranking)**:
  - Takes query and document pair into Transformer layers together.
  - Fully attends across all query tokens and document tokens ($O(L^2)$ attention).
  - High computational latency (~50-150ms for 50 docs).
  - Provides state-of-the-art ranking accuracy.

---

### Q4: How does Maximal Marginal Relevance (MMR) mathematically prevent redundancy in retrieved chunks?
**Answer**:
MMR uses an iterative greedy selection process:
$$\text{MMR} = \operatorname{argmax}_{d_i \in R \setminus S} \left[ \lambda \operatorname{Sim}_1(d_i, q) - (1-\lambda) \max_{d_j \in S} \operatorname{Sim}_2(d_i, d_j) \right]$$
1. It first selects the single candidate document $d_1$ with the highest similarity to query $q$.
2. For all subsequent candidates $d_i$, it computes their similarity to query $q$ ($\operatorname{Sim}_1$), and subtracts their maximum similarity to any already chosen document in $S$ ($\max \operatorname{Sim}_2$).
3. If candidate $d_2$ is identical to $d_1$, $\operatorname{Sim}_2(d_2, d_1) \approx 1.0$, which drastically reduces its composite score and eliminates it from selection.

---

### Q5: What are the primary hyperparameters for MMR, and how do you tune them?
**Answer**:
1. **`fetch_k` (Candidate Pool Size)**: Number of initial documents fetched using fast vector search (e.g., 20 to 50).
2. **`k` (Final Chunks Returned)**: Number of diverse top chunks selected to pass to the LLM (e.g., 3 to 5).
3. **`lambda_mult` ($\lambda$)**: Diversity weight.
   - Set to `0.7 - 0.8` when the user query is specific and needs mostly high relevance.
   - Set to `0.4 - 0.6` when the query is exploratory or comparative (e.g., *"Compare features of Product A and Product B"*).

---

### Q6: What is Contextual Compression in LangChain, and how does it save LLM cost and improve latency?
**Answer**:
`ContextualCompressionRetriever` wraps a base retriever and applies a post-retrieval processing filter (such as an LLM extractor or small cross-encoder):
1. The base retriever fetches large chunks.
2. The compressor strips out irrelevant sentences within those chunks, passing only the exact 2-3 sentences pertinent to the query.
3. This reduces prompt token count by 50-80%, reducing LLM cost, decreasing time-to-first-token (TTFT), and preventing context clutter.

---

### Q7: In a high-traffic production system with a 200ms latency budget, how would you design the retrieval pipeline?
**Answer**:
1. **Query Pre-processing (~10ms)**: Query cleaning, embedding generation via quantized local bi-encoder (`bge-small` on ONNX/TensorRT).
2. **Parallel Hybrid Search (~20-30ms)**: Concurrently query HNSW vector index (FAISS/Qdrant) and BM25 index (Elasticsearch/Tantivy), retrieving top-30 candidates each.
3. **RRF Fusion (~2ms)**: Combine candidate lists to top-30 items.
4. **Fast Cross-Encoder Reranking (~40-60ms)**: Run a lightweight, quantized Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2` with ONNX Runtime on GPU/CPU) over the 30 candidates to output top-5.
5. **LLM Generation (~100ms)**: Stream response from LLM.
Total retrieval latency: ~80ms, comfortably within budget.

---

### Q8: What is the "Zero-Hit Problem" in Sparse Search and how does Hybrid Search resolve it?
**Answer**:
The Zero-Hit Problem occurs when a user query contains valid synonyms or natural phrasing (e.g., *"How to stop car engine vibration?"*) but the document corpus uses formal technical terminology (e.g., *"Mitigating internal combustion powertrain oscillation"*).
Because there is 0% vocabulary overlap, BM25 returns zero results.
Hybrid search resolves this because the Dense retriever recognizes semantic proximity and returns relevant candidate chunks despite the absence of exact keyword matches.

---

### Q9: How do you tune the weights in `EnsembleRetriever(weights=[w_dense, w_sparse])`?
**Answer**:
Run an offline evaluation grid search across test queries with labeled ground truth:
1. Vary $w_{\text{dense}} \in [0.0, 1.0]$ in steps of 0.1 with $w_{\text{sparse}} = 1.0 - w_{\text{dense}}$.
2. Calculate **Recall@k** and **NDCG@k** for each configuration.
3. Typical findings:
   - For general FAQ / customer support: `[0.7 Dense, 0.3 Sparse]`.
   - For technical documentation / API docs / legal code: `[0.4 Dense, 0.6 Sparse]`.

---

### Q10: What is Query Rewriting / Query Expansion and how does it complement Hybrid Search?
**Answer**:
Raw user queries are often ambiguous, vague, or conversation-dependent (e.g., *"What about its warranty?"*).
- **Query Expansion**: Uses an LLM to generate multiple alternate phrasings, sub-queries, or hypothetical document embeddings (HyDE).
- **Multi-Query Retriever**: Executes hybrid search across all generated variations and aggregates unique candidate chunks via RRF before re-ranking, boosting retrieval recall for difficult or underspecified queries.
