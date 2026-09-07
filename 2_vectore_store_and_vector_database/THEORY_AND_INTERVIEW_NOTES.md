# Module 2: Vector Stores & Vector Databases in RAG

---

## 📖 Table of Contents
1. [Core Concepts: Vector Stores vs. Vector Databases](#-core-concepts-vector-stores-vs-vector-databases)
   - Fundamental Architectural Differences
   - Key Requirements of an Enterprise Vector Database
2. [Approximate Nearest Neighbor (ANN) Algorithms Deep Dive](#-approximate-nearest-neighbor-ann-algorithms-deep-dive)
   - Exact Flat Search ($O(N \cdot d)$)
   - Inverted File Index (IVF & IVFFlat)
   - Hierarchical Navigable Small World (HNSW)
   - Product Quantization (PQ & IVFPQ)
   - Locality Sensitive Hashing (LSH)
3. [Vector Database Landscape & Comparison](#-vector-database-landscape--comparison)
   - ChromaDB, FAISS, Pinecone, DataStax AstraDB, Qdrant, Milvus, Weaviate
4. [Metadata Filtering Architectures](#-metadata-filtering-architectures)
   - Pre-Filtering vs. Post-Filtering vs. Single-Stage Filtered HNSW
5. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
6. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 Core Concepts: Vector Stores vs. Vector Databases

While developers often use the terms interchangeably, there is a fundamental distinction between a **Vector Store / Index Library** and a **Vector Database**:

```
┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│         VECTOR STORE / INDEX LIBRARY          │                VECTOR DATABASE                │
│             (e.g., FAISS, Annoy)              │     (e.g., Pinecone, Milvus, Qdrant, Astra)   │
├───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ • In-memory indexing data structure           │ • Full distributed database engine            │
│ • Ephemeral; requires manual file persistence │ • Persistent storage with replication & WAL   │
│ • No real-time CRUD (rebuild index on insert) │ • Real-time dynamic updates, upserts, deletes │
│ • No native metadata filtering (or basic)     │ • Advanced hybrid metadata filtering & index  │
│ • Single-node execution; no horizontal scale  │ • Horizontal sharding, high availability, SLA │
│ • No RBAC, multi-tenancy, or telemetry        │ • Multi-tenancy, enterprise RBAC, audit logs  │
└───────────────────────────────────────────────┴───────────────────────────────────────────────┘
```

```
                              VECTOR DATABASE INTERNALS
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API / Query Gateway                            │
├──────────────────────────────────────┬──────────────────────────────────────┤
│          Metadata Engine             │            Vector Engine             │
│   - B-Tree / Inverted Indexes        │   - HNSW / IVF / ScaNN Graph         │
│   - Scalar predicates (SQL / JSON)   │   - Distance metric hardware SIMD    │
├──────────────────────────────────────┴──────────────────────────────────────┤
│                         Storage & Replication Engine                        │
│   - Write-Ahead Log (WAL)            - LSM Trees / Columnar Segments        │
│   - Raft Consensus / Paxos           - S3 / Blob Tiered Archival            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Approximate Nearest Neighbor (ANN) Algorithms Deep Dive

Searching through billions of vectors using brute-force exact search is computationally prohibitive ($O(N \cdot d)$). ANN algorithms trade a tiny fraction of recall (e.g., finding the true 1st nearest neighbor 98% of the time) for **100x-1000x faster sub-millisecond retrieval**.

### 1. Exact Flat Search (`IndexFlatL2` / `IndexFlatIP`)
* Computes exact distance from query to every single vector in the dataset.
* **Pros**: 100% recall guarantee.
* **Cons**: Search time grows linearly $O(N)$. Completely unscalable beyond 100k vectors.

### 2. Inverted File Index (`IndexIVFFlat`)
* **Concept**: Uses $k$-means clustering to partition vector space into $N_{\text{list}}$ Voronoi cells. Each cell is represented by a centroid vector.
* **Search Process**:
  1. Compares query vector to $N_{\text{list}}$ centroids.
  2. Identifies the closest $N_{\text{probe}}$ centroids.
  3. Searches only the vectors belonging to those $N_{\text{probe}}$ Voronoi cells.
* **Trade-off**: Larger `nprobe` increases recall at the cost of higher query latency.

```
                         VORONOI PARTITIONING (IVF)
                ┌───────────────────┬───────────────────┐
                │   •    Centroid A │   •               │
                │     •             │         •         │
                │        [Query]    │      Centroid B   │
                │           •       │                   │
                ├───────────────────┼───────────────────┤
                │      •            │     •             │
                │ Centroid C        │ Centroid D   •    │
                │        •          │          •        │
                └───────────────────┴───────────────────┘
```

### 3. Hierarchical Navigable Small World (HNSW)
HNSW is the gold standard for high-performance vector search in production (used by Chroma, Pinecone, Qdrant, Milvus).

* **Intuition**: Combines the concept of **Skip Lists** (multi-layer linked lists) with **Navigable Small World Graphs** (graphs where any two nodes can be connected in a small number of hops).
* **Architecture**:
  * **Top Layers**: Sparse graphs with long-distance links for fast logarithmic jumping across vector space.
  * **Bottom Layer ($Layer_0$)**: Dense graph connecting all vectors with short-distance links for fine-grained local neighborhood exploration.
* **Complexity**: $O(\log N)$ search time.
* **Key Hyperparameters**:
  * $M$: Maximum number of bi-directional edges per node (typical: 16 to 64). Higher $M$ = better recall, more RAM.
  * $efConstruction$: Search depth during index building (typical: 100 to 400).
  * $efSearch$: Search depth during runtime query execution (typical: 32 to 128).

```
                            HNSW GRAPH STRUCTURE
   Layer 2 (Express)        [Node A] ────────────────────────► [Node Z]
                                │                                  │
                                ▼                                  ▼
   Layer 1 (Regional)       [Node A] ──────► [Node M] ───────► [Node Z]
                                │                │                 │
                                ▼                ▼                 ▼
   Layer 0 (All Nodes)      [A]──[B]──[C]──► [M]──[N]──[O]──► [Y]──[Z]
```

### 4. Product Quantization (PQ & `IVFPQ`)
* **Concept**: Lossy compression algorithm for vectors.
* **Mechanism**: Splits a $d$-dimensional vector into $m$ equal sub-vectors. Quantizes each sub-vector to the nearest centroid among $k^*$ learned cluster centroids (represented by an 8-bit byte).
* **Benefit**: Compresses a 1536-dim vector (6144 bytes) down to 64 or 128 bytes (**97% memory reduction**), enabling billions of vectors to fit into RAM.

---

## 📊 Vector Database Landscape & Comparison

| Vector Database | Type | Primary ANN Index | Hosting / Deployment | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **ChromaDB** | Vector Store | HNSW (via hnswlib) | Local / Embedded / Client-Server | Rapid prototyping, local Python development |
| **FAISS** | Library | Flat, IVF, HNSW, PQ | Local (CPU / GPU C++ lib) | High-throughput offline batch indexing, research |
| **Pinecone** | Vector DB | Proprietary HNSW | Fully Managed Serverless Cloud | Production enterprise SaaS, zero infra maintenance |
| **DataStax AstraDB** | Hybrid NoSQL+Vector | Cassandra Vector (HNSW) | Managed Cloud (AWS/GCP/Azure) | Enterprise apps needing both ACID CQL and Vector |
| **Qdrant** | Vector DB | HNSW + Scalar Quant | Open Source / Cloud (Rust) | High performance, rich payload filtering, edge |
| **Milvus / Zilliz** | Distributed Vector DB | Knowhere (HNSW, IVF, SCANN) | Kubernetes Distributed / Cloud | Massive scale (100M to Billions of vectors) |
| **Weaviate** | Vector DB | HNSW | Open Source / Cloud (Go) | Native multi-modal, GraphQL & hybrid search |

---

## 🔍 Metadata Filtering Architectures

In enterprise search, queries rarely search the raw entire corpus without filters (e.g., *"Show me compliance docs where `department == 'HR'` and `year >= 2023`"*).

```
1. PRE-FILTERING                    2. POST-FILTERING                3. SINGLE-STAGE FILTERED HNSW
┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
│ Filter Metadata      │          │ Vector Search Top-K  │          │ Traverse HNSW Graph  │
│ (e.g., 5% match)     │          │ (e.g., K=100)        │          │ Check filter flag    │
└──────────┬───────────┘          └──────────┬───────────┘          │ on every visited node│
           │                                 │                      └──────────┬───────────┘
           ▼                                 ▼                                 │
┌──────────────────────┐          ┌──────────────────────┐                     ▼
│ Search Vectors       │          │ Apply Filter         │             Accurate, High Recall
│ over remaining 5%    │          │ (Risk: 0 matches left│             Sub-millisecond
│ (Slow if unindexed)  │          │  if top-K had wrong  │
└──────────────────────┘          │  metadata!)          │
                                  └──────────────────────┘
```

> **Industry Best Practice**: Modern vector databases (Qdrant, Pinecone, Milvus) implement **Single-Stage Filtered Graph Traversal**, where the graph traversal algorithm evaluates the metadata bitmap filter directly during neighbor expansion, ensuring both 100% filter adherence and high recall.

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content="LangChain enables seamless RAG pipelines.", metadata={"topic": "ai", "year": 2024}),
    Document(page_content="PostgreSQL pgvector allows SQL vector search.", metadata={"topic": "database", "year": 2023}),
    Document(page_content="Chroma is a lightweight embedded vector store.", metadata={"topic": "database", "year": 2024})
]

# 1. ChromaDB: Local Persistent Vector Store
chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_collection"
)
# Search with Metadata Filtering
results = chroma_db.similarity_search_with_score(
    query="What vector database can I embed in Python?",
    k=2,
    filter={"topic": "database"}
)
for doc, score in results:
    print(f"Chroma Match: {doc.page_content} | Score: {score:.4f} | Meta: {doc.metadata}")

# 2. FAISS: High Performance In-Memory Vector Store
faiss_db = FAISS.from_documents(docs, embeddings)
faiss_db.save_local("./faiss_index")  # Serialize to disk

# Reload FAISS index
loaded_faiss = FAISS.load_local(
    folder_path="./faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
faiss_results = loaded_faiss.similarity_search("How to build RAG pipelines?", k=1)
print(f"FAISS Match: {faiss_results[0].page_content}")
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: What is the core difference between a Vector Index (FAISS) and a Vector Database (Pinecone/Milvus)?
**Answer**:
- **Vector Index (FAISS)**: A specialized library that organizes vectors in memory using algorithms like IVF or HNSW for fast nearest-neighbor computation. It lacks database features: no distributed sharding, no write-ahead logging (WAL), no transactional consistency, no multi-tenant isolation, and no live updates (modifying vectors often requires rebuilding the index).
- **Vector Database (Pinecone/Milvus/Qdrant)**: A full database management system wrapping vector indexing engines with horizontal scaling, metadata indexing, live CRUD operations, role-based access control (RBAC), data replication, automatic backups, and high-availability SLAs.

---

### Q2: How does the HNSW algorithm achieve logarithmic $O(\log N)$ search complexity?
**Answer**:
HNSW constructs a hierarchical multi-layer graph inspired by Skip Lists.
- The **top layers** contain sparse graphs with long-range edges between distant vector clusters.
- The search starts at the top layer with an entry point, performing greedy routing to find the local minimum.
- Once a local minimum is reached in layer $l$, the search drops down to the corresponding node in layer $l-1$ and resumes greedy search with tighter connections.
- The final layer ($Layer_0$) explores the dense neighborhood of vectors to return the top-$k$ nearest neighbors.
Because the number of layers scales logarithmically with $N$, search time is $O(\log N)$.

---

### Q3: Why is Post-Filtering dangerous in production RAG systems, and how do modern vector databases solve it?
**Answer**:
- **Post-Filtering Problem**: The system first retrieves the top $k$ nearest neighbors based purely on vector similarity, and then applies metadata filters (e.g., `tenant_id == 'Finance'`). If none of the top $k$ vectors happen to belong to the 'Finance' tenant, the query returns **0 results**, leading to false empty retrievals even when matching documents exist in the database.
- **Modern Solution**: Single-stage filtered HNSW traversal. The database maintains an inverted index / roaring bitmap of metadata. During graph traversal, candidate nodes that fail the filter are simply excluded from the nearest-neighbor priority queue while continuing graph traversal.

---

### Q4: Explain the difference between `IVFFlat` and `IVFPQ` in FAISS. When should you use which?
**Answer**:
- **`IVFFlat`**: Partitions vector space into Voronoi cells using $k$-means centroids. Stores the **full, uncompressed vectors** in each inverted list.
  - *Use Case*: High recall requirements where vector dataset easily fits in server RAM.
- **`IVFPQ`**: Combines IVF partitioning with **Product Quantization (PQ)**. Vector coordinates are compressed into 8-bit centroid indices (quantized codes).
  - *Use Case*: Large-scale datasets (10M to 1B vectors) where RAM is constrained. PQ reduces memory by 90-97% at the expense of a slight drop in recall.

---

### Q5: What are the key hyperparameters of HNSW, and how do they balance recall vs. latency vs. index build time?
**Answer**:
1. **$M$ (8 to 64)**: Max number of outgoing connections per node. Higher $M$ increases recall and graph connectivity for high-dimensional data, but increases index build time and RAM usage.
2. **$efConstruction$ (100 to 400)**: Number of candidate neighbors evaluated during index construction. Higher values produce a higher quality graph at the cost of slower build times.
3. **$efSearch$ (32 to 128)**: Size of the dynamic candidate list during query time. Increasing $efSearch$ increases recall without altering memory footprint, but linearly increases search latency.

---

### Q6: How do you handle real-time vector updates and deletes in an HNSW-based vector database?
**Answer**:
Deleting a node from an HNSW graph is non-trivial because removing a node can sever paths and disconnect sub-graphs.
- **Soft Delete / Tombstoning**: Modern vector databases mark the deleted vector ID in a deletion bitmap. The graph traversal skips tombstoned nodes during search.
- **Background Compaction / Re-linking**: Periodic background garbage collection merges disconnected edges and rebuilds degraded segments without taking the database offline.

---

### Q7: When is an embedded vector store (Chroma/FAISS) suitable vs. a managed cloud vector database (Pinecone/AstraDB/Milvus)?
**Answer**:
- **Embedded (Chroma/FAISS)**: Suitable for single-user desktop apps, prototypes, edge devices, local development, or small static datasets (<100k vectors) where running a dedicated database server adds unnecessary operational overhead.
- **Managed Cloud (Pinecone/AstraDB/Milvus)**: Mandatory for production enterprise applications with multi-tenant traffic, millions of vectors, real-time concurrent write/read workloads, strict uptime SLAs, disaster recovery, and compliance requirements.

---

### Q8: What is the purpose of Namespaces in vector databases like Pinecone and Qdrant?
**Answer**:
Namespaces provide logical partitioning of vectors within a single index.
- **Multi-Tenancy**: Isolate customer data so that queries from Customer A can never leak or scan data from Customer B (`namespace="tenant_123"`).
- **Domain Segmentation**: Separate different document corpuses (e.g., `namespace="hr_docs"`, `namespace="engineering_docs"`) to accelerate search and prevent index bloat.

---

### Q9: What is the "Cold Start" problem during IVF index creation, and why does IVF require training?
**Answer**:
Unlike HNSW which constructs incrementally, IVF requires a representative sample of vectors to **train the $k$-means coarse quantizer** and compute the $N_{\text{list}}$ cluster centroids before vectors can be assigned to Voronoi inverted lists.
If the training sample is non-representative or too small, centroid placement is sub-optimal, causing imbalanced cluster sizes and severe recall degradation.

---

### Q10: How does DataStax AstraDB integrate vector search into Apache Cassandra?
**Answer**:
AstraDB extends Apache Cassandra's distributed, masterless NoSQL architecture by integrating Vector Search (Vector Search plugin using JVector and HNSW). It allows storing vector embeddings alongside traditional relational/NoSQL columns in the same table row, enabling developers to perform unified CQL/JSON queries combining ACID properties, scalar lookups, and vector similarity search without synchronizing external vector databases.
