# Module 4: Advanced Chunking & Preprocessing Techniques in RAG

---

## 📖 Table of Contents
1. [The Chunking Dilemma in RAG](#-the-chunking-dilemma-in-rag)
   - Why Chunking is the Most Critical Lever in RAG
   - Small Chunks vs. Large Chunks Trade-Off Analysis
2. [Traditional & Structural Chunking Techniques](#-traditional--structural-chunking-techniques)
   - Fixed-Size Token / Character Splitters
   - Recursive Character Text Splitter (`RecursiveCharacterTextSplitter`)
   - Document-Structure Splitters (Markdown, HTML, Code)
3. [Semantic Chunking Deep Dive](#-semantic-chunking-deep-dive)
   - Mathematical Intuition & Algorithm Workflow
   - Cosine Distance Spike Detection
   - Breakpoint Threshold Types (Percentile, Standard Deviation, Interquartile)
4. [Advanced Retrieval & Context Augmentation Strategies](#-advanced-retrieval--context-augmentation-strategies)
   - Parent Document Retrieval (Small-to-Big Retrieval)
   - Sentence Window Retrieval
   - Document Summary Indexing
5. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
6. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 The Chunking Dilemma in RAG

In a RAG pipeline, **Chunking** is the process of breaking long documents into smaller, coherent text passages before embedding and storing them.

```
                          THE CHUNKING TRADEOFF SPECTRUM
   ┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
   │         SMALL CHUNKS (50-200 Tokens)   │  │        LARGE CHUNKS (800-2000 Tokens)  │
   ├────────────────────────────────────────┤  ├────────────────────────────────────────┤
   │ ➕ High semantic retrieval precision   │  │ ➕ Rich surrounding context for LLM    │
   │ ➕ Fine-grained pinpoint answers       │  │ ➕ Less risk of context fragmentation   │
   │ ➖ Loses global contextual coherence   │  │ ➖ Embedding becomes "diluted" / noisy  │
   │ ➖ Requires more chunks to answer query│  │ ➖ Low cosine match for specific facts  │
   └────────────────────────────────────────┘  └────────────────────────────────────────┘
```

> **The Fundamental Dilemma**:
> - **Embeddings** prefer **small, dense, focused chunks** for accurate similarity search.
> - **LLMs** prefer **large, rich, contextual documents** for comprehensive and accurate reasoning without hallucinations.

---

## ✂️ Traditional & Structural Chunking Techniques

### 1. Fixed-Size Character / Token Splitting
* **Mechanism**: Slices text strictly every $N$ characters/tokens (e.g., every 500 characters).
* **Flaw**: Frequently cuts sentences mid-word or mid-thought, severing nouns from verbs and corrupting semantic meaning.

### 2. Recursive Character Text Splitter (`RecursiveCharacterTextSplitter`)
* **Mechanism**: The industry default splitter. It attempts to split text using a hierarchical list of separators in descending order of semantic granularity:
  1. `"\n\n"` (Double newline: paragraph boundaries)
  2. `"\n"` (Single newline: sentence / list boundaries)
  3. `" "` (Whitespace: word boundaries)
  4. `""` (Individual characters: emergency fallback)
* **Parameters**:
  * `chunk_size`: Target maximum size of each chunk.
  * `chunk_overlap`: Number of characters/tokens shared between consecutive chunks to prevent information loss across boundaries.

```
                             RECURSIVE CHUNKING FLOW
       [ Paragraph 1 ] \n\n [ Paragraph 2 ] \n\n [ Paragraph 3 ]
              │                     │                     │
              ▼                     ▼                     ▼
     Fits in chunk_size?   Fits in chunk_size?   Exceeds chunk_size?
          ( YES )                ( YES )               ( NO )
             │                      │                     │
             ▼                      ▼                     ▼
       Keep Paragraph         Keep Paragraph        Split by '\n' (Sentences)
```

### 3. Document-Structure Chunking
* **Markdown**: `MarkdownHeaderTextSplitter` splits along `#`, `##`, `###` headers, preserving sections and injecting header hierarchy into chunk metadata.
* **Code**: `PythonCodeTextSplitter` / `LanguageTextSplitter` splits along classes, methods, and functions to maintain syntactically valid code blocks.

---

## 🔬 Semantic Chunking Deep Dive

Traditional chunkers split on arbitrary length boundaries (e.g., 500 characters). **Semantic Chunking** dynamically determines chunk boundaries based on **semantic shifts** in meaning between consecutive sentences.

```
                            SEMANTIC CHUNKING PIPELINE
┌────────────────────────┐
│ 1. Raw Text Document   │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 2. Sentence Splitting  │ (e.g., NLTK / Regex / spaCy into S_1, S_2, S_3...)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 3. Window Grouping     │ (Combine sentences: S_i = [S_{i-1}, S_i, S_{i+1}])
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 4. Vector Embedding    │ (Embed each sentence window with Bi-Encoder)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 5. Cosine Distance     │ (Calculate d_i = 1 - CosSim(vec_i, vec_{i+1}))
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 6. Breakpoint Detection│ (Compare d_i against statistical threshold T)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 7. Dynamic Chunks      │ (Split at distance spikes; merge coherent sentences)
└────────────────────────┘
```

### Mathematical Formulation of Breakpoints
Given consecutive sentence embeddings $\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n$, compute the cosine distance between each adjacent pair:

$$d_i = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_{i+1}}{\|\mathbf{e}_i\| \|\mathbf{e}_{i+1}\|}$$

A **breakpoint (chunk split)** is triggered at index $i$ if $d_i > T_{\text{threshold}}$.

```
   Cosine Distance
        1.0 ┼                                   • Split Point (d_3)
            │                                  / \
        0.8 ┼                                 /   \
            │                                /     \
        0.6 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ • ─ ─ ─ • ─ ─ Threshold T = 0.6
            │          •                   /       \
        0.4 ┼         / \                 /         \
            │  •     /   \               /           •
        0.2 ┼───\───/─────•─────────────/─────────────\───•────
            └───┴───┴─────┴─────────────┴─────────────┴───┴──► Sentence Index
               S1  S2    S3 (Topic A)   S4 (Topic B) S5
               └──────┬──────┘          └──────┬──────┘
                  Chunk 1                     Chunk 2
```

### Breakpoint Threshold Calculation Strategies

| Strategy | Mathematical Formula | Description | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Percentile** | $T = P_{\alpha}(\mathbf{D})$ (e.g., $\alpha = 95$) | Splits text only at the top $(100-\alpha)\%$ most extreme semantic shifts. | Documents with sharp topic transitions (e.g., news digests, multi-topic articles). |
| **Standard Deviation** | $T = \mu + k \cdot \sigma$ (e.g., $k = 1.5$) | Splits where distance exceeds $k$ standard deviations above the mean distance. | Technical manuals and textbooks with steady, gradual transitions. |
| **Interquartile Range** | $T = Q_3 + 1.5 \cdot \text{IQR}$ | Uses robust statistics invariant to outliers ($IQR = Q_3 - Q_1$). | Messy documents with noisy, inconsistent sentence lengths. |
| **Gradient** | $T = \frac{\Delta d_i}{\Delta i}$ | Identifies inflection points in the rate of semantic change. | Narrative storytelling and transcripts. |

---

## 🚀 Advanced Retrieval & Context Augmentation Strategies

To resolve the **Chunking Dilemma**, modern RAG architectures decouple the **Retrieved Unit** from the **Context Passed to the LLM**:

```
                         ADVANCED RETRIEVAL PARADIGMS

1. PARENT DOCUMENT RETRIEVAL                 2. SENTENCE WINDOW RETRIEVAL
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ Large Parent Document (2000 tokens)  │    │ [ Window: Prev Sentence 1 + 2 ]      │
├───────────┬───────────┬──────────────┤    ├──────────────────────────────────────┤
│  Child 1  │  Child 2  │   Child 3    │    │ ► Core Target Sentence (Embedded) ◄ │
│ (200 tok) │ (200 tok) │  (200 tok)   │    ├──────────────────────────────────────┤
└─────┬─────┴───────────┴──────────────┘    │ [ Window: Next Sentence 1 + 2 ]      │
      │                                     └──────────────────┬───────────────────┘
      ▼ (Matches Query)                                        ▼
Embed & Retrieve Child 1                             Retrieve Core Sentence
      │                                                        │
      ▼ (Lookup Parent ID in Key-Value store)                  ▼
Send Entire Parent Document to LLM!                  Send 5-Sentence Window to LLM!
```

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

raw_text = """
LangChain is a framework for building applications with LLMs. It provides modular abstractions for chains and agents.
Retrieval-Augmented Generation (RAG) combines search systems with generative AI.

The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
Constructed from 1887 to 1889, it is named after the engineer Gustave Eiffel.
"""

# 1. Recursive Character Chunking
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""]
)
chunks = recursive_splitter.split_text(raw_text)
print(f"Recursive Chunks Count: {len(chunks)}")

# 2. Semantic Chunking (Using Percentile Threshold)
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # 'percentile', 'standard_deviation', 'interquartile'
    breakpoint_threshold_amount=90
)
semantic_docs = semantic_splitter.create_documents([raw_text])
print(f"\nSemantic Chunks Count: {len(semantic_docs)}")
for i, doc in enumerate(semantic_docs):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content}")

# 3. Markdown Header-Based Chunking
markdown_text = """
# Machine Learning
ML is a subfield of AI.
## Supervised Learning
Involves labeled training data.
## Unsupervised Learning
Discovers patterns in unlabeled data.
"""
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(markdown_text)
for chunk in md_chunks:
    print(f"\nMetadata: {chunk.metadata} | Content: {chunk.page_content}")
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: Why does fixed-character chunking (e.g., every 500 characters) cause retrieval failures in RAG?
**Answer**:
Fixed-character chunking ignores linguistic structure.
1. **Mid-sentence amputation**: Cuts through entities (e.g., splitting `"Barack "` into Chunk 1 and `"Obama"` into Chunk 2).
2. **Contextual severance**: Splits cause-and-effect clauses or conditions from their subjects, leading to corrupted embedding vectors that fail similarity search or hallucinated LLM answers.

---

### Q2: How does Semantic Chunking decide where to split a document?
**Answer**:
Semantic Chunking follows a 4-step process:
1. Splits raw text into atomic sentences.
2. Encodes each sentence (or sliding sentence group) into dense vector embeddings using a bi-encoder.
3. Computes the cosine distance between each consecutive pair of sentences ($d_i = 1 - \cos(\mathbf{e}_i, \mathbf{e}_{i+1})$).
4. Applies a statistical threshold (e.g., 95th percentile or $\mu + 1.5\sigma$). When the distance between sentence $i$ and $i+1$ exceeds this threshold, it signifies a semantic shift, triggering a chunk split.

---

### Q3: What is the computational trade-off of Semantic Chunking compared to Recursive Chunking?
**Answer**:
- **Recursive Chunking**: Extremely fast, $O(N)$ string slicing operations on CPU with zero ML model inference required.
- **Semantic Chunking**: Requires generating vector embeddings for every individual sentence in the corpus during ingestion. For 100,000 documents, this requires millions of embedding API calls or significant local GPU compute time, dramatically increasing ingestion latency and cost.

---

### Q4: Explain the difference between `percentile` and `standard_deviation` thresholding in Semantic Chunking.
**Answer**:
- **`percentile` (e.g., 90th percentile)**: Enforces that roughly $10\%$ of all sentence transitions will trigger a split, regardless of how homogenous or diverse the document is.
- **`standard_deviation` (e.g., $\mu + 1.5\sigma$)**: Dynamically adjusts to document variance. In a document that discusses a single topic uniformly, no distance value may exceed $\mu + 1.5\sigma$, creating large, unified chunks. In highly erratic documents, it creates multiple granular chunks.

---

### Q5: How does the Parent Document Retriever (Small-to-Big Retrieval) solve the Chunking Dilemma?
**Answer**:
The Parent Document Retriever creates a two-tiered hierarchy:
1. Large **Parent Documents** (e.g., 2000 tokens) are stored in an in-memory or Key-Value Docstore (Redis / DynamoDB / SQLite).
2. Small **Child Chunks** (e.g., 200 tokens) are embedded and stored in the Vector Database with metadata containing their `parent_id`.
3. At query time, vector search matches against the high-precision Child Chunks.
4. Instead of sending the small child chunk to the LLM, the system retrieves and forwards the full Parent Document from the Docstore, giving the LLM rich context without sacrificing embedding precision.

---

### Q6: What is Sentence Window Retrieval and how is it implemented?
**Answer**:
In Sentence Window Retrieval, the document is split into individual sentences, and only the single sentence is embedded into the vector database.
However, each vector's metadata stores a window of surrounding context (e.g., the 3 sentences preceding and 3 sentences succeeding it).
When the vector matches a user query, the retriever expands the single sentence into the 7-sentence window before feeding it to the LLM context prompt.

---

### Q7: Why is `chunk_overlap` necessary in recursive chunking, and what happens if it is set to 0?
**Answer**:
If `chunk_overlap = 0`, any semantic relationship spanning across the boundary between Chunk $A$ and Chunk $B$ is permanently broken. If a user query requires understanding the connection between the last sentence of Chunk $A$ and the first sentence of Chunk $B$, neither chunk independently contains enough information to trigger a high cosine match. An overlap of 10-20% ensures boundary continuity.

---

### Q8: How should chunking strategies differ for Source Code vs. Legal Contracts vs. FAQ Knowledge Bases?
**Answer**:
- **Source Code**: Use AST/Syntax-aware splitters (`LanguageTextSplitter`) to split strictly at function/class/method boundaries; never split mid-function.
- **Legal Contracts**: Use hierarchical clause-based splitters (`MarkdownHeaderTextSplitter` or Regex on `"Section 1.1"`, `"Article IV"`) with Parent-Document retrieval to retain definitions and governing law context.
- **FAQ Knowledge Bases**: Chunk by QA pair (one question + one answer per document), often indexing the question as the embedding vector and the answer as metadata payload.

---

### Q9: How do you determine the optimal `chunk_size` for a production RAG application?
**Answer**:
Run an empirical grid-search evaluation using synthetic or golden QA datasets and a framework like **Ragas** or **TruLens**:
1. Test candidate chunk sizes: $[128, 256, 512, 1024]$ tokens with $15\%$ overlap.
2. Evaluate **Context Recall** (did the retriever find the ground-truth chunk?) and **Faithfulness** (did the LLM answer without hallucinating?).
3. Monitor token latency and cost trade-offs.

---

### Q10: What is "Lost in the Middle" syndrome in LLM context windows, and how does chunk ordering mitigate it?
**Answer**:
Research shows that LLMs recall information with high accuracy when it is positioned at the **very beginning** or the **very end** of the context prompt, but performance drops significantly for facts located in the middle of long context windows.
- **Mitigation**: After retrieving and reranking top-$k$ chunks, re-order the retrieved chunks using a **"Long-Context Reorder"** strategy: place the highest-ranked chunk first, the second-highest last, and lower-ranked chunks in the middle.
