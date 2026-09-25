# 📥 Module 0: Data Ingestion & Document Parsing for Production RAG

Welcome to **Module 0: Data Ingestion & Document Parsing**. This module serves as the foundational data engineering layer for building production-grade **Retrieval-Augmented Generation (RAG)** systems. In RAG pipelines, generation quality is strictly bounded by ingestion fidelity (**"Garbage In, Garbage Out"**).

---

## 📑 Table of Contents
1. [Module Architecture & Ingestion Pipeline](#-module-architecture--ingestion-pipeline)
2. [Folder Structure & Notebook Roadmap](#-folder-structure--notebook-roadmap)
3. [Document Loaders & Parsing Modalities Comparison](#-document-loaders--parsing-modalities-comparison)
4. [🎯 Comprehensive Technical Interview Questions & Answers](#-comprehensive-technical-interview-questions--answers)
   - [Section A: Fundamentals & LangChain Document Schema](#section-a-fundamentals--langchain-document-schema)
   - [Section B: Deep Dive on PDF & Complex Document Parsing](#section-b-deep-dive-on-pdf--complex-document-parsing)
   - [Section C: Tabular, Structured & Semi-Structured Data (CSV, Excel, JSON, SQL)](#section-c-tabular-structured--semi-structured-data-csv-excel-json-sql)
   - [Section D: Web Scraping, OCR & Multimodal Ingestion](#section-d-web-scraping-ocr--multimodal-ingestion)
   - [Section E: Enterprise Scale, Security, and System Design](#section-e-enterprise-scale-security-and-system-design)
5. [Code Implementation Quick Reference](#-code-implementation-quick-reference)

---

## 🏗️ Module Architecture & Ingestion Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               DATA INGESTION ETL PIPELINE                              │
├───────────────────┬────────────────────────────┬────────────────────┬──────────────────┤
│ 1. Raw Sources    │ 2. Document Loaders        │ 3. Normalization   │ 4. Downstream    │
│ - PDF (Native/OCR)│ - TextLoader / PyMuPDF     │ - LangChain Doc    │ - Chunking       │
│ - DOCX / TXT / MD │ - Docx2txtLoader / Pandoc  │   {page_content,   │ - Embeddings     │
│ - CSV / Excel     │ - CSVLoader / Pandas       │    metadata}       │ - Vector Stores  │
│ - JSON / JSONL    │ - JSONLoader (jq schema)   │ - Metadata Filter  │ - Hybrid Indexes │
│ - SQL Databases   │ - SQLDatabaseLoader        │ - Content Cleaning │                  │
│ - Web / HTML      │ - WebBaseLoader / BS4      │ - Hash & Dedupe    │                  │
└───────────────────┴────────────────────────────┴────────────────────┴──────────────────┘
```

---

## 📂 Folder Structure & Notebook Roadmap

| File / Notebook | Modality | Key Technologies & Concepts Covered |
| :--- | :--- | :--- |
| [`1-dataingestion.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/1-dataingestion.ipynb) | Multi-Source Overview | `TextLoader`, `PyPDFLoader`, `WebBaseLoader`, `ArxivLoader`, `WikipediaLoader` |
| [`2-dataparsingpdf.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/2-dataparsingpdf.ipynb) | PDF Parsing | `PyPDF`, `PyMuPDF (fitz)`, `pdfplumber`, `UnstructuredPDFLoader`, Layout Parsing |
| [`3-dataparsingdoc.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/3-dataparsingdoc.ipynb) | Word Documents | `Docx2txtLoader`, `UnstructuredWordDocumentLoader`, XML Paragraph Extraction |
| [`4-csvexcelparsing.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/4-csvexcelparsing.ipynb) | Tabular Data | `CSVLoader`, `UnstructuredExcelLoader`, Row-to-Sentence Transformation, Pandas |
| [`5-jsonparsing.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/5-jsonparsing.ipynb) | Semi-Structured Data | `JSONLoader`, `jq` Schema Expressions, Nested Key Extraction, JSONL |
| [`6-databaseparsing.ipynb`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/6-databaseparsing.ipynb) | Relational SQL | `SQLDatabaseLoader`, Table-to-Document Dumps vs. Dynamic Text-to-SQL |
| [`THEORY_AND_INTERVIEW_NOTES.md`](file:///d:/Udemy/Rag_krish%20naik/0_DataInagestion/THEORY_AND_INTERVIEW_NOTES.md) | Theory & Notes | Core Ingestion Concepts, Benchmarks, and Top 10 High-Yield Questions |

---

## 📊 Document Loaders & Parsing Modalities Comparison

| Loader | Underlying Engine | Speed Benchmark | Tables / Layout Quality | Best Production Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`PyPDFLoader`** | `pypdf` (Python) | Moderate (~10-20 ms/page) | Poor (ignores table structure) | Simple, single-column linear text documents |
| **`PyMuPDFLoader`** | `fitz` / MuPDF (C-lib) | **Ultra-Fast (< 2 ms/page)** | Moderate (provides bounding boxes) | High-volume batch text extraction |
| **`PDFPlumberLoader`** | `pdfplumber` (Python) | Slower (~100-300 ms/page) | **High** (extracts cell borders & tables) | Financial reports, invoices, data sheets |
| **`UnstructuredPDFLoader`**| `unstructured` + OCR | Heavyweight (~500+ ms/page) | **Highest** (LayoutLM + OCR fallback) | Complex scanned documents & multi-column layouts |
| **`Docx2txtLoader`** | `docx2txt` | Fast (< 5 ms/doc) | Basic text stream | Microsoft Word documents without deep formatting |
| **`CSVLoader`** | Standard `csv` / Python | Instantaneous | 1 chunk per row | Row-level lookup without mathematical aggregations |
| **`JSONLoader`** | `jq` engine | Fast | Precise targeted field extraction | API payloads, log streams, nested configurations |

---

## 🎯 Comprehensive Technical Interview Questions & Answers

### Section A: Fundamentals & LangChain Document Schema

#### Q1: What is the LangChain `Document` object abstraction, and what role does the `metadata` dictionary play in production RAG?
**Answer:**
In LangChain, every raw data input is normalized into a standard `Document` instance:
```python
class Document:
    page_content: str   # The exact text payload to be chunked, embedded, and passed to LLM context
    metadata: dict      # Key-value dictionary containing contextual and operational attributes
```
In production RAG systems, the `metadata` dictionary is critical for:
1. **Metadata Pre-Filtering & Hybrid Search**: Allows restricting the vector search space using scalar filters before computing expensive distance metrics (e.g., `where={"department": "Finance", "year": 2024}`). This drastically reduces latency and eliminates irrelevant cross-domain chunks.
2. **Access Control (RBAC / Multi-Tenancy)**: Stores security tags (`tenant_id`, `role_required`) so unauthorized users cannot retrieve confidential chunks.
3. **Source Attribution & Citation Grounding**: Supplies exact `source` URLs, file paths, and `page` numbers to the LLM prompt, enabling verifiable in-text citations.
4. **Time-Decayed / Recency Ranking**: Stores `created_at` or `last_modified` timestamps so outdated documents can be filtered out or penalized during retrieval.

---

#### Q2: What is "Metadata Pollution" during data ingestion, and how do you mitigate it?
**Answer:**
**Metadata Pollution** happens when raw, bloated, unindexed, or high-cardinality non-scalar fields (e.g., entire base64 image strings, arbitrary nested dictionaries, transient request headers) are dumped into the `metadata` dictionary.
- **Problems Caused**:
  - Inflates memory footprint in vector databases (like Chroma, FAISS, Pinecone, or Qdrant).
  - Slows down scalar filtering queries.
  - Exceeds payload size limits when fetching Top-K candidates.
- **Mitigation Strategy**:
  - Define a strict metadata schema during ingestion (e.g., using Pydantic models).
  - Explicitly sanitize/prune metadata during loader execution:
    ```python
    def sanitize_metadata(doc: Document) -> Document:
        allowed_keys = {"source", "page", "author", "doc_type", "created_date"}
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in allowed_keys and isinstance(v, (str, int, float, bool))}
        return doc
    ```

---

#### Q3: What is the difference between `.load()` and `.lazy_load()` in LangChain document loaders, and why is this critical for scalability?
**Answer:**
- **`.load()`**: Reads all documents into memory at once and returns a `List[Document]`.
  - *Risk*: Ingesting a 10,000-page PDF or 100,000 CSV rows causes an Out-Of-Memory (**OOM**) crash because the entire raw text and metadata objects reside concurrently in RAM.
- **`.lazy_load()`**: Returns a Python **generator / iterator** (`Iterator[Document]`) that yields documents one-by-one or chunk-by-chunk on demand.
  - *Advantage*: Enables constant $O(1)$ memory usage during ingestion, streaming documents straight through the chunker, embedding model, and vector store upsert batcher.

---

### Section B: Deep Dive on PDF & Complex Document Parsing

#### Q4: Why does naive text extraction fail on multi-column PDF layouts (e.g., academic papers, newsletters), and how do you solve it?
**Answer:**
**The Root Cause**:
PDF is a print presentation coordinate system (storing text as $(x, y)$ draw glyph instructions), not a semantic markdown or HTML document tree. Naive parsers read text strictly in horizontal bounding box order ($y$-axis top-to-bottom, $x$-axis left-to-right). 

When encountering a two-column paper:
```
Column 1 (Line 1)   |   Column 2 (Line 1)
Column 1 (Line 2)   |   Column 2 (Line 2)
```
A naive parser reads: `Column 1 (Line 1) -> Column 2 (Line 1) -> Column 1 (Line 2) -> Column 2 (Line 2)`. This interleaves sentences from separate paragraphs, completely destroying syntax and semantic vector representation.

**Solutions**:
1. **Layout-Aware Parsers**: Use `PyMuPDF` with layout sorting (`page.get_text("blocks", sort=True)`) or `UnstructuredPDFLoader` (utilizing LayoutLM / YOLO document models to detect column boundaries before reading).
2. **Vision-Language Model Parsing**: Convert PDF pages to high-resolution images and send them to Vision LLMs (GPT-4o / Claude 3.5 Sonnet) with instructions to convert the layout into structured Markdown (`### Headers`, tables, continuous reading order).
3. **Dedicated OCR/Document Intelligence**: Use AWS Textract, Google Document AI, or Azure Document Intelligence for enterprise documents.

---

#### Q5: Compare `PyPDF`, `PyMuPDF (fitz)`, `pdfplumber`, and `Unstructured` in a production evaluation matrix.
**Answer:**

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 PDF PARSER SELECTION TREE              │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                              Is the document scanned / image-only?
                                       /              \
                                     YES               NO
                                     /                  \
                        ┌────────────────────────┐  Does it contain complex tables
                        │  Unstructured + OCR /  │  or multi-column layouts?
                        │  Vision LLM (GPT-4o)   │        /                \
                        └────────────────────────┘      YES                 NO
                                                        /                    \
                                        ┌─────────────────────────┐  ┌───────────────────────┐
                                        │ pdfplumber / Docling /  │  │ PyMuPDF (fitz)        │
                                        │ Table Transformer       │  │ (Max speed & low RAM) │
                                        └─────────────────────────┘  └───────────────────────┘
```

- **`PyMuPDF` (`fitz`)**: Best for speed-critical pipelines processing standard digital PDFs. 10x-20x faster than pure Python loaders.
- **`pdfplumber`**: Best when tabular accuracy is paramount and bounding box lines need to be resolved into markdown/CSV strings.
- **`Unstructured`**: Best for heterogeneous, messy archives containing mixtures of scanned pages, embedded images, headers, footers, and complex multi-modal layouts.
- **`PyPDF`**: Best only for zero-dependency lightweight scripting or unit testing.

---

#### Q6: How do you extract and preserve tables from documents so vector embeddings and LLMs can understand them?
**Answer:**
Flattening a table into raw text destroys row-column relationships. Ingesting:
`Header1 Header2 Val1 Val2` produces ambiguous vector embeddings.

**Best Practices for Table Ingestion**:
1. **Markdown Table Conversion**: Extract tables as explicit Markdown:
   ```markdown
   | Quarter | Revenue | Profit |
   |---------|---------|--------|
   | Q1 2024 | $10M    | $2.5M  |
   ```
2. **Row-Level Semantic Summary (Dual Representation)**:
   - For Vector Search: Convert each row into a descriptive sentence: *"In Q1 2024, the revenue was $10M and the profit was $2.5M."*
   - For LLM Context: Pass the entire raw Markdown table as the parent document when any of its row-level chunks are retrieved (using **Parent-Document Retrieval**).
3. **Table Summarization with LLM**: Generate a natural language summary of the entire table during ingestion, embed the summary for search, and store the structured table in metadata or docstore.

---

### Section C: Tabular, Structured & Semi-Structured Data (CSV, Excel, JSON, SQL)

#### Q7: When should you use a Document Loader for Tabular Data (CSV/Excel/SQL) versus a Text-to-SQL / Database Agent?
**Answer:**

| Criterion | Ingestion into Vector Store (RAG) | Text-to-SQL / Database Agent |
| :--- | :--- | :--- |
| **Query Nature** | Semantic / Conceptual / Fuzzy search (e.g., *"Find user complaints about battery drain"*) | Exact, Aggregations, Math, Sorting (e.g., *"What was the average sales price in Q2?"*) |
| **Data Size** | Small to moderate tabular subsets | Millions to billions of relational rows |
| **Failure Mode** | Hallucinates numerical calculations and cannot aggregate | Syntactically invalid SQL or schema hallucination |
| **Implementation** | `CSVLoader`, `DataFrameLoader` with row-to-sentence transformation | `SQLDatabaseChain`, LangGraph SQL Agent with schema DDL injection |

> [!IMPORTANT]
> **Rule of Thumb**: Never rely on Vector Embeddings alone to perform `COUNT`, `SUM`, `AVG`, or multi-table `JOIN` operations. Use semantic RAG for qualitative text search and Text-to-SQL for quantitative database analytics.

---

#### Q8: How does `JSONLoader` work with `jq` schema, and why is targeted extraction essential for semi-structured JSON data?
**Answer:**
Standard JSON objects contain extensive syntactic boilerplate (`{}`, `[]`, field names, quotes, null values). Ingesting raw JSON strings directly into vector stores wastes token budget and dilutes embedding quality.

`JSONLoader` uses `jq` query syntax to isolate specific target text fields and map other keys into metadata:
```python
from langchain_community.document_loaders import JSONLoader

# Extracts only the 'bio' and 'skills' text, ignoring internal server IDs and timestamps
loader = JSONLoader(
    file_path="users.json",
    jq_schema=".users[] | {bio: .bio, skills: (.skills | join(', '))}",
    text_content=False
)
docs = loader.load()
```
**Benefits**:
- Zero token wastage on structural JSON formatting.
- Clean text chunks optimized for dense bi-encoder embedding models.
- Predictable metadata key extraction.

---

### Section D: Web Scraping, OCR & Multimodal Ingestion

#### Q9: What are the primary pitfalls when ingesting web pages using `WebBaseLoader` or BeautifulSoup, and how do you handle dynamic JavaScript sites?
**Answer:**
**Pitfalls**:
1. **Boilerplate Noise**: Web pages are littered with navigation bars, cookie banners, footers, advertisements, and copyright notices. If embedded, these contaminate the vector database and cause irrelevant retrieval matches.
2. **Client-Side Rendering (SPA / JavaScript)**: `WebBaseLoader` (built on `urllib`/`requests` + `BeautifulSoup`) only fetches static HTML. If the page is rendered dynamically with React, Vue, or Angular, it receives an empty `<div id="root"></div>`.
3. **Rate Limiting & Bot Detection**: Cloudflare, Akamai, or 429 Too Many Requests errors.

**Production Solutions**:
- **Clean HTML with Readability / Trafilatura**: Use `trafilatura` or `langchain_community.document_loaders.NewsURLLoader` which automatically strips navigation, ads, and footers, extracting only main article content.
- **Headless Browsers for Dynamic Pages**: Use `PlaywrightURLLoader` or `SeleniumURLLoader` to execute JavaScript and wait for DOM elements to load before scraping.
- **Async Batch Scraping**: Use `AsyncHtmlLoader` with a `Html2TextTransformer` pipeline.

---

#### Q10: How do you design an ingestion pipeline for multimodal documents containing embedded diagrams, charts, and infographics?
**Answer:**
1. **Dual-Extraction Pipeline**:
   - **Text Path**: Extract narrative text using high-speed parsers (`PyMuPDF`).
   - **Visual Path**: Extract embedded images and charts using `fitz` / `pdfplumber` bounding box extraction.
2. **Vision-LLM Captioning & Summarization**:
   - Send extracted charts/diagrams to a Vision LLM (GPT-4o or Claude 3.5 Sonnet) with a prompt: *"Generate a comprehensive markdown table and summary of the data/process depicted in this chart."*
3. **Context Association**:
   - Embed the generated visual description as a text chunk.
   - Inject the image path or presigned S3 URL into the chunk's `metadata: {"image_url": "s3://...", "parent_section": "Financial Overview"}`.
4. **ColPali / Vision Retrievers**:
   - In modern state-of-the-art systems, use vision retriever models (like ColPali) that directly index entire page screenshots as multi-vector embeddings without needing separate OCR pipelines.

---

### Section E: Enterprise Scale, Security, and System Design

#### Q11: How do you design an Incremental Ingestion & Change Data Capture (CDC) pipeline to prevent re-indexing millions of documents every day?
**Answer:**
**Problem**: In an enterprise knowledge base (SharePoint, Notion, Google Drive), re-embedding unchanged documents is cost-prohibitive and leads to duplicate vectors in the vector store.

**Architecture**:
```
  New/Updated Doc ──► Compute SHA-256 Hash ──► Compare with SQL Record Manager
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 │ Match (No Change)                │ New / Modified / Deleted         │
                 ▼                                  ▼                                  ▼
               SKIP                          Chunk & Embed                     Delete Old Vectors
                                             Upsert to Vector DB               from Vector DB
                                             Update Record Manager             Prune Record Manager
```

**Implementation using LangChain Indexing API**:
```python
from langchain.indexes import SQLRecordManager, index

# 1. Initialize Record Manager
record_manager = SQLRecordManager(
    db_url="postgresql+psycopg2://user:pass@localhost:5432/rag_indexing",
    namespace="sharepoint_kb"
)
record_manager.create_schema()

# 2. Run incremental index with cleanup mode
# Mode 'incremental': updates modified docs, deletes stale child chunks
# Mode 'full': deletes all vectors not present in current ingestion batch
index(
    docs_source=loader.lazy_load(),
    record_manager=record_manager,
    vector_store=vector_store,
    cleanup="incremental",
    source_id_key="source"
)
```

---

#### Q12: How do you handle encoding errors (e.g., `UnicodeDecodeError`, ISO-8859-1 vs UTF-8) and corrupt files during high-volume batch ingestion?
**Answer:**
1. **Dynamic Character Encoding Detection**:
   - Wrap file reading with `charset-normalizer` or `chardet` before passing to loaders:
     ```python
     import chardet

     def detect_encoding(file_path: str) -> str:
         with open(file_path, "rb") as f:
             raw_data = f.read(4096)  # Read first 4KB
             return chardet.detect(raw_data)["encoding"] or "utf-8"
     ```
2. **Dead-Letter Queue (DLQ) & Fault Tolerance**:
   - Ingestion pipelines must never crash completely due to a single corrupted PDF or encoding failure.
   - Use `try/except` wrappers that catch parser exceptions, log the file path and stack trace, and push the failed document to an ingestion error log or DLQ (e.g., SQS / Redis Stream) for manual inspection.
3. **Text Sanitization**:
   - Strip null bytes (`\x00`), non-printable ASCII control characters, and fix broken unicode ligatures (e.g., `\ufb01` $\to$ `fi`) using `unicodedata.normalize('NFKD', text)`.

---

#### Q13: How do you implement Row-Level Security (RLS) / Role-Based Access Control (RBAC) at the data ingestion stage?
**Answer:**
**Strategy**:
1. **Security Tagging during Ingestion**:
   - When loading documents from enterprise sources (e.g., Google Drive, Confluence), fetch the Access Control List (ACL) of the source file.
   - Inject allowed groups, roles, or user IDs into the `metadata`:
     ```python
     doc.metadata["allowed_roles"] = ["admin", "hr_team"]
     doc.metadata["tenant_id"] = "enterprise_corp_a"
     ```
2. **Query-Time Post/Pre-Filtering**:
   - When a user submits a query, extract their verified JWT authentication claims (e.g., `user_roles = ["hr_team"]`).
   - Construct the vector search scalar filter:
     ```python
     retriever = vector_store.as_retriever(
         search_kwargs={
             "filter": {
                 "tenant_id": current_user.tenant_id,
                 "allowed_roles": {"$in": current_user.roles}
             }
         }
     )
     ```
3. **Security Advantage**: Guarantees at the database level that vector similarity calculations never return unauthorized document chunks to the LLM context.

---

## 💻 Code Implementation Quick Reference

```python
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    WebBaseLoader
)

# 1. Robust UTF-8 Text Loading
text_loader = TextLoader("data/sample.txt", encoding="utf-8")
text_docs = text_loader.load()

# 2. Ultra-Fast PDF Loading (PyMuPDF)
pdf_loader = PyMuPDFLoader("data/attention_is_all_you_need.pdf")
pdf_docs = pdf_loader.load()
print(f"Loaded {len(pdf_docs)} pages from PDF. Page 1 metadata: {pdf_docs[0].metadata}")

# 3. Microsoft Word Document Loading
docx_loader = Docx2txtLoader("data/annual_report.docx")
docx_docs = docx_loader.load()

# 4. CSV Loading with Custom Metadata Source Column
csv_loader = CSVLoader(
    file_path="data/products.csv",
    source_column="product_id",
    csv_args={"delimiter": ","}
)
csv_docs = csv_loader.load()

# 5. Targeted Semi-Structured JSON Loading with jq
json_loader = JSONLoader(
    file_path="data/events.json",
    jq_schema=".events[].description",
    text_content=True
)
json_docs = json_loader.load()

# 6. Web Ingestion with BeautifulSoup Parsing
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
web_docs = web_loader.load()
```

---

## 🔗 Related Modules & Next Steps
- 👉 **[Module 1: Vector Embeddings](../1_embeddings/THEORY_AND_INTERVIEW_NOTES.md)**: Transform loaded documents into high-dimensional dense vector embeddings.
- 👉 **[Module 2: Vector Stores & Vector Databases](../2_vectore_store_and_vector_database/THEORY_AND_INTERVIEW_NOTES.md)**: Index and persist chunked documents using HNSW and IVF indexes.
- 👉 **[Module 4: Advanced Chunking Strategies](../4_Advanced_chunking_and_preprocessing_techniques/THEORY_AND_INTERVIEW_NOTES.md)**: Split parsed documents using Recursive, Semantic, and Parent-Document strategies.
