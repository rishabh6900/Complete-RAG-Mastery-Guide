# Module 0: Data Ingestion & Document Parsing in RAG

---

## 📖 Table of Contents
1. [Core Concepts & Theoretical Foundations](#-core-concepts--theoretical-foundations)
   - What is Data Ingestion in RAG?
   - LangChain Document Schema Deep Dive
   - Importance of Metadata in Enterprise Search
2. [Document Parsing Modalities & Loaders](#-document-parsing-modalities--loaders)
   - Plain Text & Markdown Parsing
   - PDF Document Parsing (`PyPDF`, `PyMuPDF`, `PDFPlumber`, `Unstructured`)
   - Microsoft Word Documents (`Docx2txtLoader`, `UnstructuredWordDocumentLoader`)
   - Structured & Tabular Data (`CSVLoader`, Excel with Pandas)
   - Semi-Structured JSON Data (`JSONLoader` with `jq` schema)
   - SQL & Relational Databases (`SQLDatabaseLoader` vs Text-to-SQL)
3. [Production Best Practices & Edge Cases](#-production-best-practices--edge-cases)
4. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
5. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 Core Concepts & Theoretical Foundations

### What is Data Ingestion in RAG?
Data Ingestion is the **Extract, Transform, Load (ETL)** pipeline designed specifically for Generative AI applications. In a Retrieval-Augmented Generation (RAG) system, the generation quality of the LLM is strictly bounded by the fidelity of the ingested and retrieved context (**"Garbage In, Garbage Out"**).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION PIPELINE                            │
├───────────────┬─────────────────────────┬─────────────────┬─────────────────┤
│  Raw Sources  │    Document Loaders     │ Text Splitters  │ Vector Store    │
│  - PDFs       │  - Extract Text         │ - Chunking      │ - Embeddings    │
│  - DOCX/TXT   │  - Extract Metadata     │ - Overlap       │ - Metadata      │
│  - CSV/JSON   │  - OCR / Layout Parsing │ - Token limits  │ - Indexes       │
│  - SQL/Web    │  - Clean Artifacts      │                 │                 │
└───────┬───────┴────────────┬────────────┴────────┬────────┴────────┬────────┘
        │                    │                     │                 │
        ▼                    ▼                     ▼                 ▼
   Unstructured        LangChain Doc:          Normalized         Indexed
      Files            {page_content,           Chunks           Embeddings
                         metadata}
```

### LangChain Document Schema Deep Dive
In LangChain, every piece of ingested data is normalized into a standard `Document` abstraction:

```python
class Document:
    page_content: str       # The raw textual payload that gets embedded
    metadata: dict          # Key-value dictionary containing contextual metadata
```

#### Why Metadata is Crucial in Production RAG:
1. **Source Attribution & Citations**: Enables the LLM to output verifiable citations (e.g., *"According to page 12 of the 2024 Annual Report..."*).
2. **Metadata Filtering (Pre/Post-Filtering)**: Restricts vector search space using scalar filters before computing expensive distance metrics (e.g., `user_id == '123'`, `created_year >= 2024`, `department == 'HR'`).
3. **Multi-Tenancy & Data Isolation**: Guarantees strict security boundaries between corporate tenants without needing separate vector databases.
4. **Time-decayed & Freshness Retrieval**: Allows penalizing or filtering out deprecated documents based on `timestamp` metadata.

---

## 📂 Document Parsing Modalities & Loaders

### 1. Plain Text & Markdown Parsing
* **Loaders**: `TextLoader`, `UnstructuredMarkdownLoader`.
* **Behavior**: Reads UTF-8 text straight into memory. Markdown parsers preserve heading hierarchies (`#`, `##`, `###`), which can be leveraged for Header-based chunking (`MarkdownHeaderTextSplitter`).

### 2. PDF Document Parsing
PDFs are layout-oriented fixed-coordinate formats, not stream-oriented semantic documents. Choosing the right loader depends on performance and complexity:

| Loader | Underlying Engine | Pros | Cons | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`PyPDFLoader`** | `pypdf` | Lightweight, native Python, zero external dependencies | Struggles with multi-column text, ignores tables, slow on large PDFs | Simple linear text PDFs, quick prototypes |
| **`PyMuPDFLoader`** | `fitz` (MuPDF C-lib) | **10x-20x faster** than `pypdf`, extracts coordinates, images, and fonts | Requires C-bindings | Production ingestion of massive standard PDF archives |
| **`PDFPlumberLoader`** | `pdfplumber` | High-fidelity table extraction, precise bounding box coordinates | High CPU & memory usage | Ingesting financial statements, balance sheets, invoices |
| **`UnstructuredPDFLoader`** | `unstructured` + `Tesseract` | Advanced layout detection (detects headers, footers, narrative text, tables, OCR fallback) | Heavyweight dependency, slow processing speed | Complex scanned documents with mixed layouts & images |

```
                       ┌─────────────────────────┐
                       │    Is the PDF Scanned?   │
                       └────────────┬────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ YES                           │ NO
                    ▼                               ▼
       ┌────────────────────────┐      ┌─────────────────────────┐
       │ Use OCR Parser         │      │ Does it contain complex │
       │ (Unstructured +        │      │ tables & columns?       │
       │ Tesseract / AWS Textract)     └────────────┬────────────┘
       └────────────────────────┘                   │
                                     ┌──────────────┴──────────────┐
                                     │ YES                         │ NO
                                     ▼                             ▼
                        ┌─────────────────────────┐   ┌─────────────────────────┐
                        │ pdfplumber / LayoutLM   │   │ PyMuPDF (fitz) for speed│
                        │ Markdown table extractor│   │ PyPDF for simplicity    │
                        └─────────────────────────┘   └─────────────────────────┘
```

### 3. Microsoft Word Documents (`.docx`, `.doc`)
* **Loaders**: `Docx2txtLoader`, `UnstructuredWordDocumentLoader`.
* **Behavior**: Parses XML-based `.docx` packages into paragraph streams. `Docx2txtLoader` is fast and lightweight, while `Unstructured` captures bullet points, headers, and embedded tables.

### 4. Tabular Data (CSV & Excel)
* **Loaders**: `CSVLoader`, `UnstructuredExcelLoader`, or Custom Pandas Loaders.
* **The Tabular Ingestion Dilemma**:
  * Raw vector search over row-by-row stringified CSVs often fails because semantic embeddings do not understand column aggregations (`SUM`, `AVG`, `GROUP BY`).
  * **Solution**: Convert rows into descriptive key-value sentences:
    ```
    "Product: Laptop | Category: Electronics | Price: $1200 | Stock: 45 units"
    ```
  * For analytical queries, use **Text-to-SQL / Pandas Agents** rather than naive vector search.

### 5. Semi-Structured JSON Data
* **Loader**: `JSONLoader` (powered by `jq` filter syntax).
* **Behavior**: Allows deterministic extraction of deep nested structures:
  ```python
  # Extracting all 'description' fields from an array of products
  loader = JSONLoader(
      file_path="data.json",
      jq_schema=".products[].description",
      text_content=True
  )
  ```

### 6. SQL & Relational Databases
* **Loader**: `SQLDatabaseLoader`.
* **Approach 1 (Document Extraction)**: Queries database views and dumps rows as text documents for semantic search.
* **Approach 2 (Text-to-SQL / SQL Agents)**: Passes the database schema (DDL) to the LLM to generate real-time SQL queries on the fly.

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader
)

# 1. Loading Text File
txt_loader = TextLoader("data/sample.txt", encoding="utf-8")
txt_docs = txt_loader.load()

# 2. Loading PDF with Page-Level Metadata
pdf_loader = PyPDFLoader("data/attention_is_all_you_need.pdf")
pdf_pages = pdf_loader.load()
print(f"Loaded {len(pdf_pages)} pages. Page 1 metadata: {pdf_pages[0].metadata}")

# 3. High-Speed PDF Loading with PyMuPDF
fast_pdf_loader = PyMuPDFLoader("data/attention_is_all_you_need.pdf")
fast_pdf_pages = fast_pdf_loader.load()

# 4. Loading Word Documents
docx_loader = Docx2txtLoader("data/report.docx")
docx_docs = docx_loader.load()

# 5. Loading CSV with Custom Source Column for Metadata
csv_loader = CSVLoader(
    file_path="data/products.csv",
    source_column="product_id",
    csv_args={"delimiter": ","}
)
csv_docs = csv_loader.load()

# 6. Loading JSON with jq Schema
json_loader = JSONLoader(
    file_path="data/users.json",
    jq_schema=".users[].bio",
    text_content=True
)
json_docs = json_loader.load()
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: What is the LangChain `Document` object, and why is the `metadata` dictionary critical in production RAG?
**Answer**:
A LangChain `Document` object consists of two attributes: `page_content` (the string text chunk to be embedded) and `metadata` (a key-value dictionary containing contextual info such as `source`, `page`, `author`, `creation_date`, `access_role`).
`metadata` is critical in production for:
1. **Metadata Pre-filtering**: Restricting semantic search by scalar attributes (e.g., date ranges, tenant IDs, document types), drastically improving search latency and precision.
2. **Access Control (RBAC)**: Ensuring users can only retrieve chunks they are authorized to see based on security tags in metadata.
3. **Traceability & Citations**: Allowing the LLM response to provide exact source URLs or page numbers to reduce hallucinations.

---

### Q2: Why does naive PDF text extraction often fail on multi-column research papers or financial reports?
**Answer**:
PDF is a print presentation format storing text strings at explicit $(x, y)$ coordinate points rather than semantic reading streams.
- **Multi-column text**: Naive extractors read text horizontally across the page ($x_1 \to x_2$), interleaving sentences from Column 1 with Column 2, completely destroying the grammatical syntax and semantic meaning.
- **Embedded Tables**: Tables get flattened into unstructured word sequences, disassociating table headers from row values.
- **Solution**: Use layout-aware or vision-based parsers (`pdfplumber`, `PyMuPDF` with layout bounding boxes, `Unstructured` with LayoutLM, or multi-modal Vision models like GPT-4o / Claude 3.5 Sonnet to convert pages into structured Markdown).

---

### Q3: How do `PyPDFLoader`, `PyMuPDFLoader`, and `UnstructuredPDFLoader` differ in terms of performance and trade-offs?
**Answer**:
- **`PyPDFLoader`**: Pure Python implementation. Easy to install, but slow on large files and cannot handle complex layouts, OCR, or tables.
- **`PyMuPDFLoader` (`fitz`)**: Written in C (MuPDF). Incredibly fast (10x-20x faster than pure Python parsers), handles standard text and bounding box extractions with minimal CPU overhead.
- **`UnstructuredPDFLoader`**: Heavyweight pipeline integrating layout segmentation models, table transformers, and Tesseract OCR. Slower and resource-intensive, but essential for messy, scanned, or complex legacy PDFs.

---

### Q4: When should you use a Document Loader vs. a Text-to-SQL pipeline for tabular/structured data (CSV, Excel, SQL)?
**Answer**:
- **Use Document Loader (RAG)**: When the query is **semantic or fuzzy** (e.g., *"Find customer reviews complaining about battery life and overheating"*).
- **Use Text-to-SQL / Database Agent**: When the query requires **mathematical aggregation, sorting, or exact relational lookups** (e.g., *"What is the total revenue for Q3 2024 grouped by region?"*). Embeddings cannot reliably perform math or count operations over vectorized tabular chunks.

---

### Q5: What is "Metadata Pollution" and how do you prevent it during data ingestion?
**Answer**:
Metadata pollution occurs when unnecessary, bloated, or non-scalar metadata (e.g., raw binary data, massive nested JSON trees, base64 images) is stored alongside document chunks in the vector store.
- **Consequences**: Inflates vector database memory/storage, degrades scalar filtering performance, and increases payload transmission latency.
- **Prevention**: Sanitize and prune metadata dictionaries during the ingestion ETL step, keeping only indexed scalar fields (strings, ints, floats, booleans, timestamps).

---

### Q6: How do you extract structured content from JSON files without embedding irrelevant syntax like braces and keys?
**Answer**:
Use `JSONLoader` with targeted `jq` schema selectors. Instead of embedding the entire raw JSON object string (which burns context tokens on brackets, commas, and structural keys), specify `jq_schema` to extract only the target text fields (e.g., `jq_schema='.articles[].content'`). You can also map key fields (like `id` or `category`) into the chunk's `metadata`.

---

### Q7: How do you handle non-English or multilingual documents during data ingestion?
**Answer**:
1. **Encoding Handling**: Ensure loaders explicitly enforce `encoding="utf-8"` or use charset-detectors (`chardet`) to prevent UTF-8/ISO decode errors.
2. **Language Identification**: Run language detection (e.g., `langdetect` or `fasttext`) and inject `lang: "es" | "en" | "zh"` into metadata.
3. **Multilingual Embedding Models**: Use language-appropriate embedding models (e.g., `text-embedding-3-small` or `paraphrase-multilingual-mpnet-base-v2`) to preserve cross-lingual semantic alignment.

---

### Q8: What strategy should you use to ingest continuously updating data sources (e.g., daily updated Confluence or SharePoint docs)?
**Answer**:
Use **Incremental Ingestion with Content Hashing / CDC (Change Data Capture)**:
1. Generate an MD5 or SHA256 hash of each document's raw content before chunking.
2. Maintain an Ingestion Index / Record Manager (`langchain.indexes.SQLRecordManager`).
3. If the hash matches an existing record in the vector database, skip processing.
4. If the document has been modified, delete outdated chunks by parent document ID and upsert the newly chunked vectors.
5. If a document is deleted from source, remove all associated vectors from the vector store.

---

### Q9: How do you parse documents containing embedded images, charts, and diagrams for a multimodal RAG system?
**Answer**:
1. **Vision-LLM Extraction**: Use an OCR/Vision model (e.g., GPT-4o, Claude 3.5 Sonnet) to inspect pages and generate descriptive Markdown summaries of charts, diagrams, and figures.
2. **Dual-Path Ingestion**: Extract textual narrative via high-speed text parsers, and extract image objects using `pdfplumber` / `fitz`. Pass images to a vision model to generate captions, and embed the captions with metadata linking to the image asset URL.
3. **ColPali / Vision-based Embeddings**: Use modern vision-retriever models that directly embed entire page images without traditional OCR.

---

### Q10: How do you ensure high throughput when ingesting millions of enterprise documents?
**Answer**:
1. **Asynchronous & Distributed Batching**: Utilize multiprocessing or distributed task queues (Celery, Ray, AWS Lambda, Apache Airflow).
2. **Lazy Loading**: Use `.lazy_load()` instead of `.load()` in LangChain to stream documents into memory as an iterator rather than loading millions of files into RAM at once.
3. **Batch Embedding Calls**: Group text chunks into optimal batch sizes (e.g., 512 to 2048 chunks per API call) to saturate API throughput limits without hitting rate limits.
