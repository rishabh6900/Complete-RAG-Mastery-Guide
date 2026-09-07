# Module 6: Query Enhancement & Transformation Techniques in RAG

---

## 📖 Table of Contents
1. [Core Concepts: Why Raw User Queries Fail](#-core-concepts-why-raw-user-queries-fail)
   - The Semantic Gap & Query Ambiguity
   - The Query Transformation Paradigm
2. [Query Enhancement Strategies Deep Dive](#-query-enhancement-strategies-deep-dive)
   - Multi-Query Retrieval (Query Expansion)
   - Hypothetical Document Embeddings (HyDE)
   - Step-Back Prompting (Abstraction to First Principles)
   - Sub-Query Decomposition (Complex Multi-Hop Queries)
   - Query Rewriting & Contextual Disambiguation
3. [LangChain Code Implementation Reference](#-langchain-code-implementation-reference)
4. [🎯 Top 10 Technical Interview Questions & Answers](#-top-10-technical-interview-questions--answers)

---

## 🧠 Core Concepts: Why Raw User Queries Fail

In real-world RAG applications, users rarely type clean, dense, keyword-rich queries. Instead, user queries are often:
1. **Ambiguous or Vague**: *"What is its policy?"* (Lacks subject entity).
2. **Vocabulary Mismatch**: Users use colloquial language, while documents use formal legal/technical terminology.
3. **Multi-Faceted / Complex**: *"Compare the battery degradation of Model X vs Model Y and how it affects warranty claims."* (Single vector search cannot match both topics simultaneously).

```
                              QUERY ENHANCEMENT TAXONOMY
                                 ┌───────────────────┐
                                 │  Raw User Query   │
                                 └─────────┬─────────┘
                                           │
         ┌──────────────────┬──────────────┼──────────────┬──────────────────┐
         ▼                  ▼              ▼              ▼                  ▼
  ┌──────────────┐   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ Multi-Query  │   │     HyDE     │ │  Step-Back   │ │  Sub-Query   │ │   Context    │
  │ (Expansion)  │   │ (Hypothetical│ │  Prompting   │ │Decomposition │ │  Rewriting   │
  │              │   │   Answer)    │ │(High-level)  │ │ (Multi-Hop)  │ │(Chat Memory) │
  └──────────────┘   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🔍 Query Enhancement Strategies Deep Dive

### 1. Multi-Query Retrieval (Query Expansion)
* **Concept**: Uses an LLM to generate 3 to 5 semantically distinct variations/perspectives of the original query.
* **Mechanism**: Executes vector search on each generated query, then combines and deduplicates the retrieved chunks using **Reciprocal Rank Fusion (RRF)**.

### 2. Hypothetical Document Embeddings (HyDE)
* **Concept**: Instead of embedding the user's short query, an LLM generates a *hypothetical answer passage* (even if it contains hallucinations).
* **Mechanism**: The hypothetical answer is embedded. Because it matches the tone, vocabulary, and document length of real indexed passages, search recall is dramatically higher.

### 3. Step-Back Prompting
* **Concept**: When a query is too specific (e.g., *"Why did physics experiment X fail under condition Y in 1984?"*), the LLM generates a higher-level, foundational question (*"What are the core physical laws governing phenomenon X?"*).
* **Mechanism**: Retrieves both the foundational concepts and the specific details to enable holistic reasoning.

### 4. Sub-Query Decomposition
* **Concept**: Deconstructs complex, multi-hop queries into independent sub-questions:
  * *Complex Query*: *"How does LangChain's memory management compare to LlamaIndex's chat engine?"*
  * *Sub-Query 1*: *"How does LangChain manage conversation memory?"*
  * *Sub-Query 2*: *"How does LlamaIndex implement chat engines?"*

---

## ⚙️ LangChain Code Implementation Reference

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# 1. Multi-Query Expansion Generator
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Generate 3 distinct versions of the given question 
to retrieve relevant documents from a vector database. Provide each question on a new line.

Original Question: {question}
Alternate Questions:"""
)
multi_query_chain = multi_query_prompt | llm | StrOutputParser()

# 2. HyDE (Hypothetical Document Embedding) Generator
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Write a detailed, technical passage answering the question below. 
Do not include conversational filler; write directly as if excerpted from an authoritative manual.

Question: {question}
Passage:"""
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

# Example Test
query = "What causes Chroma vector database search latency to spike?"
print("--- HyDE Generated Document ---")
print(hyde_chain.invoke({"question": query}))
```

---

## 🎯 Top 10 Technical Interview Questions & Answers

### Q1: How does HyDE (Hypothetical Document Embeddings) solve the "Asymmetric Search" problem?
**Answer**:
In search, user queries are short, question-oriented sentences (e.g., *"What is gradient descent?"*), while document chunks are long, factual assertions (e.g., *"Gradient descent is a first-order iterative optimization algorithm..."*).
Because of this stylistic and structural asymmetry, raw query embeddings may not align tightly with factual chunks. HyDE converts the question into a synthetic factual passage before embedding, restoring structural and stylistic symmetry in the vector space.

---

### Q2: What are the primary failure modes of HyDE and when should you avoid using it?
**Answer**:
1. **Unfamiliar / Niche Proprietary Domains**: If the LLM has zero pre-trained knowledge about a proprietary codebase or secret corporate policy, its hypothetical answer will hallucinate completely incorrect concepts and misleading vocabulary, causing vector search to retrieve irrelevant chunks.
2. **Extra Latency**: Requires a full LLM generation call before retrieval can even start (~200-500ms overhead).

---

### Q3: What is Step-Back Prompting and how does it prevent LLMs from getting lost in details?
**Answer**:
Step-Back Prompting prompts an LLM to abstract a specific question into its underlying foundational principles. By retrieving context for both the high-level concept and the specific question, the final generation has both the overarching theoretical rules and the specific factual details needed to avoid reasoning errors.

---

### Q4: How does Multi-Query Retrieval differ from standard Top-K retrieval?
**Answer**:
Standard Top-K relies on a single vector probe from one point in vector space. Multi-Query expands the prompt into multiple diverse vectors, firing queries from different semantic directions into the database, dramatically increasing retrieval recall across varied phrasings. Results are aggregated and deduplicated via Reciprocal Rank Fusion (RRF).

---

### Q5: How do you handle conversational coreference resolution in multi-turn RAG (e.g., "What about its pricing?")?
**Answer**:
Use a **Query Condensation / History-Aware Rewriter**:
Pass the recent chat history and the latest user turn into an LLM prompt:
*"Given the chat history and latest user query, rewrite the query into a standalone question that can be understood without the conversation history."*
The rewritten query (*"What is the enterprise pricing for Snowflake Data Cloud?"*) is then passed to the retriever.
