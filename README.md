# AI-MATH-AGENT

**Math Agentic RAG** is a Human-in-the-Loop, Agentic Retrieval-Augmented Generation system designed for solving mathematical questions using domain-specific datasets like GSM8K, MathQA, MathQSA, and ORCA-Math. It features robust fallback logic, web search integration, real-time feedback, and alternative embedding/backends for flexible deployment across environments.

---

## 🚀 Features

- 📚 Multi-source math knowledge base (GSM8K, MathQA, MathQSA, ORCA)
- 🔎 Semantic search using `SentenceTransformers` and FAISS
- 🧩 RAG pipeline with LangChain-style retriever + LLM
- 🌐 Web search fallback (Exa API optional)
- 🔁 Feedback loop for model correction and improvement


---
## 🧪 Datasets Used

| Dataset    | Format        | Description                             |
|------------|----------------|-----------------------------------------|
| GSM8K      | Parquet        | Grade school math questions             |
| MathQA     | JSON           | Math word problems and rationales       |
| MathQSA    | CSV            | Annotated short answer math questions   |
| ORCA-Math  | Parquet        | High-quality math QA corpus             |

---
## Model Stack 
sentence-transformers for embeddings

faiss for vector search

LangChain + custom logic for RAG

Optional exa API for web fallback
---
export EXA_API_KEY=your_exa_api_key (link: https://exa.ai)
---
## Developed By
Nanduvamsikrishna Yanamandala
