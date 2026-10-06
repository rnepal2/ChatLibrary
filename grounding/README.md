# grounding — reusable corrective-RAG machinery

Extracted from the ChatLibrary corrective-RAG pipeline (Sept 2024). Unlike
the original `src/`, this package has **no dependency on Azure OpenAI,
ChromaDB, LangGraph, or Streamlit** — it takes any chat model that supports
structured output and any retriever/generator functions you supply.

## What it does

A corrective-RAG loop with LLM-based graders at every decision point:

1. **Retrieve** candidates via your `retrieve(question)` function.
2. **Grade** each document's relevance to the question; keep only the relevant ones.
3. If none survive, **rewrite the query** and re-retrieve (up to `max_rewrites`).
4. **Generate** an answer from the surviving documents.
5. **Check** the answer is grounded in the documents (hallucination grader)
   and that it actually addresses the question (answer grader).
6. Regenerate or rewrite-and-retry on failure; fall back to a default
   "cannot answer from the library" response when retries are exhausted.

Every run returns a `RAGResult` with the answer, backing documents, whether
the answer was library-backed, a retry count, and an audit trail.

## Usage

```python
from langchain_openai import ChatOpenAI
from grounding import CorrectiveRAG, FeedbackLog

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def retrieve(question):
    # your vector store lookup; documents need .page_content and .metadata
    return my_collection.similarity_search(question, k=5)

def generate(question, documents):
    context = "\n\n".join(d.page_content for d in documents)
    return llm.invoke(
        f"Answer the question based only on these documents.\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}"
    ).content

rag = CorrectiveRAG(llm, retrieve=retrieve, generate=generate)
result = rag.run("What are the prompt-engineering guidelines?")

print(result.answer)
print("backed by library:", result.backed_by_library)
for ref in result.documents:
    print("-", ref.metadata.get("filepath"), ref.metadata.get("split_id"))

# Optional: log the exchange and collect feedback (star ratings, comments)
log = FeedbackLog("logs/userlog.parquet")
chat_id = log.record(result.answer and "What are the prompt-engineering guidelines?",
                     result.answer, username="rabindra",
                     references=[d.metadata for d in result.documents])
log.update_feedback(chat_id, stars=5, comment="Precise, well-cited.")
```

## Module layout

- `graders.py` — structured-output graders: question router, document
  relevance, hallucination check, answer-quality check, question rewriter.
- `loop.py` — `CorrectiveRAG` orchestrator and `RAGResult`.
- `feedback.py` — parquet-backed conversation + star-rating feedback log.

## Dependencies

`langchain-core`, `pydantic` (+ `pandas` and `pyarrow` only if you use
`FeedbackLog`).
