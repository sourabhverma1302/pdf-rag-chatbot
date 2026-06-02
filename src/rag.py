from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    # Newer LangChain split-out package
    from langchain_chroma import Chroma  # type: ignore
except ModuleNotFoundError:
    # Backwards-compatible import (deprecated upstream, but still works)
    from langchain_community.vectorstores import Chroma  # type: ignore
from langchain_core.documents import Document
from dotenv import load_dotenv
import time
import uuid
load_dotenv()
    
DB_DIR = "chroma_db"
PDF_COLLECTION = "pdf_chunks"
MEMORY_COLLECTION = "chat_memory"
PDF_RELEVANCE_K = 4
# Chroma returns a distance score (smaller = more similar). Tune if needed.
PDF_DISTANCE_THRESHOLD = 0.6
MEMORY_HISTORY_LIMIT = 30

def get_vectorstore(collection_name: str):
    embeddings = OpenAIEmbeddings()

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

def get_memory_history(session_id: str, limit: int = MEMORY_HISTORY_LIMIT) -> list[tuple[str, str]]:
    """
    Returns recent (question, answer) pairs from the persisted chat memory.
    Best-effort: uses Chroma's underlying collection API when available.
    """
    memory_db = get_vectorstore(MEMORY_COLLECTION)

    # Try to fetch raw docs + metadata so we can sort by time (filtered to this session).
    try:
        raw = memory_db._collection.get(  # type: ignore[attr-defined]
            include=["documents", "metadatas"],
            where={"session_id": session_id},
        )
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        rows: list[tuple[float, str]] = []
        for doc, meta in zip(docs, metas, strict=False):
            ts = 0.0
            if isinstance(meta, dict):
                ts = float(meta.get("ts", 0.0) or 0.0)
            if isinstance(doc, str):
                rows.append((ts, doc))

        rows.sort(key=lambda x: x[0], reverse=True)
        rows = rows[: max(0, limit)]

        pairs: list[tuple[str, str]] = []
        for _ts, text in rows:
            q = ""
            a = ""
            for line in text.splitlines():
                if line.startswith("Question:"):
                    q = line.replace("Question:", "", 1).strip()
                elif line.startswith("Answer:"):
                    a = line.replace("Answer:", "", 1).strip()
            if q or a:
                pairs.append((q, a))
        return pairs
    except Exception:
        # Fallback: no history (avoid breaking the app)
        return []

def query_rag(question: str, session_id: str):
    pdf_db = get_vectorstore(PDF_COLLECTION)
    memory_db = get_vectorstore(MEMORY_COLLECTION)

    # --- Retrieve candidates from PDF (with scores if supported) ---
    pdf_docs: list[Document] = []
    pdf_is_relevant = False
    try:
        pdf_with_scores = pdf_db.similarity_search_with_score(question, k=PDF_RELEVANCE_K)
        pdf_docs = [d for d, _score in pdf_with_scores]
        best_distance = min((score for _d, score in pdf_with_scores), default=float("inf"))
        pdf_is_relevant = best_distance <= PDF_DISTANCE_THRESHOLD and len(pdf_docs) > 0
    except Exception:
        # Fallback if the vectorstore doesn't support scoring
        pdf_docs = pdf_db.similarity_search(question, k=PDF_RELEVANCE_K)
        pdf_is_relevant = len(pdf_docs) > 0

    # Only use memory from this session (best-effort; depends on backend support)
    try:
        memory_docs = memory_db.similarity_search(
            question,
            k=4,
            filter={"session_id": session_id},
        )
    except Exception:
        memory_docs = memory_db.similarity_search(question, k=4)

    pdf_context = "\n\n".join([d.page_content for d in pdf_docs])
    memory_context = "\n\n".join([d.page_content for d in memory_docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if pdf_is_relevant:
        prompt = f"""
You are a helpful assistant.

Use the PDF context to answer the user's question as accurately as possible.
If the PDF does not contain enough information, say so clearly and then give a best-effort general answer (do not invent PDF facts).

PDF context:
{pdf_context}

Past Q&A memory (may be empty):
{memory_context}

Question:
{question}
"""
    else:
        prompt = f"""
You are a helpful assistant having a normal conversation.

The PDF context is not relevant to this question, so ignore it.
Use general knowledge and be honest about limitations (for real-time info like current weather, you can't browse).

Past Q&A memory (may be empty):
{memory_context}

Question:
{question}
"""

    answer = llm.invoke(prompt).content

    # Store this Q&A so future questions can retrieve it (retrieval-based "learning")
    memory_db.add_documents(
        [
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={"type": "qa_memory", "ts": time.time(), "session_id": session_id},
                id=str(uuid.uuid4()),
            )
        ]
    )

    return answer