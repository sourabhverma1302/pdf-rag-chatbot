from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
try:
    # Newer LangChain split-out package
    from langchain_chroma import Chroma  # type: ignore
except ModuleNotFoundError:
    # Backwards-compatible import (deprecated upstream, but still works)
    from langchain_community.vectorstores import Chroma  # type: ignore
import os

DB_DIR = "chroma_db"
PDF_COLLECTION = "pdf_chunks"

def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    print("file_path",file_path)
    print("loader",loader)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()

    # ✅ IMPORTANT: reuse existing DB if it exists
    if os.path.exists(DB_DIR):
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=PDF_COLLECTION
        )

        db.add_documents(chunks)   # append new data
    else:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR,
            collection_name=PDF_COLLECTION
        )

    # Some Chroma versions auto-persist; older ones require persist()
    if hasattr(db, "persist"):
        db.persist()