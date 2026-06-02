import streamlit as st
import tempfile
import uuid
from ingest import ingest_pdf
from rag import get_memory_history, query_rag

from dotenv import load_dotenv
load_dotenv()
 


st.set_page_config(page_title="RAG PDF Bot")

st.title("📄 RAG Chatbot with PDF Upload")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("Processing PDF...")

    ingest_pdf(file_path)

    st.success("PDF indexed successfully!")

# Session-only memory/history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.button("New chat (clear session memory)"):
    st.session_state.session_id = str(uuid.uuid4())


# Chat (auto-clear input after submit)
with st.form("qa_form", clear_on_submit=True):
    question = st.text_input("Ask a question from your PDF:")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    answer = query_rag(question, st.session_state.session_id)
    st.write("**You:**", question)
    st.write("**Bot:**", answer)


# Persisted chat memory history (from Chroma)
history = get_memory_history(st.session_state.session_id)
if history:
    st.subheader("Chat history")
    for q, a in reversed(history):
        if q:
            st.write("**You:**", q)
        if a:
            st.write("**Bot:**", a)