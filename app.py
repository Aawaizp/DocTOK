import streamlit as st
import os
from modules.pdf_loader import load_pdf
from modules.text_chunker import chunk_documents
from modules.embeddings import get_embedding_model
from modules.vector_store import get_vector_store
from modules.retriever import retrieve_documents
from modules.vector_store import store_documents
from modules.rag_chain import ask_llm
st.title("📄 Doctok")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    os.makedirs("uploaded_pdfs", exist_ok=True)

    with open(f"uploaded_pdfs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF saved successfully!")


    documents = load_pdf(f"uploaded_pdfs/{uploaded_file.name}")

    chunks = chunk_documents(documents)

    st.write("Total Chunks:", len(chunks))

    vector_db = get_vector_store()

    added = store_documents(chunks)

    st.success(f"{added} new chunks stored.")

    st.write(chunks[0].metadata)

    query = st.text_input("Ask a question")

    if query:
        answer = ask_llm(query)

        st.subheader("Answer")

        st.write(answer)

# How should team members join?