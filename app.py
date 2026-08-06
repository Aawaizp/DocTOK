import streamlit as st
import os
from modules.pdf_loader import load_pdf
from modules.text_chunker import chunk_documents
from modules.embeddings import get_embedding_model
from modules.vector_store import get_vector_store
from modules.retriever import retrieve_documents
from modules.vector_store import store_documents
from modules.rag_chain import ask_llm

if "messages" not in st.session_state:
    st.session_state.messages = []
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

    

    vector_db = get_vector_store()

    added = store_documents(chunks)

    st.success(f"{added} new chunks stored.")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything about your document...")

    if query:

        # Show user message
        st.chat_message("user").markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Generate answer
        answer = ask_llm(query)

        # Show assistant message
        st.chat_message("assistant").markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
# How should team members join?