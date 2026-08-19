import shutil

import streamlit as st
from langchain_chroma import Chroma

from modules.embeddings import get_embedding_model
from config import CHROMA_PATH


@st.cache_resource(show_spinner=False)
def get_vector_store():
    try:
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_model(),
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_model(),
            collection_metadata={"hnsw:space": "cosine"},
        )


def store_documents(chunks):
    db = get_vector_store()
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    existing = db.get(ids=ids)
    existing_ids = set(existing["ids"])

    new_chunks, new_ids = [], []
    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        db.add_documents(documents=new_chunks, ids=new_ids)

    return len(new_chunks)


def list_indexed_filenames():
    db = get_vector_store()
    try:
        data = db.get(include=["metadatas"])
    except Exception:
        return []
    metadatas = data.get("metadatas") or []
    return sorted({m.get("filename") for m in metadatas if m and m.get("filename")})


def delete_file(filename: str):
    """Remove every chunk belonging to `filename` from the Chroma collection."""
    db = get_vector_store()
    db._collection.delete(where={"filename": filename})