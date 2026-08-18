import shutil

from langchain_chroma import Chroma

from modules.embeddings import get_embedding_model
from config import CHROMA_PATH


def get_vector_store():
    try:
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_model(),
            # Forces cosine distance so similarity_search_with_relevance_scores
            # returns properly normalized 0-1 scores. Without this, Chroma
            # defaults to L2 distance and the relevance-score conversion
            # produces negative/unbounded garbage.
            # NOTE: only applies when the collection is first created --
            # delete chroma_db/ and re-index after changing this.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        # Corrupted/incompatible on-disk DB — wipe and rebuild rather than crash.
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

    new_chunks = []
    new_ids = []

    for chunk in chunks:

        chunk_id = chunk.metadata["chunk_id"]

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        db.add_documents(
            documents=new_chunks,
            ids=new_ids
        )

    return len(new_chunks)


def list_indexed_filenames():
    """
    Return a sorted list of distinct `filename` values currently indexed
    in Chroma. Powers the multi-PDF selector in the UI -- reads directly
    from collection metadata rather than tracking filenames separately,
    so it stays accurate across sessions (Chroma is persisted to disk).
    """

    db = get_vector_store()

    try:
        data = db.get(include=["metadatas"])
    except Exception:
        return []

    metadatas = data.get("metadatas") or []

    filenames = {
        m.get("filename")
        for m in metadatas
        if m and m.get("filename")
    }

    return sorted(filenames)