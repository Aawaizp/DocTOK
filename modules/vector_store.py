from langchain_chroma import Chroma

from modules.embeddings import get_embedding_model


CHROMA_PATH = "chroma_db"


def get_vector_store():

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model()
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