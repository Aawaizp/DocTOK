from modules.vector_store import get_vector_store


def retrieve_documents(query, k=5):
    """
    Retrieve the most relevant chunks.
    """

    db = get_vector_store()

    return db.similarity_search(
        query=query,
        k=k
    )