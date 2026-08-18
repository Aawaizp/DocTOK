import os

from groq import Groq

from config import TOP_K, MODEL_NAME
from modules.vector_store import get_vector_store


# Minimum relevance score required for a chunk to be used.
# Higher = stricter retrieval.
SCORE_THRESHOLD = 0.35


# Query rewriting only needs *a* working key, not the full rotation logic
# from rag_chain.py -- grab the first available GROQ_API_KEY_* from the
# environment.
_api_key = next(
    (v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v),
    None
)
if not _api_key:
    raise RuntimeError("No GROQ_API_KEY_* found in environment for query rewriting.")

_client = Groq(api_key=_api_key)


def rewrite_query(question: str) -> str:
    """
    Rewrite a user's question into a clear search query.

    If rewriting fails, use the original question.
    """

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following as a clear search query. "
                        "Preserve the original meaning and intent. "
                        "Do not add information. "
                        "Return only the rewritten query.\n\n"
                        f"Question: {question}"
                    ),
                }
            ],
        )

        rewritten = response.choices[0].message.content.strip()

        return rewritten if rewritten else question

    except Exception:
        # Retrieval should still work if query rewriting fails.
        return question


def _build_filename_filter(filenames):
    """
    Build a Chroma `where` filter scoping search to specific filenames.
    Returns None if no scoping is needed (search the whole collection).
    """

    if not filenames:
        return None

    filenames = list(filenames)

    if len(filenames) == 1:
        return {"filename": filenames[0]}

    return {"filename": {"$in": filenames}}


def retrieve_documents(
    query: str,
    k: int | None = None,
    score_threshold: float = SCORE_THRESHOLD,
    filenames: list[str] | None = None,
):
    """
    Retrieve relevant document chunks.

    Steps:
    1. Rewrite the user's question.
    2. Search ChromaDB using semantic similarity, optionally scoped to
       `filenames` (metadata filter applied at the vector-search level,
       not after the fact -- cheaper and avoids wasting the k budget on
       chunks from PDFs the user didn't select).
    3. Calculate relevance scores.
    4. Remove chunks below the relevance threshold.
    5. Return only Document objects.
    """

    # Step 1: Improve the search query
    search_query = rewrite_query(query)

    # Step 2: Get ChromaDB
    db = get_vector_store()

    # Step 3: Use configured TOP_K unless caller provides another value
    if k is None:
        k = TOP_K

    where_filter = _build_filename_filter(filenames)

    # Step 4: Semantic search with relevance scores, scoped by filename
    results = db.similarity_search_with_relevance_scores(
        query=search_query,
        k=k,
        filter=where_filter,
    )

    # Step 5: Keep only relevant chunks
    filtered_results = [
        (doc, score)
        for doc, score in results
        if score >= score_threshold
    ]

    # Step 6: Print scores for debugging/tuning
    print("\n--- Retrieval Results ---")
    print(f"Original query: {query}")
    print(f"Search query:   {search_query}")
    print(f"Filename filter: {filenames if filenames else 'ALL'}")

    for doc, score in filtered_results:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        print(
            f"Page {page} | "
            f"Score: {score:.3f} | "
            f"{filename}"
        )

    print(f"Retrieved: {len(filtered_results)} / {len(results)} chunks")
    print("-------------------------\n")

    # Important:
    # Return only Document objects because rag_chain.py
    # currently expects doc.page_content and doc.metadata.
    return [
        doc
        for doc, score in filtered_results
    ]