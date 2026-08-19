import os

from groq import Groq
from langchain_core.documents import Document

from config import TOP_K, MODEL_NAME
from modules.vector_store import get_vector_store

SCORE_THRESHOLD = 0.35
REWRITE_WORD_THRESHOLD = 6
SUMMARY_K = 8

DEBUG = os.getenv("DOCTOK_DEBUG", "false").lower() == "true"

SUMMARY_KEYWORDS = (
    "summarize", "summary", "key points", "main points", "main steps",
    "overview", "main idea", "tl;dr", "what is this document about",
    "what is this about", "steps", "outline", "table of contents",
    "walk me through", "explain the document", "what does this cover",
)

_api_key = next(
    (v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v),
    None
)
if not _api_key:
    raise RuntimeError("No GROQ_API_KEY_* found in environment for query rewriting.")

_client = Groq(api_key=_api_key)


def rewrite_query(question: str) -> str:
    if len(question.split()) >= REWRITE_WORD_THRESHOLD:
        return question
    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the following as a clear search query. "
                    "Preserve the original meaning and intent. "
                    "Do not add information. Return only the rewritten query.\n\n"
                    f"Question: {question}"
                ),
            }],
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def _build_filename_filter(filenames):
    if not filenames:
        return None
    filenames = list(filenames)
    if len(filenames) == 1:
        return {"filename": filenames[0]}
    return {"filename": {"$in": filenames}}


def _is_summary_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SUMMARY_KEYWORDS)


def _get_summary_chunks(filenames, k=SUMMARY_K):
    db = get_vector_store()
    where = _build_filename_filter(filenames)

    data = db.get(where=where, include=["metadatas", "documents"])
    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    by_file = {}
    for meta, text in zip(metadatas, documents):
        fname = meta.get("filename", "unknown")
        by_file.setdefault(fname, []).append((meta, text))

    for fname in by_file:
        by_file[fname].sort(key=lambda item: item[0].get("chunk_index", 0))

    files = list(by_file.keys()) or ["unknown"]
    per_file = max(1, k // len(files))

    results = []
    for fname in files:
        for meta, text in by_file.get(fname, [])[:per_file]:
            results.append(Document(page_content=text, metadata=meta))

    return results


def retrieve_documents(
    query: str,
    k: int | None = None,
    score_threshold: float = SCORE_THRESHOLD,
    filenames: list[str] | None = None,
):
    if k is None:
        k = TOP_K

    if _is_summary_query(query):
        docs = _get_summary_chunks(filenames, k=SUMMARY_K)
        if DEBUG:
            print(f"[retrieval:summary] {len(docs)} chunks | filter={filenames or 'ALL'}")
        return docs

    search_query = rewrite_query(query)
    db = get_vector_store()
    where_filter = _build_filename_filter(filenames)

    results = db.similarity_search_with_relevance_scores(
        query=search_query, k=k, filter=where_filter,
    )

    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

    if DEBUG:
        print(f"[retrieval] query={query!r} -> {search_query!r} | filter={filenames or 'ALL'} "
              f"| {len(filtered_results)}/{len(results)} passed threshold")
        for doc, score in filtered_results:
            print(f"    p.{doc.metadata.get('page', 0) + 1} score={score:.3f} {doc.metadata.get('filename')}")

    if filtered_results:
        return [doc for doc, score in filtered_results]

    if DEBUG:
        print("[retrieval] no chunks passed threshold -- falling back to summary chunks")
    return _get_summary_chunks(filenames, k=SUMMARY_K)