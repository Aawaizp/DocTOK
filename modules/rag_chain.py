import os
import itertools
from groq import Groq, APIStatusError, RateLimitError

from modules.retriever import retrieve_documents
from config import MODEL_NAME, TEMPERATURE

NOT_FOUND_MESSAGE = "I couldn't find that information in the uploaded documents."

_API_KEYS = [
    v for k, v in os.environ.items()
    if k.startswith("GROQ_API_KEY") and v
]
if not _API_KEYS:
    raise RuntimeError("No GROQ_API_KEY_* values found in environment.")

_key_cycle = itertools.cycle(_API_KEYS)
_clients = {key: Groq(api_key=key) for key in _API_KEYS}


def _build_prompt(question: str, context: str) -> str:
    return f"""You are a helpful AI assistant that explains information from documents clearly and thoroughly.

Answer ONLY using the provided context. Do not use outside knowledge.

Write your answer as a well-developed paragraph (or a few short paragraphs
if the question has multiple parts). Explain the concept fully using the
details, examples, and specifics given in the context -- do not just give
a one-line definition when the context supports a fuller explanation.
Avoid restating the question; go straight into the explanation.

If the answer is not available in the context, reply with exactly this
and nothing else:
"{NOT_FOUND_MESSAGE}"

Context:
{context}

Question:
{question}

Answer:"""


def ask_llm_stream(question: str, filenames: list[str] | None = None):
    """
    Retrieve relevant chunks for `question` and stream the LLM's answer
    token-by-token instead of waiting for the full response.

    Returns (stream_or_None, sources):
    - If no relevant docs are found, returns (None, []) -- caller should
      display NOT_FOUND_MESSAGE directly, no LLM call is made.
    - Otherwise returns a generator yielding text chunks, plus the source
      docs used for context (for citation display).
    """
    docs = retrieve_documents(question, filenames=filenames)

    if not docs:
        return None, []

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = _build_prompt(question, context)

    last_error = None
    for _ in range(len(_API_KEYS)):
        key = next(_key_cycle)
        client = _clients[key]
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            def _token_generator(stream=stream):
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            return _token_generator(), docs

        except RateLimitError as e:
            last_error = e
            continue
        except APIStatusError as e:
            raise RuntimeError(
                f"Groq API error (status {e.status_code}) with model={MODEL_NAME!r}: {e.message}"
            ) from e

    raise RuntimeError(f"All Groq API keys rate-limited. Last error: {last_error}")


def ask_llm(question: str, filenames: list[str] | None = None) -> dict:
    """
    Non-streaming convenience wrapper around ask_llm_stream, kept for any
    callers that want the full answer as a single string (e.g. scripts,
    tests) rather than a token stream.
    """
    stream, docs = ask_llm_stream(question, filenames=filenames)

    if stream is None:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    full_answer = "".join(stream)

    return {"answer": full_answer, "sources": docs}