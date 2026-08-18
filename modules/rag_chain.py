import os
import itertools
from groq import Groq, APIStatusError, RateLimitError

from modules.retriever import retrieve_documents
from config import MODEL_NAME, TEMPERATURE

NOT_FOUND_MESSAGE = "I couldn't find that information in the uploaded documents."

# Rotate across all available Groq keys instead of relying on just one.
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


def ask_llm(question: str, filenames: list[str] | None = None) -> dict:
    """
    Retrieve relevant chunks for `question`, ask the LLM to answer
    strictly from that context, and return the answer plus source docs.

    `filenames`: optional list of PDF filenames to restrict retrieval
    to (multi-PDF selection from the UI). None/empty means search all
    indexed documents.
    """
    docs = retrieve_documents(question, filenames=filenames)

    if not docs:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = _build_prompt(question, context)

    last_error = None
    # Try each key once in case one is rate-limited or invalid.
    for _ in range(len(_API_KEYS)):
        key = next(_key_cycle)
        client = _clients[key]
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "answer": response.choices[0].message.content,
                "sources": docs,
            }
        except RateLimitError as e:
            last_error = e
            continue  # try next key
        except APIStatusError as e:
            # Auth/model errors won't be fixed by switching keys — surface immediately.
            raise RuntimeError(
                f"Groq API error (status {e.status_code}) with model={MODEL_NAME!r}: {e.message}"
            ) from e

    raise RuntimeError(f"All Groq API keys rate-limited. Last error: {last_error}")