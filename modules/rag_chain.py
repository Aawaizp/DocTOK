import os

from dotenv import load_dotenv
from groq import Groq

from modules.retriever import retrieve_documents

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY_2")
)


def ask_llm(question):

    docs = retrieve_documents(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
reply exactly:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content