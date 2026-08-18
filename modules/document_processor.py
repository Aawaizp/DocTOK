import os

from modules.pdf_loader import load_pdf
from modules.text_chunker import chunk_documents
from modules.vector_store import store_documents


UPLOAD_DIR = "uploaded_pdfs"


def process_pdf(uploaded_file):
    """
    Save, load, chunk and index one PDF.
    """

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_pdf(file_path)

    chunks = chunk_documents(documents)

    added = store_documents(chunks)

    return {
        "filename": uploaded_file.name,
        "pages": len(documents),
        "chunks": len(chunks),
        "indexed": added
    }


def process_pdfs(uploaded_files):
    """
    Save, load, chunk and index multiple PDFs.

    Each file is processed independently via process_pdf, so a failure
    on one file doesn't prevent the others from indexing. Returns a list
    of per-file result dicts in the same order as uploaded_files, with an
    "error" key set instead of the usual fields if that file failed.
    """

    results = []

    for uploaded_file in uploaded_files:
        try:
            info = process_pdf(uploaded_file)
        except Exception as e:
            info = {
                "filename": uploaded_file.name,
                "error": str(e)
            }
        results.append(info)

    return results