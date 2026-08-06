from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF and return one Document object per page.
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()