import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text before chunking.
    """

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    return text.strip()