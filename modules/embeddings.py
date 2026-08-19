import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    Load the local embedding model once and reuse it across reruns.
    Re-loading model weights on every question was adding real latency --
    cached as a Streamlit resource so it only loads once per server process.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)