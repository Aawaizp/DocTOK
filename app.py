import streamlit as st

from modules.rag_chain import ask_llm, NOT_FOUND_MESSAGE
from modules.document_processor import process_pdfs
from modules.vector_store import list_indexed_filenames

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocTOK",
    page_icon="📄",
    layout="wide"
)
st.title("📄 DocTOK")

# Max distinct sources to show per answer -- retrieval may pull several
# chunks to build good context, but a long source list for a short
# answer looks noisy. Chroma returns results ranked by relevance, so
# capping here keeps the most relevant ones.
MAX_SOURCES_SHOWN = 4

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    # Tracks which files have already been indexed this session,
    # so we don't re-run indexing on every Streamlit rerun.
    st.session_state.processed_files = set()


def render_sources(sources):
    """Render a deduped, capped list of (filename, page) source citations."""
    displayed = []
    seen = set()
    for doc in sources:
        source = (doc.metadata["filename"], doc.metadata["page"] + 1)
        if source not in seen:
            seen.add(source)
            displayed.append(source)
        if len(displayed) >= MAX_SOURCES_SHOWN:
            break

    st.markdown("### 📚 Sources")
    for filename, page in displayed:
        st.markdown(f"- **{filename}** — Page {page}")

    remaining = len({
        (doc.metadata["filename"], doc.metadata["page"] + 1) for doc in sources
    }) - len(displayed)
    if remaining > 0:
        st.caption(f"+ {remaining} more source{'s' if remaining != 1 else ''}")


# ── File upload + indexing (multi-PDF) ───────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_files
    ]

    if new_files:
        with st.spinner(f"Processing {len(new_files)} file(s)..."):
            results = process_pdfs(new_files)

        for info in results:
            st.session_state.processed_files.add(info["filename"])

            if info.get("error"):
                st.error(f"❌ {info['filename']} failed to index: {info['error']}")
            elif info["indexed"] > 0:
                st.success(f"✅ {info['filename']} indexed successfully.")
            else:
                st.info(f"📚 {info['filename']} is already indexed.")

# ── PDF selector (query all or a subset of indexed documents) ────────────
all_filenames = list_indexed_filenames()

if all_filenames:
    selected_filenames = st.multiselect(
        "Choose PDFs to query",
        options=all_filenames,
        default=all_filenames,
        help="Search across all uploaded PDFs, or narrow to specific ones."
    )

    # ── Render chat history (including past sources) ─────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Only show sources if this answer wasn't a "not found" response.
            if (
                message["role"] == "assistant"
                and message.get("sources")
                and message["content"] != NOT_FOUND_MESSAGE
            ):
                render_sources(message["sources"])

    # ── Chat input ─────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your document(s)...")

    if query:
        if not selected_filenames:
            st.warning("Select at least one PDF above to ask a question.")
            st.stop()

        # Show + store user message
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        try:
            with st.spinner("Thinking..."):
                result = ask_llm(query, filenames=selected_filenames)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        answer = result["answer"]
        sources = result["sources"]

        with st.chat_message("assistant"):
            st.markdown(answer)

            # Only show sources if the model actually answered from context.
            if sources and answer != NOT_FOUND_MESSAGE:
                render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
else:
    st.info("👆 Upload one or more PDFs to get started.")