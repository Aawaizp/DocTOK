import os
import streamlit as st

# On Streamlit Cloud, keys live in st.secrets (dashboard-configured), not a
# .env file. Mirror them into os.environ before importing any module that
# reads GROQ_API_KEY_* at import time -- keeps local (.env) and hosted
# (secrets.toml) behavior identical with zero code branching downstream.
try:
    for _key, _value in st.secrets.items():
        if _key.startswith("GROQ_API_KEY") and _value:
            os.environ[_key] = str(_value)
except Exception:
    pass  # no secrets.toml locally -- .env already populated os.environ

from modules.rag_chain import ask_llm_stream, NOT_FOUND_MESSAGE
from modules.document_processor import process_pdfs
from modules.vector_store import list_indexed_filenames, delete_file

st.set_page_config(page_title="DocTOK", page_icon="📄", layout="wide")

MAX_SOURCES_SHOWN = 4
EXAMPLE_QUESTIONS = ["Summarize this document", "What are the key points?", "List the main steps"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

:root {
    --accent: #1E8E3E;
    --accent-dark: #17752F;
    --accent-light: #E6F4EA;
    --text: #202124;
    --text-secondary: #5F6368;
    --border: #DADCE0;
}

html, body, [class*="css"], p, span, div, label { font-family: 'Roboto', Arial, sans-serif; color: var(--text); }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #FFFFFF !important; }
.main .block-container { max-width: 900px; padding-top: 1.5rem; }

h1 { font-weight: 500; font-size: 1.6rem; margin-bottom: 0; }
.subtitle { color: var(--text-secondary); font-size: 0.88rem; margin: 0; }

/* Buttons -- base style. No forced nowrap/ellipsis here; that was
   clipping normal button labels. Ellipsis is applied separately, only
   to the file-chip buttons that actually need truncation. */
div[data-testid="stButton"] button {
    border-radius: 6px;
    font-size: 0.84rem;
    min-height: 34px;
    padding: 4px 14px;
    line-height: 1.2;
    box-shadow: none;
}

button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
}

button[kind="primary"] {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent) !important;
}
button[kind="primary"]:hover { background: var(--accent-dark) !important; }

/* File chip buttons: single line, ellipsis for long filenames only */
.file-chip-marker ~ div[data-testid="stButton"] button {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: left;
    display: block;
}

/* Red "danger" button, scoped to its own container via marker */
.danger-btn-marker ~ div[data-testid="stButton"] button {
    background: #FFFFFF !important;
    color: #D93025 !important;
    border: 1px solid #D93025 !important;
}
.danger-btn-marker ~ div[data-testid="stButton"] button:hover {
    background: #FCE8E6 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 6px;
    background: #FAFAFA;
    padding: 10px 14px;
    border: 1px solid var(--border);
    margin-bottom: 8px;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { border-radius: 6px !important; }

.source-badge {
    display: inline-block; background: #FFFFFF; color: var(--accent);
    border: 1px solid var(--accent); border-radius: 6px; padding: 4px 10px;
    margin: 3px 6px 3px 0; font-size: 0.76rem; font-weight: 500;
}
.more-sources { color: var(--text-secondary); font-size: 0.76rem; margin-left: 2px; }

section[data-testid="stSidebar"] { background: #FAFAFA !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem; font-weight: 500; color: var(--text-secondary) !important;
    text-transform: uppercase; letter-spacing: 0.04em; margin-top: 1.2rem; margin-bottom: 0.4rem;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    background: #FFFFFF !important;
}

.stat-pill {
    display: inline-block; background: var(--accent-light); color: var(--accent) !important;
    border-radius: 4px; padding: 4px 10px; font-size: 0.76rem; font-weight: 500; margin: 4px 0 2px 0;
}

div[data-testid="stChatInput"] textarea { border-radius: 6px; }
hr { border-color: var(--border); margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = {}
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

_indexed_now = list_indexed_filenames()

col_title, col_clear = st.columns([4, 1.4])
with col_title:
    st.markdown("# 📄 DocTOK")
    st.markdown('<p class="subtitle">Ask questions across your uploaded PDFs</p>', unsafe_allow_html=True)
with col_clear:
    if _indexed_now and st.session_state.messages:
        st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="danger-btn-marker"></div>', unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_sources(sources):
    seen, ordered = set(), []
    for doc in sources:
        s = (doc.metadata["filename"], doc.metadata["page"] + 1)
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    shown = ordered[:MAX_SOURCES_SHOWN]
    remaining = len(ordered) - len(shown)
    badges = "".join(f'<span class="source-badge">{fn} · p.{pg}</span>' for fn, pg in shown)
    if remaining > 0:
        badges += f'<span class="more-sources">+{remaining} more</span>'
    st.markdown(badges, unsafe_allow_html=True)


def run_query(query, selected_filenames):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Thinking..."):
            try:
                stream, sources = ask_llm_stream(query, filenames=selected_filenames)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if stream is None:
            st.markdown(NOT_FOUND_MESSAGE)
            answer = NOT_FOUND_MESSAGE
        else:
            answer = st.write_stream(stream)
            if sources:
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


with st.sidebar:
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed", key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)..."):
                results = process_pdfs(new_files)
            for info in results:
                st.session_state.processed_files.add(info["filename"])
                if info.get("error"):
                    st.toast(f"❌ {info['filename']} failed", icon="⚠️")
                else:
                    st.session_state.selected_files[info["filename"]] = True
                    st.toast(f"Indexed {info['filename']}", icon="✅")
            st.session_state.uploader_key += 1
            st.rerun()

    all_filenames = list_indexed_filenames()

    for fname in list(st.session_state.selected_files.keys()):
        if fname not in all_filenames:
            del st.session_state.selected_files[fname]
    for fname in all_filenames:
        st.session_state.selected_files.setdefault(fname, True)

    if all_filenames:
        st.markdown(f'<span class="stat-pill">{len(all_filenames)} file(s) indexed</span>', unsafe_allow_html=True)
        st.markdown("### Search scope")

        for fname in all_filenames:
            is_selected = st.session_state.selected_files[fname]
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown('<div class="file-chip-marker"></div>', unsafe_allow_html=True)
                label = f"✓  {fname}" if is_selected else fname
                if st.button(
                    label, key=f"toggle_{fname}", use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.selected_files[fname] = not is_selected
                    st.rerun()
            with c2:
                if st.button("✕", key=f"delete_{fname}", use_container_width=True, help=f"Remove {fname}"):
                    delete_file(fname)
                    st.session_state.processed_files.discard(fname)
                    st.session_state.selected_files.pop(fname, None)
                    st.toast(f"Removed {fname}", icon="🗑️")
                    st.rerun()

selected_filenames = [f for f, sel in st.session_state.selected_files.items() if sel]

if not all_filenames:
    st.info("Upload one or more PDFs from the sidebar to get started.")
else:
    if not st.session_state.messages:
        st.markdown("##### Try asking")
        cols = st.columns(len(EXAMPLE_QUESTIONS) + 1)
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            with col:
                if st.button(q, use_container_width=False):
                    st.session_state.pending_query = q
                    st.rerun()

    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "📄"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if (
                message["role"] == "assistant"
                and message.get("sources")
                and message["content"] != NOT_FOUND_MESSAGE
            ):
                render_sources(message["sources"])

    query = st.chat_input("Ask anything about your document(s)...")

    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    if query:
        if not selected_filenames:
            st.warning("Select at least one PDF in the sidebar to ask a question.")
            st.stop()
        run_query(query, selected_filenames)