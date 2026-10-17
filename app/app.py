"""Streamlit front end for the Smart Document Q&A Assistant.

Run from the project root:

    streamlit run app/app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Make `src` importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.document_loader import load_pdf  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.rag_chain import DocumentQAChain  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402
from src.vector_store import build_vector_store, save_vector_store  # noqa: E402

SAMPLE_PDF = PROJECT_ROOT / "data" / "sample_policy.pdf"

st.set_page_config(
    page_title="Smart Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def cached_embeddings():
    """Load the embedding model once per server process."""
    return get_embeddings()


def api_key_is_set() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return bool(key) and not key.startswith("your-")


def build_assistant(
    pdfs: List[Tuple[str, Path]], chunk_size: int, chunk_overlap: int, top_k: int
):
    """Load -> split -> embed -> index -> wrap in a QA chain."""
    pages = []
    for display_name, path in pdfs:
        pages.extend(load_pdf(path, source_name=display_name))
    if not pages:
        raise ValueError(
            "No readable text found. Scanned PDFs need OCR before they can be used."
        )

    chunks = split_documents(pages, chunk_size, chunk_overlap)
    store = build_vector_store(chunks, cached_embeddings())
    save_vector_store(store)
    return DocumentQAChain(store, k=top_k), len(pages), len(chunks)


def activate(pdfs, chunk_size, chunk_overlap, top_k, names):
    """Index documents and store the resulting assistant in session state."""
    try:
        with st.spinner("Reading, splitting and indexing your documents..."):
            qa, n_pages, n_chunks = build_assistant(
                pdfs, chunk_size, chunk_overlap, top_k
            )
    except Exception as exc:  # surface any loader / embedding problem to the user
        st.sidebar.error(f"Could not process documents: {exc}")
        return

    st.session_state.qa = qa
    st.session_state.doc_names = names
    st.session_state.stats = (n_pages, n_chunks)
    st.session_state.messages = []
    st.sidebar.success(f"Ready: {n_pages} pages, {n_chunks} chunks indexed.")


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("qa", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_names", [])
st.session_state.setdefault("stats", (0, 0))

# ----------------------------------------------------------------------------
# Sidebar: documents and settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")

    if not api_key_is_set():
        st.error(
            "ANTHROPIC_API_KEY is missing. Add your key to the `.env` file "
            "and restart the app."
        )

    uploads = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )

    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        top_k = st.slider("Chunks retrieved per question", 1, 10, 4)
        st.caption("Changes apply the next time you process documents.")

    process_clicked = st.button(
        "Process uploaded documents",
        type="primary",
        disabled=not uploads or not api_key_is_set(),
        use_container_width=True,
    )
    sample_clicked = st.button(
        "Use sample policy instead",
        disabled=not SAMPLE_PDF.exists() or not api_key_is_set(),
        use_container_width=True,
    )

    if process_clicked and uploads:
        with tempfile.TemporaryDirectory() as tmp:
            saved = []
            for upload in uploads:
                safe_name = Path(upload.name).name
                target = Path(tmp) / safe_name
                target.write_bytes(upload.getbuffer())
                saved.append((safe_name, target))
            activate(saved, chunk_size, chunk_overlap, top_k, [n for n, _ in saved])

    if sample_clicked:
        activate(
            [(SAMPLE_PDF.name, SAMPLE_PDF)],
            chunk_size,
            chunk_overlap,
            top_k,
            [SAMPLE_PDF.name],
        )

    if st.session_state.qa:
        st.divider()
        st.markdown("**Active documents**")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")
        pages, chunks = st.session_state.stats
        st.caption(f"{pages} pages · {chunks} chunks")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ----------------------------------------------------------------------------
# Main area: chat
# ----------------------------------------------------------------------------
st.title("📄 Smart Document Q&A Assistant")
st.caption(
    "Ask questions about your policies, technical documentation and reports. "
    "Answers come only from the documents you provide."
)

if st.session_state.qa is None:
    st.info(
        "👈 Upload PDFs and click **Process uploaded documents**, or try the "
        "**sample policy** to see how it works."
    )
    st.markdown("**Example questions for the sample policy**")
    st.markdown(
        "- How many days of annual leave do employees get?\n"
        "- What is the daily meal allowance during business travel?\n"
        "- How soon must a lost laptop be reported?\n"
        "- Who needs to approve an expense over $2,500?"
    )


def render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']}")
            st.caption(s["snippet"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.qa is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                result = st.session_state.qa.ask(question)
            sources = result.source_summaries()
            st.markdown(result.answer)
            render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")
