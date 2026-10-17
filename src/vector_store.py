"""FAISS vector store helpers: build, save and load an index."""

from pathlib import Path
from typing import List, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PathLike = Union[str, Path]

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"


def build_vector_store(chunks: List[Document], embeddings: Embeddings) -> FAISS:
    """Embed the chunks and index them in a new in-memory FAISS store."""
    if not chunks:
        raise ValueError(
            "No text chunks to index. The PDF may be scanned images without "
            "selectable text."
        )
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(store: FAISS, index_dir: PathLike = DEFAULT_INDEX_DIR) -> Path:
    """Persist the index to disk so it does not need to be rebuilt."""
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return path


def load_vector_store(
    embeddings: Embeddings, index_dir: PathLike = DEFAULT_INDEX_DIR
) -> FAISS:
    """Load a previously saved index.

    FAISS stores metadata with pickle, so only load indexes that this project
    created itself (``allow_dangerous_deserialization`` acknowledges that).
    """
    path = Path(index_dir)
    if not (path / "index.faiss").exists():
        raise FileNotFoundError(f"No saved index found in {path}")
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )
