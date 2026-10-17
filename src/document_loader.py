"""Load PDF files into LangChain Document objects."""

from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

PathLike = Union[str, Path]


def load_pdf(file_path: PathLike, source_name: Optional[str] = None) -> List[Document]:
    """Load a single PDF, returning one Document per page.

    Args:
        file_path: Path to the PDF on disk.
        source_name: Name to record in metadata (useful for uploaded files that
            are stored under a temporary path). Defaults to the file name.

    Each Document gets ``source`` (file name) and ``page`` (1-based) metadata so
    answers can cite where the information came from.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    pages = PyPDFLoader(str(path)).load()
    name = source_name or path.name

    for page in pages:
        page.metadata["source"] = name
        # PyPDFLoader pages are 0-based; humans count from 1.
        page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

    # Drop pages with no extractable text (e.g. scanned images without OCR).
    return [p for p in pages if p.page_content.strip()]


def load_directory(directory: PathLike) -> List[Document]:
    """Load every PDF found directly inside ``directory``."""
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    documents: List[Document] = []
    for pdf in sorted(folder.glob("*.pdf")):
        documents.extend(load_pdf(pdf))
    return documents
