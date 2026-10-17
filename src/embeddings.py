"""Embedding model setup.

Anthropic does not offer an embeddings endpoint, so we use a small open-source
sentence-transformers model that runs locally. It is downloaded once from
Hugging Face on first use and needs no API key.
"""

import os
from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Return a (cached) embedding model.

    The model name comes from the argument, then the EMBEDDING_MODEL
    environment variable, then the default.
    """
    name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=name,
        encode_kwargs={"normalize_embeddings": True},
    )
