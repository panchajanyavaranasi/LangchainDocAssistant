"""Retrieval-Augmented Generation chain that answers questions from documents.

Flow: question -> retrieve the most relevant chunks -> give them to Claude as
context -> Claude answers using only that context and cites sources.

Run a quick command-line test from the project root:

    python -m src.rag_chain
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a Smart Document Q&A Assistant that helps employees \
find information in company documents such as policies, technical documentation \
and reports.

Rules:
- Answer ONLY using the context excerpts provided below.
- If the context does not contain the answer, say: "I couldn't find that in the \
uploaded documents." Do not guess or use outside knowledge.
- Be concise and direct. Use bullet points for lists or multi-part answers.
- Mention the source file and page for key facts, like (policy.pdf, p. 3).
- Quote numbers, dates and thresholds exactly as written in the context."""

HUMAN_PROMPT = """Context excerpts:
{context}

Question: {question}"""


@dataclass
class QAResult:
    """An answer plus the document chunks it was based on."""

    answer: str
    sources: List[Document] = field(default_factory=list)

    def source_summaries(self) -> List[Dict[str, str]]:
        """Compact, de-duplicated view of the sources for display."""
        seen, summaries = set(), []
        for doc in self.sources:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            snippet = " ".join(doc.page_content.split())
            summaries.append(
                {
                    "source": str(doc.metadata.get("source", "unknown")),
                    "page": str(doc.metadata.get("page", "?")),
                    "snippet": snippet[:300] + ("..." if len(snippet) > 300 else ""),
                }
            )
        return summaries


def format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into one labelled context block."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


class DocumentQAChain:
    """Question answering over a FAISS vector store using Claude."""

    def __init__(
        self,
        vector_store: FAISS,
        model: Optional[str] = None,
        k: int = 4,
        temperature: float = 0.0,
    ):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": k})
        self.llm = ChatAnthropic(
            model=model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            temperature=temperature,
            max_tokens=1024,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        # LCEL pipeline: prompt -> model -> plain string
        self.chain = prompt | self.llm | StrOutputParser()

    def ask(self, question: str) -> QAResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(
            {"context": format_context(docs), "question": question}
        )
        return QAResult(answer=answer, sources=docs)


def main() -> None:
    """Tiny CLI: index everything in data/ and answer questions interactively."""
    from pathlib import Path

    from dotenv import load_dotenv

    from src.document_loader import load_directory
    from src.embeddings import get_embeddings
    from src.text_splitter import split_documents
    from src.vector_store import build_vector_store

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    print("Loading and indexing documents in data/ ...")
    docs = load_directory(project_root / "data")
    chunks = split_documents(docs)
    store = build_vector_store(chunks, get_embeddings())
    print(f"Indexed {len(docs)} pages as {len(chunks)} chunks.\n")

    qa = DocumentQAChain(store)
    print("Ask a question (or type 'exit').")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        result = qa.ask(question)
        print(f"\n{result.answer}\n")
        for s in result.source_summaries():
            print(f"  - {s['source']}, page {s['page']}")


if __name__ == "__main__":
    main()
