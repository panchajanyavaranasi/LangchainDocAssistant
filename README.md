# Smart Document Q&A Assistant using LangChain

An AI assistant that reads your documents and answers questions using only the information found in them, with the source file and page cited for every answer.

## The problem

Employees spend too much time searching through lengthy PDFs, company policies, technical documentation and reports to find one specific fact ("How many days of leave do I get?", "What's the expense limit for hotels?").

## The solution

Upload your PDFs, ask questions in plain English, and get a direct answer grounded in the documents. This uses **Retrieval-Augmented Generation (RAG)**:

```
 PDFs ──► load ──► split into chunks ──► embed ──► FAISS index
                                                       │
 Question ──► embed ──► find most similar chunks ◄─────┘
                               │
                               ▼
                  Claude answers using ONLY those chunks
                               │
                               ▼
                   Answer + source file and page
```

## Project structure

```
langchain-document-assistant/
├── data/
│   └── sample_policy.pdf      # Fictional company policy for trying it out
├── src/
│   ├── document_loader.py     # PDF -> LangChain Documents (one per page)
│   ├── text_splitter.py       # Documents -> overlapping chunks
│   ├── embeddings.py          # Local sentence-transformers embedding model
│   ├── vector_store.py        # Build / save / load the FAISS index
│   └── rag_chain.py           # Retriever + prompt + Claude (LCEL chain)
├── app/
│   └── app.py                 # Streamlit chat interface
├── .env                       # Your API key and model settings (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Framework | LangChain (LCEL) | Composable retrieval and prompting |
| LLM | Claude via `langchain-anthropic` | Strong at following "answer only from context" instructions |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, runs on CPU, no second API key |
| Vector store | FAISS | Fast, local, no server to run |
| UI | Streamlit | Upload + chat in about 200 lines |

## Setup

**Requirements:** Python 3.10 or newer and an [Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key: open .env and replace the placeholder
#    ANTHROPIC_API_KEY=sk-ant-...
```

The first run downloads the embedding model (about 90 MB) from Hugging Face. After that it works offline, apart from the calls to Claude.

## Run it

**Web app (recommended)**

```bash
streamlit run app/app.py
```

Then in the browser, upload PDFs and click **Process uploaded documents**, or click **Use sample policy instead**. Ask away.

**Command line (quick test)**

```bash
python -m src.rag_chain
```

This indexes everything in `data/` and opens an interactive prompt.

## Try these questions on the sample policy

- How many days of annual leave do employees get, and how many can be carried over?
- What is the daily meal allowance during business travel?
- Who has to approve an expense over $2,500?
- How soon must a lost laptop be reported?
- How many days a week must I be in the office?
- What is the parental leave for a primary caregiver?
- What's the capital of France? *(should reply that it isn't in the documents)*

## How each module works

- **`document_loader.py`**: Uses `PyPDFLoader` to read each page and tags it with `source` (file name) and `page` (1-based) so answers can be cited. Pages without text are skipped.
- **`text_splitter.py`**: `RecursiveCharacterTextSplitter` cuts pages into about 1,000-character chunks with 200 characters of overlap, preferring paragraph and sentence boundaries.
- **`embeddings.py`**: Wraps a local Hugging Face embedding model. Anthropic has no embeddings endpoint, which is why this runs locally.
- **`vector_store.py`**: Builds the FAISS index from chunks and saves/loads it from `vectorstore/`.
- **`rag_chain.py`**: `DocumentQAChain` retrieves the top-k chunks, formats them into a prompt, and calls Claude through an LCEL pipeline (`prompt | llm | parser`). The system prompt tells Claude to answer only from the context and to say so when the answer is missing.
- **`app.py`**: Streamlit UI with multi-file upload, chat history, expandable source snippets and adjustable chunking/retrieval settings.

## Configuration

Set in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none (required) | Your Anthropic key |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Claude model used for answers |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

Tuning in the app's **Advanced settings**:

- **Chunk size / overlap**: larger chunks give more context per hit, smaller ones give more precise retrieval.
- **Chunks retrieved**: raise it if answers seem to miss information spread across several sections.

## Troubleshooting

- **"ANTHROPIC_API_KEY is missing"**: edit `.env`, save, and restart Streamlit.
- **"No readable text found"**: the PDF is probably a scan. Run OCR on it first (for example with `ocrmypdf`).
- **"I couldn't find that in the uploaded documents"**: either the answer genuinely isn't there, or retrieval missed it. Try rephrasing, or increase the number of chunks retrieved.
- **`ModuleNotFoundError: src`**: run commands from the project root, not from inside `app/` or `src/`.
- **Slow first start**: the embedding model is downloading; later starts are fast.

## Security notes

- Never commit `.env`; it is already in `.gitignore`.
- The saved FAISS index uses pickle for metadata. Only load indexes your own app created.
- Document text is sent to Anthropic's API to generate answers. Check your company's data policy before uploading confidential material.

## Ideas for next steps

- Conversation memory so follow-up questions ("and for part-timers?") work
- Support for `.docx`, `.txt` and web pages
- Hybrid search (keyword + vector) and a re-ranking step
- Streaming answers token by token
- A persistent multi-user index with access control
- Evaluation set of question/answer pairs to track answer quality

## License

Free to use and adapt for your own projects.
