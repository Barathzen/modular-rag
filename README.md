# Basic RAG

A minimal retrieval-augmented generation pipeline in Python:

1. **Chunking** — split `docs/*.txt` and `docs/*.md` into overlapping pieces
   (`_split_text` in `main.py`, pure Python).
2. **Embedding** — embed chunks with the NVIDIA embedding model
   (`NVIDIAEmbeddings` from `langchain-nvidia-ai-endpoints`).
3. **Storage** — keep vectors in `langchain_core`'s `InMemoryVectorStore`
   (requires `numpy`, see `pyproject.toml`).
4. **Retrieval** — similarity search for the top-k chunks of a question.
5. **Generation** — `ChatNVIDIA` answers grounded in the retrieved chunks.

## Run

```bash
.venv/bin/python main.py
# or
uv run main.py
```

It asks for a question and prints the answer plus the source files used.

Set `API_KEY` (or `NVIDIA_API_KEY`) in `.env`.

## Embedding models

The pipeline first tries hosted NVIDIA embedding models reported by the
endpoint and falls back to local deterministic embeddings (hashed
bag-of-words, L2-normalized) if none are available. Both are real options:
hosted models are preferred when they work.

## Files

- `main.py` — the pipeline: `load_documents`, `build_store`,
  `answer_question`, `main`.
- `docs/` — sample documents the pipeline reads (`rag.md`, `project.md`).
- `README.md` — this file.

## Tuning (top of `main.py`)

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking granularity.
- `TOP_K` — how many chunks are retrieved per question.
- `GENERATION_PROMPT` — how the model is asked to answer.
