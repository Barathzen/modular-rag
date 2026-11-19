# Project Overview

This repository is a small Python project (`rag`) set up with LangChain
and the NVIDIA AI endpoints. The `main.py` script implements a basic
retrieval-augmented generation pipeline:

- `load_documents()` reads `.txt` and `.md` files from `docs/` and splits
  them into chunks with a pure-Python recursive splitter.
- `build_store()` embeds the chunks with `NVIDIAEmbeddings` and stores
  them in an in-memory vector store (`InMemoryVectorStore`).
- `answer_question()` retrieves the top-k most relevant chunks for a
  question and generates an answer with `ChatNVIDIA`.

## Running it

```bash
.venv/bin/python main.py
```

or

```bash
uv run main.py
```

Make sure an API key is available: the `.env` file provides `API_KEY`
(the code also accepts `NVIDIA_API_KEY`).

## Tuning

- `CHUNK_SIZE` and `CHUNK_OVERLAP` control the chunking granularity.
- `TOP_K` controls how many chunks are retrieved per question.
- `GENERATION_PROMPT` controls how the model is asked to answer.
