# Retrieval-Augmented Generation

RAG is a technique that lets a language model answer questions about data
it was not trained on. Instead of relying only on its own training, the
model first *retrieves* relevant passages from a corpus of documents and
*generates* the answer from the retrieved context.

## The basic pipeline

1. **Chunking** — long documents are split into overlapping pieces so each
   can be embedded as one unit.
2. **Embedding** — every chunk is converted into a vector (a list of
   numbers) that captures its meaning, so similar texts end up close
   together in vector space.
3. **Storage & retrieval** — the vectors live in a vector store. A
   question is embedded the same way and the store returns the k chunks
   whose vectors are closest to it.
4. **Generation** — the retrieved chunks are put into a prompt as
   context, and the language model writes the answer from that context.

## Why it works

- Embedding similarity finds passages that share meaning even when the
  exact words differ, so keyword search is not required.
- Grounding the model in the retrieved context reduces hallucination for
  facts that live in the documents.
- Chunking with overlap keeps answers from losing information at
  boundaries between pieces.

See `main.py` for the implementation in this repository.
