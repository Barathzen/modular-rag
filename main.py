
from dotenv import load_dotenv

load_dotenv()

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import (
    create_vector_store,
    index_documents
)
from src.llm import create_llm
from src.rag_pipeline import answer_question


def main():

    print("\n===== MODULAR RAG SYSTEM =====\n")

    # STEP 1: Load documents
    print("[1] Loading documents...")

    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    # STEP 2: Split documents
    print("\n[2] Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # STEP 3: Create vector store
    print("\n[3] Creating vector store...")

    store = create_vector_store()

    # STEP 4: Index documents
    print("\n[4] Indexing documents...")

    index_documents(store, chunks)

    # STEP 5: Initialize LLM
    print("\n[5] Initializing LLM...")

    llm = create_llm()

    # STEP 6: Ask question
    question = input("\nAsk a question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # STEP 7: Execute RAG pipeline
    print("\n[6] Retrieving and generating answer...\n")

    result = answer_question(
        llm,
        store,
        question
    )

    print("=" * 50)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")

    for source in result["sources"]:
        print("-", source)

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
