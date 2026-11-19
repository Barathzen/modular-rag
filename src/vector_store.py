
from langchain_core.vectorstores import InMemoryVectorStore

from src.embeddings import create_embedder
from src.config import TOP_K


def create_vector_store():
    """
    Initialize an in-memory vector store.
    """

    embeddings = create_embedder()

    vector_store = InMemoryVectorStore(
        embedding=embeddings
    )

    return vector_store


def index_documents(vector_store, chunks):
    """
    Convert chunks into embeddings and store them.
    """

    if not chunks:
        print("No chunks available for indexing.")
        return

    vector_store.add_documents(chunks)

    print(f"Successfully indexed {len(chunks)} chunks.")


def retrieve_documents(vector_store, question, k=TOP_K):
    """
    Retrieve the most relevant document chunks.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(question)

    return retrieved_docs
    