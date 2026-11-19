
from src.config import TOP_K
from src.vector_store import retrieve_documents


GENERATION_PROMPT = """
You are a helpful assistant.

Answer the question using ONLY the
provided context.

If the answer is not present in the
context, clearly say that you do not
know based on the available documents.

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""


def answer_question(llm, store, question):

    # STEP 1: Retrieve relevant chunks
    retrieved_docs = retrieve_documents(
        store,
        question,
        k=TOP_K
    )

    if not retrieved_docs:

        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "context": ""
        }

    # STEP 2: Build context
    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # STEP 3: Build prompt
    prompt = GENERATION_PROMPT.format(
        context=context,
        question=question
    )

    # STEP 4: Generate answer
    response = llm.invoke(prompt)

    # STEP 5: Extract source information
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in retrieved_docs
    })

    return {
        "answer": response.content,
        "sources": sources,
        "context": context
    }
