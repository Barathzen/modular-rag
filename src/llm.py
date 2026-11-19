
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.embeddings import get_api_key
from src.config import (
    LLM_MODEL,
    MAX_COMPLETION_TOKENS
)


def create_llm():
    """
    Initialize and return the NVIDIA chat model.
    """

    llm = ChatNVIDIA(
        model=LLM_MODEL,
        api_key=get_api_key(),
        max_completion_tokens=MAX_COMPLETION_TOKENS
    )

    return llm
