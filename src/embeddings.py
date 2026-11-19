
import hashlib
import math
import os
import re

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from src.config import EMBEDDING_MODEL

load_dotenv()


def get_api_key():

    return (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("API_KEY")
    )


class LocalEmbeddings(Embeddings):

    DIMENSION = 384

    def _embed(self, text: str) -> list[float]:

        vector = [0.0] * self.DIMENSION

        tokens = re.findall(
            r"[a-z0-9]+",
            text.lower()
        )

        for token in tokens:

            digest = hashlib.sha256(
                token.encode("utf-8")
            ).digest()

            index = (
                int.from_bytes(digest[:8], "big")
                % self.DIMENSION
            )

            vector[index] += 1.0

        norm = math.sqrt(
            sum(value * value for value in vector)
        )

        if norm > 0:
            vector = [value / norm for value in vector]

        return vector

    def embed_documents(self, texts):

        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(self, text):

        return self._embed(text)


def create_embedder():

    api_key = get_api_key()

    try:

        embedder = NVIDIAEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key
        )

        # Check whether the hosted model works
        embedder.embed_query("test")

        print("Using NVIDIA embeddings")

        return embedder

    except Exception as exc:

        print("NVIDIA embeddings unavailable:", exc)
        print("Using local embeddings")

        return LocalEmbeddings()
