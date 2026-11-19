from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4

EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b"

LLM_MODEL = "moonshotai/kimi-k3.1-pro"

MAX_COMPLETION_TOKENS = 16384

