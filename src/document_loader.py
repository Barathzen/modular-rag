from pathlib import Path 
from langchain_core.documents import Document
from src.config import DOCS_DIR 

SUPPORTED_EXTENSIONS = {".txt", ".md"}

def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    documents = []
    if not docs_dir.exists():
        return documents
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue 
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue 
        text = path.read_text(encoding="utf-8")

        if not text.strip():
            continue 

        document = Document(
            page_content=text,
            metadata={
                "source": str(path.relative_to(docs_dir.parent)),
                "file_name": path.name
            }
        )

        documents.append(document)
    return documents 