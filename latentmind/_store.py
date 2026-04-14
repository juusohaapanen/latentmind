"""ChromaDB collection factory."""

from pathlib import Path
import chromadb

LATENTMIND_DIR = Path.home() / ".latentmind"  # default global persistence


def get_chroma_collection(persist_dir: Path) -> chromadb.Collection:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name="memories",
        metadata={"hnsw:space": "cosine"},
    )
