import os
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Directory to persist ChromaDB data
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
CHROMA_DIR = os.path.abspath(CHROMA_DIR)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "reports")


def get_embeddings():
    """Returns a singleton embeddings instance using Google Generative AI embeddings."""
    # Model name per langchain-google-genai; default embedding model works for text.
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_chroma_client():
    """Create or return a persistent ChromaDB client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))


def get_collection():
    client = get_chroma_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


# -------- Ingestion Utilities --------

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def add_texts_to_memory(texts: List[str], metadatas: List[dict] | None = None, ids: List[str] | None = None) -> None:
    collection = get_collection()
    if ids is None:
        # Generate simple deterministic IDs if not provided
        ids = [f"doc_{i}" for i in range(len(texts))]
    if metadatas is None:
        metadatas = [{} for _ in texts]
    vectors = embed_texts(texts)
    collection.add(documents=texts, metadatas=metadatas, embeddings=vectors, ids=ids)


# -------- Retrieval Utilities --------

def similarity_search(query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
    """
    Returns list of tuples: (text, distance, metadata)
    """
    collection = get_collection()
    embeddings = get_embeddings()
    q = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[q], n_results=k, include=["distances", "metadatas", "documents"])  # type: ignore
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, dists, metas))
