import time
from typing import List
from tools.memory import split_text, add_texts_to_memory


def memory_ingestion_agent(topic: str, report_text: str):
    """
    Splits the final report into chunks, embeds, and stores them in ChromaDB.
    """
    print("\n🧠 Memory Ingestion Agent: Storing report into memory...")
    chunks: List[str] = split_text(report_text)
    timestamp = int(time.time())
    metadatas = [{"source": "report", "topic": topic, "ts": timestamp} for _ in chunks]
    ids = [f"{topic}-{timestamp}-{i}" for i in range(len(chunks))]
    try:
        add_texts_to_memory(chunks, metadatas=metadatas, ids=ids)
        print(f"✅ Stored {len(chunks)} chunks to ChromaDB.")
    except Exception as e:
        print(f"❌ Memory Ingestion error: {e}")
