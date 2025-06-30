# run_build_vectorstore.py
from loaders import build_vectorstore, load_documents
from collections import Counter


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents/chunks before embedding...")

    build_vectorstore()

    print("Vectorstore built successfully.")


def chunk_stats(documents):
    counter = Counter()
    for doc in documents:
        src = doc.metadata.get("source", "Unknown")
        counter[src] += 1
    for src, count in counter.items():
        print(f"{src}: {count} chunks")