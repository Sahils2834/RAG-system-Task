
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Indexed chunks: {collection.count()}")

    while True:
        question = input(
            "\nAsk a question about your documents "
            "(or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        question_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(4, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        print("\n========== RETRIEVED CHUNKS ==========")

        for index, document in enumerate(
            results["documents"][0], start=1
        ):
            metadata = results["metadatas"][0][index - 1]
            distance = results["distances"][0][index - 1]

            print(f"\nResult {index}")
            print(f"Source: {metadata['source']}")
            print(f"Page: {metadata['page']}")
            print(f"Chunk: {metadata['chunk_index']}")
            print(f"Cosine distance: {distance:.4f}")
            print(f"Text: {document[:1000]}")


if __name__ == "__main__":
    main()
