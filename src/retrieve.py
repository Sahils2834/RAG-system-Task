
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 12
DEFAULT_TOP_K = 4

# Initialize the database and embedding model once.
_db = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _db.get_collection(COLLECTION_NAME)

_embedder = SentenceTransformer(EMBEDDING_MODEL)
_reranker = None


def get_reranker():
    """Load the Cross-Encoder only when needed."""
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = CANDIDATE_K,
    use_reranker: bool = True,
) -> list[dict]:
    """Retrieve candidates, rerank them, and return the best passages."""
    count = _collection.count()

    if count == 0:
        return []

    candidate_k = min(max(top_k, candidate_k), count)

    question_vector = _embedder.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = _collection.query(
        query_embeddings=question_vector,
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

    for index, document in enumerate(
        results["documents"][0]
    ):
        metadata = results["metadatas"][0][index] or {}

        candidates.append({
            "text": document,
            "source": metadata.get("source", "Unknown source"),
            "page": metadata.get("page", "?"),
            "chunk_index": metadata.get("chunk_index", "?"),
            "distance": float(
                results["distances"][0][index]
            ),
            "rerank_score": None,
        })

    if use_reranker and len(candidates) > 1:
        reranker = get_reranker()

        pairs = [
            (question, item["text"])
            for item in candidates
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

    return candidates[:min(top_k, len(candidates))]


def main():
    print(f"Indexed chunks: {_collection.count()}")
    print(f"Reranker: {RERANKER_MODEL}")

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

        try:
            results = retrieve(question)

        except Exception as exc:
            print(f"Retrieval failed: {exc}")
            print(
                "If this is the first reranker run, "
                "check internet access for model download."
            )
            continue

        print(
            "\n========== RETRIEVED + RERANKED CHUNKS =========="
        )

        for index, item in enumerate(results, start=1):
            print(f"\nResult {index}")
            print(
                f"Source: {item['source']} | "
                f"Page: {item['page']} | "
                f"Chunk: {item['chunk_index']}"
            )

            print(
                "Cosine distance "
                f"(lower is closer): {item['distance']:.4f}"
            )

            if item["rerank_score"] is not None:
                print(
                    "Cross-encoder score "
                    "(ranking score, not probability): "
                    f"{item['rerank_score']:.4f}"
                )

            print(f"Text: {item['text'][:1000]}")


if __name__ == "__main__":
    main()
