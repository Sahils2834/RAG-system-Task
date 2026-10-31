
from pathlib import Path
from functools import lru_cache
from time import perf_counter

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 3
CANDIDATE_K = 6


# --------------------------------------------------
# Initialize ChromaDB and embedding model once
# --------------------------------------------------

_db = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _db.get_collection(COLLECTION_NAME)

_embedder = SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    """
    Load the reranker only when first needed.
    Cache the loaded model for subsequent queries.
    """
    print("Loading CrossEncoder reranker for the first time...")

    model = CrossEncoder(RERANKER_MODEL)

    print("CrossEncoder loaded successfully.")

    return model


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = CANDIDATE_K,
    use_reranker: bool = True,
    return_timings: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Retrieve relevant document chunks using ChromaDB.

    Steps:
    1. Embed the question.
    2. Retrieve candidate chunks from ChromaDB.
    3. Optionally rerank candidates with a CrossEncoder.
    4. Return the best passages.

    Set return_timings=True to also return stage timings.
    """

    total_start = perf_counter()

    if not question or not question.strip():
        empty_result = []
        timings = {
            "embedding_seconds": 0.0,
            "vector_search_seconds": 0.0,
            "reranking_seconds": 0.0,
            "retrieval_seconds": 0.0,
        }

        return (empty_result, timings) if return_timings else empty_result

    if top_k < 1 or candidate_k < 1:
        raise ValueError("top_k and candidate_k must be positive.")

    count = _collection.count()

    if count == 0:
        empty_result = []
        timings = {
            "embedding_seconds": 0.0,
            "vector_search_seconds": 0.0,
            "reranking_seconds": 0.0,
            "retrieval_seconds": round(perf_counter() - total_start, 4),
        }

        return (empty_result, timings) if return_timings else empty_result

    candidate_k = min(max(top_k, candidate_k), count)

    # Step 1: Embed the question.
    embedding_start = perf_counter()

    question_vector = _embedder.encode(
        [question.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    embedding_seconds = perf_counter() - embedding_start

    # Step 2: Search ChromaDB.
    search_start = perf_counter()

    results = _collection.query(
        query_embeddings=question_vector,
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    vector_search_seconds = perf_counter() - search_start

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    retrieved_documents = documents[0] if documents else []
    retrieved_metadatas = metadatas[0] if metadatas else []
    retrieved_distances = distances[0] if distances else []

    candidates = []

    for index, document in enumerate(retrieved_documents):
        metadata = (
            retrieved_metadatas[index]
            if index < len(retrieved_metadatas)
            else {}
        ) or {}

        distance = (
            retrieved_distances[index]
            if index < len(retrieved_distances)
            else None
        )

        candidates.append({
            "text": document,
            "source": metadata.get("source", "Unknown source"),
            "page": metadata.get("page", "?"),
            "chunk_index": metadata.get("chunk_index", "?"),
            "distance": (
                float(distance) if distance is not None else None
            ),
            "rerank_score": None,
        })

    # Step 3: Optionally rerank the retrieved candidates.
    reranking_start = perf_counter()

    if use_reranker and len(candidates) > 1:
        reranker = get_reranker()

        pairs = [
            (question.strip(), item["text"])
            for item in candidates
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
            batch_size=8,
        )

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

    reranking_seconds = perf_counter() - reranking_start

    selected = candidates[:min(top_k, len(candidates))]

    timings = {
        "embedding_seconds": round(embedding_seconds, 4),
        "vector_search_seconds": round(vector_search_seconds, 4),
        "reranking_seconds": round(reranking_seconds, 4),
        "retrieval_seconds": round(perf_counter() - total_start, 4),
    }

    if return_timings:
        return selected, timings

    return selected


# --------------------------------------------------
# Interactive terminal testing
# --------------------------------------------------

def main():
    print(f"Indexed chunks: {_collection.count()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranker model: {RERANKER_MODEL}")
    print("Reranking is lazy-loaded and cached.")

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
            results, timings = retrieve(
                question,
                top_k=3,
                candidate_k=6,
                use_reranker=True,
                return_timings=True,
            )

        except Exception as exc:
            print(f"Retrieval failed: {exc}")
            continue

        print("\n========== RETRIEVED CHUNKS ==========")

        for index, item in enumerate(results, start=1):
            print(f"\nResult {index}")
            print(
                f"Source: {item['source']} | "
                f"Page: {item['page']} | "
                f"Chunk: {item['chunk_index']}"
            )
            print(f"Vector distance: {item['distance']}")

            if item["rerank_score"] is not None:
                print(f"Reranker score: {item['rerank_score']:.4f}")

            print(f"Text: {item['text'][:1000]}")

        print("\n========== TIMINGS ==========")

        for name, seconds in timings.items():
            print(f"{name}: {seconds:.4f}s")


if __name__ == "__main__":
    main()
