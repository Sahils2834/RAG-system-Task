from pathlib import Path
import re
import hashlib

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    This is a simple, inspectable baseline. Later we can improve
    it with paragraph-aware or token-aware chunking.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += step

    return chunks


def load_pdf_pages():
    """Extract readable text from every PDF, preserving page metadata."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. Add your PDFs to the data folder."
        )

    pages = []
    total_pages = 0
    total_words = 0

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:
            print(f"  PDF pages: {len(pdf)}")

            for page_index, page in enumerate(pdf):
                text = clean_text(page.get_text("text"))
                total_pages += 1

                if not text:
                    print(
                        f"  Warning: page {page_index + 1} has no "
                        "extractable text. It may need OCR."
                    )
                    continue

                word_count = len(text.split())
                total_words += word_count

                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })

    print(f"\nTotal PDF pages: {total_pages}")
    print(f"Pages with extracted text: {len(pages)}")
    print(f"Extracted words: {total_words}")

    if total_pages == 0 or not pages:
        raise ValueError(
            "No readable text was extracted. Check your PDFs."
        )

    if total_words < 2000:
        print(
            "WARNING: The extracted collection is quite small. "
            "Check whether the PDFs contain all intended documentation."
        )

    return pages


def build_chunks(pages):
    """Chunk each page and attach source and page information."""
    records = []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks):
            unique_text = (
                f"{page_data['source']}|"
                f"{page_data['page']}|"
                f"{chunk_index}|{chunk}"
            )
            chunk_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()[:24]

            records.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": chunk_index,
                },
            })

    return records


def main():
    pages = load_pdf_pages()
    records = build_chunks(pages)

    print(f"\nTotal chunks created: {len(records)}")

    if not records:
        raise ValueError("No chunks were created.")

    print("\n========== CHUNK PREVIEW ==========")

    for index, record in enumerate(records[:3], start=1):
        print(f"\nChunk {index}")
        print("Source:", record["metadata"]["source"])
        print("Page:", record["metadata"]["page"])
        print("Word count:", len(record["text"].split()))
        print("Text preview:", record["text"][:600])

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches to avoid processing the entire dataset at once.
    batch_size = 64

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        texts = [record["text"] for record in batch]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=[record["id"] for record in batch],
            documents=texts,
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings,
        )

        print(
            f"Stored {min(start + batch_size, len(records))}"
            f"/{len(records)} chunks"
        )

    print("\n========== INGESTION COMPLETE ==========")
    print("Collection:", COLLECTION_NAME)
    print("Stored chunks:", collection.count())
    print("Database location:", CHROMA_DIR)


if __name__ == "__main__":
    main()
