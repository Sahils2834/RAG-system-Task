
from pathlib import Path
import hashlib
import re

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 60
MIN_CHUNK_WORDS = 25


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Create paragraph-aware, overlapping word-based chunks."""
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    if not paragraphs:
        return []

    # Split paragraphs that exceed the chunk size.
    units = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) <= CHUNK_SIZE_WORDS:
            units.append(words)
        else:
            for start in range(0, len(words), step):
                piece = words[start:start + CHUNK_SIZE_WORDS]

                if piece:
                    units.append(piece)

                if start + CHUNK_SIZE_WORDS >= len(words):
                    break

    chunks = []
    current = []

    for words in units:
        if current and len(current) + len(words) > CHUNK_SIZE_WORDS:
            chunks.append(" ".join(current))

            overlap = (
                current[-CHUNK_OVERLAP_WORDS:]
                if CHUNK_OVERLAP_WORDS
                else []
            )
            current = overlap[:]

        if current and len(current) + len(words) > CHUNK_SIZE_WORDS:
            chunks.append(" ".join(current))
            current = []

        current.extend(words)

    if current:
        chunks.append(" ".join(current))

    # Remove short fragments and duplicate chunks.
    cleaned_chunks = []
    seen = set()

    for chunk in chunks:
        chunk = chunk.strip()
        key = re.sub(r"\s+", " ", chunk).casefold()

        if (
            len(chunk.split()) >= MIN_CHUNK_WORDS
            and key not in seen
        ):
            cleaned_chunks.append(chunk)
            seen.add(key)

    return cleaned_chunks


def load_pdf_pages() -> list[dict]:
    """Extract text from PDFs while retaining page information."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. "
            "Add your PDFs to the data folder."
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
                        f"  Warning: page {page_index + 1} "
                        "has no extractable text. OCR may be needed."
                    )
                    continue

                total_words += len(text.split())

                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })

    print(f"\nTotal PDF pages: {total_pages}")
    print(f"Pages with extracted text: {len(pages)}")
    print(f"Extracted words: {total_words}")

    if not pages:
        raise ValueError("No readable text was extracted.")

    return pages


def build_chunks(pages: list[dict]) -> list[dict]:
    """Create chunks with stable IDs and source metadata."""
    records = []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks):
            identity = (
                f"{page_data['source']}|"
                f"{page_data['page']}|"
                f"{chunk_index}|{chunk}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:24]

            records.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": page_data["source"],
                    "page": int(page_data["page"]),
                    "chunk_index": int(chunk_index),
                    "word_count": len(chunk.split()),
                },
            })

    return records


def main():
    pages = load_pdf_pages()
    records = build_chunks(pages)

    if not records:
        raise ValueError("No chunks were created.")

    print(f"\nTotal chunks created: {len(records)}")

    for index, record in enumerate(records[:3], start=1):
        print(f"\nChunk {index}")
        print("Source:", record["metadata"]["source"])
        print("Page:", record["metadata"]["page"])
        print("Word count:", record["metadata"]["word_count"])
        print("Text preview:", record["text"][:500])

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Rebuild the collection using the updated chunking strategy.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "chunking": "paragraph-aware-350w-overlap60",
        },
    )

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
            metadatas=[
                record["metadata"] for record in batch
            ],
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
