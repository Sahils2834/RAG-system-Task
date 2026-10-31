
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from src.retrieve import retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(
    title="Document RAG API",
    description="Ask questions about your indexed PDF documents.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=12,
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Document RAG API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": client is not None,
        "model": MODEL_NAME,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the backend.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a question.",
        )

    request_start = time.perf_counter()

    try:
        # ------------------------------------------
        # Step 1: Retrieve and rerank relevant chunks
        # ------------------------------------------

        results, retrieval_timings = retrieve(
            question,
            top_k=3,
            candidate_k=6,
            use_reranker=True,
            return_timings=True,
        )

        if not results:
            return {
                "answer": (
                    "I couldn't find relevant passages in the indexed "
                    "documents. Please try rephrasing your question."
                ),
                "sources": [],
                "latency_seconds": round(
                    time.perf_counter() - request_start, 3
                ),
                "timings": {
                    **retrieval_timings,
                    "generation_seconds": 0.0,
                },
            }

        # ------------------------------------------
        # Step 2: Build the context and source list
        # ------------------------------------------

        context_blocks = []
        sources = []

        for index, result in enumerate(results, start=1):
            source_id = f"S{index}"
            source = result.get("source", "Unknown source")
            page = result.get("page", "?")
            text = result.get("text", "")

            context_blocks.append(
                f"[{source_id}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{text}"
            )

            sources.append({
                "id": source_id,
                "source": str(source),
                "page": page,
                "text": text,
                "distance": result.get("distance"),
                "rerank_score": result.get("rerank_score"),
            })

        # ------------------------------------------
        # Step 3: Limit conversation history
        # ------------------------------------------

        history_text = "\n".join(
            f"{message.role.upper()}: {message.content[:2000]}"
            for message in request.history[-6:]
        )

        # ------------------------------------------
        # Step 4: Build the grounded prompt
        # ------------------------------------------

        prompt = f"""
You are a helpful assistant that answers questions using the
provided document excerpts.

Rules:
- Ground document-specific factual claims in the excerpts.
- Cite supporting passages with identifiers such as [S1] or [S2].
- Never invent source identifiers.
- If the excerpts do not contain enough evidence, say so clearly.
- Conversation history helps interpret follow-up questions but is
  not evidence for document-specific factual claims.
- Be concise, accurate, and easy to understand.
- Do not claim a passage supports something it does not say.

Conversation history:
{history_text or "(No previous messages)"}

Document excerpts:
{chr(10).join(context_blocks)}

Current question:
{question}

Answer using the excerpts and cite relevant source identifiers.
"""

        # ------------------------------------------
        # Step 5: Generate the answer with Gemini
        # ------------------------------------------

        generation_start = time.perf_counter()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        generation_seconds = time.perf_counter() - generation_start

        answer = response.text or (
            "The model returned an empty response. Please try again."
        )

        total_seconds = time.perf_counter() - request_start

        timings = {
            **retrieval_timings,
            "generation_seconds": round(generation_seconds, 4),
            "total_seconds": round(total_seconds, 4),
        }

        print(
            f"RAG timings | "
            f"embedding={timings['embedding_seconds']:.3f}s | "
            f"vector_search={timings['vector_search_seconds']:.3f}s | "
            f"reranking={timings['reranking_seconds']:.3f}s | "
            f"generation={timings['generation_seconds']:.3f}s | "
            f"total={timings['total_seconds']:.3f}s"
        )

        return {
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(total_seconds, 3),
            "timings": timings,
        }

    except HTTPException:
        raise

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()

        print(f"RAG API error: {type(exc).__name__}: {message}")

        if (
            "429" in message
            or "resource_exhausted" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota is exhausted or rate-limited. "
                    "Retrieval may still work. Check your Gemini quota "
                    "and try again when requests are available."
                ),
            ) from exc

        if (
            "503" in message
            or "unavailable" in lowered
            or "overloaded" in lowered
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while answering your question. "
                "Check the backend terminal for details."
            ),
        ) from exc
