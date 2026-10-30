
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
    version="1.0.0",
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

    try:
        # Retrieve relevant chunks from your existing ChromaDB.
        results = retrieve(
            question,
            top_k=4,
            candidate_k=12,
            use_reranker=True,
        )

        if not results:
            return {
                "answer": (
                    "I couldn't find relevant passages in the indexed "
                    "documents. Please try rephrasing your question."
                ),
                "sources": [],
                "latency_seconds": 0,
            }

        context_blocks = []
        sources = []

        for index, result in enumerate(results, start=1):
            source_id = f"S{index}"
            source = result.get("source", "unknown")
            page = result.get("page")
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

        history_text = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in request.history[-10:]
        )

        prompt = f"""
You are a helpful assistant answering questions about documents.

RULES:
- Ground factual claims in the supplied document context.
- Use exact citation identifiers such as [S1] for supporting claims.
- Never invent sources or citation identifiers.
- If the documents do not provide enough evidence, clearly say so.
- Conversation history helps interpret follow-up questions, but is not
  evidence. Use the current document context to support factual claims.
- If the user asks a follow-up, use the conversation history to
  understand what they mean.
- Do not claim that a source supports a statement unless its content
  actually supports that statement.

CONVERSATION HISTORY:
{history_text or "(No previous messages)"}

DOCUMENT CONTEXT:
{chr(10).join(context_blocks)}

CURRENT QUESTION:
{question}

Answer clearly. Cite factual statements using the source identifiers.
"""

        start = time.perf_counter()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        latency = round(time.perf_counter() - start, 3)
        answer = response.text or "The model returned an empty response."

        return {
            "answer": answer,
            "sources": sources,
            "latency_seconds": latency,
        }

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()

        if (
            "429" in message
            or "resource_exhausted" in lowered
            or "quota" in lowered
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota is exhausted or rate-limited. "
                    "Your retrieval database is unaffected. Please "
                    "check your Gemini API quota and try again later."
                ),
            ) from exc

        if (
            "503" in message
            or "unavailable" in lowered
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                ),
            ) from exc

        print(f"RAG API error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while answering your question.",
        ) from exc
