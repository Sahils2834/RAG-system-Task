
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.retrieve import retrieve


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None

TOP_K = 3
CANDIDATE_K = 6

MAX_CONTEXT_CHARS_PER_SOURCE = 3500
MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 1000

MAX_OUTPUT_TOKENS = 4096
THINKING_LEVEL = "LOW"


# ==================================================
# 2. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="DocuMind - Document RAG API",
    description="Ask questions about your indexed PDF documents.",
    version="1.5.0",
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


# ==================================================
# 3. REQUEST SCHEMAS
# ==================================================

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=12,
    )


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def build_history(history: list[HistoryMessage]) -> str:
    """Keep recent conversation messages within safe limits."""

    recent_messages = history[-MAX_HISTORY_MESSAGES:]

    return "\n".join(
        f"{message.role.upper()}: "
        f"{message.content[:MAX_HISTORY_CHARS]}"
        for message in recent_messages
    )


def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """Build document context and source metadata."""

    context_blocks = []
    sources = []

    for index, result in enumerate(results, start=1):

        source_id = f"S{index}"

        source = result.get("source", "Unknown source")
        page = result.get("page", "?")
        full_text = result.get("text", "") or ""

        # Limit prompt context while preserving the complete
        # retrieved passage in the API response.

        context_blocks.append(
            f"[{source_id}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{full_text[:MAX_CONTEXT_CHARS_PER_SOURCE]}"
        )

        sources.append({
            "id": source_id,
            "source": str(source),
            "page": page,
            "text": full_text,
            "distance": result.get("distance"),
            "rerank_score": result.get("rerank_score"),
        })

    return "\n\n---\n\n".join(context_blocks), sources


def build_prompt(
    question: str,
    history_text: str,
    document_context: str,
) -> str:

    return f"""
You are DocuMind, an intelligent document research assistant.

Your task is to answer the user's question accurately and
completely using the retrieved document excerpts.

STRICT RULES:

1. Answer every part of the user's question.

2. Ground document-specific factual claims in the excerpts.

3. Cite supporting passages using identifiers such as [S1], [S2].

4. Never invent citations or unsupported information.

5. If the excerpts do not contain enough evidence,
   clearly explain what information is missing.

6. Do not replace missing document evidence with
   general knowledge.

7. Conversation history helps interpret follow-up questions.
   It is not evidence.

8. Use clear headings, bullet points and tables
   when they improve readability.

9. If asked to compare concepts:
   - Explain each concept separately.
   - Describe their differences.
   - Explain their use cases.
   - Include relevant examples from the documents.

10. Finish every requested section, sentence,
    list and table.

11. Be detailed, accurate and avoid repetition.

12. Do not stop after an introductory paragraph.

CONVERSATION HISTORY:

{history_text or "(No previous messages)"}

RETRIEVED DOCUMENT EXCERPTS:

{document_context}

CURRENT QUESTION:

{question}

Provide a complete, well-structured answer with citations throughout.

Include all relevant information available in the excerpts.
"""


# ==================================================
# 5. GEMINI RESPONSE DIAGNOSTICS
# ==================================================

def inspect_gemini_response(response) -> tuple[str, str]:
    """Inspect Gemini output and identify why generation stopped."""

    print("\n========== GEMINI RESPONSE DIAGNOSTICS ==========")

    candidates = response.candidates or []
    finish_reason = "UNKNOWN"

    if candidates:

        candidate = candidates[0]

        finish_reason = str(candidate.finish_reason or "UNKNOWN")

        print("Finish reason:", finish_reason)
        print("Safety ratings:", candidate.safety_ratings)

    else:
        print("No response candidates were returned.")

    usage = response.usage_metadata

    if usage:

        print("Prompt tokens:", usage.prompt_token_count)
        print("Output tokens:", usage.candidates_token_count)
        print("Thought tokens:", getattr(
            usage,
            "thoughts_token_count",
            "Not available",
        ))
        print("Total tokens:", usage.total_token_count)

    answer = response.text or ""

    print("Returned answer length:", len(answer))
    print("Returned answer preview:", answer[:500])

    print("=================================================\n")

    return answer, finish_reason


# ==================================================
# 6. API ROUTES
# ==================================================

@app.get("/")
def root():

    return {
        "name": "DocuMind Document RAG API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "gemini_configured": client is not None,
        "model": MODEL_NAME,
        "thinking_level": THINKING_LEVEL,
        "retrieval": "ChromaDB + CrossEncoder",
    }


# ==================================================
# 7. CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    if client is None:

        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured in the backend .env file.",
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
        # STEP 1: RETRIEVAL + RERANKING
        # ------------------------------------------

        retrieval_start = time.perf_counter()

        results = retrieve(
            question,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            use_reranker=True,
        )

        retrieval_seconds = (
            time.perf_counter() - retrieval_start
        )

        if not results:

            total_seconds = (
                time.perf_counter() - request_start
            )

            return {
                "answer": (
                    "I couldn't find relevant passages in the indexed "
                    "documents. Please try rephrasing your question."
                ),
                "sources": [],
                "latency_seconds": round(total_seconds, 3),
                "timings": {
                    "retrieval_seconds": round(
                        retrieval_seconds, 4
                    ),
                    "generation_seconds": 0.0,
                    "total_seconds": round(total_seconds, 4),
                },
                "finish_reason": "NO_RELEVANT_RESULTS",
            }

        # ------------------------------------------
        # STEP 2: BUILD DOCUMENT CONTEXT
        # ------------------------------------------

        document_context, sources = build_context(results)

        # ------------------------------------------
        # STEP 3: CONVERSATION HISTORY
        # ------------------------------------------

        history_text = build_history(request.history)

        # ------------------------------------------
        # STEP 4: BUILD PROMPT
        # ------------------------------------------

        prompt = build_prompt(
            question=question,
            history_text=history_text,
            document_context=document_context,
        )

        # ------------------------------------------
        # STEP 5: GEMINI GENERATION
        # ------------------------------------------

        generation_start = time.perf_counter()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL,
                ),
            ),
        )

        generation_seconds = (
            time.perf_counter() - generation_start
        )

        # ------------------------------------------
        # STEP 6: INSPECT RESPONSE
        # ------------------------------------------

        answer, finish_reason = inspect_gemini_response(response)

        if not answer.strip():

            answer = (
                "Gemini returned an empty answer. "
                "Please try again."
            )

        if "MAX_TOKENS" in finish_reason.upper():

            answer += (
                "\n\n> **Generation limit reached:** Gemini stopped "
                "after reaching its output-token limit. This answer "
                "may be incomplete. Try asking a more focused question."
            )

        # ------------------------------------------
        # STEP 7: PERFORMANCE METRICS
        # ------------------------------------------

        total_seconds = (
            time.perf_counter() - request_start
        )

        timings = {
            "retrieval_seconds": round(
                retrieval_seconds, 4
            ),
            "generation_seconds": round(
                generation_seconds, 4
            ),
            "total_seconds": round(
                total_seconds, 4
            ),
        }

        print("\n========== DOCUMIND PERFORMANCE ==========")

        print(
            f"Retrieval + reranking: "
            f"{retrieval_seconds:.3f}s"
        )

        print(
            f"Gemini generation: "
            f"{generation_seconds:.3f}s"
        )

        print(
            f"Total request: "
            f"{total_seconds:.3f}s"
        )

        print("Thinking level:", THINKING_LEVEL)

        print("==========================================\n")

        # ------------------------------------------
        # STEP 8: RETURN RESPONSE TO FRONTEND
        # ------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(total_seconds, 3),
            "timings": timings,
            "finish_reason": finish_reason,
        }

    # ------------------------------------------
    # STEP 9: ERROR HANDLING
    # ------------------------------------------

    except HTTPException:
        raise

    except Exception as exc:

        message = str(exc)
        lowered = message.lower()

        print(
            f"DocuMind API error: "
            f"{type(exc).__name__}: {message}"
        )

        if (
            "429" in message
            or "resource_exhausted" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
        ):

            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota is exhausted or "
                    "the request was rate-limited. "
                    "Check your Gemini API quota and try again."
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
                "An internal error occurred while answering "
                "your question. Check the backend terminal."
            ),
        ) from exc
