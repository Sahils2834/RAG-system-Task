# DocuMind --- Document RAG Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application that
answers questions using passages retrieved from a local collection of
PDF documents. It combines dense-vector retrieval, optional CrossEncoder
reranking, and Gemini-based answer generation in a React chat interface.

> **Important:** Answers are AI-generated. Verify important claims
> against the cited PDF passages.

## Features

-   Extracts text from PDF documents and splits it into overlapping
    chunks.
-   Generates text embeddings with `all-MiniLM-L6-v2`.
-   Stores document chunks and metadata in a persistent local ChromaDB
    collection.
-   Retrieves candidate passages using vector similarity.
-   Optionally reranks candidates with
    `cross-encoder/ms-marco-MiniLM-L-6-v2`.
-   Sends retrieved context to Gemini to generate an answer with source
    labels.
-   Exposes a FastAPI backend and a React + Vite frontend.
-   Displays source document names, page numbers, and retrieved
    passages.

## Architecture

``` text
PDF files
   |
   v
src/ingest.py
   |  extract text, chunk, embed
   v
ChromaDB (chroma_db/)
   |
   v
React frontend (frontend/)
   |  POST /api/chat
   v
FastAPI (src/api.py)
   |
   +--> src/retrieve.py
   |      1. Embed question
   |      2. Retrieve candidates from ChromaDB
   |      3. Rerank candidates with CrossEncoder
   |
   +--> Gemini API
          Generate answer using retrieved context
   |
   v
Answer + source metadata returned to the frontend
```

## Project structure

``` text
document-rag/
├── data/                  # Source PDFs to ingest
├── evaluation/
│   └── questions.json     # Evaluation questions (if configured)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── .env               # Frontend API URL; no secret keys here
│   └── package.json
├── chroma_db/             # Persistent local vector database (generated)
├── results/               # Evaluation outputs (generated, if used)
├── src/
│   ├── api.py             # FastAPI routes
│   ├── evaluate.py        # Retrieval/generation evaluation
│   ├── generate.py        # CLI generation / RAG comparison
│   ├── ingest.py          # PDF ingestion and indexing
│   └── retrieve.py        # Dense retrieval and reranking
├── tests/                 # Tests (if present)
├── .env                   # Gemini API key and model configuration
├── .gitignore
├── requirements.txt
└── README.md
```

The exact files and generated folders can vary by local setup.

## Requirements

-   Python 3.10 or later (use the Python version tested with your
    dependencies)
-   Node.js and npm
-   A Gemini API key
-   PDF documents that you are permitted to use

## Setup

### 1. Create and activate a Python virtual environment

From the repository root in Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the project's Python
executable directly instead:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install dependencies:

``` powershell
python -m pip install -r requirements.txt
```

If `requirements.txt` does not yet contain the API dependencies, install
them with:

``` powershell
python -m pip install fastapi "uvicorn[standard]" python-dotenv google-genai
```

### 2. Configure environment variables

Create a root `.env` file:

``` env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Use a model name available to your Gemini API project. Never commit
`.env` or expose the API key in frontend code.

### 3. Add documents

Place the PDF files you want to search in `data/`. Use documents you are
authorized to process.

### 4. Ingest and index PDFs

Run from the repository root:

``` powershell
python -m src.ingest
```

This extracts document text, creates chunks, embeds them, and stores the
vectors and metadata in `chroma_db/`. Re-run ingestion after changing
the source documents or ingestion settings. Do not delete the database
unless you intend to rebuild the index.

### 5. Start the FastAPI backend

In a terminal at the repository root:

``` powershell
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

-   API root: `http://127.0.0.1:8000/`
-   Health check: `http://127.0.0.1:8000/api/health`
-   Interactive API docs: `http://127.0.0.1:8000/docs`

The health endpoint checks whether the API is running and whether a
Gemini key is configured. It does not prove that a generation request
will succeed.

### 6. Start the React frontend

Open another terminal:

``` powershell
cd frontend
npm install
```

Create `frontend/.env`:

``` env
VITE_API_URL=http://127.0.0.1:8000
```

Then run:

``` powershell
npm run dev
```

Open the local URL printed by Vite, commonly `http://localhost:5173`.

Keep the backend and frontend terminals running while using DocuMind.

## How retrieval works

1.  **Ingestion:** PDF text is extracted and split into overlapping
    chunks.
2.  **Embedding:** Each chunk is represented as a dense vector using
    `all-MiniLM-L6-v2`.
3.  **Initial retrieval:** The question is embedded and ChromaDB returns
    a set of candidate chunks using vector distance.
4.  **Reranking:** The CrossEncoder scores each
    `(question, candidate passage)` pair and reorders the candidate
    list.
5.  **Context construction:** The top passages are added to the Gemini
    prompt with source identifiers such as `[S1]`.
6.  **Generation:** Gemini produces an answer based on the question,
    conversation history, and retrieved passages.
7.  **Display:** The UI shows the answer, source metadata, and retrieved
    excerpts.

### Understanding the scores

-   **Vector distance/similarity:** Used during initial vector search to
    find semantically similar candidates. In this project's ChromaDB
    output, the field is called `distance`; lower is closer for the
    configured distance metric. Do not label a distance as a similarity
    score without checking the metric and conversion.
-   **CrossEncoder reranking score:** Computed from the question and
    each candidate passage together. It is used to reorder the retrieved
    candidates. The score is a ranking value, not necessarily a
    calibrated probability.
-   **Latency:** The API's `latency_seconds` measures the Gemini
    generation call in the current implementation; it may not include
    embedding, ChromaDB retrieval, reranking, or frontend/network time.

## Evaluation: RAG vs. plain Gemini

### Goal

Check whether providing retrieved document context helps answer a
question grounded in the actual indexed PDFs.

### Test question

> According to the documents in my knowledge base, what is the
> difference between the similarity score used to rank semantic search
> results and the reranking score produced by a CrossEncoder? Explain
> how each is used in the retrieval pipeline.

Use the **same exact question** for both runs.

### Observed plain-Gemini result

The plain Gemini response gave a detailed general explanation of
bi-encoder/vector similarity and CrossEncoder reranking. However, it
referred to Reliance Industries annual-report excerpts and a
compiler-phases overview, which were not the relevant DocuMind technical
source documents in the displayed comparison. It did not establish its
answer using the user's indexed PDF passages.

**Interpretation:** The answer may be technically useful as general
background, but it is not evidence of document-grounded retrieval.

### Observed DocuMind result

DocuMind reported that the supplied excerpts did not mention similarity
scores, CrossEncoders, or retrieval pipelines. It then described the
two-stage retrieval/reranking workflow using the passages it cited.

**Interpretation:** The response showed an important grounding behavior
by flagging a lack of direct evidence. However, because it continued
with a general explanation, the prompt should make a clearer distinction
between statements supported by retrieved passages and outside
background knowledge. The answer should not be treated as fully
source-grounded unless the cited passages actually support each claim.

### Comparison

  -----------------------------------------------------------------------
  Criterion               Plain Gemini            DocuMind RAG
  ----------------------- ----------------------- -----------------------
  Uses passages retrieved No                      Yes, by design
  from the local ChromaDB                         
  index                                           

  Provides a detailed     Yes                     Yes
  general explanation                             

  Demonstrates support    Not in the observed     It identified that
  from the relevant       answer                  direct support was
  indexed PDFs                                    missing; cited passages
                                                  must be checked

  Uses the local document Not demonstrated        Partially demonstrated;
  set as evidence                                 source support needs
                                                  verification

  Overall finding         General-knowledge       Better awareness of
                          answer, not a valid     missing evidence, but
                          document-grounded       the response prompt and
                          baseline                evidence need
                                                  tightening
  -----------------------------------------------------------------------

### Conclusion

This run does **not yet prove that DocuMind answers the question more
accurately than plain Gemini**. It demonstrates the intended RAG
workflow and highlights why source-grounding checks matter. To make a
strong submission, select a question whose answer is explicitly present
in one of the indexed PDFs, run it through both systems, and compare
each claim against the exact source passage and page. Report the actual
results without assuming RAG always wins.

### Recommended follow-up test

The embeddings guide contains material on semantic search and vector
spaces. A more directly supported question to test is:

> According to the embeddings guide, what is the difference between
> symmetric and asymmetric semantic search, and when is each used?

Before using this as final evidence, inspect the retrieved source
passage and confirm that it answers both parts of the question. For the
plain-Gemini run, do not upload or paste the PDF; for DocuMind, use the
indexed PDF and save the displayed source name and page number.

## Troubleshooting

### Gemini quota or rate limit

A `429 RESOURCE_EXHAUSTED` response means the Gemini API request was
rate-limited or the project's quota was exhausted. This is not a
ChromaDB indexing failure. Check the current quota and retry when
allowed. Avoid repeated test requests while the quota is exhausted.

### Slow first request

The first request may be slower because the embedding model and
CrossEncoder weights need to be loaded. The CrossEncoder may also
download weights on its first use. Later requests in the same running
process should avoid that initial model-loading cost.

### Hugging Face unauthenticated warning

The warning about unauthenticated Hugging Face Hub requests is generally
a download-rate notice, not an application error. It does not by itself
mean retrieval failed.

### No relevant passages

-   Confirm PDFs exist in `data/`.
-   Re-run ingestion if the documents changed.
-   Confirm the expected ChromaDB collection exists and contains chunks.
-   Inspect the passages returned by `python -m src.retrieve` or the
    project's retrieval test before relying on generated answers.

## Security and Git

-   Keep `.env`, `.venv/`, `node_modules/`, model caches, and generated
    ChromaDB files out of Git.
-   Do not commit private or copyrighted documents unless you have
    permission to publish them.
-   Ensure `.gitignore` excludes local data/database files if they
    should not be public. If the assignment requires reproducibility,
    provide permitted sample documents or clear instructions for
    supplying them.

## Future improvements

-   Add a fixed evaluation set with questions and expected supporting
    passages.
-   Measure retrieval metrics such as Recall@K and MRR independently
    from answer quality.
-   Evaluate citation support and whether each answer claim is entailed
    by its cited passage.
-   Add an explicit "documents do not contain enough evidence" mode that
    avoids unsupported general-knowledge additions.
-   Measure end-to-end latency separately from Gemini generation
    latency.
-   Add automated tests for API errors, empty retrieval results, and
    missing configuration.

## License and data

Add a license and document the provenance and usage rights of the PDFs
before making the repository public.

Backend cmd:python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
frontend command :cd frontend
                  npm install
                  npm run dev    