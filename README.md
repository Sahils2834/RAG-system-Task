# DocuMind — Document RAG Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application that answers questions using relevant passages retrieved from a local collection of PDF documents.

It combines dense-vector semantic retrieval, CrossEncoder reranking, and Gemini-based answer generation in a React chat interface.

The application allows users to index document collections, ask natural-language questions, retrieve relevant passages, and receive contextual answers with source references and page numbers.

> **Important:** Answers are AI-generated. Always verify important claims against the cited PDF passages.

---

## Features

- Extracts text from PDF documents.
- Splits extracted content into overlapping text chunks.
- Generates text embeddings using `all-MiniLM-L6-v2`.
- Stores document chunks and metadata in a persistent ChromaDB collection.
- Retrieves relevant passages using dense-vector similarity search.
- Reranks retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generates contextual answers using the Gemini API.
- Maintains conversation history for follow-up questions.
- Provides source identifiers, document names, page numbers, and retrieved passages.
- Uses FastAPI for backend services.
- Provides a React and Vite-based chat interface.
- Includes performance diagnostics for retrieval and generation.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| LLM | Google Gemini API |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranking Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Semantic Search | Sentence Transformers |
| API Communication | REST API |
| Environment Management | Python dotenv |

---

## System Architecture

```text
                  PDF DOCUMENTS
                        |
                        v
                  PDF INGESTION
                   src/ingest.py
                        |
             Extract text from PDFs
                        |
                Split into chunks
                        |
                Generate embeddings
                        |
                        v
                   CHROMADB
                  chroma_db/
                        |
                        v
                  USER QUESTION
                        |
                        v
                 REACT FRONTEND
                  frontend/
                        |
                   POST /api/chat
                        |
                        v
                  FASTAPI BACKEND
                   src/api.py
                        |
                        v
                  RETRIEVAL ENGINE
                  src/retrieve.py
                        |
              Generate query embedding
                        |
                        v
               CHROMADB VECTOR SEARCH
                        |
               Retrieve candidate chunks
                        |
                        v
                CROSSENCODER RERANKING
                        |
                 Select top passages
                        |
                        v
                BUILD DOCUMENT CONTEXT
                        |
                        v
                   GEMINI API
                        |
              Generate grounded answer
                        |
                        v
                ANSWER + SOURCES
                        |
                        v
                 REACT FRONTEND
```

---

## Project Structure

```text
document-rag/
│
├── data/
│   └── Source PDF documents
│
├── evaluation/
│   └── questions.json
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   │
│   ├── .env
│   └── package.json
│
├── chroma_db/
│   └── Persistent vector database
│
├── results/
│   └── Evaluation outputs
│
├── src/
│   ├── api.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── ingest.py
│   └── retrieve.py
│
├── tests/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

The exact generated folders and files may vary depending on the local configuration.

---

## Requirements

- Python 3.10 or later.
- Node.js and npm.
- A Google Gemini API key.
- PDF documents that you are authorized to process.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sahils2834/RAG-system-Task.git

cd RAG-system-Task
```

### 2. Create a Python Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly.

### 3. Install Backend Dependencies

```bash
python -m pip install -r requirements.txt
```

If the required API dependencies are not included in `requirements.txt`, install:

```bash
python -m pip install fastapi "uvicorn[standard]" python-dotenv google-genai
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Replace the placeholder with your actual Gemini API key.

**Security note:** Never commit your `.env` file or expose API keys in frontend code.

### 5. Add PDF Documents

Place the PDF documents you want DocuMind to process inside the `data/` directory.

Example:

```text
data/
├── rag_research_paper.pdf
├── embeddings_guide.pdf
├── gemini_api_guide.pdf
└── chroma_guide.pdf
```

The filenames above are examples. Add the documents available in your own collection.

### 6. Ingest and Index PDF Documents

Run the following command from the project root:

```bash
python -m src.ingest
```

The ingestion pipeline:

1. Extracts text from PDF documents.
2. Splits the extracted text into overlapping chunks.
3. Generates embeddings for each chunk.
4. Stores embeddings and metadata in ChromaDB.

The resulting vector database is stored locally in:

```text
chroma_db/
```

Run ingestion again whenever you add or change source documents or ingestion settings.

### 7. Start the FastAPI Backend

From the project root:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

The backend runs at:

`http://127.0.0.1:8000`

Available endpoints:

| Endpoint | Description |
|---|---|
| `/` | API root |
| `/api/health` | Backend health check |
| `/api/chat` | Chat and RAG endpoint |
| `/docs` | Interactive API documentation |

The health endpoint checks whether the backend is running and whether a Gemini API key is configured. It does not guarantee that every generation request will succeed.

### 8. Start the React Frontend

Open a second terminal:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Create a `.env` file inside the `frontend/` directory:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open the local URL displayed by Vite, usually:

`http://localhost:5173`

Keep both backend and frontend terminals running while using DocuMind.

---

## How the RAG Pipeline Works

DocuMind uses a multi-stage retrieval and generation process.

### 1. Document Ingestion

PDF documents are processed to extract their text content.

The extracted text is divided into smaller overlapping chunks to make retrieval more focused and efficient.

### 2. Embedding Generation

Each text chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

These embeddings represent the semantic meaning of the text.

### 3. Initial Semantic Retrieval

When a user submits a question:

- The question is converted into an embedding.
- ChromaDB searches for semantically similar document chunks.
- A set of candidate passages is retrieved.

### 4. CrossEncoder Reranking

The retrieved candidates are passed to:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The CrossEncoder evaluates the relationship between the user's question and each candidate passage.

It assigns ranking scores and reorders the retrieved passages according to their relevance.

### 5. Context Construction

The highest-ranked passages are assembled into a contextual prompt.

Each passage receives a source identifier, such as:

- `[S1]`
- `[S2]`
- `[S3]`

These identifiers allow Gemini to refer to the retrieved passages in its response.

### 6. Answer Generation

The contextual prompt is sent to Gemini.

The model generates an answer using the current question, relevant retrieved passages, and conversation history.

### 7. Source Display

The backend returns:

- Generated answer.
- Source document names.
- Page numbers.
- Retrieved passages.
- Retrieval and generation performance timings.
- Generation finish reason.

The React frontend displays the answer and its associated source information.

---

## Understanding Retrieval Scores and Performance

### Vector Distance

Vector distance is used during the initial ChromaDB search to identify semantically similar candidate passages.

The API returns this value under the `distance` field.

For the configured distance metric, a lower distance indicates a closer match.

A distance value should not automatically be interpreted as a similarity percentage.

### CrossEncoder Reranking Score

The CrossEncoder calculates a relevance score using the question and candidate passage together.

This score helps reorder the initial retrieval results.

It is a ranking value and should not automatically be interpreted as a calibrated probability.

### Performance Metrics

DocuMind records:

- Retrieval and reranking time.
- Gemini generation time.
- Total request time.
- Prompt token count, where available.
- Output token count, where available.
- Generation finish reason.

These metrics help identify performance bottlenecks and monitor the response-generation process.

---

# Evaluation: RAG vs. Without RAG

## 1. Objective

The purpose of this experiment is to demonstrate the difference between conventional LLM-based question answering and Retrieval-Augmented Generation.

The experiment uses a question whose answer is available in the indexed `embeddings_guide.pdf`.

The same question is used for both approaches.

### Evaluation Question

> According to the Embeddings Guide, explain symmetric and asymmetric semantic search in detail. Include their definitions, differences, suitable models, encoding methods, use cases, and examples. Present the comparison in a table and cite the relevant pages.

---

## 2. Experiment A: Without RAG (Plain Gemini)

### Methodology

In the non-RAG approach:

1. The question is submitted directly to Gemini.
2. No local PDF documents are retrieved.
3. ChromaDB is not used.
4. CrossEncoder reranking is not performed.
5. Gemini generates a response using its existing model knowledge.

### Expected Characteristics

A plain LLM can provide a general explanation of symmetric and asymmetric semantic search.

For example:

- Symmetric semantic search involves queries and corpus entries with approximately similar lengths and content.
- Asymmetric semantic search involves a short query retrieving a longer passage.
- Symmetric search can be used for similar-question retrieval.
- Asymmetric search can be used for information retrieval and question answering.

However, without the PDF context, the response cannot establish that its statements, examples, and page references correspond to the actual indexed Embeddings Guide.

**Limitation:** This describes the expected non-RAG baseline. A separate measured output for this exact question has not been included in the experiment results.

---

## 3. Experiment B: With RAG (DocuMind)

### Methodology

The same evaluation question is submitted through DocuMind.

The system:

1. Embeds the user's question.
2. Searches the indexed Embeddings Guide using ChromaDB.
3. Retrieves relevant document chunks.
4. Reranks the candidate passages using the CrossEncoder.
5. Sends the retrieved passages to Gemini.
6. Generates a contextual answer.
7. Returns source references and page numbers.

### Observed Result

DocuMind successfully generated a detailed response explaining both symmetric and asymmetric semantic search.

The response included the following information.

### A. Symmetric Semantic Search

**Definition:**

Symmetric semantic search involves queries and corpus entries that have approximately the same length and amount of content.

In these tasks, the query and corpus entries can potentially be interchanged.

**Suitable models:**

- Pretrained sentence embedding models.
- `distilbert-multilingual-nli-stsb-quora-ranking`.
- `SPECTER`.

**Encoding method:**

The general `SentenceTransformer.encode()` method can be used.

**Use cases:**

- Similar-question retrieval.
- Duplicate question detection.
- Similar scientific publication retrieval.

**Example:**

Query:

"How to learn Python online?"

Retrieved entry:

"How to learn Python on the web?"

### B. Asymmetric Semantic Search

**Definition:**

Asymmetric semantic search involves a short query, such as a question or keywords, being used to retrieve a longer paragraph or document passage.

Unlike symmetric search, the query and corpus entries generally cannot be interchanged.

**Suitable models:**

- Pretrained MS MARCO models.

**Encoding methods:**

- `SentenceTransformer.encode_query()` for query encoding.
- `SentenceTransformer.encode_document()` for corpus document encoding.

These methods can use predefined query and document prompts and support appropriate task routing in models equipped with a Router module.

**Use cases:**

- Information retrieval.
- Question answering.
- Retrieving explanatory passages from longer documents.

**Example:**

Query:

"What is Python?"

Retrieved passage:

"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."

### C. Retrieved Source References

The generated DocuMind response included these source references:

| Source | Page | Retrieved information |
|---|---:|---|
| embeddings_guide.pdf | 10 | Symmetric and asymmetric search definitions and encoding approaches |
| embeddings_guide.pdf | 9 | Characteristics of symmetric semantic search |
| embeddings_guide.pdf | 16 | Model examples and applications |

The response included explanations, examples, a comparison table, and citations to the relevant PDF pages.

---

## 4. Comparison: Without RAG vs. With RAG

| Evaluation Criterion | Without RAG | DocuMind with RAG |
|---|---|---|
| Uses local indexed PDFs | No | Yes |
| Retrieves document passages | No | Yes |
| Performs vector similarity search | No | Yes |
| Uses CrossEncoder reranking | No | Yes |
| Generates natural-language answers | Yes | Yes |
| Uses document-specific context | Not provided | Yes |
| Provides source document references | Not guaranteed | Yes |
| Provides PDF page references | Not guaranteed | Yes |
| Supports document-grounded responses | Not through retrieval | Yes, through retrieved context |
| Maintains conversation context | Depends on implementation | Supported |

---

## 5. Experimental Conclusion

The experiment demonstrates the difference between general-purpose LLM generation and a document-based Retrieval-Augmented Generation workflow.

Without RAG, Gemini can generate a general explanation based on its existing knowledge. However, it does not retrieve evidence from the local PDF collection.

With RAG, DocuMind retrieves relevant passages from the indexed Embeddings Guide using ChromaDB, reranks the retrieved candidates using a CrossEncoder, and provides the selected passages to Gemini for answer generation.

The observed DocuMind response successfully covered the requested concepts and included source document references and page numbers.

This demonstrates the intended retrieval and source-grounding capabilities of DocuMind.

However, citations and generated claims should still be checked against the original passages. The experiment demonstrates the RAG workflow; it does not establish that RAG is universally more accurate than plain Gemini.

---

## 6. Reproducing the Experiment

To reproduce the DocuMind experiment:

1. Place `embeddings_guide.pdf` in the `data/` directory.
2. Run document ingestion if the document has not already been indexed.

```bash
python -m src.ingest
```

3. Start the FastAPI backend.

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

4. Start the React frontend.

```bash
cd frontend
npm install
npm run dev
```

5. Open DocuMind in the browser.
6. Submit the evaluation question.
7. Inspect the generated answer, retrieved source passages, and page references.

For a non-RAG baseline, submit the exact same question directly to Gemini without providing the PDF or retrieved document context.

---

## Troubleshooting

### Gemini API Quota or Rate Limit

A `429 RESOURCE_EXHAUSTED` response indicates that the Gemini API request was rate-limited or the available quota was exhausted.

This does not necessarily indicate a problem with ChromaDB or the retrieval pipeline.

Check your Gemini API quota and retry when requests are available.

### Gemini Temporarily Unavailable

A `503` response may indicate that the model service is temporarily unavailable or overloaded.

Retry after some time.

### Slow First Request

The first request can take longer because the embedding model and CrossEncoder weights may need to be loaded.

Subsequent requests in the same running process should avoid most of this initial model-loading overhead.

### Hugging Face Authentication Warning

An unauthenticated Hugging Face Hub warning is generally related to download rate limits.

It does not by itself indicate that the application or retrieval pipeline has failed.

### No Relevant Passages Retrieved

If DocuMind cannot find relevant passages:

- Confirm that the PDFs exist in `data/`.
- Run the ingestion pipeline again if documents have changed.
- Verify that the ChromaDB collection contains indexed chunks.
- Check the retrieved passages before relying on the generated answer.

---

## Security and Git Guidelines

- Never commit `.env` files or API keys.
- Keep `.venv/` and `node_modules/` out of Git.
- Avoid committing generated model caches and local database files unless required.
- Do not publish private or copyrighted PDF documents without permission.
- Use `.gitignore` to exclude sensitive files and unnecessary generated content.
- Provide instructions for supplying sample documents when reproducibility is required.

---

## Future Improvements

Potential future improvements include:

- Adding a fixed evaluation dataset with expected supporting passages.
- Measuring retrieval metrics such as Recall@K and Mean Reciprocal Rank (MRR).
- Evaluating citation correctness and source support.
- Improving handling of questions with insufficient document evidence.
- Measuring end-to-end response latency.
- Supporting additional document formats.

---

## License and Data Usage

A suitable software license and documentation of document provenance should be added before public distribution.

Only publish documents for which you have the necessary permissions.

---

## Repository

**GitHub:** https://github.com/Sahils2834/RAG-system-Task.git

**Project:** DocuMind — Document RAG Assistant

**Developer:** Sahil Shete