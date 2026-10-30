# Document RAG System — Semantic Search and Gemini-Powered Question Answering

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions using information retrieved from a collection of PDF documents.

The system combines semantic search with Google's Gemini API to retrieve relevant document passages and generate context-aware answers. It also compares RAG-generated answers with answers produced by Gemini without access to the indexed documents.

The main objective is to investigate whether grounding a language model in retrieved documents improves its ability to answer document-specific questions and identify supporting sources.

## 2. Objectives

- Extract text from PDF documents.
- Split extracted text into manageable chunks with overlapping context.
- Generate semantic embeddings using Sentence Transformers.
- Store document chunks and their metadata in ChromaDB.
- Retrieve relevant passages for a user query.
- Generate answers using the Google Gemini API.
- Include source references in generated answers.
- Compare document-grounded responses with plain Gemini responses.
- Evaluate retrieval relevance, answer correctness, and source attribution.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application and pipeline implementation |
| PyMuPDF | Extracting text from PDF documents |
| Sentence Transformers | Generating semantic embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Persistent vector database and similarity search |
| Google GenAI SDK | Communicating with the Gemini API |
| Gemini 3.8 Flash | Language model configured for answer generation |
| python-dotenv | Loading environment variables |
| Git | Version control |

## 4. System Architecture

The application follows a basic RAG pipeline.

1. **Document ingestion:** PDF files are loaded and their text is extracted page by page.
2. **Text chunking:** Extracted text is divided into smaller overlapping chunks.
3. **Embedding generation:** Sentence Transformers converts chunks into numerical vector representations.
4. **Vector storage:** Chunks, embeddings, and metadata are stored in a persistent ChromaDB collection.
5. **Query embedding:** A user's question is converted into an embedding using the same embedding model.
6. **Semantic retrieval:** ChromaDB returns the most relevant document chunks.
7. **Prompt construction:** Retrieved passages are added to the prompt with source labels and page metadata.
8. **Answer generation:** Gemini generates a response based on the retrieved context.
9. **Source reporting:** The application displays retrieved filenames, page numbers, chunk information, and distances.
10. **Baseline comparison:** Plain Gemini answers the same question without the indexed PDF passages.

### Architecture Diagram

```text
              PDF Documents
                    |
                    v
           Text Extraction
                    |
                    v
             Text Chunking
                    |
                    v
          Embedding Generation
          all-MiniLM-L6-v2
                    |
                    v
               ChromaDB
             Vector Storage
                    |
             User Question
                    |
                    v
            Query Embedding
                    |
                    v
            Similarity Search
                    |
                    v
          Top Relevant Chunks
                    |
                    v
          Context + User Query
                    |
                    v
             Gemini API
                    |
                    v
        Answer with Source Labels
                    |
                    v
           RAG vs Plain Gemini
              Comparison
```

## 5. Project Structure

```text
document-rag/
├── data/
│   ├── chroma_guide.pdf
│   ├── embeddings_guide.pdf
│   ├── gemini_api_guide.pdf
│   └── rag_research_paper.pdf
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── retrieve.py
│   └── generate.py
├── tests/
├── results/
├── chroma_db/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

The `chroma_db/` directory is generated during ingestion and stores the persistent vector database. It should generally be excluded from Git because it can be recreated from the source documents.

## 6. Dataset and Document Collection

The system indexes four PDF documents.

| Document | Pages |
|---|---:|
| chroma_guide.pdf | 35 |
| embeddings_guide.pdf | 17 |
| gemini_api_guide.pdf | 14 |
| rag_research_paper.pdf | 19 |
| **Total** | **85** |

### Ingestion Statistics

| Metric | Observed result |
|---|---:|
| PDF documents | 4 |
| Total pages | 85 |
| Pages with extracted text | 85 |
| Extracted words | 19,283 |
| Chunks created and stored | 95 |
| Embedding model | all-MiniLM-L6-v2 |
| ChromaDB collection | rag_documents |
| Retrieval configuration | Top 4 chunks |

These figures describe the current ingestion run and may change if the source documents or chunking configuration change.

## 7. Implementation Details

### Text Extraction and Chunking

PyMuPDF extracts text from each PDF page. The extracted text is split into word-based chunks to make relevant passages easier to retrieve.

Current configuration:

- Chunk size: 500 words
- Chunk overlap: 80 words
- Chunking: Page-based
- Metadata: Source filename, page number, and chunk index

Chunk overlap helps preserve context across chunk boundaries.

### Embedding Generation

The `all-MiniLM-L6-v2` Sentence Transformers model generates vector representations for document chunks and user queries.

The embeddings are normalized, and the ChromaDB collection uses cosine distance for similarity search.

### Vector Storage and Retrieval

ChromaDB stores document chunks and associated metadata in a persistent collection named `rag_documents`.

For each query, the application retrieves up to four relevant chunks. The retrieved metadata includes the source PDF and page number, which are used to identify the supporting material.

### Answer Generation

The retrieved passages are incorporated into a prompt sent to Gemini.

The prompt instructs the model to:

- Answer using the retrieved excerpts.
- Avoid inventing information not supported by the context.
- Indicate when the excerpts do not contain sufficient information.
- Cite the supplied source labels.
- Treat retrieved document text as reference material rather than instructions.

### Baseline Comparison

The application also sends the question to Gemini without retrieved document context.

This baseline helps demonstrate the difference between document-grounded answering and answering from the model's general knowledge.

## 8. Experimental Results

Four document-question-answering tests were performed. The following findings are based on the observed application outputs.

### Test 1: ChromaDB Capabilities

**Question:** According to the indexed documents, what specific capabilities does the Chroma introduction describe, and which PDF and page contain that information?

**RAG result:**

The generated response described:

- Local and self-hosted deployment options.
- Managed or serverless Chroma Cloud.
- The Chroma SDK.
- Scalable database capabilities.
- Agentic search and code search with AST-aware chunking.

**Retrieved evidence:**

- Document: `chroma_guide.pdf`
- Page: 2
- Supporting label in the generated answer: `[Source 3]`

**Plain Gemini result:**

Plain Gemini provided general information about ChromaDB but could not identify the local PDF page from the indexed collection.

**Observation:**

The RAG response identified a document-specific source and provided a more targeted answer. The cited introduction page should be checked against the exact claims before treating every item as independently verified.

### Test 2: Sentence Transformer Characteristics

**Question:** What are the four characteristics of Sentence Transformer bi-encoder models described in the Quickstart section, and which PDF page contains them?

**RAG result:**

The answer identified four characteristics:

1. Producing fixed-size vector representations for supported inputs.
2. Efficient embedding generation and fast similarity calculations.
3. Supporting tasks such as semantic similarity, semantic search, clustering, classification, and paraphrase mining.
4. Serving as the first stage of a two-stage retrieval process, with a Cross-Encoder reranking the top results.

**Retrieved evidence:**

- Document: `embeddings_guide.pdf`
- Page: 1
- Retrieved distance: 0.4304

**Plain Gemini result:**

Plain Gemini described general bi-encoder properties but could not identify the specific four points in the indexed Quickstart section or its page number.

**Observation:**

The RAG response was aligned with the expected content of the Quickstart section. Exact wording and all four points should be verified against page 1 of the source PDF.

### Test 3: Gemini API Interaction

**Question:** What Python approach does `gemini_api_guide.pdf` recommend for creating an interaction with Gemini, including the model name and method used?

**RAG result:**

The generated answer identified:

- SDK: `google-genai`
- Method: `client.interactions.create()`
- Model: `gemini-3.8-flash`

The answer included both standard and streaming interaction examples.

**Retrieved evidence:**

- Document: `gemini_api_guide.pdf`
- Page: 3
- Retrieved distance for the cited chunk: 0.4339

**Plain Gemini result:**

Plain Gemini returned an older `google-generativeai` SDK example using `GenerativeModel` and `generate_content()`, with older model names such as `gemini-1.5-flash`.

**Observation:**

This test illustrates the value of retrieving version-specific documentation. The RAG answer followed the indexed guide, while the baseline supplied a different API approach that did not answer the document-specific question.

### Test 4: ChromaDB Collections and Embeddings

**Question:** Explain the difference between a Chroma collection and the embeddings stored in it, using the indexed documents and page citations.

**RAG result:**

The generated answer explained that:

- A collection organizes records, IDs, documents, metadata, and embeddings.
- An embedding is a numerical vector representation associated with a record.
- Embeddings can be generated by a configured embedding function or supplied manually.
- Precomputed embeddings can be stored without Chroma re-embedding the text.

**Retrieved evidence:**

- Document: `chroma_guide.pdf`
- Pages retrieved: 11, 13, 16, and 6
- Closest retrieved distance: 0.3059

**Plain Gemini result:**

Plain Gemini explained the same general distinction but did not identify supporting pages in the local PDF.

**Observation:**

Both answers communicated the general concept. The RAG response additionally referenced the indexed material and its page metadata.

### Test 5: RAG Fundamentals

**Question:** What problem does Retrieval-Augmented Generation solve, and how does retrieving external documents help a language model?

**RAG result:**

The generated answer discussed:

- Limitations of knowledge stored only in model parameters.
- Hallucinations and difficulty updating internal knowledge.
- Lack of transparent provenance.
- Combining parametric memory with non-parametric external memory.
- Updating an external index without retraining the language model.
- Improved factuality, specificity, inspectability, and task performance as discussed in the research paper.

**Retrieved evidence:**

- Document: `rag_research_paper.pdf`
- Pages retrieved: 9, 7, 1, and 8
- Closest retrieved distance: 0.3741

**Plain Gemini result:**

Plain Gemini provided a general explanation of RAG, including external retrieval, grounding, knowledge updates, and source traceability. It did not identify the specific passages in the local research paper.

**Observation:**

This test shows that both approaches can explain a general technical concept. The distinctive advantage of RAG is access to the supplied research document and its associated source metadata, not necessarily a longer or more sophisticated answer.

## 9. Results Summary

| Evaluation criterion | Observed result |
|---|---|
| PDF ingestion | Successful |
| Text extraction | 85 of 85 pages |
| Chunk creation | 95 chunks |
| Persistent vector storage | Successful |
| Semantic retrieval | Relevant chunks retrieved in the tested queries |
| Source filename and page metadata | Available |
| Gemini answer generation | Successful in completed test runs |
| Source labels in answers | Present |
| Plain Gemini baseline | Completed |
| Quantitative accuracy evaluation | Not yet performed |
| Automated citation verification | Not yet implemented |

The tests provide qualitative evidence that the retrieval and generation pipeline functions as intended. They do not establish a numerical accuracy improvement, a reduction in hallucinations, or a statistically significant advantage over the baseline.

## 10. Limitations

1. **Small document collection:** The current evaluation uses four PDFs and a limited number of manually selected questions.
2. **Qualitative evaluation:** Answers have not yet been scored using a standardized rubric or an independent ground-truth dataset.
3. **Citation verification:** Source labels and page metadata are available, but citation correctness is not automatically checked.
4. **Chunking quality:** Page-based chunking can produce short or noisy chunks, including navigation or footer text.
5. **Retrieval limits:** Retrieving only four chunks may omit useful information for questions that require multiple passages.
6. **Model availability:** Gemini API requests may fail temporarily because of service load, rate limits, or model availability.
7. **Latency and cost:** Embedding, retrieval, and generation latency and API costs have not yet been benchmarked.
8. **Baseline differences:** Plain Gemini does not have access to the indexed documents, so the comparison evaluates document-specific grounding rather than model quality under identical information conditions.

## 11. Future Improvements

- Improve PDF text cleaning and chunk boundaries.
- Evaluate alternative chunk sizes and overlaps.
- Add configurable retrieval depth.
- Introduce reranking using a Cross-Encoder.
- Implement retry logic and clearer API error handling.
- Add automated tests for ingestion, retrieval, and generation.
- Build a question-answer evaluation dataset with expected answers and source pages.
- Measure retrieval Recall@k, answer correctness, citation accuracy, latency, and API cost.
- Add a web interface for document upload and question answering.
- Support multiple collections or user-specific document libraries.
- Add document-level access controls and safer handling of untrusted retrieved content.

## 12. Setup and Execution

### Prerequisites

- Python installed on Windows.
- A valid Google Gemini API key.
- Internet access for model/API operations.

### Installation

Open a terminal in the project root:

```bash
cd /d "A:\Valentius Kryptix\document-rag"
```

Create and activate a virtual environment if one is not already active:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Replace the placeholder with your own API key. Never commit the actual key to version control.

### Ingest the PDFs

```bash
python -m src.ingest
```

This extracts the document text, creates chunks and embeddings, and stores the results in ChromaDB.

Run ingestion again when the source PDFs or ingestion configuration changes.

### Retrieve Relevant Chunks

```bash
python -m src.retrieve
```

Follow the prompts to test semantic retrieval.

### Ask Questions Using RAG

```bash
python -m src.generate
```

Enter a question to receive a document-grounded answer, inspect the retrieved sources, and compare the response with plain Gemini.

Type `exit` to stop.

## 13. Security and Version Control

The `.gitignore` file should exclude at least:

```gitignore
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

Do not commit API keys, virtual environments, or generated vector database files.

The source PDFs may also be excluded if they are copyrighted, confidential, or too large for the repository. Document any requirements for obtaining the source files separately.

## 14. Conclusion

This project demonstrates a working PDF-based Retrieval-Augmented Generation pipeline using Python, Sentence Transformers, ChromaDB, and the Gemini API.

The implementation extracts and indexes document content, retrieves semantically relevant passages, and uses those passages to generate answers with source references. Initial tests show that RAG can provide document-specific context and page references that the plain Gemini baseline cannot derive from the local collection alone.

Further work should focus on automated evaluation, citation verification, retrieval-quality improvements, and robustness testing before drawing quantitative conclusions about answer accuracy or reliability.


Backend cmdd:python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
frontend command :cd frontend
                  npm install
                  npm run dev      