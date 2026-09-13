# privatedoc-ai

Privacy-first document intelligence and RAG system using Python, FastAPI, and local LLMs.

## v1

The first version provides a small local API that can:

- store plain-text documents in memory
- split documents into overlapping chunks
- retrieve relevant chunks with lexical search
- generate an answer through Ollama when it is running
- return a clear fallback when no local LLM is available

The current in-memory store is intentional for v1. PostgreSQL and embedding-based retrieval can be added behind the same API after the workflow is validated.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive documentation is at `/docs`.

To enable local answer generation, start Ollama and optionally configure:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2
```

## Endpoints

- `GET /health` checks that the service is running.
- `POST /documents` accepts `{ "title": "...", "content": "..." }`.
- `GET /documents` lists uploaded documents.
- `POST /search` accepts `{ "query": "...", "limit": 5 }`.
- `POST /ask` accepts `{ "question": "...", "limit": 5 }`.
- `DELETE /documents/{document_id}` removes a document.

Run the tests with:

```bash
python -m pytest -q
```
