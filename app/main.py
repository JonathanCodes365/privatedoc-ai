from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class DocumentSummary(BaseModel):
    id: str
    title: str
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: str
    title: str
    chunk: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    generated: bool


@dataclass
class Document:
    id: str
    title: str
    chunks: list[str]


documents: dict[str, Document] = {}


def tokenize(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def split_into_chunks(content: str) -> list[str]:
    words = content.split()
    if not words:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + CHUNK_SIZE]))
        if start + CHUNK_SIZE >= len(words):
            break
    return chunks


def search_documents(query: str, limit: int) -> list[SearchResult]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    results = []
    for document in documents.values():
        for chunk in document.chunks:
            chunk_tokens = tokenize(chunk)
            score = len(query_tokens & chunk_tokens) / len(query_tokens)
            if score:
                results.append(
                    SearchResult(
                        document_id=document.id,
                        title=document.title,
                        chunk=chunk,
                        score=round(score, 3),
                    )
                )
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def ollama_answer(question: str, sources: list[SearchResult]) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    context = "\n\n".join(source.chunk for source in sources)
    prompt = (
        "Answer the question using only the provided context. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())
        answer = result.get("response")
        return answer.strip() if isinstance(answer, str) and answer.strip() else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def fallback_answer(sources: list[SearchResult]) -> str:
    if not sources:
        return "I could not find relevant information in the uploaded documents."
    return "Relevant context found, but no local LLM is available to generate an answer."


app = FastAPI(title="PrivateDoc AI", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentSummary, status_code=201)
def create_document(document: DocumentCreate) -> DocumentSummary:
    document_id = str(uuid4())
    chunks = split_into_chunks(document.content)
    stored = Document(id=document_id, title=document.title, chunks=chunks)
    documents[document_id] = stored
    return DocumentSummary(id=document_id, title=stored.title, chunks=len(chunks))


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(id=document.id, title=document.title, chunks=len(document.chunks))
        for document in documents.values()
    ]


@app.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest) -> list[SearchResult]:
    return search_documents(request.query, request.limit)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    sources = search_documents(request.question, request.limit)
    answer = ollama_answer(request.question, sources) if sources else None
    return AskResponse(
        answer=answer or fallback_answer(sources),
        sources=sources,
        generated=answer is not None,
    )


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    if documents.pop(document_id, None) is None:
        raise HTTPException(status_code=404, detail="Document not found")