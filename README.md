# RAG Lab API

An Advanced Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **Pinecone**, and **OpenAI**.

---

## Overview

RAG Lab API goes beyond basic vector search by applying **LLM-powered query enhancement** before retrieval. Instead of embedding the raw user query directly, the pipeline first generates a hypothetical answer using an LLM, then uses both the original query and the generated answer as retrieval signals. This surfaces semantically richer, more relevant results from the knowledge base.

Documents are ingested as PDFs, chunked with overlap, embedded, and stored in Pinecone. Queries are then answered using retrieved context — with source pages cited in the response.

---

## Key Features

### Query Enhancement Techniques

#### 1. Query Expansion via Generated Answer (HyDE) — Implemented

**Endpoint:** `POST /query`

**How it works:**

1. User submits a query
2. LLM generates a short hypothetical answer to the query (used as an embedding signal only — never shown to the user)
3. Both the original query and the hypothetical answer are embedded concurrently
4. Pinecone is queried with both vectors (`top_k` results each)
5. Results are fused using max-score deduplication and ranked
6. Top chunks are passed to the LLM for the final, grounded answer

**Why this improves retrieval:**
Stored chunks are written as answers, not questions. Embedding a hypothetical answer brings the query vector into the same semantic space as the stored content — significantly improving cosine similarity matching compared to embedding the raw question alone.

**Request:**
```json
POST /query
{
  "query": "What is the D/C DISC style blend?",
  "document_id": "optional-uuid-to-scope-to-one-doc",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The D/C blend indicates a person who is both results-driven and analytical... (Page 2)",
  "sources": [
    { "document_id": "abc123", "page": 2, "score": 0.8912 },
    { "document_id": "abc123", "page": 7, "score": 0.8541 }
  ]
}
```

---

#### 2. Contextual Compression — Coming Soon

#### 3. Self-Query / Metadata Filtering — Coming Soon

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-file` | Upload a PDF and index it into Pinecone |
| `GET` | `/files` | List all uploaded documents |
| `DELETE` | `/delete-file/{document_id}` | Delete a document record and its file |
| `POST` | `/query` | Query the knowledge base with HyDE expansion |

---

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/earlhansg/rag-lab-api.git
cd rag-lab-api
```

### 2. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables) below).

### 4. Run the server

```bash
python main.py
```

API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_ENV=us-east-1
```

| Variable | Where to get it |
|----------|----------------|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `PINECONE_ENV` | Pinecone dashboard → your index region (e.g. `us-east-1`) |

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Pinecone Setup

The API automatically creates the index on first startup if it does not exist.

**Index configuration used:**

| Setting | Value |
|---------|-------|
| Index name | `rag-lab` |
| Dimension | `1536` |
| Metric | `cosine` |
| Cloud / Region | `aws / us-east-1` |
| Embedding model | `text-embedding-3-small` |

To use a different region or cloud provider, update `ServerlessSpec` in `app/rag/pinecone.py`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI |
| Vector database | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| PDF parsing | PyPDF2 |
| Database | SQLite via SQLAlchemy (async) |
| Package manager | uv |
