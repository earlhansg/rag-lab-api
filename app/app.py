from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.db import create_db_and_tables, Document, get_async_session
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import os
import shutil
from app.rag import embed_and_save_to_pinecone, retrieve_with_expansion, answer_with_context
from uuid import UUID

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

UPLOAD_FOLDER = "app/knowledge"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), session: AsyncSession = Depends(get_async_session)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed."}
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    doc = Document(filename=file.filename, path=file_path, status="uploaded")
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Generate embeddings and save to Pinecone
    await embed_and_save_to_pinecone(str(doc.id), file_path)

    return {"message": "File uploaded successfully!", "document_id": doc.id}

@app.get("/files")
async def get_all_files(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Document))
    documents = result.scalars().all()
    
    return {
        "files": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "path": doc.path,
                "status": doc.status,
                "namespace": doc.namespace,
                "created_at": doc.created_at,
            }
            for doc in documents
        ]
    }

class QueryRequest(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int = 5


@app.post("/query")
async def query_documents(body: QueryRequest):
    chunks = await retrieve_with_expansion(
        query=body.query,
        top_k=body.top_k,
        document_id=body.document_id,
    )
    answer = await answer_with_context(body.query, chunks)
    return {
        "answer": answer,
        "sources": [
            {"document_id": c["document_id"], "page": c["page"], "score": round(c["score"], 4)}
            for c in chunks
        ],
    }


@app.delete("/delete-file/{document_id}")
async def delete_all_files(document_id: UUID, session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        return {"error": "Document not found."}
    
    # Delete the file from disk
    if os.path.exists(document.path):
        os.remove(document.path)
    
    # Delete the document record from the database
    await session.delete(document)
    await session.commit()
    
    return {"message": "File deleted successfully!"}