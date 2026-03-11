from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.db import create_db_and_tables, Document, get_async_session
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
import shutil

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