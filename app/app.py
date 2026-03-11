from fastapi import FastAPI, UploadFile, File
from app.db import create_db_and_tables
from contextlib import asynccontextmanager
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
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed."}
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully!"}