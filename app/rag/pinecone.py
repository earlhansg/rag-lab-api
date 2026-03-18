from pinecone import Pinecone, ServerlessSpec
from openai import AsyncOpenAI
import os
import logging
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
INDEX_NAME = "rag-lab"

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks to preserve context

#Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists with correct dimension, recreate if mismatched
existing = {idx.name: idx for idx in pc.list_indexes()}
if INDEX_NAME in existing:
    actual_dim = existing[INDEX_NAME].dimension
    if actual_dim != EMBEDDING_DIMENSION:
        logger.warning(
            f"Index '{INDEX_NAME}' has dimension {actual_dim} but expected {EMBEDDING_DIMENSION}. "
            f"Deleting and recreating."
        )
        pc.delete_index(INDEX_NAME)
        existing.pop(INDEX_NAME)

if INDEX_NAME not in existing:
    logger.info(f"Creating Pinecone index '{INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}.")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

#OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_chunks_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from each page and split into overlapping chunks.
    Returns a list of dicts with 'text', 'page', and 'chunk_index'.
    """
    reader = PdfReader(file_path)
    chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if not page_text:
            continue

        # Slide a window of CHUNK_SIZE across the page text with CHUNK_OVERLAP
        start = 0
        while start < len(page_text):
            end = start + CHUNK_SIZE
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

            if end >= len(page_text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP  # slide forward with overlap

    return chunks


async def embed_and_save_to_pinecone(document_id: str, file_path: str):
    chunks = extract_chunks_from_pdf(file_path)
    if not chunks:
        logger.warning(f"No text extracted from {file_path}")
        return

    logger.info(f"Document {document_id}: {len(chunks)} chunks to embed.")
    vectors = []

    for chunk in chunks:
        response = await openai_client.embeddings.create(
            input=chunk["text"],
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(embedding)}, expected {EMBEDDING_DIMENSION}"
            )
        vectors.append({
            "id": f"{document_id}-chunk-{chunk['chunk_index']}",
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        })

    # Upsert in batches of 100 (Pinecone recommended batch size)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors) for document {document_id}")

    logger.info(f"Document {document_id}: all {len(vectors)} vectors upserted successfully.")