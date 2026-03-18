import asyncio
import hashlib
import logging
from app.rag.pinecone import index, openai_client, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

TOP_K = 5  # results per vector search

# ---------------------------------------------------------------------------
# Simple in-process cache: query_hash -> hypothetical answer string
# For production, swap this dict for Redis with a TTL.
# ---------------------------------------------------------------------------
_hyde_cache: dict[str, str] = {}


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Step 1 — Generate a hypothetical answer (HyDE)
# ---------------------------------------------------------------------------
async def generate_hypothetical_answer(query: str) -> str:
    """
    Ask the LLM to write a short, general answer to the query as if it already
    knew the answer. The result is used purely as an embedding signal — it is
    never shown to the user — so we keep it concise and factual to reduce
    hallucination noise.
    """
    key = _cache_key(query)
    if key in _hyde_cache:
        logger.info("HyDE cache hit for query.")
        return _hyde_cache[key]

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",  # cheap + fast; only used for embedding signal
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short, factual paragraph (3-5 sentences) that would "
                    "directly answer the question below. Be general — do not invent "
                    "specific names, numbers, or dates. Focus on concepts and structure."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        temperature=0.3,  # low temp = less hallucination
    )

    hypothetical = response.choices[0].message.content.strip()
    _hyde_cache[key] = hypothetical
    logger.info(f"HyDE generated ({len(hypothetical)} chars).")
    return hypothetical


# ---------------------------------------------------------------------------
# Step 2 — Embed a text string
# ---------------------------------------------------------------------------
async def _embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step 3 — Multi-vector retrieval + result fusion
# ---------------------------------------------------------------------------
async def retrieve_with_expansion(
    query: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> list[dict]:
    """
    Embed both the original query and the hypothetical answer, query Pinecone
    with each, then merge results by deduplication + score fusion.

    Args:
        query:       The raw user question.
        top_k:       Number of results to retrieve per vector.
        document_id: Optional — filter retrieval to a specific document.
    """
    hypothetical = await generate_hypothetical_answer(query)

    # Embed both vectors concurrently
    query_vec, hyde_vec = await asyncio.gather(
        _embed(query),
        _embed(hypothetical),
    )

    pinecone_filter = {"document_id": {"$eq": document_id}} if document_id else None

    # Query Pinecone with both vectors
    def _query(vector: list[float]) -> list[dict]:
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        return result.matches

    query_matches, hyde_matches = await asyncio.gather(
        asyncio.to_thread(_query, query_vec),
        asyncio.to_thread(_query, hyde_vec),
    )

    # --- Result fusion ---
    # Keep the best score per chunk ID across both result sets.
    # Using max-score fusion: if a chunk appears in both, keep the higher score.
    seen: dict[str, dict] = {}
    for match in query_matches + hyde_matches:
        chunk_id = match.id
        if chunk_id not in seen or match.score > seen[chunk_id]["score"]:
            seen[chunk_id] = {
                "id": chunk_id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "page": match.metadata.get("page"),
                "document_id": match.metadata.get("document_id"),
            }

    # Sort by score descending, return top_k
    fused = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(
        f"Retrieved {len(query_matches)} (query) + {len(hyde_matches)} (HyDE) → "
        f"{len(fused)} after fusion."
    )
    return fused


# ---------------------------------------------------------------------------
# Step 4 — Final answer generation using retrieved context
# ---------------------------------------------------------------------------
async def answer_with_context(query: str, chunks: list[dict]) -> str:
    """
    Build a context string from the retrieved chunks and call the LLM for the
    final, grounded answer shown to the user.
    """
    if not chunks:
        return "I could not find relevant information to answer your question."

    context = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in chunks
    )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using "
                    "ONLY the context provided below. If the answer is not in the "
                    "context, say so. Cite the page number when possible."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
