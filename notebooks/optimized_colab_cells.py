"""
=============================================================
OPTIMIZED COLAB CELLS - Copy từng CELL vào notebook tương ứng
=============================================================
Thay thế CELL 6 và CELL 7 trong Model_Serving_API(Colab).ipynb

THAY ĐỔI CHÍNH:
- CELL 6: Thêm embedding cache + BM25 pre-filter
- CELL 7: Thêm endpoint /index, tối ưu /retrieve (không cần nhận chunks)
=============================================================
"""

# ============================================================
# CELL 6 (THAY THẾ): EMBEDDING CACHE + BM25 + DENSE RETRIEVAL
# ============================================================
# Dán toàn bộ nội dung dưới vào CELL 6

CELL_6 = """
import numpy as np
from typing import List, Dict, Any

# ----------------------------------------------------------------
# SERVER-SIDE EMBEDDING CACHE
# Lưu embeddings sau khi index → mỗi câu hỏi chỉ encode 1 vector
# ----------------------------------------------------------------
_cache: Dict[str, Any] = {
    "chunks": [],            # chunk gốc (plain text)
    "enhanced_chunks": [],   # chunk đã thêm heading prefix
    "embeddings": None,      # np.ndarray shape (N, D), normalized
    "bm25": None,            # BM25 index (nếu có rank_bm25)
}

# Thử import BM25 (cài thêm nếu chưa có: !pip install -q rank_bm25)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
    print("✅ rank_bm25 sẵn sàng (Hybrid Search ON)")
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 chưa cài → dùng Dense-only Search (pip install rank_bm25 để bật Hybrid)")


def _tokenize(text: str) -> List[str]:
    """Tokenize đơn giản cho BM25 (tách chữ, lowercase)."""
    import re
    return re.findall(r"\w+", text.lower())


def build_index(chunks: List[str], chunk_headings: List[str]) -> int:
    """
    Encode toàn bộ chunks và lưu vào cache server-side.
    Gọi 1 lần sau khi upload document.
    Trả về số chunks đã index.
    """
    if not chunks:
        _cache["chunks"] = []
        _cache["enhanced_chunks"] = []
        _cache["embeddings"] = None
        _cache["bm25"] = None
        return 0

    # Bước 1: Contextual enhancement (ghép heading vào chunk)
    enhanced = []
    for i, chunk in enumerate(chunks):
        heading = chunk_headings[i] if chunk_headings and i < len(chunk_headings) else ""
        if heading:
            enhanced.append(f"Tiêu đề: {heading}\\nNội dung: {chunk}")
        else:
            enhanced.append(chunk)

    # Bước 2: Dense embeddings (BGE-M3)
    print(f"🔄 Encoding {len(enhanced)} chunks với bge-m3...")
    embs = embedding_model.encode(
        enhanced,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    print(f"✅ Embedding xong: shape={embs.shape}")

    # Bước 3: BM25 index (nếu có thư viện)
    bm25_index = None
    if BM25_AVAILABLE:
        tokenized = [_tokenize(c) for c in enhanced]
        bm25_index = BM25Okapi(tokenized)
        print("✅ BM25 index xong")

    # Lưu vào cache
    _cache["chunks"] = chunks
    _cache["enhanced_chunks"] = enhanced
    _cache["embeddings"] = embs
    _cache["bm25"] = bm25_index

    return len(chunks)


def retrieve_from_cache(
    question: str,
    top_k: int = 5,
    candidate_pool_size: int = 20,
) -> List[str]:
    """
    Retrieve từ cache embeddings (không encode lại chunks).

    Pipeline:
      1. BM25 pre-filter (nếu có) → top candidate_pool_size candidates
      2. Bi-Encoder dense search (dùng cached embeddings) → top candidate_pool_size
      3. Merge + deduplicate candidates
      4. Cross-Encoder reranking → top_k cuối cùng
    """
    chunks = _cache["chunks"]
    enhanced_chunks = _cache["enhanced_chunks"]
    chunk_embs = _cache["embeddings"]

    if not chunks or chunk_embs is None:
        return []

    actual_top_k = min(top_k, len(chunks))
    if len(chunks) <= actual_top_k:
        return chunks  # Ít hơn top_k → trả hết

    actual_pool = min(candidate_pool_size, len(chunks))

    # ── Stage 1: Dense Search (Bi-Encoder) ──────────────────────
    # Chỉ encode câu hỏi (1 vector, cực nhanh)
    query_emb = embedding_model.encode(question, normalize_embeddings=True)
    dense_sims = np.dot(chunk_embs, query_emb)  # cosine similarity
    dense_top_idx = set(np.argsort(dense_sims)[::-1][:actual_pool].tolist())

    # ── Stage 2: BM25 Keyword Search (Hybrid) ───────────────────
    bm25_top_idx = set()
    if _cache["bm25"] is not None:
        tokenized_query = _tokenize(question)
        bm25_scores = _cache["bm25"].get_scores(tokenized_query)
        bm25_top_idx = set(np.argsort(bm25_scores)[::-1][:actual_pool].tolist())

    # ── Stage 3: Merge candidates ────────────────────────────────
    candidate_indices = list(dense_top_idx | bm25_top_idx)  # union
    # Nếu quá nhiều ứng viên, cắt bớt theo điểm dense
    if len(candidate_indices) > actual_pool * 2:
        candidate_indices = sorted(
            candidate_indices,
            key=lambda i: dense_sims[i],
            reverse=True,
        )[:actual_pool * 2]

    candidate_chunks = [enhanced_chunks[i] for i in candidate_indices]

    # ── Stage 4: Cross-Encoder Reranking ─────────────────────────
    pairs = [[question, cand] for cand in candidate_chunks]
    ce_scores = reranker_model.predict(pairs, batch_size=16)  # batch để nhanh hơn
    reranked_idx = np.argsort(ce_scores)[::-1][:actual_top_k]

    # Map ngược về index gốc, sắp xếp theo thứ tự xuất hiện trong document
    final_indices = sorted([candidate_indices[i] for i in reranked_idx])

    return [chunks[i] for i in final_indices]


# Hàm cũ (giữ để backward compatible)
def select_relevant_chunks(
    chunks: List[str],
    chunk_headings: List[str],
    question: str,
    top_k: int = 5,
    candidate_pool_size: int = 20,
) -> List[str]:
    build_index(chunks, chunk_headings)
    return retrieve_from_cache(question, top_k, candidate_pool_size)


print("✅ Embedding Cache + BM25 Hybrid Retrieval đã sẵn sàng!")
print(f"   BM25 status: {'ON' if BM25_AVAILABLE else 'OFF (cài rank_bm25 để bật)'}")
"""

# ============================================================
# CELL 7 (THAY THẾ): FASTAPI VỚI /index + /retrieve TỐI ƯU
# ============================================================
# Dán toàn bộ nội dung dưới vào CELL 7

CELL_7 = """
import threading
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="RAG Serving API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 200

class SummarizeResponse(BaseModel):
    summary: str
    chunk_count: int

class IndexRequest(BaseModel):
    \"\"\"Nhận toàn bộ chunks khi upload document (1 lần).\"\"\"
    chunks: List[str]
    chunk_headings: Optional[List[str]] = []

class IndexResponse(BaseModel):
    indexed_count: int
    message: str

class RetrieveRequest(BaseModel):
    \"\"\"
    Chỉ nhận câu hỏi - KHÔNG cần gửi chunks nữa (đã cache server-side).
    Vẫn giữ trường chunks/chunk_headings để backward compatible với v2.
    \"\"\"
    question: str
    top_k: Optional[int] = 5
    chunks: Optional[List[str]] = None          # deprecated, giữ cho tương thích
    chunk_headings: Optional[List[str]] = []    # deprecated, giữ cho tương thích

class RetrieveResponse(BaseModel):
    relevant_chunks: List[str]

# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def health_check():
    cached_count = len(_cache["chunks"])
    return {
        "status": "ok",
        "version": "3.0.0",
        "models": ["summarization", "bge-m3", "bge-reranker-v2-m3"],
        "cached_chunks": cached_count,
        "bm25_enabled": BM25_AVAILABLE,
    }


@app.post("/index", response_model=IndexResponse)
def api_index(req: IndexRequest):
    \"\"\"
    [MỚI] Nhận chunks từ backend, encode và lưu cache.
    Gọi 1 lần sau upload → retrieve nhanh cho mọi câu hỏi sau đó.
    \"\"\"
    if not req.chunks:
        raise HTTPException(status_code=400, detail="Danh sách chunks rỗng.")
    try:
        count = build_index(req.chunks, req.chunk_headings or [])
        return IndexResponse(
            indexed_count=count,
            message=f"Đã index thành công {count} chunks. Sẵn sàng trả lời câu hỏi."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi index: {str(e)}")


@app.post("/summarize", response_model=SummarizeResponse)
def api_summarize(req: SummarizeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản rỗng.")
    try:
        chunks = [c.strip() for c in req.text.split("|||") if c.strip()]
        if not chunks:
            chunks = [req.text]

        summaries = [summarize_text(chunk, req.max_length) for chunk in chunks]
        final_text = " ".join(summaries)

        if len(chunks) > 1 and len(final_text) > 500:
            final_summary = summarize_text(final_text, req.max_length)
        else:
            final_summary = final_text

        return SummarizeResponse(summary=final_summary, chunk_count=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def api_retrieve(req: RetrieveRequest):
    \"\"\"
    Retrieve dùng cache embeddings.
    Nếu gửi kèm chunks (backward compat v2), sẽ build index trước.
    \"\"\"
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được rỗng.")
    try:
        # Backward compatible: nếu gửi chunks thì index trước
        if req.chunks:
            build_index(req.chunks, req.chunk_headings or [])

        if not _cache["chunks"]:
            raise HTTPException(
                status_code=400,
                detail="Chưa có dữ liệu trong cache. Hãy gọi /index trước."
            )

        results = retrieve_from_cache(
            question=req.question,
            top_k=req.top_k,
        )
        return RetrieveResponse(relevant_chunks=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi retrieve: {str(e)}")
"""
