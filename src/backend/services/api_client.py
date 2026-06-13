import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8001").rstrip("/")
OLLAMA_REFINE_URL = os.getenv("OLLAMA_REFINE_URL", "").rstrip("/")
COLAB_TIMEOUT = httpx.Timeout(180.0, connect=30.0)
OLLAMA_TIMEOUT = httpx.Timeout(90.0, connect=15.0)


async def call_colab_index(chunks: list, chunk_headings: list | None = None) -> dict:
    """
    Gửi toàn bộ chunks lên Colab để encode và cache server-side.
    Chỉ cần gọi 1 lần sau khi upload document.
    Trả về dict: {"indexed_count": int, "message": str}
    Nếu Colab chưa có endpoint /index (version cũ), silently return.
    """
    if chunk_headings is None:
        chunk_headings = []

    try:
        async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
            resp = await client.post(
                f"{COLAB_API_URL}/index",
                json={
                    "chunks": chunks,
                    "chunk_headings": chunk_headings,
                },
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        # Colab cũ chưa có /index → bỏ qua, retrieve sẽ fallback gửi chunks
        if exc.response.status_code == 404:
            return {"indexed_count": 0, "message": "endpoint /index chưa tồn tại (Colab cũ)"}
        raise
    except Exception:
        # Không chặn upload nếu index thất bại
        return {"indexed_count": 0, "message": "index thất bại, sẽ fallback khi retrieve"}


async def call_colab_retrieve(
    question: str,
    chunks: list | None = None,
    chunk_headings: list | None = None,
    top_k: int = 5,
) -> list:
    """
    Retrieve chunks liên quan từ Colab.

    Với Colab v3 (đã gọi /index): chỉ gửi câu hỏi → payload nhỏ → nhanh.
    Với Colab v2 (chưa có /index): gửi kèm chunks như cũ (backward compat).
    """
    if chunk_headings is None:
        chunk_headings = []

    payload: dict = {
        "question": question,
        "top_k": top_k,
    }

    # Chỉ đính kèm chunks nếu được cung cấp (fallback cho Colab v2)
    if chunks:
        payload["chunks"] = chunks
        payload["chunk_headings"] = chunk_headings

    async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
        resp = await client.post(
            f"{COLAB_API_URL}/retrieve",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("relevant_chunks", [])


async def call_colab_summarize(text_chunk: str, max_length: int = 200) -> str:
    """Call the Colab model server summarization endpoint (single chunk)."""

    async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
        resp = await client.post(
            f"{COLAB_API_URL}/summarize",
            json={"text": text_chunk, "max_length": max_length},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary", "")


async def call_colab_summarize_batch(
    chunks: list[str],
    max_length: int = 200,
) -> list[str]:
    """
    Gửi tất cả chunks trong 1 request duy nhất lên Colab để tóm tắt batch.
    Nhanh hơn gọi N lần riêng lẻ vì giảm network round-trip qua Pinggy tunnel.
    Trả về list summaries tương ứng theo thứ tự chunks đầu vào.
    Nếu Colab chưa hỗ trợ /summarize_batch (endpoint cũ), fallback về N lần riêng lẻ.
    """
    if not chunks:
        return []

    try:
        async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
            resp = await client.post(
                f"{COLAB_API_URL}/summarize_batch",
                json={"chunks": chunks, "max_length": max_length},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("summaries", [])
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # Fallback: Colab cũ chưa có /summarize_batch → gọi song song
            semaphore = asyncio.Semaphore(5)

            async def _single(chunk: str) -> str:
                async with semaphore:
                    return await call_colab_summarize(chunk, max_length)

            results = await asyncio.gather(
                *[_single(c) for c in chunks], return_exceptions=True
            )
            return [r if isinstance(r, str) else "" for r in results]
        raise


async def call_ollama_answer(
    question: str,
    context: str,
    model: str = "gpt-oss:20b",
) -> str:
    """
    Ask Ollama to turn retrieved context into a natural-language answer.

    The prompt keeps the answer grounded in the supplied context and preserves
    Vietnamese user-facing behavior.
    """

    if not OLLAMA_REFINE_URL:
        return "Ollama not configured"

    prompt = (
        "Bạn là một trợ lý AI trung thực và lịch sự. "
        "Hãy trả lời câu hỏi dựa hoàn toàn trên NGỮ CẢNH (context) được cung cấp. "
        "KHÔNG được thêm thông tin ngoài ngữ cảnh, nếu không tìm thấy trả về: 'Không tìm thấy thông tin'.\n\n"
        "LUẬT:\n"
        "1) Chỉ sử dụng thông tin trong NGỮ CẢNH.\n"
        "2) Nếu thông tin không đủ để trả lời, trả về chính xác: "
        "'Không tìm thấy thông tin liên quan trong tài liệu'.\n"
        "3) Trả lời ngắn gọn, đầy đủ câu, không lặp lại câu hỏi.\n\n"
        "---\n"
        f"NGỮ CẢNH:\n{context}\n---\n"
        f"CÂU HỎI: {question}\n\nHãy trả lời:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 256,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_REFINE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
