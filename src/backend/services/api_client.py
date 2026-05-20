import os

import httpx
from dotenv import load_dotenv

load_dotenv()

COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8001").rstrip("/")
OLLAMA_REFINE_URL = os.getenv("OLLAMA_REFINE_URL", "").rstrip("/")
COLAB_TIMEOUT = httpx.Timeout(180.0, connect=30.0)
OLLAMA_TIMEOUT = httpx.Timeout(90.0, connect=15.0)


async def call_colab_summarize(text_chunk: str, max_length: int = 200) -> str:
    """Call the Colab model server summarization endpoint."""

    async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
        resp = await client.post(
            f"{COLAB_API_URL}/summarize",
            json={"text": text_chunk, "max_length": max_length},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary", "")


async def call_colab_retrieve(
    question: str,
    chunks: list,
    chunk_headings: list | None = None,
    top_k: int = 3,
) -> list:
    """Call the Colab model server retrieval endpoint."""

    if chunk_headings is None:
        chunk_headings = []

    async with httpx.AsyncClient(timeout=COLAB_TIMEOUT) as client:
        resp = await client.post(
            f"{COLAB_API_URL}/retrieve",
            json={
                "question": question,
                "chunks": chunks,
                "chunk_headings": chunk_headings,
                "top_k": top_k,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("relevant_chunks", [])


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
