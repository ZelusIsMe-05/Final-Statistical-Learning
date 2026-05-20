import asyncio
import os
from collections import OrderedDict

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile

from core.schemas import AskResponse, HealthResponse, QuestionRequest, TextUploadRequest
from core.store import document_store
from services.api_client import call_colab_retrieve, call_colab_summarize, call_ollama_answer
from services.document_processor import process_document
from utils.grounding import is_answer_grounded

load_dotenv()
COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8001").rstrip("/")

router = APIRouter()


def prepend_bullet_prefix(summary: str, bullet: str) -> str:
    """Put the source chunk's list marker back before its summary."""

    clean_summary = summary.strip()
    clean_bullet = (bullet or "").strip()
    if not clean_bullet:
        return clean_summary
    if clean_summary.startswith(clean_bullet):
        return clean_summary
    return f"{clean_bullet} {clean_summary}"


def update_loaded_document(result: dict) -> None:
    """Persist processed document data in the in-memory document store."""

    document_store.update({
        "full_text": result["full_text"],
        "chunks": result["chunks"],
        "chunk_headings": result.get("chunk_headings", []),
        "chunk_bullets": result.get("chunk_bullets", []),
        "chunk_count": result["chunk_count"],
        "source": result["source"],
        "summary": "",
    })


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return backend and model-server health information."""

    colab_ok = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{COLAB_API_URL}/")
            colab_ok = response.status_code == 200
    except Exception:
        colab_ok = False

    return HealthResponse(
        status="ok" if colab_ok else "colab_offline",
        colab_api=COLAB_API_URL,
        document_loaded=bool(document_store["full_text"]),
        chunk_count=document_store["chunk_count"],
    )


@router.post("/upload/text")
async def upload_text(request: TextUploadRequest):
    """Load raw text, process it, and store it for summarization or QA."""

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được rỗng.")

    result = process_document(content=request.text)
    update_loaded_document(result)

    preview = result["full_text"]
    return {
        "message": "Đã tải và xử lý văn bản thành công.",
        "chunk_count": result["chunk_count"],
        "source": "text",
        "preview": preview[:500] + "..." if len(preview) > 500 else preview,
    }


@router.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Load a PDF file, extract text, process it, and store it."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File PDF rỗng.")

    try:
        result = process_document(pdf_bytes=pdf_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc PDF: {str(exc)}") from exc

    if not result["full_text"]:
        raise HTTPException(
            status_code=422,
            detail="Không trích xuất được text từ PDF. PDF có thể chỉ chứa ảnh (scanned).",
        )

    update_loaded_document(result)

    preview = result["full_text"]
    return {
        "message": f"Đã tải PDF '{file.filename}' thành công.",
        "chunk_count": result["chunk_count"],
        "source": "pdf",
        "preview": preview[:500] + "..." if len(preview) > 500 else preview,
    }


@router.post("/summarize")
async def summarize_document():
    """Summarize the currently loaded document by chunk and heading group."""

    if not document_store["chunks"]:
        raise HTTPException(
            status_code=400,
            detail="Chưa có tài liệu. Vui lòng tải lên PDF hoặc văn bản trước.",
        )

    chunks = document_store["chunks"]
    headings = document_store.get("chunk_headings", [])
    bullets = document_store.get("chunk_bullets", [])

    try:
        if len(chunks) == 1:
            summary = await call_colab_summarize(chunks[0])
            summary = prepend_bullet_prefix(summary, bullets[0] if bullets else "")
            label = headings[0] if headings else "Nội dung"
            summary = f"{label}\n{summary}"
        else:
            semaphore = asyncio.Semaphore(5)

            async def summarize_chunk(chunk: str):
                async with semaphore:
                    return await call_colab_summarize(chunk)

            tasks = [summarize_chunk(chunk) for chunk in chunks]
            partial_summaries = await asyncio.gather(*tasks, return_exceptions=True)

            valid_pairs = [
                (
                    headings[i] if i < len(headings) else f"Phần {i + 1}",
                    prepend_bullet_prefix(summary, bullets[i] if i < len(bullets) else ""),
                )
                for i, summary in enumerate(partial_summaries)
                if isinstance(summary, str) and summary.strip()
            ]
            if not valid_pairs:
                raise HTTPException(
                    status_code=500,
                    detail="Tất cả các chunk đều thất bại khi tóm tắt.",
                )

            grouped = OrderedDict()
            for label, body in valid_pairs:
                grouped.setdefault(label, []).append(body.strip())

            parts = []
            for label, bodies in grouped.items():
                parts.append(label + "\n" + "\n".join(bodies))

            summary = "\n\n".join(parts)

        document_store["summary"] = summary
        return {
            "summary": summary,
            "chunk_count": len(chunks),
            "source": document_store["source"],
        }

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối đến Model Server ({COLAB_API_URL}).",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Model Server timeout. Thử lại sau.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: QuestionRequest):
    """Run the retrieve-then-answer QA workflow."""

    if not document_store["chunks"]:
        raise HTTPException(
            status_code=400,
            detail="Chưa có tài liệu. Vui lòng tải lên PDF hoặc văn bản trước.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được rỗng.")

    try:
        relevant_chunks = await call_colab_retrieve(
            question=request.question,
            chunks=document_store["chunks"],
            chunk_headings=document_store.get("chunk_headings", []),
            top_k=request.top_k,
        )

        if not relevant_chunks:
            return AskResponse(
                answer="Không tìm thấy thông tin liên quan trong tài liệu.",
                relevant_chunks=[],
                relevant_chunk_count=0,
                has_context=False,
            )

        context = "\n\n".join(relevant_chunks)
        refined_answer = await call_ollama_answer(
            question=request.question,
            context=context,
        )
        has_context = is_answer_grounded(refined_answer, context, threshold=0.2)

        return AskResponse(
            answer=refined_answer,
            relevant_chunks=relevant_chunks,
            relevant_chunk_count=len(relevant_chunks),
            has_context=has_context,
        )

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối đến Model Server ({COLAB_API_URL}).",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Model Server timeout. Thử lại sau.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(exc)}") from exc


@router.delete("/document")
async def clear_document():
    """Clear the current in-memory document."""

    document_store.update({
        "full_text": "",
        "chunks": [],
        "chunk_headings": [],
        "chunk_bullets": [],
        "chunk_count": 0,
        "source": "",
        "summary": "",
    })
    return {"message": "Đã xóa tài liệu."}


@router.get("/document/info")
async def get_document_info():
    """Return metadata for the current in-memory document."""

    text = document_store["full_text"]
    return {
        "has_document": bool(text),
        "chunk_count": document_store["chunk_count"],
        "source": document_store["source"],
        "has_summary": bool(document_store["summary"]),
        "text_length": len(text),
        "preview": text[:300] + "..." if len(text) > 300 else text,
    }
