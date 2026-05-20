import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

load_dotenv()

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8001").rstrip("/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log backend startup and shutdown lifecycle events."""

    print("✅ Backend RAG đã khởi động!")
    print(f"   Model Server: {COLAB_API_URL}")
    yield
    print("🔴 Backend đang tắt...")


app = FastAPI(
    title="RAG Backend API",
    description="Backend điều phối Summarization + Short QA cho hệ thống RAG tiếng Việt",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        reload=True,
        log_level="info",
    )
