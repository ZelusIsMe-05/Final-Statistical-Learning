from typing import List, Optional

from pydantic import BaseModel


class TextUploadRequest(BaseModel):
    text: str


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class HealthResponse(BaseModel):
    status: str
    colab_api: str
    document_loaded: bool
    chunk_count: int


class AskResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    relevant_chunk_count: int
    has_context: bool
