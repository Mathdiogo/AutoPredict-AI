# ============================================================
# API Schemas - Modelos de Request/Response
# ============================================================
# Pydantic valida automaticamente os dados recebidos pela API.
# Se o cliente mandar dados errados, a API retorna 422 automaticamente.
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Corpo da requisição POST /chat"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pergunta sobre manutenção ou diagnóstico automotivo",
        examples=["Quais são as causas mais comuns de superaquecimento do motor?"],
    )
    min_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Score mínimo de relevância dos documentos (0 a 1)",
    )


class SourceDocument(BaseModel):
    """Representa um documento de contexto usado na resposta."""
    text: str
    source: str
    source_label: str
    score: float
    metadata: dict


class ChatResponse(BaseModel):
    """Resposta do endpoint POST /chat"""
    answer: str = Field(description="Resposta gerada pelo LLM")
    query: str = Field(description="Pergunta original")
    sources: list[SourceDocument] = Field(description="Documentos usados como contexto")
    model: str = Field(description="Modelo LLM utilizado")
    total_docs_retrieved: int = Field(description="Total de documentos recuperados dos 3 datasets")


class HealthResponse(BaseModel):
    """Resposta do endpoint GET /health"""
    status: str
    services: dict[str, bool]
    indexed_documents: dict[str, int]
