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
    model: Optional[str] = Field(
        default=None,
        description="Modelo LLM a usar (padrão: llama3.2:3b). Ex: 'mistral', 'gpt-4', 'claude-3-opus'",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="ID do usuário fazendo a requisição (para auditoria e governança)",
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Parâmetro top_p para amostragem do modelo (nucleus sampling)",
    )
    top_k: Optional[int] = Field(
        default=40,
        ge=1,
        le=100,
        description="Parâmetro top_k para amostragem do modelo",
    )
    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperatura para geração (0=determinístico, 2=criativo)",
    )


class SourceDocument(BaseModel):
    """Representa um documento de contexto usado na resposta."""
    text: str
    source: str
    source_label: str
    score: float
    metadata: dict


class InferenceMetrics(BaseModel):
    """Métricas de inferência e governança da resposta."""
    inference_time_seconds: float = Field(description="Tempo total de inferência em segundos")
    tokens_used: int = Field(description="Número estimado de tokens utilizados")
    chunks_retrieved: int = Field(description="Quantidade de chunks recuperados")
    collections_used: list[str] = Field(description="Collections do Milvus consultadas")
    user_id: Optional[str] = Field(description="ID do usuário da requisição")
    model_provider: str = Field(description="Provider do modelo (ollama, openai, anthropic, groq)")
    model_name: str = Field(description="Nome do modelo utilizado")
    top_p: float = Field(description="Parâmetro top_p utilizado")
    top_k: int = Field(description="Parâmetro top_k utilizado")
    temperature: float = Field(description="Temperatura utilizada")


class ChatResponse(BaseModel):
    """Resposta do endpoint POST /chat"""
    answer: str = Field(description="Resposta gerada pelo LLM")
    query: str = Field(description="Pergunta original")
    sources: list[SourceDocument] = Field(description="Documentos usados como contexto")
    model: str = Field(description="Modelo LLM utilizado")
    total_docs_retrieved: int = Field(description="Total de documentos recuperados dos 3 datasets")
    metrics: InferenceMetrics = Field(description="Métricas de inferência e governança")


class HealthResponse(BaseModel):
    """Resposta do endpoint GET /health"""
    status: str
    services: dict[str, bool]
    indexed_documents: dict[str, int]
