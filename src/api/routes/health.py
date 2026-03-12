# ============================================================
# Rota de Health Check
# ============================================================
# GET /health → Verifica se todos os serviços estão funcionando
# Útil para debugging e para o frontend saber se a API está pronta.
# ============================================================

import logging
from fastapi import APIRouter
from src.api.schemas.chat import HealthResponse
from src.database.milvus_client import MilvusClient
from src.database.postgres_client import PostgresClient
from src.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Verifica o status de todos os serviços.
    Retorna 200 se tudo OK, mesmo que algum serviço esteja com problema
    (o status de cada serviço é reportado no body).
    """
    settings = get_settings()

    # Verifica Milvus
    milvus_ok = False
    indexed_docs = {}
    try:
        milvus = MilvusClient()
        milvus_ok = milvus.ping()
        if milvus_ok:
            indexed_docs = {
                settings.milvus_collection_maintenance: milvus.get_count(settings.milvus_collection_maintenance),
                settings.milvus_collection_predictive: milvus.get_count(settings.milvus_collection_predictive),
                settings.milvus_collection_engine: milvus.get_count(settings.milvus_collection_engine),
            }
    except Exception as e:
        logger.warning(f"Health check - Milvus falhou: {e}")

    # Verifica PostgreSQL
    postgres_ok = False
    try:
        postgres = PostgresClient()
        postgres_ok = postgres.ping()
    except Exception as e:
        logger.warning(f"Health check - PostgreSQL falhou: {e}")

    # Verifica Ollama
    ollama_ok = False
    try:
        import requests
        resp = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception as e:
        logger.warning(f"Health check - Ollama falhou: {e}")

    overall = "healthy" if (milvus_ok and postgres_ok) else "degraded"

    return HealthResponse(
        status=overall,
        services={
            "milvus": milvus_ok,
            "postgres": postgres_ok,
            "ollama": ollama_ok,
        },
        indexed_documents=indexed_docs,
    )
