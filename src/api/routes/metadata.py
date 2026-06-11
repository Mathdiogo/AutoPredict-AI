# ============================================================
# Rota de Metadata - Informações sobre Datasets e Sistema
# ============================================================
# GET /metadata → Retorna metadados dos datasets, embeddings e configurações RAG
# Útil para auditoria, debugging e compliance de governança.
# ============================================================

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.database.milvus_client import MilvusClient
from src.database.postgres_client import PostgresClient
from src.config import get_settings
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


class DatasetMetadata(BaseModel):
    """Metadados de um dataset específico."""
    collection_name: str
    display_name: str
    indexed_documents: int
    description: str


class EmbeddingConfig(BaseModel):
    """Configuração do modelo de embeddings."""
    model_name: str
    dimensions: int
    description: str


class RAGConfig(BaseModel):
    """Configuração do pipeline RAG."""
    top_k_per_collection: int
    total_context_docs: int
    llm_model: str
    embedding_model: str
    min_score_default: float


class SystemMetadata(BaseModel):
    """Metadados completos do sistema."""
    datasets: list[DatasetMetadata]
    embedding_config: EmbeddingConfig
    rag_config: RAGConfig
    total_indexed_documents: int
    system_status: str
    last_updated: str


@router.get("/metadata", response_model=SystemMetadata, tags=["Metadata"])
def get_metadata():
    """
    Retorna metadados completos do sistema AutoPredict AI.
    
    Inclui:
    - Informações sobre os 3 datasets indexados
    - Configuração de embeddings (modelo, dimensões)
    - Configuração do pipeline RAG (top_k, LLM, etc.)
    - Total de documentos indexados
    - Status do sistema
    
    **Casos de uso:**
    - Auditoria de dados (compliance, governança)
    - Debugging (verificar se dados foram indexados corretamente)
    - Monitoramento (quantos documentos estão disponíveis)
    - Documentação (quais modelos estão sendo usados)
    """
    logger.info("[API] GET /metadata")
    
    settings = get_settings()
    
    # ── Busca metadados dos datasets ──────────────────────────────────────
    datasets = []
    total_docs = 0
    system_status = "operational"
    
    try:
        milvus = MilvusClient()
        
        # Dataset 1: Vehicle Maintenance
        try:
            maintenance_count = milvus.get_count(settings.milvus_collection_maintenance)
            datasets.append(DatasetMetadata(
                collection_name=settings.milvus_collection_maintenance,
                display_name="📋 Histórico de Manutenção",
                indexed_documents=maintenance_count,
                description="Registros de manutenção preventiva e corretiva, histórico de serviços, peças trocadas e quilometragem de frotas comerciais (50.000 registros originais, 5.000 indexados)."
            ))
            total_docs += maintenance_count
        except Exception as e:
            logger.warning(f"Erro ao buscar metadata de {settings.milvus_collection_maintenance}: {e}")
            system_status = "degraded"
        
        # Dataset 2: Car Predictive (Sensors)
        try:
            predictive_count = milvus.get_count(settings.milvus_collection_predictive)
            datasets.append(DatasetMetadata(
                collection_name=settings.milvus_collection_predictive,
                display_name="📊 Dados de Sensores Preditivos",
                indexed_documents=predictive_count,
                description="Leituras de sensores automotivos: temperatura, vibração, pressão dos pneus, umidade. Indicadores de anomalias para manutenção preditiva (1.100 registros)."
            ))
            total_docs += predictive_count
        except Exception as e:
            logger.warning(f"Erro ao buscar metadata de {settings.milvus_collection_predictive}: {e}")
            system_status = "degraded"
        
        # Dataset 3: Engine Fault Detection
        try:
            engine_count = milvus.get_count(settings.milvus_collection_engine)
            datasets.append(DatasetMetadata(
                collection_name=settings.milvus_collection_engine,
                display_name="⚠️ Diagnóstico de Falhas de Motor",
                indexed_documents=engine_count,
                description="Dados de vibração, temperatura, RPM e condições de motor. Classificação multiclasse de estado do motor: normal, alerta, crítico (10.000 registros originais, 5.000 indexados)."
            ))
            total_docs += engine_count
        except Exception as e:
            logger.warning(f"Erro ao buscar metadata de {settings.milvus_collection_engine}: {e}")
            system_status = "degraded"
            
    except Exception as e:
        logger.error(f"Erro ao conectar ao Milvus: {e}")
        system_status = "error"
        raise HTTPException(
            status_code=503,
            detail="Não foi possível conectar ao banco vetorial (Milvus). Verifique se os serviços estão rodando."
        )
    
    # ── Configuração de Embeddings ────────────────────────────────────────
    embedding_config = EmbeddingConfig(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dim,
        description=(
            f"Modelo {settings.embedding_model} converte texto em vetores de "
            f"{settings.embedding_dim} dimensões. Treinado em 50+ idiomas incluindo português, "
            "otimizado para busca semântica. Vetores são normalizados e indexados no Milvus "
            "com índice IVF_FLAT e métrica de similaridade COSINE."
        )
    )
    
    # ── Configuração do RAG ──────────────────────────────────────────────
    rag_config = RAGConfig(
        top_k_per_collection=settings.top_k_per_collection,
        total_context_docs=settings.top_k_per_collection * 3,  # 3 datasets
        llm_model=settings.ollama_model,
        embedding_model=settings.embedding_model,
        min_score_default=0.25,
    )
    
    # ── Retorna metadados completos ───────────────────────────────────────
    return SystemMetadata(
        datasets=datasets,
        embedding_config=embedding_config,
        rag_config=rag_config,
        total_indexed_documents=total_docs,
        system_status=system_status,
        last_updated=datetime.now().isoformat(),
    )


@router.get("/metadata/datasets", tags=["Metadata"])
def get_datasets_metadata():
    """
    Retorna apenas metadados dos datasets (versão simplificada de /metadata).
    Útil para monitoramento rápido.
    """
    logger.info("[API] GET /metadata/datasets")
    
    settings = get_settings()
    datasets = []
    
    try:
        milvus = MilvusClient()
        
        for collection, display_name in [
            (settings.milvus_collection_maintenance, "Histórico de Manutenção"),
            (settings.milvus_collection_predictive, "Sensores Preditivos"),
            (settings.milvus_collection_engine, "Diagnóstico de Motor"),
        ]:
            try:
                count = milvus.get_count(collection)
                datasets.append({
                    "collection": collection,
                    "display_name": display_name,
                    "documents": count,
                })
            except Exception as e:
                logger.warning(f"Erro ao buscar {collection}: {e}")
                datasets.append({
                    "collection": collection,
                    "display_name": display_name,
                    "documents": 0,
                    "error": str(e),
                })
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erro ao conectar ao Milvus: {str(e)}")
    
    return {
        "datasets": datasets,
        "total": sum(d.get("documents", 0) for d in datasets),
    }


@router.get("/metadata/config", tags=["Metadata"])
def get_config_metadata():
    """
    Retorna apenas as configurações do sistema (RAG + Embeddings).
    Útil para verificar quais modelos estão sendo usados.
    """
    logger.info("[API] GET /metadata/config")
    
    settings = get_settings()
    
    return {
        "rag": {
            "llm_model": settings.ollama_model,
            "llm_url": settings.ollama_url,
            "top_k_per_collection": settings.top_k_per_collection,
            "total_context_docs": settings.top_k_per_collection * 3,
        },
        "embeddings": {
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dim,
        },
        "vector_db": {
            "host": settings.milvus_host,
            "port": settings.milvus_port,
            "collections": [
                settings.milvus_collection_maintenance,
                settings.milvus_collection_predictive,
                settings.milvus_collection_engine,
            ],
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
        },
    }
