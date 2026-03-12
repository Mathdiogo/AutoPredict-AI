# ============================================================
# AutoPredict AI - Configuração Central
# ============================================================
# Este arquivo lê as variáveis de ambiente do .env automaticamente.
# Em vez de usar os.getenv() espalhado pelo código, tudo vem daqui.
# ============================================================

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_user: str = "minioadmin"
    minio_password: str = "minioadmin123"
    minio_secure: bool = False  # True em produção com HTTPS

    # Nomes dos buckets (camadas Medallion)
    minio_bucket_bronze: str = "bronze"  # Dados brutos (CSV original)
    minio_bucket_silver: str = "silver"  # Dados limpos
    minio_bucket_gold: str = "gold"      # Dados prontos para RAG

    # --- PostgreSQL ---
    postgres_url: str = "postgresql://autopredict:autopredict123@localhost:5432/autopredict"

    # --- Milvus ---
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Nomes das coleções (uma por dataset = multi-collection RAG)
    milvus_collection_maintenance: str = "vehicle_maintenance"
    milvus_collection_predictive: str = "car_predictive"
    milvus_collection_engine: str = "engine_fault"

    # --- Ollama (LLM) ---
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # --- RAG ---
    # Quantos documentos buscar em CADA dataset por pergunta
    # Total de contexto = top_k_per_collection * 3 datasets
    top_k_per_collection: int = 3

    # --- Embeddings ---
    # Modelo que converte texto em vetores numéricos
    # all-MiniLM-L6-v2: leve, rápido, 384 dimensões, ótimo para português
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # Deve bater com o modelo acima

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignora variáveis do .env que não estão aqui


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna a instância de configuração (singleton com cache).
    Use assim em outros módulos:
        from src.config import get_settings
        settings = get_settings()
    """
    return Settings()
