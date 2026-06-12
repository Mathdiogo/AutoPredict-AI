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
    minio_bucket_governance: str = "governance"  # Documentação de governança

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
    ollama_model: str = "llama3.2:1b"  # Modelo mais leve para evitar crashes

    # --- Cloud LLMs (Opcional - para comparação) ---
    openai_api_key: str = ""  # Opcional: sk-...
    anthropic_api_key: str = ""  # Opcional: sk-ant-...
    groq_api_key: str = ""  # Opcional: gsk_... (GRATUITO!)
    
    # --- Model Pool (Modelos disponíveis) ---
    available_models: list[str] = [
        "llama3.2:1b",
        "llama3.2:3b", 
        "mistral:7b",
        "qwen2.5:3b",
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "groq-llama-3.3-70b"
    ]
    
    # --- Governança e Tokens ---
    max_tokens_per_request: int = 1000  # Limite máximo de tokens por requisição
    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_top_k: int = 40

    # --- MLflow ---
    # Dentro do container: http://mlflow:5000
    # Fora do container (localhost): http://localhost:5001
    mlflow_tracking_uri: str = "http://localhost:5001"

    # --- RAG ---
    # Quantos documentos buscar em CADA dataset por pergunta
    # Total de contexto = top_k_per_collection * 3 datasets
    # Maior valor = mais contexto para o LLM, mas mais lento
    top_k_per_collection: int = 5

    # --- Embeddings ---
    # Modelo que converte texto em vetores numéricos
    # paraphrase-multilingual-MiniLM-L12-v2:
    #   - 384 dimensões (compatível com schema Milvus existente)
    #   - Treinado em 50+ idiomas incluindo português
    #   - Muito superior ao all-MiniLM-L6-v2 para textos em português
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
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
