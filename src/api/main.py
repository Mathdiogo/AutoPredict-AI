# ============================================================
# FastAPI - Aplicação Principal
# ============================================================
# Este arquivo cria e configura a aplicação FastAPI.
#
# FastAPI é um framework web moderno para Python que:
#   - Gera documentação automática (Swagger em /docs)
#   - Valida dados com Pydantic automaticamente
#   - Suporta async/await nativamente
#   - É extremamente rápido (baseado em Starlette + Pydantic)
#
# Acesse a documentação interativa em: http://localhost:8000/docs
# Lá você pode testar todos os endpoints sem precisar do frontend!
# ============================================================

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import chat, health

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Código que roda ao INICIAR e ao ENCERRAR a aplicação.
    É aqui que fazemos a inicialização "pesada" (conectar ao Milvus, etc.)
    """
    # ── Startup ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  AutoPredict AI - API iniciando...")
    logger.info("=" * 50)

    # Pré-inicializa o pipeline RAG (carrega o modelo de embedding)
    # Isso evita que a primeira pergunta demore muito
    try:
        from src.rag.pipeline import get_pipeline
        get_pipeline()
        logger.info("Pipeline RAG inicializado com sucesso!")
    except Exception as e:
        logger.warning(f"Pipeline RAG não pôde ser inicializado agora: {e}")
        logger.warning("O pipeline será inicializado na primeira requisição.")

    logger.info("API pronta! Acesse: http://localhost:8000/docs")
    logger.info("=" * 50)

    yield  # A aplicação roda aqui

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("API encerrando...")


# ── Criação da aplicação ──────────────────────────────────────
app = FastAPI(
    title="AutoPredict AI",
    description=(
        "API para diagnóstico preditivo automotivo usando RAG (Retrieval-Augmented Generation).\n\n"
        "Consulta simultaneamente 3 datasets de manutenção automotiva para responder perguntas "
        "sobre diagnóstico, manutenção preditiva e falhas de veículos."
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # Redoc (alternativa ao Swagger)
    lifespan=lifespan,
)

# ── CORS (permite que o frontend Gradio chame a API) ─────────
# Em produção, substitua "*" pela URL exata do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Registra as rotas ─────────────────────────────────────────
app.include_router(chat.router)
app.include_router(health.router)


# ── Rota raiz ─────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Endpoint raiz, útil para verificar se a API está no ar."""
    return {
        "message": "AutoPredict AI está rodando!",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
    }
