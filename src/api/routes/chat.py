# ============================================================
# Rota de Chat - Endpoint Principal da API
# ============================================================
# POST /chat → Recebe pergunta, retorna resposta RAG
# GET  /chat/examples → Retorna exemplos de perguntas
# ============================================================

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas.chat import ChatRequest, ChatResponse, SourceDocument
from src.rag.pipeline import get_pipeline, RAGResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Endpoint principal do chatbot.

    Fluxo interno:
      1. Recebe a pergunta
      2. Busca documentos relevantes nos 3 datasets (Milvus)
      3. Envia pergunta + contexto para o LLM (Ollama)
      4. Retorna resposta + fontes usadas

    Exemplo de request:
    ```json
    {
      "question": "Quais falhas são mais comuns em motores a diesel?",
      "min_score": 0.25
    }
    ```
    """
    logger.info(f"[API] POST /chat - '{request.question[:60]}'")

    try:
        pipeline = get_pipeline()
        result: RAGResponse = pipeline.query(
            question=request.question,
            min_score=request.min_score,
        )

        # Converte os sources para o schema Pydantic
        sources = [
            SourceDocument(
                text=s["text"],
                source=s["source"],
                source_label=s["source_label"],
                score=s["score"],
                metadata=s["metadata"],
            )
            for s in result.sources
        ]

        return ChatResponse(
            answer=result.answer,
            query=result.query,
            sources=sources,
            model=result.model,
            total_docs_retrieved=result.total_docs_retrieved,
        )

    except Exception as e:
        logger.error(f"[API] Erro no chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get(
    "/chat/stream",
    tags=["Chat"],
    summary="Chat com streaming (resposta em tempo real)",
)
def chat_stream(question: str, min_score: float = 0.25):
    """
    Versão streaming do chat.
    Os tokens são enviados conforme são gerados pelo LLM.

    Use assim no frontend:
    ```js
    const resp = await fetch('/chat/stream?question=...')
    const reader = resp.body.getReader()
    // lê tokens um por um
    ```
    """
    if not question or len(question) < 3:
        raise HTTPException(status_code=422, detail="Pergunta muito curta")

    pipeline = get_pipeline()

    def generate():
        try:
            for token in pipeline.stream_query(question, min_score=min_score):
                yield token
        except Exception as e:
            logger.error(f"[API] Erro no streaming: {e}")
            yield f"\n[Erro: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/chat/examples", tags=["Chat"])
def get_examples():
    """
    Retorna exemplos de perguntas para demonstração.
    Útil para o frontend sugerir perguntas ao usuário.
    """
    return {
        "examples": [
            "Quais são as causas mais comuns de superaquecimento do motor?",
            "O que significa o código de falha P0300?",
            "Meu carro tem 80.000km, o que devo verificar preventivamente?",
            "Quando devo trocar o óleo do motor?",
            "Quais sensores indicam desgaste no sistema de freios?",
            "Como a pressão baixa do óleo afeta o motor?",
            "Quais falhas são mais comuns em veículos com mais de 100.000km?",
            "O que causa vibração excessiva no veículo?",
        ]
    }
