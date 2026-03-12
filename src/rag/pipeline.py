# ============================================================
# RAG Pipeline - Orquestrador Principal
# ============================================================
# Este é o ponto de entrada do sistema RAG.
# Junta o Retriever (busca) + Generator (resposta).
#
# FLUXO COMPLETO:
#
#   Usuário: "Quais falhas são comuns em veículos com mais de 100.000km?"
#        │
#        ▼
#   [1] pipeline.query(pergunta)
#        │
#        ▼
#   [2] Retriever.retrieve(pergunta)
#   ├── embed(pergunta) → [0.23, -0.71, ...]
#   ├── milvus.search("vehicle_maintenance", vetor, k=3)
#   ├── milvus.search("car_predictive", vetor, k=3)
#   └── milvus.search("engine_fault", vetor, k=3)
#        │
#        ▼   9 documentos relevantes dos 3 datasets
#        │
#   [3] Generator.generate(pergunta, documentos)
#   └── Monta prompt com contexto → Envia para Ollama → Resposta
#        │
#        ▼
#   Resposta fundamentada nos dados reais!
# ============================================================

import logging
from dataclasses import dataclass, field
from src.rag.retriever import Retriever, RetrievedDocument
from src.rag.generator import Generator, GeneratorResponse

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Resposta completa do pipeline RAG, pronta para a API."""
    answer: str
    query: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    total_docs_retrieved: int = 0

    @classmethod
    def from_generator_response(cls, gen_response: GeneratorResponse) -> "RAGResponse":
        """Converte GeneratorResponse para o formato da API."""
        sources = []
        for doc in gen_response.sources:
            sources.append({
                "text": doc.text[:300],  # Preview do texto
                "source": doc.source,
                "source_label": doc.source_label,
                "score": round(doc.score, 4),
                "metadata": doc.metadata,
            })

        return cls(
            answer=gen_response.answer,
            query=gen_response.query,
            sources=sources,
            model=gen_response.model_used,
            total_docs_retrieved=len(gen_response.sources),
        )


class RAGPipeline:
    """
    Orquestra o pipeline completo: Retrieval → Generation.
    Esta classe é usada pela FastAPI e pelo frontend Gradio.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question: str, min_score: float = 0.25) -> RAGResponse:
        """
        Responde uma pergunta usando RAG multi-dataset.

        Args:
            question: Pergunta em texto natural
            min_score: Score mínimo de relevância para incluir documentos

        Returns:
            RAGResponse com a resposta e os documentos usados
        """
        logger.info(f"[RAG] Nova query: '{question[:80]}'")

        # Passo 1: Recupera documentos relevantes dos 3 datasets
        documents = self.retriever.retrieve_with_threshold(
            query=question,
            min_score=min_score,
        )

        if not documents:
            logger.warning("[RAG] Nenhum documento recuperado. Respondendo sem contexto.")
            # Se não encontrou nada, ainda tenta responder (o LLM usará seu conhecimento geral)
            documents = self.retriever.retrieve(question)

        # Passo 2: Gera resposta com o LLM
        gen_response = self.generator.generate(
            query=question,
            documents=documents,
        )

        # Converte para o formato da API
        result = RAGResponse.from_generator_response(gen_response)
        logger.info(f"[RAG] Resposta gerada ({result.total_docs_retrieved} docs de contexto)")
        return result

    def stream_query(self, question: str, min_score: float = 0.25):
        """
        Versão com streaming: retorna tokens conforme são gerados.
        Ideal para o Gradio mostrar a resposta "digitando".

        Usage:
            for token in pipeline.stream_query("Meu motor esquenta muito"):
                print(token, end="")
        """
        documents = self.retriever.retrieve_with_threshold(
            query=question,
            min_score=min_score,
        )

        if not documents:
            documents = self.retriever.retrieve(question)

        yield from self.generator.stream_generate(
            query=question,
            documents=documents,
        )


# Singleton do pipeline (inicializar é lento, reutilizamos)
_pipeline_instance: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Retorna a instância singleton do RAGPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("[RAG] Inicializando pipeline...")
        _pipeline_instance = RAGPipeline()
        logger.info("[RAG] Pipeline pronto!")
    return _pipeline_instance
