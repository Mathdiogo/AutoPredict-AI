# ============================================================
# RAG Retriever - Busca Multi-Dataset no Milvus
# ============================================================
# Esta é a parte do "R" no RAG (Retrieval).
#
# O que acontece aqui:
#   1. Recebe a pergunta do usuário como texto
#   2. Converte a pergunta em vetor (embedding)
#   3. Busca simultaneamente nas 3 coleções do Milvus
#   4. Retorna os documentos mais relevantes de cada dataset
#   5. Os resultados ficam identificados por fonte para o LLM
#
# CONCEITO - Por que buscar nos 3 datasets?
#   Pergunta: "Motor superaquecendo depois de 80.000km"
#
#   → Dataset 1 (Manutenção): "Veículo com 75.000km: termostato
#     substituído por superaquecimento recorrente"
#     (histórico similar!)
#
#   → Dataset 2 (Preditivo): "Temp: 108°C, Pressão óleo: 1.8bar
#     - FALHA DETECTADA" (padrão de sensor que indica o problema)
#
#   → Dataset 3 (Falhas): "P0217 - Temperatura do motor excessiva.
#     Verificar termostato, bomba d'água e radiador"
#     (diagnóstico técnico)
#
# Com os 3 contextos, o LLM dá uma resposta muito mais completa!
# ============================================================

import logging
from dataclasses import dataclass
from src.database.milvus_client import MilvusClient
from src.embeddings.embedder import get_embedder
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Representa um documento recuperado do Milvus."""
    text: str            # Texto descritivo do chunk
    source: str          # Nome da coleção Milvus (ex: "vehicle_maintenance")
    source_label: str    # Nome amigável para exibição (ex: "Histórico de Manutenção")
    metadata: dict       # Metadados extras (vehicle_id, fault_code, etc.)
    score: float         # Score de similaridade (0 a 1, maior = mais relevante)


# Nomes amigáveis para cada coleção
SOURCE_LABELS = {
    "vehicle_maintenance": "📋 Histórico de Manutenção",
    "car_predictive": "📊 Dados de Sensores Preditivos",
    "engine_fault": "⚠️ Diagnóstico de Falhas",
}


class Retriever:
    """
    Realiza busca semântica multi-coleção no Milvus.
    """

    def __init__(self):
        self.milvus = MilvusClient()
        self.embedder = get_embedder()
        self.settings = get_settings()

    def retrieve(self, query: str, top_k_per_collection: int | None = None) -> list[RetrievedDocument]:
        """
        Busca documentos relevantes nos 3 datasets para uma pergunta.

        Args:
            query: A pergunta do usuário em texto natural
            top_k_per_collection: Quantos docs buscar por dataset.
                                  None = usa o valor do config.

        Returns:
            Lista de documentos ordenados por relevância,
            vindos dos 3 datasets.
        """
        settings = self.settings
        k = top_k_per_collection or settings.top_k_per_collection

        # Converte a pergunta em vetor
        logger.info(f"[Retriever] Query: '{query[:80]}...' (buscando k={k} por coleção)")
        query_embedding = self.embedder.embed_text(query)

        # Busca nas 3 coleções em sequência
        # (pymilvus não é async, então não dá pra paralelizar nativamente)
        collections = [
            settings.milvus_collection_maintenance,
            settings.milvus_collection_predictive,
            settings.milvus_collection_engine,
        ]

        all_documents: list[RetrievedDocument] = []

        for collection_name in collections:
            try:
                results = self.milvus.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    top_k=k,
                )

                for doc in results:
                    all_documents.append(
                        RetrievedDocument(
                            text=doc["text"],
                            source=collection_name,
                            source_label=SOURCE_LABELS.get(collection_name, collection_name),
                            metadata=doc.get("metadata", {}),
                            score=doc["score"],
                        )
                    )

                logger.info(f"[Retriever] {collection_name}: {len(results)} docs encontrados")

            except Exception as e:
                logger.error(f"[Retriever] Erro ao buscar em '{collection_name}': {e}")
                # Continua com os outros datasets mesmo se um falhar

        # Ordena todos os resultados por score (mais relevante primeiro)
        all_documents.sort(key=lambda d: d.score, reverse=True)

        logger.info(f"[Retriever] Total: {len(all_documents)} documentos relevantes encontrados")
        return all_documents

    def retrieve_with_threshold(self, query: str, min_score: float = 0.3) -> list[RetrievedDocument]:
        """
        Busca com filtro de score mínimo.
        Documentos abaixo do threshold são descartados (não são relevantes).

        Args:
            min_score: Score mínimo de similaridade (0 a 1)
        """
        docs = self.retrieve(query)
        filtered = [d for d in docs if d.score >= min_score]

        if not filtered:
            logger.warning(
                f"[Retriever] Nenhum documento acima do threshold {min_score}. "
                f"Usando todos os {len(docs)} resultados."
            )
            return docs  # Retorna todos mesmo assim para não deixar o LLM sem contexto

        return filtered
