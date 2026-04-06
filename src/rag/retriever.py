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

import numpy as np

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

    def _mmr_rerank(
        self,
        query_embedding: list[float],
        docs: list["RetrievedDocument"],
        k: int,
        lambda_: float = 0.65,
    ) -> list["RetrievedDocument"]:
        """
        Maximum Marginal Relevance: seleciona k documentos que maximizam
        simultaneamente relevância com a query E diversidade entre si.

        lambda_=0.65 → 65% relevância + 35% diversidade.
        """
        if len(docs) <= k:
            return docs

        # Re-embeda os textos candidatos para cálculo de similaridade interna
        texts = [d.text for d in docs]
        doc_embs = np.array(self.embedder.embed_batch(texts))
        q_emb = np.array(query_embedding)

        selected: list[int] = []
        remaining = list(range(len(docs)))

        while len(selected) < k and remaining:
            best_idx, best_score = None, -float("inf")
            for idx in remaining:
                relevance = float(np.dot(q_emb, doc_embs[idx]))
                if not selected:
                    mmr_score = relevance
                else:
                    sims = np.dot(doc_embs[selected], doc_embs[idx])
                    mmr_score = lambda_ * relevance - (1 - lambda_) * float(np.max(sims))
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [docs[i] for i in selected]

    def retrieve(self, query: str, top_k_per_collection: int | None = None) -> list["RetrievedDocument"]:
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

        logger.info(f"[Retriever] Query: '{query[:80]}...' (buscando k={k} por coleção)")
        query_embedding = self.embedder.embed_text(query)

        # Busca o dobro de candidatos por coleção para alimentar o MMR
        candidate_k = k * 2

        collections = [
            settings.milvus_collection_maintenance,
            settings.milvus_collection_predictive,
            settings.milvus_collection_engine,
        ]

        all_candidates: list[RetrievedDocument] = []

        for collection_name in collections:
            try:
                results = self.milvus.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    top_k=candidate_k,
                )

                for doc in results:
                    all_candidates.append(
                        RetrievedDocument(
                            text=doc["text"],
                            source=collection_name,
                            source_label=SOURCE_LABELS.get(collection_name, collection_name),
                            metadata=doc.get("metadata", {}),
                            score=doc["score"],
                        )
                    )

                logger.info(f"[Retriever] {collection_name}: {len(results)} candidatos")

            except Exception as e:
                logger.error(f"[Retriever] Erro ao buscar em '{collection_name}': {e}")

        if not all_candidates:
            return []

        # Aplica MMR para k resultados por coleção com diversidade garantida
        # (total = k * n_collections documentos finais)
        target_k = k * len(collections)
        mmr_results = self._mmr_rerank(query_embedding, all_candidates, k=target_k)

        logger.info(
            f"[Retriever] MMR: {len(all_candidates)} candidatos → {len(mmr_results)} selecionados"
        )
        return mmr_results

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
