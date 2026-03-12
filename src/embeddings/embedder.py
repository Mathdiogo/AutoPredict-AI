# ============================================================
# Embedder - Converte Texto em Vetores Numéricos
# ============================================================
# CONCEITO FUNDAMENTAL:
#   Um "embedding" é uma representação numérica de um texto.
#   Textos com significado similar ficam "próximos" no espaço vetorial.
#
#   Exemplo:
#     "motor superaquecido"  → [0.23, -0.71, 0.44, ...]
#     "temperatura do motor alta" → [0.21, -0.69, 0.47, ...]  ← próximo!
#     "banco de dados SQL"   → [-0.88, 0.12, -0.33, ...]      ← distante!
#
# O modelo all-MiniLM-L6-v2:
#   - Leve e rápido (roda em CPU sem problemas)
#   - Produz vetores de 384 dimensões
#   - Treinado em similaridade semântica multilíngue
#   - Na primeira execução, o modelo é baixado automaticamente (~80MB)
#
# Analogia: é como um dicionário que traduz palavras para coordenadas GPS.
# ============================================================

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton que gerencia o modelo de embedding.
    O modelo é carregado UMA vez na memória e reutilizado.
    Isso é importante pois carregar um modelo é lento.
    """

    _instance: "Embedder | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        """Carrega o modelo na primeira chamada (lazy loading)."""
        if self._model is None:
            settings = get_settings()
            logger.info(f"Carregando modelo de embedding: {settings.embedding_model}")
            # Na primeira execução, faz download automático do modelo
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("Modelo de embedding carregado com sucesso!")

    def embed_text(self, text: str) -> list[float]:
        """
        Converte um único texto em um vetor de 384 floats.
        Usado para gerar o embedding da pergunta do usuário.
        """
        self._load_model()
        # normalize_embeddings=True garante que os vetores têm comprimento 1
        # Isso é necessário para que a métrica COSINE funcione corretamente
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Converte uma lista de textos em vetores (mais eficiente que um por um).
        Usado durante o pipeline de ingestão dos datasets.

        Args:
            texts: Lista de textos para embedar
            batch_size: Quantos textos processar por vez (ajuste conforme RAM disponível)

        Returns:
            Lista de vetores (um por texto)
        """
        self._load_model()
        logger.info(f"Gerando embeddings para {len(texts)} textos...")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,  # Mostra barra de progresso no terminal
        )
        logger.info("Embeddings gerados com sucesso!")
        return embeddings.tolist()


def get_embedder() -> Embedder:
    """Retorna a instância singleton do Embedder."""
    return Embedder()
