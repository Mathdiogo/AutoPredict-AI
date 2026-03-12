# ============================================================
# Milvus Client - Banco Vetorial
# ============================================================
# Milvus é um banco de dados especializado em vetores numéricos.
# É a peça central do RAG: permite buscar, por SIMILARIDADE SEMÂNTICA,
# os documentos mais relevantes para uma determinada pergunta.
#
# COMO FUNCIONA:
#   1. Você tem um texto: "Motor superaquecendo com pressão baixa"
#   2. O embedder converte em vetor: [0.23, -0.71, 0.44, ... ] (384 números)
#   3. Milvus compara com todos os vetores da coleção
#   4. Retorna os K mais "próximos" (semanticamente similares)
#
# COLEÇÕES (uma por dataset):
#   - vehicle_maintenance  → Histórico de manutenção
#   - car_predictive       → Dados de sensores + falhas
#   - engine_fault         → Códigos de falha de motor
#
# Buscando nas 3 ao mesmo tempo = Multi-Collection RAG!
# ============================================================

import json
import logging
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from src.config import get_settings

logger = logging.getLogger(__name__)

# Schema compartilhado por todas as coleções
# Todos os datasets seguem a mesma estrutura no Milvus
VECTOR_DIM = 384  # Dimensão do modelo all-MiniLM-L6-v2


def _build_schema(description: str) -> CollectionSchema:
    """
    Define a estrutura de dados de uma coleção Milvus.

    Campos:
      id          → Identificador único (gerado automaticamente)
      text        → O texto original do chunk (para mostrar na resposta)
      source      → De qual dataset veio (ex: "vehicle_maintenance")
      metadata    → Informações extras em JSON (ex: ID do veículo, data, etc.)
      embedding   → O vetor numérico (é nesse campo que a busca acontece)
    """
    fields = [
      FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
      FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
      FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
      FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
      FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    return CollectionSchema(fields=fields, description=description, enable_dynamic_field=False)


class MilvusClient:
    """
    Wrapper sobre pymilvus para operações nas coleções de vetores.
    Gerencia criação de coleções, indexação e busca semântica.
    """

    def __init__(self):
      settings = get_settings()
      self.settings = settings
      self._connect()
      self._ensure_collections()

    def _connect(self):
      """Estabelece conexão com o Milvus."""
      s = self.settings
      connections.connect(
        alias="default",
        host=s.milvus_host,
        port=str(s.milvus_port),
      )
      logger.info(f"Conectado ao Milvus em {s.milvus_host}:{s.milvus_port}")

    def _ensure_collections(self):
      """Cria as 3 coleções se não existirem, com índice de busca."""
      collections_config = {
        self.settings.milvus_collection_maintenance: "Histórico de manutenção de veículos",
        self.settings.milvus_collection_predictive: "Dados preditivos de sensores automotivos",
        self.settings.milvus_collection_engine: "Falhas e diagnósticos de motor",
      }

      for name, description in collections_config.items():
        if not utility.has_collection(name):
          schema = _build_schema(description)
          collection = Collection(name=name, schema=schema)

          # HNSW é o índice recomendado para alta performance
          # M=16 = cada nó tem até 16 conexões no grafo
          # efConstruction=200 = qualidade do índice (mais alto = melhor mas mais lento pra criar)
          index_params = {
            "metric_type": "COSINE",   # Similaridade por ângulo entre vetores
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
          }
          collection.create_index(field_name="embedding", index_params=index_params)
          logger.info(f"Coleção '{name}' criada no Milvus com índice HNSW")
        else:
          logger.info(f"Coleção '{name}' já existe no Milvus")

    def _get_collection(self, name: str) -> Collection:
      """Retorna uma coleção carregada na memória (necessário para busca)."""
      collection = Collection(name)
      collection.load()
      return collection

    def insert(self, collection_name: str, texts: list[str], embeddings: list[list[float]], metadata_list: list[dict]) -> list[int]:
      """
      Insere documentos em uma coleção.

      Args:
        collection_name: Nome da coleção (use as constantes do config)
        texts: Lista de textos dos chunks
        embeddings: Lista de vetores (um por texto)
        metadata_list: Lista de dicionários com metadados extras

      Returns:
        Lista de IDs gerados pelo Milvus
      """
      collection = self._get_collection(collection_name)

      data = [
        texts,                                           # campo "text"
        [collection_name] * len(texts),                  # campo "source"
        [json.dumps(m, ensure_ascii=False) for m in metadata_list],  # campo "metadata"
        embeddings,                                      # campo "embedding"
      ]

      result = collection.insert(data)
      collection.flush()  # Garante que os dados foram gravados
      logger.info(f"Inseridos {len(texts)} documentos em '{collection_name}'")
      return result.primary_keys

    def search(self, collection_name: str, query_embedding: list[float], top_k: int = 3) -> list[dict]:
      """
      Busca os K documentos mais similares ao embedding da pergunta.

      Retorna lista de dicts com: text, source, metadata, score (0 a 1)
      Quanto maior o score, mais relevante é o documento.
      """
      collection = self._get_collection(collection_name)

      # ef = 100: balanço entre velocidade e precisão na busca
      search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

      results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "source", "metadata"],
      )

      documents = []
      for hits in results:  # results[0] = resultados para o primeiro (único) query
        for hit in hits:
          metadata = {}
          try:
            metadata = json.loads(hit.entity.get("metadata", "{}"))
          except json.JSONDecodeError:
            pass

          documents.append({
            "text": hit.entity.get("text", ""),
            "source": collection_name,
            "metadata": metadata,
            "score": float(hit.score),  # Score de similaridade COSINE (0 a 1)
          })

      return documents

    def get_count(self, collection_name: str) -> int:
      """Retorna quantos documentos existem em uma coleção."""
      collection = self._get_collection(collection_name)
      return collection.num_entities

    def drop_collection(self, collection_name: str):
      """
      Remove uma coleção inteira (APAGA TUDO!).
      Use apenas para reprocessar os dados do zero.
      """
      if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.warning(f"Coleção '{collection_name}' removida do Milvus!")

    def ping(self) -> bool:
      """Testa se o Milvus está acessível."""
      try:
        utility.list_collections()
        return True
      except Exception as e:
        logger.error(f"Milvus não está acessível: {e}")
        return False