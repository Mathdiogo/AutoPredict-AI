#!/usr/bin/env python3
# ============================================================
# Indexa Documento de Governança no Milvus
# ============================================================
# Este script adiciona o documento SYSTEM_INFO.md como uma
# collection especial no Milvus para que o sistema possa
# responder perguntas sobre si mesmo usando RAG.
# ============================================================

import sys
import logging
from pathlib import Path

# Adiciona src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.milvus_client import MilvusClient
from src.embeddings.embedder import get_embedder
from src.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """
    Divide o texto em chunks menores para melhor recuperação.
    
    Args:
        text: Texto completo para dividir
        max_chunk_size: Tamanho máximo de cada chunk em caracteres
        
    Returns:
        Lista de chunks de texto
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Se o parágrafo sozinho já é maior que max_chunk_size, divide por linhas
        if len(para) > max_chunk_size:
            lines = para.split('\n')
            for line in lines:
                if len(current_chunk) + len(line) + 1 > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    current_chunk += "\n" + line if current_chunk else line
        else:
            # Tenta adicionar o parágrafo ao chunk atual
            if len(current_chunk) + len(para) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
    
    # Adiciona o último chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def index_system_info():
    """Indexa o documento SYSTEM_INFO.md no Milvus."""
    
    logger.info("=" * 60)
    logger.info("Indexando Documento de Governança no Milvus")
    logger.info("=" * 60)
    
    # Lê o documento
    doc_path = Path(__file__).parent.parent / "docs" / "governance" / "SYSTEM_INFO.md"
    
    if not doc_path.exists():
        logger.error(f"Documento não encontrado: {doc_path}")
        return False
    
    logger.info(f"Lendo documento: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Divide em chunks
    chunks = chunk_text(content, max_chunk_size=500)
    logger.info(f"Documento dividido em {len(chunks)} chunks")
    
    # Conecta ao Milvus
    settings = get_settings()
    from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType
    
    # Nome da collection para informações do sistema
    collection_name = "system_info"
    
    # Cria collection se não existir
    if not utility.has_collection(collection_name):
        logger.info(f"Criando collection '{collection_name}'...")
        
        # Define schema (mesmo padrão das outras collections)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description="Informações do sistema AutoPredict AI")
        
        collection = Collection(name=collection_name, schema=schema)
        
        # Cria índice HNSW
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Collection '{collection_name}' criada com índice HNSW")
    else:
        logger.info(f"Collection '{collection_name}' já existe")
        # Limpa dados antigos
        logger.info("Removendo dados antigos...")
        collection = Collection(name=collection_name)
        collection.delete(expr="id >= 0")  # Deleta todos os registros
        collection.flush()
    
    # Gera embeddings
    logger.info("Gerando embeddings...")
    embedder = get_embedder()
    embeddings = embedder.embed_batch(chunks)  # Usa embed_batch do embedder atual
    
    # Prepara metadados
    import json
    metadata_list = []
    for i in range(len(chunks)):
        metadata_list.append(json.dumps({
            "file": "SYSTEM_INFO.md",
            "chunk_id": i,
            "total_chunks": len(chunks),
        }, ensure_ascii=False))
    
    # Insere no Milvus usando o formato do MilvusClient.insert
    logger.info(f"Inserindo {len(chunks)} documentos no Milvus...")
    collection = Collection(name=collection_name)
    
    data = [
        chunks,                           # campo "text"
        ["system_info"] * len(chunks),    # campo "source"
        metadata_list,                    # campo "metadata"
        embeddings,                       # campo "embedding"
    ]
    
    collection.insert(data)
    collection.flush()  # Força persistência
    
    logger.info("=" * 60)
    logger.info("✅ Documento de governança indexado com sucesso!")
    logger.info(f"   Collection: {collection_name}")
    logger.info(f"   Documentos: {len(entities)}")
    logger.info("=" * 60)
    logger.info("\nAgora o sistema pode responder perguntas como:")
    logger.info("  - Quem é você?")
    logger.info("  - Quais modelos foram treinados?")
    logger.info("  - Quais estratégias de governança você usa?")
    logger.info("  - Quais datasets você utiliza?")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = index_system_info()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Erro ao indexar documento: {e}", exc_info=True)
        sys.exit(1)
