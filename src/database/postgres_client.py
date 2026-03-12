# ============================================================
# PostgreSQL Client - Banco de Metadados
# ============================================================
# Usamos o PostgreSQL para registrar O QUE foi processado e QUANDO.
# Não guardamos os dados brutos aqui (isso é função do MinIO e Milvus).
#
# Tabelas criadas:
#   ingestion_log → Histórico de cada execução do pipeline
#   documents     → Registro de cada documento indexado no Milvus
#
# Isso permite saber "o dataset X foi ingerido em Y" sem reprocessar.
# ============================================================

import logging
from datetime import datetime
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.orm import DeclarativeBase, Session
from src.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class IngestionLog(Base):
    """Registra cada execução do pipeline de ingestão."""
    __tablename__ = "ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False)
    layer = Column(String(10), nullable=False)        # bronze, silver ou gold
    status = Column(String(20), nullable=False)       # success, error
    records_processed = Column(Integer, default=0)
    error_message = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Registro de cada documento/chunk indexado no Milvus."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    milvus_id = Column(Integer, nullable=True)           # ID retornado pelo Milvus
    collection_name = Column(String(100), nullable=False) # em qual coleção Milvus
    dataset_source = Column(String(100), nullable=False)  # de qual dataset veio
    text_preview = Column(String(500), nullable=False)    # primeiros 500 chars do texto
    created_at = Column(DateTime, default=datetime.utcnow)


class PostgresClient:
    """
    Wrapper sobre SQLAlchemy para operações no PostgreSQL.
    Cria as tabelas automaticamente na primeira execução.
    """

    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,   # Verifica conexão antes de usar
            pool_size=5,
            max_overflow=10,
        )
        self._create_tables()

    def _create_tables(self):
        """Cria todas as tabelas se não existirem (safe: não apaga dados)."""
        Base.metadata.create_all(self.engine)
        logger.info("Tabelas do PostgreSQL verificadas/criadas")

    def log_ingestion(
        self,
        dataset_name: str,
        layer: str,
        status: str,
        records_processed: int = 0,
        error_message: str | None = None,
    ) -> int:
        """
        Registra uma execução do pipeline no log.
        Retorna o ID do registro criado.
        """
        with Session(self.engine) as session:
            log = IngestionLog(
                dataset_name=dataset_name,
                layer=layer,
                status=status,
                records_processed=records_processed,
                error_message=error_message,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log.id

    def log_document(
        self,
        collection_name: str,
        dataset_source: str,
        text_preview: str,
        milvus_id: int | None = None,
    ):
        """Registra um documento que foi indexado no Milvus."""
        with Session(self.engine) as session:
            doc = Document(
                milvus_id=milvus_id,
                collection_name=collection_name,
                dataset_source=dataset_source,
                text_preview=text_preview[:500],  # Limita a 500 chars
            )
            session.add(doc)
            session.commit()

    def get_ingestion_history(self, dataset_name: str | None = None) -> list[dict]:
        """Retorna o histórico de ingestões (útil para debug)."""
        with Session(self.engine) as session:
            query = session.query(IngestionLog)
            if dataset_name:
                query = query.filter(IngestionLog.dataset_name == dataset_name)
            logs = query.order_by(IngestionLog.created_at.desc()).limit(50).all()
            return [
                {
                    "id": log.id,
                    "dataset": log.dataset_name,
                    "layer": log.layer,
                    "status": log.status,
                    "records": log.records_processed,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ]

    def get_document_count(self, collection_name: str | None = None) -> int:
        """Retorna quantos documentos estão indexados."""
        with Session(self.engine) as session:
            query = session.query(Document)
            if collection_name:
                query = query.filter(Document.collection_name == collection_name)
            return query.count()

    def ping(self) -> bool:
        """Testa a conexão com o banco."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"PostgreSQL não está acessível: {e}")
            return False
