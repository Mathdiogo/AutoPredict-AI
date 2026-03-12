# ============================================================
# MinIO Client - Data Lake
# ============================================================
# MinIO é um servidor de armazenamento de objetos compatível com S3.
# Usamos ele como "Data Lake" para guardar os datasets em 3 camadas:
#
#   BRONZE → CSV bruto, exatamente como baixado do Kaggle
#   SILVER → CSV limpo (sem nulos, colunas padronizadas)
#   GOLD   → Dados processados, prontos para o RAG
#
# Analogia: É como um Google Drive/S3 que roda localmente no Docker.
# ============================================================

import io
import logging
from minio import Minio
from minio.error import S3Error
from src.config import get_settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    Wrapper sobre o cliente oficial do MinIO.
    Gerencia upload, download e listagem de arquivos nos buckets.
    """

    def __init__(self):
        settings = get_settings()
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_user,
            secret_key=settings.minio_password,
            secure=settings.minio_secure,
        )
        self.buckets = [
            settings.minio_bucket_bronze,
            settings.minio_bucket_silver,
            settings.minio_bucket_gold,
        ]
        self._ensure_buckets_exist()

    def _ensure_buckets_exist(self):
        """Cria os buckets Bronze, Silver e Gold se não existirem."""
        for bucket in self.buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"Bucket '{bucket}' criado no MinIO")

    def upload_file(self, bucket: str, object_name: str, file_path: str) -> bool:
        """
        Faz upload de um arquivo local para um bucket.

        Args:
            bucket: Nome do bucket (bronze, silver ou gold)
            object_name: Caminho dentro do bucket (ex: "vehicle_maintenance.csv")
            file_path: Caminho local do arquivo
        """
        try:
            self.client.fput_object(bucket, object_name, file_path)
            logger.info(f"Upload: {file_path} → {bucket}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Erro no upload para MinIO: {e}")
            return False

    def upload_dataframe(self, bucket: str, object_name: str, df_csv: str) -> bool:
        """
        Faz upload de um DataFrame já serializado como CSV (string).
        Útil para salvar dados processados sem criar arquivo temporário.
        """
        try:
            data = df_csv.encode("utf-8")
            self.client.put_object(
                bucket,
                object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type="text/csv",
            )
            logger.info(f"Upload DataFrame: → {bucket}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Erro no upload do DataFrame: {e}")
            return False

    def download_file(self, bucket: str, object_name: str, dest_path: str) -> bool:
        """
        Faz download de um objeto do bucket para um arquivo local.
        """
        try:
            self.client.fget_object(bucket, object_name, dest_path)
            logger.info(f"Download: {bucket}/{object_name} → {dest_path}")
            return True
        except S3Error as e:
            logger.error(f"Erro no download do MinIO: {e}")
            return False

    def read_csv_as_string(self, bucket: str, object_name: str) -> str | None:
        """
        Lê um CSV diretamente do bucket e retorna como string.
        Evita criar arquivos temporários em disco.
        """
        try:
            response = self.client.get_object(bucket, object_name)
            content = response.read().decode("utf-8")
            response.close()
            return content
        except S3Error as e:
            logger.error(f"Erro ao ler {bucket}/{object_name}: {e}")
            return None

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Lista todos os objetos em um bucket (com filtro opcional por prefixo)."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Erro ao listar {bucket}: {e}")
            return []

    def object_exists(self, bucket: str, object_name: str) -> bool:
        """Verifica se um objeto existe no bucket."""
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False
