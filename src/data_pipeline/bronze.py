# ============================================================
# Camada Bronze - Ingestão de Dados Brutos
# ============================================================
# CONCEITO (Medallion Architecture):
#
#   BRONZE = "assim como veio do Kaggle"
#   Não modificamos NADA no dado. Apenas subimos para o MinIO.
#   Isso garante que sempre podemos reprocessar a partir do original.
#
# O que esta camada faz:
#   1. Lê o CSV do disco local (./data/)
#   2. Faz upload para o bucket "bronze" no MinIO
#   3. Registra o evento no PostgreSQL
# ============================================================

import logging
import os
from pathlib import Path
from src.database.minio_client import MinIOClient
from src.database.postgres_client import PostgresClient
from src.config import get_settings

logger = logging.getLogger(__name__)

# Mapeamento: nome do dataset → arquivo CSV esperado em ./data/
DATASET_FILES = {
    "vehicle_maintenance": "vehicle_maintenance_data.csv",
    "car_predictive": "cars_hyundai.csv",
    "engine_fault": "engine_fault_detection_dataset.csv",
}


def ingest_to_bronze(data_dir: str = "/app/data") -> dict[str, bool]:
    """
    Lê os CSVs da pasta local e os envia para o bucket Bronze do MinIO.

    Args:
        data_dir: Diretório onde estão os CSVs baixados do Kaggle

    Returns:
        Dict com resultado por dataset: {"vehicle_maintenance": True, ...}
    """
    settings = get_settings()
    minio = MinIOClient()
    postgres = PostgresClient()
    results = {}

    for dataset_name, filename in DATASET_FILES.items():
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            logger.warning(
                f"Arquivo não encontrado: {file_path}\n"
                f"Baixe o dataset do Kaggle e coloque em {data_dir}/"
            )
            postgres.log_ingestion(
                dataset_name=dataset_name,
                layer="bronze",
                status="error",
                error_message=f"Arquivo {filename} não encontrado em {data_dir}",
            )
            results[dataset_name] = False
            continue

        logger.info(f"[Bronze] Ingerindo: {dataset_name} ({filename})")

        # Conta as linhas para o log (lê o arquivo)
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f) - 1  # -1 para o header

        # Upload para MinIO (bucket bronze)
        success = minio.upload_file(
            bucket=settings.minio_bucket_bronze,
            object_name=filename,
            file_path=str(file_path),
        )

        postgres.log_ingestion(
            dataset_name=dataset_name,
            layer="bronze",
            status="success" if success else "error",
            records_processed=line_count if success else 0,
        )

        results[dataset_name] = success
        if success:
            logger.info(f"[Bronze] ✓ {dataset_name}: {line_count} registros")

    return results
