#!/usr/bin/env python3
# ============================================================
# Upload da Documentação de Governança para o MinIO
# ============================================================
# Envia os arquivos de docs/governance/ para o bucket
# "governance" no MinIO (Data Lake), conforme requisito acadêmico.
#
# Uso:
#   docker exec -w /app autopredict-api python -m src.data_pipeline.upload_governance_to_minio
# ============================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings
from src.database.minio_client import MinIOClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOVERNANCE_FILES = [
    "README.md",
    "bronze_layer.md",
    "silver_layer.md",
    "gold_layer.md",
    "SYSTEM_INFO.md",
]


def upload_governance_to_minio(docs_dir: Path | None = None) -> dict:
    """
    Faz upload da documentação de governança para o bucket MinIO.

    Estrutura no MinIO:
        governance/
        ├── README.md
        ├── bronze_layer.md
        ├── silver_layer.md
        ├── gold_layer.md
        ├── SYSTEM_INFO.md
        └── _manifest.json
    """
    settings = get_settings()
    docs_dir = docs_dir or (Path(__file__).parent.parent.parent / "docs" / "governance")
    bucket = settings.minio_bucket_governance
    prefix = "governance"

    if not docs_dir.exists():
        raise FileNotFoundError(f"Pasta de governança não encontrada: {docs_dir}")

    missing = [f for f in GOVERNANCE_FILES if not (docs_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Documentos faltando em {docs_dir}: {', '.join(missing)}")

    client = MinIOClient()
    uploaded = []
    failed = []

    for filename in GOVERNANCE_FILES:
        object_name = f"{prefix}/{filename}"
        file_path = docs_dir / filename
        if client.upload_file(bucket, object_name, str(file_path)):
            uploaded.append(object_name)
        else:
            failed.append(object_name)

    manifest = json.dumps(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "bucket": bucket,
            "prefix": f"{prefix}/",
            "files": [f for f in uploaded],
        },
        ensure_ascii=False,
        indent=2,
    )
    manifest_key = f"{prefix}/_manifest.json"
    if client.upload_text(bucket, manifest_key, manifest, content_type="application/json"):
        uploaded.append(manifest_key)
    else:
        failed.append(manifest_key)

    result = {
        "status": "success" if not failed else "partial",
        "bucket": bucket,
        "prefix": f"{prefix}/",
        "uploaded": uploaded,
        "failed": failed,
        "console_url": "http://localhost:9001",
    }

    logger.info("=" * 60)
    logger.info("Governança enviada ao MinIO")
    logger.info("  Bucket: %s", bucket)
    logger.info("  Pasta:  %s/", prefix)
    logger.info("  Arquivos: %d", len(uploaded))
    logger.info("  Console: http://localhost:9001 → Buckets → %s → %s/", bucket, prefix)
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    try:
        upload_governance_to_minio()
    except Exception as exc:
        logger.error("Falha no upload de governança: %s", exc, exc_info=True)
        raise SystemExit(1) from exc
